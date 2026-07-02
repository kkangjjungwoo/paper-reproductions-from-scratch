from dataclasses import dataclass
from types import NoneType
import torch.nn as nn
import torch
import math
from typing import Tuple


class CustomTokenEmbedding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(
            torch.empty(vocab_size, hidden_size, dtype=dtype, device=device)
        )
        """
        torch.empty(vocab_size, hidden_size, ...)
        (vocab_size, hidden_size) 크기의 빈 텐서를 만듭니다.
        메모리만 잡고, 값은 초기화하지 않습니다 (쓰레기 값).
        나중에 forward에서 쓰기 전에 보통 nn.init 등으로 초기화합니다.
        Parameter는 파라미터로
        """
        with torch.no_grad():
            nn.init.normal_(self.weight, mean=0.0, std=0.02)
        """
        임베딩 정의에서 autograd 키면 이게 다 오버헤드인가?
        좋은 질문입니다. 결론부터 말하면:
        네, 초기화 단계에서 autograd를 켜 두면 그 구간은 전부 “쓸데없는 기록”에 가깝습니다.
        다만 한 번만 돌고, 학습 루프랑 비교하면 오버헤드는 거의 무시할 수준입니다.
        """

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.embedding(self.weight, input_ids)
        """
        shape: [batch_size, seq_len]
        의미
        batch_size
        한 번에 처리하는 문장(시퀀스) 개수
        seq_len
        문장 하나당 토큰 개수
        예:

        input_ids = [
            [101, 2054, 2003, 102],   # 문장 1 (토큰 4개)
            [101, 1045, 2293, 102],   # 문장 2 (토큰 4개)
        ]
        # shape: (2, 4)
        # batch_size=2, seq_len=4
        batch_size = 2 → 문장 2개
        seq_len = 4 → 각 문장이 토큰 4개
        반환값
        타입: torch.Tensor (실수)
        shape: indices shape + (hidden_size,)
        예:

        indices.shape  # (2, 4)
        weight.shape   # (50000, 768) 단어 50000개, 각 차원 768
        out = torch.embedding(weight, indices) 각 토큰마다 768차원 부여해주는 함수
        out.shape      # (2, 4, 768)
        indices에 있던 각 정수마다 hidden_size 길이 벡터가 붙습니다.
"""


class CustomRotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_dim: int,
        base: float = 10000.0,
        dtype: torch.dtype = torch.float32,
        initial_context_length: int = 4096,
        scaling_factor: float = 1.0,
        ntk_alpha: float = 1.0,
        ntk_beta: float = 32.0,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.dtype = dtype
        self.initial_context_length = initial_context_length
        self.scaling_factor = scaling_factor
        self.ntk_alpha = ntk_alpha
        self.ntk_beta = ntk_beta
        self.device = device

    def _compute_concentration_and_inv_freq(self) -> Tuple[float, torch.Tensor]:

        freq = self.base ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=self.device)
            / self.head_dim
        )

        if self.scaling_factor > 1.0:
            concentration = 0.1 * math.log(self.scaling_factor) + 1.0
            d_half = self.head_dim / 2
            low = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_beta * 2 * math.pi))
            ) / math.log(self.base)
            high = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_alpha * 2 * math.pi))
            ) / math.log(self.base)

            interpolation = 1.0 / (self.scaling_factor * freq)
            extrapolation = 1.0 / freq

            ramp = (
                torch.arange(d_half, dtype=torch.float32, device=freq.device) - low
            ) / (high - low)
            mask = 1 - ramp.clamp(0, 1)
            inv_freq = interpolation * (1 - mask) + extrapolation * mask
        else:
            concentration = 1.0
            inv_freq = 1.0 / freq
        return concentration, inv_freq

    def _compute_cos_sin(self, num_tokens: int) -> Tuple[torch.Tensor, torch.Tensor]:

        concentration, inv_freq = self._compute_concentration_and_inv_freq()
        positions = torch.arange(num_tokens, dtype=torch.float32, device=self.device)
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        cos = torch.cos(freqs) * concentration
        sin = torch.sin(freqs) * concentration
        return cos, sin

    def forward(self, num_tokens: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cos, sin = self._compute_cos_sin(num_tokens)
        return cos.unsqueeze(0).to(self.dtype), sin.unsqueeze(0).to(self.dtype)

    # [nn.Module 상속 이유 요약]
    # 1. .to("cuda") 호출 시 하위 모듈과 내부 텐서들이 자동으로 GPU로 일괄 이동됨.
    # 2. register_buffer 사용 시 파라미터가 없어도 state_dict에 포함되어 안전하게 저장/로드됨.
    # 3. PyTorch 시스템(Hook, Autograd 등)에 등록되어 디버깅과 모델 조립이 원활해짐.


def apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)

    x1, x2 = torch.chunk(x, 2, dim=-1)

    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin

    return torch.cat((o1, o2), dim=-1)


class CustomCasualMask:
    @staticmethod
    def create_casual_mask(
        seq_len: int,
        sliding_window: int | None = None,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        mask = torch.zeros(seq_len, seq_len, dtype=dtype, device=device)
        future_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1
        )
        mask = mask.masked_fill(future_mask, float("-inf"))

        if sliding_window is not None and sliding_window > 0:
            past_mask = torch.tril(
                torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
                diagonal=-sliding_window,
            )
            mask = mask.masked_fill(past_mask, float("-inf"))
        return mask.unsqueeze(0).unsqueeze(0)

        """
        [동작 원리 상세 가이드]

        future mask (casual한 마스크) 만들고 past mask (너무 옛날 것은 잊도록) 만들어서 합쳐서 마스킹 안되게 진짜 집중해서 볼 것

        즉 sliding_window = None 이면 causal mask로서 역할, 값 들어가면 casual mask의 값만큼의 과거를 안보게 하는 mask로서 역할 

        1. torch.triu(..., diagonal=1) : 미래 단어 차단 지도 생성
        - triu (Upper): 주대각선 기준 오른쪽 위 영역만 남김
        - diagonal=1: 기준선을 위(오른쪽)로 1칸 올림 -> 자기 자신 제외 미래만 True
        
        >>> future_mask (seq_len=4)
        tensor([[False,  True,  True,  True],   # 0번 행: 미래인 1,2,3번 차단 대상
                [False, False,  True,  True],   # 1번 행: 미래인 2,3번 차단 대상
                [False, False, False,  True],   # 2번 행: 미래인 3번 차단 대상
                [False, False, False, False]])  # 3번 행: 미래가 없음

        2. torch.tril(..., diagonal=-sliding_window) : 너무 먼 과거 단어 차단 지도 생성
        - tril (Lower): 주대각선 기준 왼쪽 아래 영역만 남김
        - diagonal=-2: 기준선을 아래(왼쪽)로 2칸 내림 -> 최근 2개보다 더 먼 과거만 True
        
        >>> past_mask (seq_len=4, sliding_window=2)
        tensor([[False, False, False, False],   # 0번 행: 내 밑으로 2칸 내려갈 공간 없음
                [False, False, False, False],   # 1번 행: 내 밑으로 2칸 내려갈 공간 없음
                [ True, False, False, False],   # 2번 행: 2칸 전인 0번 열만 차단 대상
                [ True,  True, False, False]])  # 3번 행: 2칸 이상 전인 0,1번 열 차단 대상

        3. 최종 mask 모양 (masked_fill 이후)
        - future_mask와 past_mask가 True인 자리가 전부 float("-inf")로 채워짐
        
        [인덱스 매칭 구조]
                        [0번:나]    [1번:오늘]   [2번:밥을]   [3번:먹었다]
        [0번: 나]      [  0.0 ]    [ -inf ]    [ -inf ]    [ -inf ]
        [1번: 오늘]    [  0.0 ]    [  0.0 ]    [ -inf ]    [ -inf ]
        [2번: 밥을]    [ -inf ]    [  0.0 ]    [  0.0 ]    [ -inf ]  <-- 0번 단어(나) 차단됨!
        [3번: 먹었다]  [ -inf ]    [ -inf ]    [  0.0 ]    [  0.0 ]  <-- 0, 1번 단어 차단됨!

        """


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:

    # MHA는 쿼리미다 KV를 두는 것
    # n_rep : 각 KV 하나 당 몇개의 쿼리와 연산하는지(GQA)

    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    """
    왜 굳이 1번째 인덱스를 Head로 만들어서 넘겨줬을까?
    이 함수(repeat_kv) 바로 직전 단계(어텐션 모듈 내부)에서 .transpose(1, 2)나 .permute(0, 2, 1, 3) 같은 함수를 써서 Seq_len과 Head 자리를 뒤바꾼 채로 이 함수에 던져주었기 때문.
    그렇게 한 이유는 딱 GQA(Grouped Query Attention) 연산을 하려면 KV 헤드를 복사하는 작업(expand와 reshape)을 바로 이 'Head 차원(1번째 인덱스)'에서 수행해야 하기 때문. 자리를 미리 안 바꿔놓으면 차원을 늘리고 합치는 연산이 엉뚱한 Seq_len 자리에 적용되어 행렬이 꼬이게 됨.

    repeat_kv 함수를 좀 더 자세히 설명하면:

    1. 입력 shape: [batch, 8, seq, 64] (8개의 KV head)
    2. [:, :, None, :, :]로 차원 추가: [batch, 8, 1, seq, 64]
    3. expand로 복제: [batch, 8, 8, seq, 64] (각 KV head가 8번 복제됨)
    4. reshape로 합치기: [batch, 64, seq, 64] (64개의 head처럼 보이게)

    이렇게 하면 64개의 Query head 각각이 대응되는 KV head와 연산할 수 있습니다.
    ex) 0~7번째 쿼리는 모두 0번째 KV head와 연산, 8~15번째 쿼리는 모두 1번째 KV head와 연산 -> 이런식으로 총 64번의 KV head(종류는 8개)

    """
    if n_rep == 1:
        return hidden_states

    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, seq_len, head_dim
    )

    return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)

    """
    torch.expand() vs. torch.repeat()
    특정 텐서의 sizes 차원의 데이터를 반복한다 vs. torch.expand(*sizes)의 경우 메모리를 참조하기 때문에, 원본을 참조하게 된다.

    1. 3차원으로 축소해서 직관적으로 보기

    원래 2개 헤드가 있고, 각 헤드마다 3글자씩 단어를 들고 있는 [2, 3] 크기의 2차원 텐서가 있다고 해봅시다.
    0번 헤드: [A, B, C]
    1번 헤드: [D, E, F]
    이 상태에서 중간에 None을 넣습니다: [:, None, :]
    그럼 모양이 [2, 1, 3]이 되면서, 각 헤드 알맹이들이 대괄호[]로 한 번 더 감싸집니다.
    Plaintext
    [
    [[A, B, C]],  <-- 0번 헤드 방 (크기 1)
    [[D, E, F]]   <-- 1번 헤드 방 (크기 1)
    ]

    2. 여기서 .expand(2, 4, 3)을 하면 어떤 일이 벌어질까?

    우리는 방금 만든 가운데 1짜리 가상 차원을 4로 늘려달라고(expand) 명령했습니다.
    그럼 컴퓨터는 "아하, 각 헤드 방 안에 들어있는 알맹이 덩어리([A, B, C] 또는 [D, E, F])를 통째로 4번씩 읽게 만들면 되는구나!"라고 이해합니다.
    결과물은 다음과 같은 모양의 [2, 4, 3] 구조가 됩니다.
    Plaintext
    [
    [ <-- 0번 헤드 구역
        [A, B, C],  # 원본
        [A, B, C],  # 가상 복사 1
        [A, B, C],  # 가상 복사 2
        [A, B, C]   # 가상 복사 3
    ],
    [ <-- 1번 헤드 구역
        [D, E, F],  # 원본
        [D, E, F],  # 가상 복사 1
        [D, E, F],  # 가상 복사 2
        [D, E, F]   # 가상 복사 3
    ]
    ]
    """

    """
    input_ids: 텍스트를 숫자로 바꾼 것 (예: [[12, 45, 789, 101, 2022]]) - 단어 사전
    position_ids: input_ids 개수만큼 0번부터 순서대로 번호표를 매겨준 것 (예: [[0, 1, 2, 3, 4]])
    """


@dataclass
# 함수 인자 ➔ self.인자 매핑 데코레이터
class AttentionConfig:
    hidden_size: int
    # 의미: 모델의 전체 두께(차원 수)입니다.
    # 설명: 우리가 FFN 배울 때 이야기했던 "단어 하나의 4,096차원" 기억하시나요? 그 전체 차원의 크기가 여기에 저장됩니다. (예: Llama 3 8B 모델의 경우 4096)
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    # 의미: 헤드 1개당 쪼개 가져가는 차원의 크기입니다.
    # 설명: 전체 hidden_size를 num_attention_heads로 나눈 값입니다. 우리가 앞서 예시로 든 64나 128 같은 숫자가 들어갑니다. (4096÷32=128)
    rope_theta: float
    rope_scaling: dict


class CustomSelfAttention(nn.Module):

    def __init__(
        self,
        config: AttentionConfig,
        layer_idx: int = 0,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = None,
        rotary_emb=None,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = 1.0 / math.sqrt(self.head_dim)
        self.rotary_emb = rotary_emb

        # Key projection: [hidden_size] → [num_kv_heads * head_dim]
        # 2880 → 8 * 64 = 512 (smaller due to GQA!)

        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=True,
            dtype=dtype,
            device=device,
        )

        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=True,
            dtype=dtype,
            device=device,
        )

        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=True,
            dtype=dtype,
            device=device,
        )
        
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=True,
            dtype=dtype,
            device=device,
        )

        

