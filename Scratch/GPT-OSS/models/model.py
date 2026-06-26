import torch.nn as nn
import torch
import math
from tying import Tuple


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

        freq = self.base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=self.device))/self.head_dim
        
        if self.scaling_factor > 1.0:
            concentration = 0.1 * math.log(self.scaling_factor) + 1.0
            d_half = self.head_dim / 2
            low = (d_half * math.log(self.initial_context_length / (self.ntk_beta * 2 * math.pi)))/math.log(self.base)
            high = (d_half * math.log(self.initial_context_length / (self.ntk_alpha * 2 * math.pi)))/math.log(self.base)

            interpolation = 1.0 / (self.scaling_factor * freq)
            extrapolation = 1.0 / freq
            
            ramp = (torch.arange(d_half, dtype = torch.float32, device=freq.device) - low) / (high - low)
            mask = 1 - ramp.clamp(0,1)
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
    
def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos.unsqueeze(-2).to(x.type)
    sin = sin.unsqueeze(-2).to(x.type)

    x1, x2 = torch.chunk(x, 2, dim=-1)

    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin

    return torch.cat((o1, o2), dim = -1)



            
    
