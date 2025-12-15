from .auto import AutoDistributedTargetModel, AutoDraftModelConfig, AutoEagle3DraftModel, AutoPosSEagle3DraftModel
from .draft.llama3_eagle import LlamaForCausalLMEagle3
from .draft.llama3_poss_eagle import LlamaForCausalLMPosSEagle3

__all__ = [
    "AutoDraftModelConfig",
    "AutoEagle3DraftModel",
    "AutoPosSEagle3DraftModel",
    "AutoDistributedTargetModel",
    "LlamaForCausalLMEagle3",
    "LlamaForCausalLMPosSEagle3",
]
