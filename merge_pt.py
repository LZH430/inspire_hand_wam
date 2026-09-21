import os
import torch
from collections import defaultdict
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
# path for checkpoint directory
CKPT_DIR = (
    "/run/determined/workdir/home/unifolm-world-model-action/"
    "results/inspirehand/checkpoints/"
    "epoch=5244-step=47200.ckpt"
)

OUTPUT = (
    "/run/determined/workdir/home/unifolm-world-model-action/"
    "step47200.pt"
)

def merge_zero3():
    print("="*100)
    print("ZERO3 MERGE")
    print("="*100)
    print("INPUT:")
    print(CKPT_DIR)
    state_dict = get_fp32_state_dict_from_zero_checkpoint(
        CKPT_DIR
    )
    print(
        "MERGED KEYS:",
        len(state_dict)
    )
    if os.path.isdir(OUTPUT):
        print(
            "REMOVE OLD DIRECTORY:",
            OUTPUT
        )
        shutil.rmtree(OUTPUT)
    torch.save(
        state_dict,
        OUTPUT
    )
    print(
        "SAVED:",
        OUTPUT
    )
    return state_dict

def check_model(sd):
    print("\n")
    print("="*100)
    print("CHECK MERGED MODEL")
    print("="*100)
    empty=[]
    spatial=[]
    ema=[]
    for k,v in sd.items():
        if torch.is_tensor(v):
            if v.numel()==0:
                empty.append(k)
        if (
            "spatial_softmax_blocks" in k
            and
            (
                "nets.weight" in k
                or
                "nets.bias" in k
            )
        ):
            spatial.append(
                (
                    k,
                    tuple(v.shape),
                    v.numel()
                )
            )
        if "dp_ema_model" in k:
            ema.append(k)
    print(
        "TOTAL KEYS:",
        len(sd)
    )
    print(
        "\nEMPTY PARAM COUNT:",
        len(empty)
    )
    if empty:
        print("EMPTY SAMPLE:")
        for x in empty[:50]:
            print(x)
    print("\n")
    print("="*100)
    print("SPATIAL SOFTMAX")
    print("="*100)
    for x in sorted(spatial):
        print(x)
    
    print("\n")
    print("="*100)
    print("EMA")
    print("="*100)
    print(
        "EMA COUNT:",
        len(ema)
    )
    print("\nDONE")
if __name__ == "__main__":
    sd = merge_zero3()
    check_model(sd)