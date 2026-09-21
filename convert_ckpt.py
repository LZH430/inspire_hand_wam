import torch
src = "step47200.pt"
dst = "step47200.ckpt"

print("loading:", src)
state_dict = torch.load(
    src,
    map_location="cpu"
)

print("original keys:", len(state_dict))

new_state_dict = {}

for k,v in state_dict.items():

    if k.startswith("module."):
        k = k[7:]

    if k.startswith("_forward_module."):
        k = k[len("_forward_module."):]

    new_state_dict[k] = v


checkpoint = {
    "state_dict": new_state_dict
}


torch.save(
    checkpoint,
    dst
)
print("="*80)
print("saved:", dst)
print("keys:", len(new_state_dict))
print("="*80)