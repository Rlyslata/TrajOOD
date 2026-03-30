from traj_ood.models import hook, traj_encoder


all_feats = []
all_labels = []

for x, y in loader:
    hook.clear()
    logits = model(x)

    feats = hook.features
    traj = traj_builder.build(feats)

    z = traj_encoder(traj)

    all_feats.append(z.detach())
    all_labels.append(y)