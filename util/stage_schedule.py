DEFAULT_STAGES = [
    {
        "load_size": 256,
        "crop_size": 224,
        "n_epochs": 0,
        "n_epochs_decay": 0,
        "batch_size": 64,
    },
    {
        "load_size": 512,
        "crop_size": 448,
        "n_epochs": 0,
        "n_epochs_decay": 0,
        "batch_size": 16,
    },
    {
        "load_size": 1024,
        "crop_size": 896,
        "n_epochs": 30,
        "n_epochs_decay": 30,
        "batch_size": 4,
    },
]
STAGE_KEYS = ("load_size", "crop_size", "n_epochs", "n_epochs_decay", "batch_size")

def parse_value(value):
    value = value.strip()
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value

def parse_stages(raw_stages):
    if not raw_stages:
        return [None]
    if raw_stages == "default":
        return DEFAULT_STAGES
    stages = []
    for raw_stage in raw_stages.split(";"):
        parts = [part.strip() for part in raw_stage.split(",") if part.strip()]
        if not parts:
            continue
        if any("=" in part for part in parts):
            stage = {}
            for part in parts:
                if "=" not in part:
                    raise ValueError(
                        "--train_stages cannot mix key=value and positional values in one stage"
                    )
                key, value = part.split("=", 1)
                stage[key.strip()] = parse_value(value)
            stages.append(stage)
            continue
        values = [int(value) for value in parts]
        if len(values) != len(STAGE_KEYS):
            raise ValueError(
                "--train_stages expects load,crop,epochs,decay,batch per stage"
            )
        stages.append(dict(zip(STAGE_KEYS, values)))
    return stages

def apply_stage(opt, stage):
    if stage is None:
        return
    for key, value in stage.items():
        setattr(opt, key, value)
