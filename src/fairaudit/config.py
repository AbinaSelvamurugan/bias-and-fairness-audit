"""Central versioned YAML configuration with runtime checks."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]

def load_configs(root=ROOT):
    root = Path(root)
    cfg = {}
    for name in ('dataset', 'model', 'fairness', 'mitigation'):
        with (root / 'configs' / f'{name}.yaml').open(encoding='utf-8') as f:
            cfg[name] = yaml.safe_load(f)
        if cfg[name].get('version') != 1:
            raise ValueError(f'Unsupported {name} configuration version')
    if not 0 < cfg['model']['test_fraction'] < 1:
        raise ValueError('test_fraction must be in (0, 1)')
    if sorted(cfg['fairness']['thresholds']) != cfg['fairness']['thresholds']:
        raise ValueError('thresholds must be sorted')
    return cfg
