# Kubeflow Pipelines (KFP) Compatibility

This document describes how to use the InternVL Evaluation system in a Kubeflow Pipelines (KFP) environment.

## Relative Path Support

The InternVL Evaluation system now supports relative paths for KFP compatibility. This means that you can run
the system in a Kubeflow Pipelines environment without having to modify the code to use absolute paths.

### Key Changes

1. All paths are now resolved relative to `INTERNVL_PROJECT_ROOT`
2. The `.env` file now uses relative paths by default
3. The `path.py` module has been updated to support both absolute and relative paths

## Module Invocation Pattern

To ensure consistent behavior across different environments, all scripts should be invoked using the module
pattern:

```bash
python -m internvl.module.script_name [args]
```

Instead of:

```bash
python internvl/module/script_name.py [args]
```

This ensures that imports work correctly regardless of the current working directory.

## Setting Up for KFP

1. Use the `.env` file with relative paths:

```bash
# Project root (base for all relative paths)
INTERNVL_PROJECT_ROOT=.

# Base paths (all relative to project root)
INTERNVL_DATA_PATH=data
INTERNVL_OUTPUT_PATH=output
INTERNVL_SOURCE_PATH=internvl
INTERNVL_PROMPTS_PATH=prompts.yaml
```

2. Model path considerations:

For the model path, you'll typically need an absolute path in KFP, as the models are often 
stored in a shared volume:

```bash
# In KFP environment
INTERNVL_MODEL_PATH=/mnt/shared/models/InternVL2_5-1B
```

3. Running scripts in KFP:

Always use the module invocation pattern within your KFP component:

```python
def internvl_component(image_path, output_path):
    import subprocess
    
    # Setup environment variables
    env = {
        "INTERNVL_PROJECT_ROOT": ".",
        "INTERNVL_DATA_PATH": "data",
        "INTERNVL_OUTPUT_PATH": "output",
        "INTERNVL_MODEL_PATH": "/mnt/shared/models/InternVL2_5-1B"
    }
    
    # Run using module invocation pattern
    cmd = [
        "python", "-m", "internvl.cli.internvl_single",
        "--image", image_path,
        "--output", output_path
    ]
    
    subprocess.run(cmd, env=env, check=True)
```

## Path Resolution

Paths are resolved using the following rules:

1. If a path is absolute, it's used as-is (for backwards compatibility)
2. If a path is relative, it's resolved relative to `INTERNVL_PROJECT_ROOT`
3. If `INTERNVL_PROJECT_ROOT` is not set, the current working directory is used

## Template Script

See `internvl/utils/dev_tools/module_template.py` for a template script that demonstrates the module invocation pattern
and path resolution.