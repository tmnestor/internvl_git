# Shared Conda Environments Guide

This guide explains how to create and manage shared Conda environments in multi-user systems, JupyterHub environments using the Amazon EFS network file system.

## Conda Environment Locations

To check where Conda looks for environments, use:

```bash
conda config --show envs_dirs
```

Typical locations include:
- Personal environments: `/home/jovyan/.conda/envs/`
- Shared environments: `/efs/shared/.conda/envs/` 

## Creating a Shared Environment

### 1. Specify the Prefix in YAML

In your environment YAML file (e.g., `internvl_env.yml`), add a prefix pointing to the shared location:

```yaml
name: internvl_env
channels:
  - defaults
dependencies:
  - python=3.11
  - pytorch=2.1.0
  # ... other dependencies
prefix: /efs/shared/.conda/envs/internvl_env
```

### 2. Create Environment Using Prefix

```bash
conda env create -f internvl_env.yml --prefix /efs/shared/.conda/envs/internvl_env
```

### 3. Update Existing Environment

When you need to update dependencies:

```bash
conda env update -f internvl_env.yml --prefix /efs/shared/.conda/envs/internvl_env --prune
```

The `--prune` flag removes dependencies that are no longer specified in the YAML file.

## Using the Shared Environment

### 1. Method 1: Activating by Path

Users can activate the environment using its full path:

```bash
conda activate /efs/shared/.conda/envs/internvl_env
```

### 2. Method 2: Adding to Known Environments

Users can add the shared directory to their environment search path:

```bash
# Add shared environment location to config
conda config --append envs_dirs /efs/shared/.conda/envs

# Now users can activate by name
conda activate internvl_env
```

## Package Management and Updates

### Administrator Responsibilities

1. **Schedule Regular Updates**:
   ```bash
   # First test the updated environment in a test location
   conda env update -f internvl_env.yml --prefix /efs/shared/.conda/envs/internvl_env_test --prune
   
   # Once tested, update the production environment
   conda env update -f internvl_env.yml --prefix /efs/shared/.conda/envs/internvl_env --prune
   ```

2. **Track Dependencies**:
   ```bash
   # Export current environment to a requirements file for reference
   conda list --explicit > internvl_env_snapshot_$(date +%Y%m%d).txt
   ```

3. **Custom Channel Configuration**:
   ```bash
   # Add internal channels when needed
   conda config --add channels https://your-internal-channel.example.com
   ```

### Best Practices

1. **Permissions Management**:
   ```bash
   # Make the environment readable by all users
   chmod -R 755 /efs/shared/.conda/envs/internvl_env
   ```

2. **Communication with Users**:
   - Establish a notification system for environment changes
   - Provide clear documentation on how to use the environment
   - Set up a regular maintenance schedule

3. **Version Control**:
   - Keep all environment files under version control
   - Document changes with each update
   - Consider maintaining multiple environment versions for compatibility

4. **Validation**:
   ```bash
   # Add a validation script to test that all core functionality works
   python -m scripts.utils.verify_env
   ```

## Troubleshooting

### Common Issues and Solutions

1. **Permission Errors**:
   ```bash
   # Fix permission issues
   sudo chown -R $USER:$GROUP /efs/shared/.conda/envs/internvl_env
   chmod -R 755 /efs/shared/.conda/envs/internvl_env
   ```

2. **Package Conflicts**:
   ```bash
   # Use the --no-deps flag when installing individual packages
   conda install --no-deps package_name
   ```

3. **Identifying Issues**:
   ```bash
   # Check environment integrity
   conda list --revisions
   ```

## Example for InternVL Project

Create a shared environment specifically for the InternVL project:

```bash
# Configure conda to use shared environment directory
conda config --append envs_dirs /efs/shared/.conda/envs

# Create the shared environment
conda env create -f internvl_env.yml --prefix /efs/shared/.conda/envs/internvl_env

# Make sure permissions are correct
chmod -R 755 /efs/shared/.conda/envs/internvl_env

# Create a validation script that users can run
echo "python -m scripts.utils.verify_env" > /efs/shared/.conda/envs/validate_internvl.sh
chmod +x /efs/shared/.conda/envs/validate_internvl.sh
```

This approach ensures a consistent, well-maintained environment that all users can access while minimizing duplication and conflicts.