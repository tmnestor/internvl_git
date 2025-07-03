# InternVL Legacy Code Archive

**Archive Date:** Thu Jul  3 11:34:05 AEST 2025
**Archived From:** InternVL PoC Environment Variable Refactor

## Why These Files Were Archived

This archive contains code that has been superseded by newer, better implementations:

### 🔄 **Extraction Methods**
- **Legacy**: JSON extraction (unreliable, required extensive post-processing)
- **Current**: Key-Value extraction (robust, production-ready)

### 📚 **Notebooks**
- **Legacy**: `internvl_codebase_demo.ipynb` (custom implementations)
- **Current**: `internvl_package_demo.ipynb` (proper package utilization)

### 🛠️ **Development Tools**
- **Legacy**: Failed tests and debugging scripts
- **Current**: Working test utilities and environment verification

### 📝 **Documentation**
- **Legacy**: Implementation plans and suggestions
- **Current**: Completed features with environment-driven configuration

## 🎯 **Current Best Practices**

✅ **Key-Value extraction** over JSON  
✅ **Environment-driven configuration** via .env  
✅ **Package-based architecture** with proper modules  
✅ **Cross-platform compatibility** (local Mac / remote GPU)  
✅ **KFP-ready structure** (data outside source)  

## 🔄 **Migration Path**

If you need to reference archived code:

1. **JSON to Key-Value**: Use `internvl.extraction.key_value_parser`
2. **Custom code**: Use package modules from `internvl.*`
3. **Environment setup**: Use `.env` configuration
4. **CLI usage**: Use new CLI with optional arguments

## 📞 **Need Help?**

The archived code is preserved for reference but is no longer maintained.
Use the current implementations in the main codebase.
