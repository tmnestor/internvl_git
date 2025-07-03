# Import Cleanup Required

The following files were archived and may have import references that need updating:

## Archived Files
- `internvl/extraction/json_extraction.py` → Use `key_value_parser.py` instead
- `internvl/extraction/json_extraction_fixed.py` → Use `key_value_parser.py` instead

## Search Commands
Run these commands to find any remaining references:

```bash
# Find JSON extraction imports
grep -r "from.*json_extraction" internvl/
grep -r "import.*json_extraction" internvl/

# Find legacy dev tools imports
grep -r "from.*dev_tools" internvl/
```

## Recommended Replacements
- Replace `json_extraction` imports with `key_value_parser`
- Update any references to use the new Key-Value extraction approach
- Remove imports to archived dev tools (they were mostly for testing)
