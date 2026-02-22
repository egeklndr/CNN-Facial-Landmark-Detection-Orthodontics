import os
import json

def fix_json_file(filepath):
    """Fix a broken JSON file by extracting first valid object"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Try to find the first complete JSON object
        brace_count = 0
        start_idx = content.find('{')
        
        if start_idx == -1:
            return False, "No JSON object found"
        
        for i in range(start_idx, len(content)):
            if content[i] == '{':
                brace_count += 1
            elif content[i] == '}':
                brace_count -= 1
                
                if brace_count == 0:
                    # Found complete JSON object
                    json_str = content[start_idx:i+1]
                    
                    # Validate it
                    try:
                        data = json.loads(json_str)
                        
                        # Write fixed version
                        with open(filepath, 'w', encoding='utf-8') as f:
                            json.dump(data, f, indent=2, ensure_ascii=False)
                        
                        return True, "Fixed"
                    except json.JSONDecodeError as e:
                        return False, f"Invalid JSON: {e}"
        
        return False, "No complete JSON object"
    
    except Exception as e:
        return False, f"Error: {e}"

def fix_all_json_files(folder):
    """Fix all JSON files in a folder"""
    
    print(f"="*60)
    print(f"FIXING JSON FILES IN: {folder}")
    print(f"="*60)
    
    if not os.path.exists(folder):
        print(f" Folder not found: {folder}")
        return
    
    files = [f for f in os.listdir(folder) if f.endswith('.json')]
    
    print(f"\nFound {len(files)} JSON files")
    
    fixed = 0
    failed = 0
    
    for filename in files:
        filepath = os.path.join(folder, filename)
        success, message = fix_json_file(filepath)
        
        if success:
            fixed += 1
            print(f" {filename}")
        else:
            failed += 1
            print(f" {filename}: {message}")
    
    print(f"\n" + "="*60)
    print(f"SUMMARY")
    print(f"="*60)
    print(f" Fixed: {fixed}/{len(files)}")
    print(f" Failed: {failed}/{len(files)}")
    
    return fixed, failed

if __name__ == "__main__":
    print("JSON FILE FIXER")
    print("="*60)
    
    # Fix both folders
    folders = ['clabel', 'plabel']
    
    total_fixed = 0
    total_failed = 0
    
    for folder in folders:
        fixed, failed = fix_all_json_files(folder)
        total_fixed += fixed
        total_failed += failed
        print()
    
    print("="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Total fixed: {total_fixed}")
    print(f"Total failed: {total_failed}")
    
    if total_failed == 0:
        print("\n ALL JSON FILES FIXED!")
    else:
        print(f"\n {total_failed} files still have issues")