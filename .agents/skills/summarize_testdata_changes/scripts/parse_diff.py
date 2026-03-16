import sys
import re
from collections import defaultdict

def parse_diff(stream):
    current_file = None
    file_changes = defaultdict(lambda: {'input': [], 'stderr': [], 'stdout': []})
    
    for line in stream:
        if line.startswith('diff --git'):
            parts = line.split()
            if len(parts) >= 4:
                current_file = parts[3][2:] if parts[3].startswith('b/') else parts[3]
        elif line.startswith('+') or line.startswith('-'):
            if not line.startswith('+++') and not line.startswith('---'):
                stripped = line[1:].strip()
                if stripped.startswith('// CHECK:STDERR'):
                    file_changes[current_file]['stderr'].append(line.rstrip('\n'))
                elif stripped.startswith('// CHECK:STDOUT'):
                    file_changes[current_file]['stdout'].append(line.rstrip('\n'))
                elif stripped.startswith('// CHECK'):
                    file_changes[current_file]['stdout'].append(line.rstrip('\n'))
                else:
                    file_changes[current_file]['input'].append(line.rstrip('\n'))
                        
    for f, c in file_changes.items():
        if not c['input'] and not c['stderr'] and not c['stdout']:
            continue
        print(f"File: {f}")
        if c['input']:
            print("  --- Input Changes ---")
            for change in c['input']:
                print(f"  {change}")
        if c['stderr']:
            print("  --- STDERR Changes ---")
            for change in c['stderr']:
                print(f"  {change}")
        if c['stdout']:
            print("  --- STDOUT Changes ---")
            for change in c['stdout']:
                print(f"  {change}")
        print("-" * 40)

if __name__ == '__main__':
    parse_diff(sys.stdin)
