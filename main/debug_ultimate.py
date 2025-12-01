import subprocess

cmd = 'python pacman.py -l tiny_maze -p UltimateSearchAgent -n 1 -q'

result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
output = result.stdout + result.stderr

print("="*80)
print("FULL OUTPUT")
print("="*80)
print(output)
print("="*80)
print("OUTPUT AS REPR (showing special chars)")
print("="*80)
print(repr(output))
print("="*80)

# Save to file for inspection
with open('ultimate_output.txt', 'w') as f:
    f.write(output)

print("\nOutput saved to ultimate_output.txt")