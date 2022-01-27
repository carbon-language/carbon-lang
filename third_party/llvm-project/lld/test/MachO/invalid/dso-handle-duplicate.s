# REQUIRES: x86

## If for some bizarre reason the input file defines its own ___dso_handle, we
## should raise an error. At least, we've implemented this behavior if the
## conflicting symbol is a global. A local symbol of the same name will still
## take priority in our implementation, unlike in ld64. But that's a pretty
## far-out edge case that should be safe to ignore.

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: not %lld -dylib %t.o -o /dev/null 2>&1 | FileCheck %s -DFILE=%t.o
# CHECK:      error: duplicate symbol: ___dso_handle
# CHECK-NEXT: >>> defined in [[FILE]]
# CHECK-NEXT: >>> defined in <internal>

.globl _main, ___dso_handle
.text
_main:
  leaq ___dso_handle(%rip), %rdx
  ret

___dso_handle:
  .space 1
