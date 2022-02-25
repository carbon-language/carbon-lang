# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
# RUN: not ld.lld --entry=func --gc-sections %t.o -o /dev/null 2>&1 | FileCheck %s

## Check we are able to find a function symbol that encloses
## a given location when reporting error messages.
# CHECK: {{.*}}.o:(function func: .text.func+0x3): relocation R_X86_64_32S out of range: -281474974609120 is not in [-2147483648, 2147483647]; references func
# CHECK-NEXT: >>> defined in {{.*}}.o

# This mergeable section will be garbage collected. We had a crash issue in that case. Test it.
.section .rodata.str1,"aMS",@progbits,1
.asciz "a"

.section .text.func, "ax", %progbits
.globl func
.type func,@function
func:
  movq $func - 0x1000000000000, %rdx
.size func, .-func
