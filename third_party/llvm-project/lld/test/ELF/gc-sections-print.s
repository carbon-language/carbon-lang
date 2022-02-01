# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
# RUN: ld.lld %t --gc-sections --print-gc-sections -o %t2 2>&1 | FileCheck -check-prefix=PRINT %s

# PRINT:      removing unused section {{.*}}:(.text.x)
# PRINT-NEXT: removing unused section {{.*}}:(.text.y)

# RUN: ld.lld %t --gc-sections --print-gc-sections --no-print-gc-sections -o %t2 >& %t.log
# RUN: echo >> %t.log
# RUN: FileCheck -check-prefix=NOPRINT %s < %t.log

# NOPRINT-NOT: removing

.globl _start
.protected a, x, y
_start:
 call a

.section .text.a,"ax",@progbits
a:
 nop

.section .text.x,"ax",@progbits
x:
 nop

.section .text.y,"ax",@progbits
y:
 nop
