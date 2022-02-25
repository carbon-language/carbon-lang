# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t1
# RUN: rm -f %t.a
# RUN: llvm-ar rcs %t.a %t1
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %S/Inputs/symver-archive1.s -o %t2.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %S/Inputs/symver-archive2.s -o %t3.o
# RUN: ld.lld -o /dev/null %t2.o %t3.o %t.a

# RUN: not ld.lld -o /dev/null %t2.o %t3.o %t1 2>&1 | FileCheck %s --check-prefix=ERR

# ERR:      error: duplicate symbol: x

## If defined xx and xx@@VER are in different files, report a duplicate definition error like GNU ld.
# ERR:      error: duplicate symbol: xx
# ERR-NEXT: >>> defined at {{.*}}2.o:(.text+0x0)
# ERR-NEXT: >>> defined at {{.*}}1:(.text+0x0)

.text
.globl x
.type x, @function
x:

.globl xx
xx = x
