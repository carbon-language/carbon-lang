# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t1.o

# RUN: echo "SECTIONS { ASSERT(1, fail) }" > %t1.script
# RUN: ld.lld -shared -o %t1 --script %t1.script %t1.o
# RUN: llvm-readobj %t1 > /dev/null

# RUN: echo "SECTIONS { ASSERT(0, fail) }" > %t3.script
# RUN: not ld.lld -o /dev/null -T %t3.script %t1.o 2>&1 | FileCheck --check-prefix=FAIL %s
# RUN: ld.lld -o /dev/null -T %t3.script %t1.o --noinhibit-exec 2>&1 | FileCheck --check-prefix=FAIL %s
# FAIL: fail

# RUN: echo "SECTIONS { . = ASSERT(0x1000, fail); }" > %t4.script
# RUN: ld.lld -shared -o %t4 --script %t4.script %t1.o
# RUN: llvm-readobj %t4 > /dev/null

# RUN: echo "SECTIONS { .foo : { *(.foo) } }" > %t5.script
# RUN: echo "ASSERT(SIZEOF(.foo) == 8, fail);" >> %t5.script
# RUN: ld.lld -shared -o %t5 --script %t5.script %t1.o
# RUN: llvm-readobj %t5 > /dev/null

## Even without SECTIONS block we still use section names
## in expressions
# RUN: echo "ASSERT(SIZEOF(.foo) == 8, fail);" > %t5.script
# RUN: ld.lld -shared -o %t5 --script %t5.script %t1.o
# RUN: llvm-readobj %t5 > /dev/null

## Test assertions inside of output section descriptions.
# RUN: echo "SECTIONS { .foo : { *(.foo) ASSERT(SIZEOF(.foo) == 8, \"true\"); } }" > %t6.script
# RUN: ld.lld -shared -o %t6 --script %t6.script %t1.o
# RUN: llvm-readobj %t6 > /dev/null

## Unlike the GNU ld, we accept the ASSERT without the semicolon.
## It is consistent with how ASSERT can be written outside of the
## output section declaration.
# RUN: echo "SECTIONS { .foo : { ASSERT(1, \"true\") } }" > %t7.script
# RUN: ld.lld -shared -o /dev/null --script %t7.script %t1.o

.section .foo, "a"
 .quad 0
