# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %p/Inputs/shared.s -o %t2.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %p/Inputs/gc-sections-shared.s -o %t3.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %p/Inputs/gc-sections-shared2.s -o %t4.o
# RUN: ld.lld -shared %t2.o -o %t2.so
# RUN: ld.lld -shared %t3.o -o %t3.so
# RUN: ld.lld -shared %t4.o -o %t4.so
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
# RUN: ld.lld --gc-sections --export-dynamic-symbol foo -o %t %t.o --as-needed %t2.so %t3.so %t4.so
# RUN: llvm-readelf -d --dyn-symbols %t | FileCheck %s

# This test the property that we have a needed line for every undefined.
# It would also be OK to keep bar2 and the need for %t2.so
# At the same time, weak symbols should not cause adding DT_NEEDED;
# this case is checked with symbol qux and %t4.so.

# CHECK-NOT: NEEDED
# CHECK:     (NEEDED) Shared library: [{{.*}}3.so]
# CHECK-NOT: NEEDED

# CHECK-DAG: FUNC    WEAK   DEFAULT  UND qux
# CHECK-DAG: NOTYPE  GLOBAL DEFAULT    6 bar
# CHECK-DAG: FUNC    GLOBAL DEFAULT  UND baz
# CHECK-DAG: NOTYPE  GLOBAL DEFAULT    6 foo

# Test with %t.o at the end too.
# RUN: ld.lld --gc-sections --export-dynamic-symbol foo -o %t --as-needed %t2.so %t3.so %t4.so %t.o
# RUN: llvm-readelf --dynamic-table --dyn-symbols %t | FileCheck --check-prefix=CHECK %s

.section .text.foo, "ax"
.globl foo
foo:
.long bar - .

.section .text.bar, "ax"
.globl bar
bar:
ret

.section .text._start, "ax"
.globl _start
.weak qux
_start:
.long baz - .
.long qux - .
ret

.section .text.unused, "ax"
.long bar2 - .
