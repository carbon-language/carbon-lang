// RUN: not llvm-mc -triple x86_64-pc-win32 -filetype=obj %s -o %t.o 2>&1 | FileCheck %s

// CHECK: Allocation size must be non-zero!

    .globl smallFunc
    .def smallFunc; .scl 2; .type 32; .endef
    .seh_proc smallFunc
    .seh_stackalloc 0
smallFunc:
    ret
    .seh_endproc
