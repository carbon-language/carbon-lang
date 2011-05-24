# RUN: llvm-mc -triple x86_64-pc-win32 %s | FileCheck %s

# CHECK: .seh_proc func
# CHECK: .seh_stackalloc 8
# CHECK: .seh_endprologue
# CHECK: .seh_endproc

    .text
    .globl func
    .def func; .scl 2; .type 32; .endef
    .seh_proc func
func:
    subq $8, %rsp
    .seh_stackalloc 8
    .seh_endprologue
    addq $8, %rsp
    ret
    .seh_endproc
