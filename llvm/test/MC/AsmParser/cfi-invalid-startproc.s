# RUN: not llvm-mc -triple=x86_64-apple-macosx10.8 -filetype=obj -o %t %s 2>&1 | FileCheck %s
# Check that the cfi_startproc is declared after the beginning of
# a procedure, otherwise it will reference an invalid symbol for
# emitting the relocation.
# <rdar://problem/15939159>

# CHECK: No symbol to start a frame
.text
.cfi_startproc
.globl _someFunction
_someFunction:
.cfi_def_cfa_offset 16
.cfi_offset %rbp, -16
.cfi_def_cfa_register rbp
  ret
.cfi_endproc
