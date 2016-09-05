# Check handling TLS related relocations and symbols when linking
# a static executable.

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux %s -o %t
# RUN: ld.lld -static %t -o %t.exe
# RUN: llvm-readobj -d %t.exe | FileCheck %s

# REQUIRES: mips

# CHECK: LoadName

  .text
  .global __start
__start:
  nop

  .global __tls_get_addr
__tls_get_addr:
  nop

  .data
loc:
  .word __tls_get_addr
