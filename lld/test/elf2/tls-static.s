// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %S/Inputs/shared.s -o %tso
// RUN: ld.lld2 -static %t -o %tout
// RUN: ld.lld2 %t -o %tout
// RUN: ld.lld2 -shared %tso -o %tshared
// RUN: not ld.lld2 -static %t %tshared -o %tout
// REQUIRES: x86

.global _start
_start:
  call __tls_get_addr
