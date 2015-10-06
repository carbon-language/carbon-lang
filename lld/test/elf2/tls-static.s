// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t
// RUN: lld -flavor gnu2 -static %t -o %tout
// REQUIRES: x86

.global _start
.text
_start:
  call __tls_get_addr
