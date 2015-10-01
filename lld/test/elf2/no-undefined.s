# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
# RUN: not lld --no-undefined -shared -flavor gnu2 %t -o %t.so 
# RUN: lld -shared -flavor gnu2 %t -o %t1.so 

.globl _shared
_shared: 
  call _unresolved
