# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t1
# RUN: not lld -flavor gnu2 %t1 -o %t2
# RUN: lld -flavor gnu2 %t1 -o %t2 -e _end

.globl _end;
_end:
