; This fails linking when it is linked with an empty file as the first object file

; RUN: as > %t1.bc < /dev/null
; RUN: as < %s > %t2.bc
; RUN: link %t[12].bc

%work = global int (int, int)* %zip

declare int %zip(int, int)
