; This fails linking when it is linked with an empty file as the first object file

; RUN: as > %t.LinkTest.bc < /dev/null
; RUN: as < %s > %t.bc
; RUN: link %t.LinkTest.bc %t.bc

%work = global int 4
%test = global int* getelementptr( int* %work, long 1)

