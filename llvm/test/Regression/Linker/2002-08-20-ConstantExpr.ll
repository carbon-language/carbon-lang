; This fails linking when it is linked with an empty file as the first object file

; RUN: as > Output/%s.LinkTest.bc < /dev/null
; RUN: as < %s > Output/%s.bc
; RUN: link Output/%s.LinkTest.bc Output/%s.bc

%work = global int 4
%test = global int* getelementptr( int* %work, long 1)

