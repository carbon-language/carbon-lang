; This fails linking when it is linked with an empty file as the first object file

; RUN: touch Output/LinkTest1.ll
; RUN: as Output/LinkTest1.ll
; RUN: as < %s > Output/LinkTest2.bc
; RUN: link Output/LinkTest[12].bc

%work = global int (int, int)* %zip

declare int %zip(int, int)
