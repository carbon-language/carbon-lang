; RUN: llc < %s -march=systemz | grep oill | count 1
; RUN: llc < %s -march=systemz | grep oilh | count 1
; RUN: llc < %s -march=systemz | grep oihl | count 1
; RUN: llc < %s -march=systemz | grep oihh | count 1

define i64 @foo1(i64 %a, i64 %b) {
entry:
    %c = or i64 %a, 1
    ret i64 %c
}

define i64 @foo2(i64 %a, i64 %b) {
entry:
    %c = or i64 %a, 131072
    ret i64 %c
}

define i64 @foo3(i64 %a, i64 %b) {
entry:
    %c = or i64 %a, 8589934592
    ret i64 %c
}

define i64 @foo4(i64 %a, i64 %b) {
entry:
    %c = or i64 %a, 562949953421312
    ret i64 %c
}
