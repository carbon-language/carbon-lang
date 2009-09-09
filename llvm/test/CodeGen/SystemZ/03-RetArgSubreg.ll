; RUN: llc < %s -march=systemz | grep lgr   | count 2
; RUN: llc < %s -march=systemz | grep nihf  | count 1
; RUN: llc < %s -march=systemz | grep lgfr  | count 1


define i32 @foo(i32 %a, i32 %b) {
entry:
    ret i32 %b
}

define i32 @foo1(i32 %a, i32 %b) zeroext {
entry:
    ret i32 %b
}

define i32 @foo2(i32 %a, i32 %b) signext {
entry:
    ret i32 %b
}
