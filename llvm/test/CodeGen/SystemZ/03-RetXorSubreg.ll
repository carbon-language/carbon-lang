; RUN: llc < %s -march=systemz | grep xgr   | count 3
; RUN: llc < %s -march=systemz | grep nihf  | count 1
; RUN: llc < %s -march=systemz | grep lgfr  | count 1


define i32 @foo(i32 %a, i32 %b) {
entry:
    %c = xor i32 %a, %b
    ret i32 %c
}

define zeroext i32 @foo1(i32 %a, i32 %b)  {
entry:
    %c = xor i32 %a, %b
    ret i32 %c
}

define signext i32 @foo2(i32 %a, i32 %b)  {
entry:
    %c = xor i32 %a, %b
    ret i32 %c
}

