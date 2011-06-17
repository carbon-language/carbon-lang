; RUN: llc < %s -march=systemz | grep sr    | count 3
; RUN: llc < %s -march=systemz | grep llgfr | count 1
; RUN: llc < %s -march=systemz | grep lgfr  | count 2

define i32 @foo(i32 %a, i32 %b) {
entry:
    %c = sub i32 %a, %b
    ret i32 %c
}

define zeroext i32 @foo1(i32 %a, i32 %b)  {
entry:
    %c = sub i32 %a, %b
    ret i32 %c
}

define signext i32 @foo2(i32 %a, i32 %b)  {
entry:
    %c = sub i32 %a, %b
    ret i32 %c
}

