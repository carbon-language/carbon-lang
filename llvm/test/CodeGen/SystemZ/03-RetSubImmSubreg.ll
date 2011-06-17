; RUN: llc < %s -march=systemz | grep ahi   | count 3
; RUN: llc < %s -march=systemz | grep afi   | count 3
; RUN: llc < %s -march=systemz | grep lgfr  | count 4
; RUN: llc < %s -march=systemz | grep llgfr | count 2


define i32 @foo1(i32 %a, i32 %b) {
entry:
    %c = sub i32 %a, 1
    ret i32 %c
}

define i32 @foo2(i32 %a, i32 %b) {
entry:
    %c = sub i32 %a, 131072
    ret i32 %c
}

define zeroext i32 @foo3(i32 %a, i32 %b)  {
entry:
    %c = sub i32 %a, 1
    ret i32 %c
}

define signext i32 @foo4(i32 %a, i32 %b)  {
entry:
    %c = sub i32 %a, 131072
    ret i32 %c
}

define zeroext i32 @foo5(i32 %a, i32 %b)  {
entry:
    %c = sub i32 %a, 1
    ret i32 %c
}

define signext i32 @foo6(i32 %a, i32 %b)  {
entry:
    %c = sub i32 %a, 131072
    ret i32 %c
}

