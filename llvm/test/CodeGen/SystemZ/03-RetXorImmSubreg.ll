; RUN: llc < %s -march=systemz | grep xilf  | count 9
; RUN: llc < %s -march=systemz | grep llgfr | count 3
; RUN: llc < %s -march=systemz | grep lgfr  | count 6

define i32 @foo1(i32 %a, i32 %b) {
entry:
    %c = xor i32 %a, 1
    ret i32 %c
}

define i32 @foo2(i32 %a, i32 %b) {
entry:
    %c = xor i32 %a, 131072
    ret i32 %c
}

define i32 @foo7(i32 %a, i32 %b) {
entry:
    %c = xor i32 %a, 123456
    ret i32 %c
}

define zeroext i32 @foo3(i32 %a, i32 %b)  {
entry:
    %c = xor i32 %a, 1
    ret i32 %c
}

define zeroext i32 @foo8(i32 %a, i32 %b)  {
entry:
    %c = xor i32 %a, 123456
    ret i32 %c
}

define signext i32 @foo4(i32 %a, i32 %b)  {
entry:
    %c = xor i32 %a, 131072
    ret i32 %c
}

define zeroext i32 @foo5(i32 %a, i32 %b)  {
entry:
    %c = xor i32 %a, 1
    ret i32 %c
}

define signext i32 @foo6(i32 %a, i32 %b)  {
entry:
    %c = xor i32 %a, 131072
    ret i32 %c
}

define signext i32 @foo9(i32 %a, i32 %b)  {
entry:
    %c = xor i32 %a, 123456
    ret i32 %c
}

