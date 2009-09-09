; RUN: llc < %s -march=systemz | grep lghi  | count 2
; RUN: llc < %s -march=systemz | grep llill | count 1
; RUN: llc < %s -march=systemz | grep llilh | count 1
; RUN: llc < %s -march=systemz | grep lgfi  | count 1
; RUN: llc < %s -march=systemz | grep llilf | count 2


define i32 @foo1() {
entry:
    ret i32 1
}

define i32 @foo2() {
entry:
    ret i32 65535 
}

define i32 @foo3() {
entry:
    ret i32 131072
}

define i32 @foo4() {
entry:
    ret i32 65537
}

define i32 @foo5() {
entry:
    ret i32 4294967295
}

define i32 @foo6() zeroext {
entry:
    ret i32 4294967295
}

define i32 @foo7() signext {
entry:
    ret i32 4294967295
}

