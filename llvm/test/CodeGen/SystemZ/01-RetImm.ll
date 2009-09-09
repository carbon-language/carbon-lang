; RUN: llc < %s -march=systemz | grep lghi  | count 1
; RUN: llc < %s -march=systemz | grep llill | count 1
; RUN: llc < %s -march=systemz | grep llilh | count 1
; RUN: llc < %s -march=systemz | grep llihl | count 1
; RUN: llc < %s -march=systemz | grep llihh | count 1
; RUN: llc < %s -march=systemz | grep lgfi  | count 1
; RUN: llc < %s -march=systemz | grep llilf | count 1
; RUN: llc < %s -march=systemz | grep llihf | count 1


define i64 @foo1() {
entry:
    ret i64 1
}

define i64 @foo2() {
entry:
    ret i64 65535 
}

define i64 @foo3() {
entry:
    ret i64 131072
}

define i64 @foo4() {
entry:
    ret i64 8589934592
}

define i64 @foo5() {
entry:
    ret i64 562949953421312
}

define i64 @foo6() {
entry:
    ret i64 65537
}

define i64 @foo7() {
entry:
    ret i64 4294967295
}

define i64 @foo8() {
entry:
    ret i64 281483566645248
}
