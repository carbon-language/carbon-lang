; RUN: llc < %s -march=arm | FileCheck %s

define i32 @f1() {
; CHECK: f1
; CHECK: mov r0, #0
        ret i32 0
}

define i32 @f2() {
; CHECK: f2
; CHECK: mov r0, #255
        ret i32 255
}

define i32 @f3() {
; CHECK: f3
; CHECK: mov r0{{.*}}256
        ret i32 256
}

define i32 @f4() {
; CHECK: f4
; CHECK: orr{{.*}}256
        ret i32 257
}

define i32 @f5() {
; CHECK: f5
; CHECK: mov r0, {{.*}}-1073741761
        ret i32 -1073741761
}

define i32 @f6() {
; CHECK: f6
; CHECK: mov r0, {{.*}}1008
        ret i32 1008
}

define void @f7(i32 %a) {
; CHECK: f7
; CHECK: cmp r0, #1, 16
        %b = icmp ugt i32 %a, 65536             ; <i1> [#uses=1]
        br i1 %b, label %r, label %r

r:              ; preds = %0, %0
        ret void
}
