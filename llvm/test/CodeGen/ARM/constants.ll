; RUN: llc < %s -march=arm -disable-cgp-branch-opts | FileCheck %s

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
; CHECK: mov r0, #1, #24
        ret i32 256
}

define i32 @f4() {
; CHECK: f4
; CHECK: orr{{.*}}#1, #24
        ret i32 257
}

define i32 @f5() {
; CHECK: f5
; CHECK: mov r0, #255, #2
        ret i32 -1073741761
}

define i32 @f6() {
; CHECK: f6
; CHECK: mov r0, #63, #28
        ret i32 1008
}

define void @f7(i32 %a) {
; CHECK: f7
; CHECK: cmp r0, #1, #16
        %b = icmp ugt i32 %a, 65536
        br i1 %b, label %r, label %r
r:
        ret void
}

%t1 = type { <3 x float>, <3 x float> }

@const1 = global %t1 { <3 x float> zeroinitializer,
                       <3 x float> <float 1.000000e+00,
                                    float 2.000000e+00,
                                    float 3.000000e+00> }, align 16
; CHECK: const1
; CHECK: .zero 16
; CHECK: float 1.0
; CHECK: float 2.0
; CHECK: float 3.0
; CHECK: .zero 4
