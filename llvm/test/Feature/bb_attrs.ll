; RUN: llvm-as < %s | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

; Test for basic block attributes.

define i32 @f1(i32 %a) {
entry:
  %cmp = icmp slt i32 %a, 37
  br i1 %cmp, label %bb, label %lpad

bb:
  ret i32 37

lpad: landingpad
  ret i32 927
}

define i32 @f2(i32 %a) {
; entry : 0
  %1 = icmp slt i32 %a, 37
  br i1 %1, label %2, label %3

; bb : 2
  ret i32 37

landingpad ; bb : 3
  ret i32 927
}
