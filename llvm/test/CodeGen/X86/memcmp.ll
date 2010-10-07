; RUN: llc %s -o - -march=x86-64 | FileCheck %s

; This tests codegen time inlining/optimization of memcmp
; rdar://6480398

@.str = private constant [23 x i8] c"fooooooooooooooooooooo\00", align 1 ; <[23 x i8]*> [#uses=1]

declare i32 @memcmp(...)

define void @memcmp2(i8* %X, i8* %Y, i32* nocapture %P) nounwind {
entry:
  %0 = tail call i32 (...)* @memcmp(i8* %X, i8* %Y, i32 2) nounwind ; <i32> [#uses=1]
  %1 = icmp eq i32 %0, 0                          ; <i1> [#uses=1]
  br i1 %1, label %return, label %bb

bb:                                               ; preds = %entry
  store i32 4, i32* %P, align 4
  ret void

return:                                           ; preds = %entry
  ret void
; CHECK: memcmp2:
; CHECK: movw    (%rdi), %ax
; CHECK: cmpw    (%rsi), %ax
}

define void @memcmp2a(i8* %X, i32* nocapture %P) nounwind {
entry:
  %0 = tail call i32 (...)* @memcmp(i8* %X, i8* getelementptr inbounds ([23 x i8]* @.str, i32 0, i32 1), i32 2) nounwind ; <i32> [#uses=1]
  %1 = icmp eq i32 %0, 0                          ; <i1> [#uses=1]
  br i1 %1, label %return, label %bb

bb:                                               ; preds = %entry
  store i32 4, i32* %P, align 4
  ret void

return:                                           ; preds = %entry
  ret void
; CHECK: memcmp2a:
; CHECK: cmpw    $28527, (%rdi)
}


define void @memcmp4(i8* %X, i8* %Y, i32* nocapture %P) nounwind {
entry:
  %0 = tail call i32 (...)* @memcmp(i8* %X, i8* %Y, i32 4) nounwind ; <i32> [#uses=1]
  %1 = icmp eq i32 %0, 0                          ; <i1> [#uses=1]
  br i1 %1, label %return, label %bb

bb:                                               ; preds = %entry
  store i32 4, i32* %P, align 4
  ret void

return:                                           ; preds = %entry
  ret void
; CHECK: memcmp4:
; CHECK: movl    (%rdi), %eax
; CHECK: cmpl    (%rsi), %eax
}

define void @memcmp4a(i8* %X, i32* nocapture %P) nounwind {
entry:
  %0 = tail call i32 (...)* @memcmp(i8* %X, i8* getelementptr inbounds ([23 x i8]* @.str, i32 0, i32 1), i32 4) nounwind ; <i32> [#uses=1]
  %1 = icmp eq i32 %0, 0                          ; <i1> [#uses=1]
  br i1 %1, label %return, label %bb

bb:                                               ; preds = %entry
  store i32 4, i32* %P, align 4
  ret void

return:                                           ; preds = %entry
  ret void
; CHECK: memcmp4a:
; CHECK: cmpl $1869573999, (%rdi)
}

define void @memcmp8(i8* %X, i8* %Y, i32* nocapture %P) nounwind {
entry:
  %0 = tail call i32 (...)* @memcmp(i8* %X, i8* %Y, i32 8) nounwind ; <i32> [#uses=1]
  %1 = icmp eq i32 %0, 0                          ; <i1> [#uses=1]
  br i1 %1, label %return, label %bb

bb:                                               ; preds = %entry
  store i32 4, i32* %P, align 4
  ret void

return:                                           ; preds = %entry
  ret void
; CHECK: memcmp8:
; CHECK: movq    (%rdi), %rax
; CHECK: cmpq    (%rsi), %rax
}

define void @memcmp8a(i8* %X, i32* nocapture %P) nounwind {
entry:
  %0 = tail call i32 (...)* @memcmp(i8* %X, i8* getelementptr inbounds ([23 x i8]* @.str, i32 0, i32 0), i32 8) nounwind ; <i32> [#uses=1]
  %1 = icmp eq i32 %0, 0                          ; <i1> [#uses=1]
  br i1 %1, label %return, label %bb

bb:                                               ; preds = %entry
  store i32 4, i32* %P, align 4
  ret void

return:                                           ; preds = %entry
  ret void
; CHECK: memcmp8a:
; CHECK: movabsq $8029759185026510694, %rax
; CHECK: cmpq	%rax, (%rdi)
}

