; RUN: llc < %s -march=mips -relocation-model=static | FileCheck %s

define i32 @main() nounwind readnone {
entry:
  %x = alloca i32, align 4                        ; <i32*> [#uses=2]
  volatile store i32 2, i32* %x, align 4
  %0 = volatile load i32* %x, align 4             ; <i32> [#uses=1]
; CHECK: lui $3, %hi($JTI0_0)
; CHECK: addiu $3, $3, %lo($JTI0_0)
; CHECK: sll $2, $2, 2
  switch i32 %0, label %bb4 [
    i32 0, label %bb5
    i32 1, label %bb1
    i32 2, label %bb2
    i32 3, label %bb3
  ]

bb1:                                              ; preds = %entry
  ret i32 2

; CHECK: $BB0_2
bb2:                                              ; preds = %entry
  ret i32 0

bb3:                                              ; preds = %entry
  ret i32 3

bb4:                                              ; preds = %entry
  ret i32 4

bb5:                                              ; preds = %entry
  ret i32 1
}
