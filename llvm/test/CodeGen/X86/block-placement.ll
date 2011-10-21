; RUN: llc -march=x86 -enable-block-placement < %s | FileCheck %s

declare void @error(i32 %i, i32 %a, i32 %b)

define i32 @test1(i32 %i, i32* %a, i32 %b) {
; Test a chain of ifs, where the block guarded by the if is error handling code
; that is not expected to run.
; CHECK: test1:
; CHECK: %entry
; CHECK: %else1
; CHECK: %else2
; CHECK: %else3
; CHECK: %else4
; CHECK: %exit
; CHECK: %then1
; CHECK: %then2
; CHECK: %then3
; CHECK: %then4
; CHECK: %then5

entry:
  %gep1 = getelementptr i32* %a, i32 1
  %val1 = load i32* %gep1
  %cond1 = icmp ugt i32 %val1, 1
  br i1 %cond1, label %then1, label %else1, !prof !0

then1:
  call void @error(i32 %i, i32 1, i32 %b)
  br label %else1

else1:
  %gep2 = getelementptr i32* %a, i32 2
  %val2 = load i32* %gep2
  %cond2 = icmp ugt i32 %val2, 2
  br i1 %cond2, label %then2, label %else2, !prof !0

then2:
  call void @error(i32 %i, i32 1, i32 %b)
  br label %else2

else2:
  %gep3 = getelementptr i32* %a, i32 3
  %val3 = load i32* %gep3
  %cond3 = icmp ugt i32 %val3, 3
  br i1 %cond3, label %then3, label %else3, !prof !0

then3:
  call void @error(i32 %i, i32 1, i32 %b)
  br label %else3

else3:
  %gep4 = getelementptr i32* %a, i32 4
  %val4 = load i32* %gep4
  %cond4 = icmp ugt i32 %val4, 4
  br i1 %cond4, label %then4, label %else4, !prof !0

then4:
  call void @error(i32 %i, i32 1, i32 %b)
  br label %else4

else4:
  %gep5 = getelementptr i32* %a, i32 3
  %val5 = load i32* %gep5
  %cond5 = icmp ugt i32 %val5, 3
  br i1 %cond5, label %then5, label %exit, !prof !0

then5:
  call void @error(i32 %i, i32 1, i32 %b)
  br label %exit

exit:
  ret i32 %b
}

!0 = metadata !{metadata !"branch_weights", i32 4, i32 64}
