; RUN: llc < %s -mtriple=x86_64-apple-darwin10.0 -relocation-model=dynamic-no-pic --asm-verbose=0 | FileCheck %s

@.str1 = internal constant [31 x i8] c"DHRYSTONE PROGRAM, SOME STRING\00", align 8
@.str3 = internal constant [31 x i8] c"DHRYSTONE PROGRAM, 2'ND STRING\00", align 8

define void @func() nounwind ssp {
entry:
  %String2Loc = alloca [31 x i8], align 1
  br label %bb

bb:
  %String2Loc9 = getelementptr inbounds [31 x i8]* %String2Loc, i64 0, i64 0
  call void @llvm.memcpy.i64(i8* %String2Loc9, i8* getelementptr inbounds ([31 x i8]* @.str3, i64 0, i64 0), i64 31, i32 1)
; CHECK: movups _.str3
  br label %bb

return:
  ret void
}

declare void @llvm.memcpy.i64(i8* nocapture, i8* nocapture, i64, i32) nounwind

; CHECK: .align  3
; CHECK-NEXT: _.str1:
; CHECK-NEXT: .asciz "DHRYSTONE PROGRAM, SOME STRING"
; CHECK: .align 3
; CHECK-NEXT: _.str3:
