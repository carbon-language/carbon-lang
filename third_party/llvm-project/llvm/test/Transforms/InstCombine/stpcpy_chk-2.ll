; Test that lib call simplification doesn't simplify __stpcpy_chk calls
; with the wrong prototype.
;
; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"

@a = common global [60 x i16] zeroinitializer, align 1
@.str = private constant [8 x i8] c"abcdefg\00"

define void @test_no_simplify() {
; CHECK-LABEL: @test_no_simplify(
  %dst = getelementptr inbounds [60 x i16], [60 x i16]* @a, i32 0, i32 0
  %src = getelementptr inbounds [8 x i8], [8 x i8]* @.str, i32 0, i32 0

; CHECK-NEXT: call i16* @__strcpy_chk
  call i16* @__strcpy_chk(i16* %dst, i8* %src, i32 8)
  ret void
}

declare i16* @__strcpy_chk(i16*, i8*, i32)
