; RUN: opt < %s -basic-aa -dse -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.8.0"

%struct.anon = type { i32, i32, i32 }
@b = global %struct.anon { i32 1, i32 0, i32 0 }, align 4
@c = common global i32 0, align 4
@a = common global %struct.anon zeroinitializer, align 4
@.str = private unnamed_addr constant [4 x i8] c"%d\0A\00", align 1

declare i32 @printf(i8* nocapture, ...) nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i1) nounwind


; Make sure that the initial memcpy call does not go away
; because the volatile load is in the way. PR12899

; CHECK: main_entry:
; CHECK-NEXT: tail call void @llvm.memcpy.p0i8.p0i8.i64

define i32 @main() nounwind uwtable ssp {
main_entry:
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 bitcast (%struct.anon* @b to i8*), i8* align 4 bitcast (%struct.anon* @a to i8*), i64 12, i1 false)
  %0 = load volatile i32, i32* getelementptr inbounds (%struct.anon, %struct.anon* @b, i64 0, i32 0), align 4
  store i32 %0, i32* @c, align 4
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 bitcast (%struct.anon* @b to i8*), i8* align 4 bitcast (%struct.anon* @a to i8*), i64 12, i1 false) nounwind
  %call = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str, i64 0, i64 0), i32 %0) nounwind
  ret i32 0
}
