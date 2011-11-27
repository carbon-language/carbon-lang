; Tests to make sure intrinsics are automatically upgraded.
; RUN: llvm-as < %s | llvm-dis | FileCheck %s


declare void @llvm.prefetch(i8*, i32, i32) nounwind

define void @p(i8* %ptr) {
; CHECK: llvm.prefetch(i8* %ptr, i32 0, i32 1, i32 1)
  tail call void @llvm.prefetch(i8* %ptr, i32 0, i32 1)
  ret void
}

declare i32 @nest_f(i8* nest, i32)
declare i8* @llvm.init.trampoline(i8*, i8*, i8*)

define void @test_trampolines() {
; CHECK: call void @llvm.init.trampoline(i8* null, i8* bitcast (i32 (i8*, i32)* @nest_f to i8*), i8* null)
; CHECK: call i8* @llvm.adjust.trampoline(i8* null)

  call i8* @llvm.init.trampoline(i8* null,
                                 i8* bitcast (i32 (i8*, i32)* @nest_f to i8*),
                                 i8* null)
  ret void
}
