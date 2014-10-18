; RUN: opt -instcombine -S < %s | FileCheck %s

define i32 @test_load_cast_combine_tbaa(float* %ptr) {
; Ensure (cast (load (...))) -> (load (cast (...))) preserves TBAA.
; CHECK-LABEL: @test_load_cast_combine_tbaa(
; CHECK: load i32* %{{.*}}, !tbaa !0
entry:
  %l = load float* %ptr, !tbaa !0
  %c = bitcast float %l to i32
  ret i32 %c
}

define i32 @test_load_cast_combine_noalias(float* %ptr) {
; Ensure (cast (load (...))) -> (load (cast (...))) preserves no-alias metadata.
; CHECK-LABEL: @test_load_cast_combine_noalias(
; CHECK: load i32* %{{.*}}, !alias.scope !2, !noalias !1
entry:
  %l = load float* %ptr, !alias.scope !2, !noalias !1
  %c = bitcast float %l to i32
  ret i32 %c
}

!0 = metadata !{ metadata !1, metadata !1, i64 0 }
!1 = metadata !{ metadata !1 }
!2 = metadata !{ metadata !2, metadata !1 }
