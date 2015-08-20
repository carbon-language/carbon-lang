; RUN: opt %loadPolly -tbaa -polly-scops -polly-ignore-aliasing -analyze < %s | FileCheck %s
;
; CHECK: Arrays {
; CHECK:   i32** MemRef_A[*][8]
; CHECK:   i32* MemRef_tmp3[*][8] [BasePtrOrigin: MemRef_A]
; CHECK:   i32 MemRef_tmp5[*][4] [BasePtrOrigin: MemRef_tmp3]
; CHECK: }
;
; CHECK: Arrays (Bounds as pw_affs) {
; CHECK:   i32** MemRef_A[*][ { [] -> [(8)] } ]
; CHECK:   i32* MemRef_tmp3[*][ { [] -> [(8)] } ] [BasePtrOrigin: MemRef_A]
; CHECK:   i32 MemRef_tmp5[*][ { [] -> [(4)] } ] [BasePtrOrigin: MemRef_tmp3]
; CHECK: }
;
;    void f(int ***A) {
;      for (int i = 0; i < 1024; i++)
;        A[42][32][i] = 0;
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(i32*** %A) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb7, %bb
  %indvars.iv = phi i64 [ %indvars.iv.next, %bb7 ], [ 0, %bb ]
  %exitcond = icmp ne i64 %indvars.iv, 1024
  br i1 %exitcond, label %bb2, label %bb8

bb2:                                              ; preds = %bb1
  %tmp = getelementptr inbounds i32**, i32*** %A, i64 42
  %tmp3 = load i32**, i32*** %tmp, align 8, !tbaa !1
  %tmp4 = getelementptr inbounds i32*, i32** %tmp3, i64 32
  %tmp5 = load i32*, i32** %tmp4, align 8, !tbaa !1
  %tmp6 = getelementptr inbounds i32, i32* %tmp5, i64 %indvars.iv
  store i32 0, i32* %tmp6, align 4, !tbaa !5
  br label %bb7

bb7:                                              ; preds = %bb2
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %bb1

bb8:                                              ; preds = %bb1
  ret void
}

!0 = !{!"clang version 3.8.0 (http://llvm.org/git/clang.git 9e282ff441e7a367dc711e41fd19d27ffc0f78d6)"}
!1 = !{!2, !2, i64 0}
!2 = !{!"any pointer", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
!5 = !{!6, !6, i64 0}
!6 = !{!"int", !3, i64 0}
