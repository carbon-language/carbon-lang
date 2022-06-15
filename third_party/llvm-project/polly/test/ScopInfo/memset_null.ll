; RUN: opt %loadPolly -polly-allow-modref-calls -polly-print-scops -disable-output < %s | FileCheck %s
; RUN: opt %loadPolly -polly-allow-modref-calls -S -polly-codegen < %s
;
; Verify we can handle a memset to "null" and that we do not model it.
; TODO: FIXME: We could use the undefined memset to optimize the code further,
;              see the TODOs in the ScopInfo.cpp.
;
; CHECK:         Statements {
; CHECK-NEXT:        Stmt_for_cond5_preheader_us221
; CHECK-NEXT:            Domain :=
; CHECK-NEXT:                { Stmt_for_cond5_preheader_us221[0] };
; CHECK-NEXT:            Schedule :=
; CHECK-NEXT:                { Stmt_for_cond5_preheader_us221[i0] -> [0] };
; CHECK-NEXT:            MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                { Stmt_for_cond5_preheader_us221[i0] -> MemRef_A[0] };
; CHECK-NEXT:    }

;
target datalayout = "e-m:e-i64:64-i128:128-n8:16:32:64-S128"

define void @test(i32* %A) {
entry:
  br i1 undef, label %for.end68, label %for.cond5.preheader.lr.ph

for.cond5.preheader.lr.ph:                        ; preds = %entry
  br label %for.cond5.preheader.us221

for.cond5.preheader.us221:                        ; preds = %for.cond5.preheader.us221, %for.cond5.preheader.lr.ph
  store i32 0, i32* %A
  call void @llvm.memset.p0i8.i64(i8* null, i8 0, i64 undef, i32 1, i1 false)
  br i1 true, label %for.end68, label %for.cond5.preheader.us221

for.end68:                                        ; preds = %for.cond5.preheader.us221, %entry
  ret void
}

declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i32, i1)
