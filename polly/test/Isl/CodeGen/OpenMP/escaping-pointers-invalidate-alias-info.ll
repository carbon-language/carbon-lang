; RUN: opt %loadPolly -basicaa -tbaa -polly-parallel -polly-codegen-isl -polly-code-generator=isl -S < %s | FileCheck %s

; CHECK: polly.split_new_and_old:
; CHECK-NOT: polly.split_new_and_old:
; CHECK: @CalculateQuant8Param.polly.subfn

; In this test case the first loop is only detected as a scop because 


target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@active_sps = external global [8 x i32]*

define void @CalculateQuant8Param() {
entry:
  %present = alloca i64, align 8
  %present19 = bitcast i64* %present to i8*
  br label %for.1

for.1:
  %indvar.1 = phi i64 [ %indvar.1.next, %for.1 ], [ 0, %entry ]
  %tmp = load [8 x i32]** @active_sps, !tbaa !1
  %arrayidx.1b = getelementptr [8 x i32]* %tmp, i32 0, i64 0
  %tmp1 = load i32* %arrayidx.1b, !tbaa !5

  %arrayidx.1a = getelementptr i8* %present19, i64 0
  %arrayidx.1c = bitcast i8* %arrayidx.1a to i32*
  store i32 %tmp1, i32* %arrayidx.1c
  %indvar.1.next = add i64 %indvar.1, 1
  br i1 false, label %for.1, label %fence

fence:
  fence seq_cst
  br label %for.2

for.2:
  %indvar.2 = phi i64 [ %indvar.2.next, %for.2 ], [ 0, %fence ]
  %uglygep = getelementptr i8* %present19, i64 %indvar.2
  %arrayidx.2 = bitcast i8* %uglygep to i32*
  store i32 42, i32* %arrayidx.2
  %indvar.2.next = add i64 %indvar.2, 1
  %exitcond18 = icmp ne i64 %indvar.2.next, 2
  br i1 %exitcond18, label %for.2, label %end

end:
  ret void
}

!1 = metadata !{metadata !2, metadata !2, i64 0}
!2 = metadata !{metadata !"any pointer", metadata !3, i64 0}
!3 = metadata !{metadata !"omnipotent char", metadata !4, i64 0}
!4 = metadata !{metadata !"Simple C/C++ TBAA"}
!5 = metadata !{metadata !6, metadata !6, i64 0}
!6 = metadata !{metadata !"int", metadata !3, i64 0}
