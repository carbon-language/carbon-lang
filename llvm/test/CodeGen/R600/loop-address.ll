;RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s

;CHECK: VTX
;CHECK: ALU_PUSH
;CHECK: JUMP @4
;CHECK: ELSE @16
;CHECK: VTX
;CHECK: LOOP_START_DX10 @15
;CHECK: LOOP_BREAK @14
;CHECK: POP @16

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-v2048:2048:2048-n32:64"
target triple = "r600--"

define void @loop_ge(i32 addrspace(1)* nocapture %out, i32 %iterations) #0 {
entry:
  %cmp5 = icmp sgt i32 %iterations, 0
  br i1 %cmp5, label %for.body, label %for.end

for.body:                                         ; preds = %for.body, %entry
  %i.07.in = phi i32 [ %i.07, %for.body ], [ %iterations, %entry ]
  %ai.06 = phi i32 [ %add, %for.body ], [ 0, %entry ]
  %i.07 = add nsw i32 %i.07.in, -1
  %arrayidx = getelementptr inbounds i32 addrspace(1)* %out, i32 %ai.06
  store i32 %i.07, i32 addrspace(1)* %arrayidx, align 4, !tbaa !4
  %add = add nsw i32 %ai.06, 1
  %exitcond = icmp eq i32 %add, %iterations
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
}

attributes #0 = { nounwind "fp-contract-model"="standard" "relocation-model"="pic" "ssp-buffers-size"="8" }

!opencl.kernels = !{!0, !1, !2, !3}

!0 = metadata !{void (i32 addrspace(1)*, i32)* @loop_ge}
!1 = metadata !{null}
!2 = metadata !{null}
!3 = metadata !{null}
!4 = metadata !{metadata !"int", metadata !5}
!5 = metadata !{metadata !"omnipotent char", metadata !6}
!6 = metadata !{metadata !"Simple C/C++ TBAA"}
