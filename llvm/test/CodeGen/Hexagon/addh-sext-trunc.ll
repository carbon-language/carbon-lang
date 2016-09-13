; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK: r{{[0-9]+}} = add(r{{[0-9]+}}.{{L|l}}, r{{[0-9]+}}.{{H|h}})

target datalayout = "e-p:32:32:32-i64:64:64-i32:32:32-i16:16:16-i1:32:32-f64:64:64-f32:32:32-v64:64:64-v32:32:32-a0:0-n16:32"
target triple = "hexagon-unknown-none"

%struct.aDataType = type { i16, i16, i16, i16, i16, i16*, i16*, i16*, i8*, i16*, i16*, i16*, i8* }

define i8* @a_get_score(%struct.aDataType* nocapture %pData, i16 signext %gmmModelIndex, i16* nocapture %pGmmScoreL16Q4) #0 {
entry:
  %numSubVector = getelementptr inbounds %struct.aDataType, %struct.aDataType* %pData, i32 0, i32 3
  %0 = load i16, i16* %numSubVector, align 2, !tbaa !0
  %and = and i16 %0, -4
  %b = getelementptr inbounds %struct.aDataType, %struct.aDataType* %pData, i32 0, i32 8
  %1 = load i8*, i8** %b, align 4, !tbaa !3
  %conv3 = sext i16 %and to i32
  %cmp21 = icmp sgt i16 %and, 0
  br i1 %cmp21, label %for.inc.preheader, label %for.end

for.inc.preheader:                                ; preds = %entry
  br label %for.inc

for.inc:                                          ; preds = %for.inc.preheader, %for.inc
  %j.022 = phi i32 [ %phitmp, %for.inc ], [ 0, %for.inc.preheader ]
  %add13 = mul i32 %j.022, 65536
  %sext = add i32 %add13, 262144
  %phitmp = ashr exact i32 %sext, 16
  %cmp = icmp slt i32 %phitmp, %conv3
  br i1 %cmp, label %for.inc, label %for.end.loopexit

for.end.loopexit:                                 ; preds = %for.inc
  br label %for.end

for.end:                                          ; preds = %for.end.loopexit, %entry
  ret i8* %1
}

attributes #0 = { nounwind readonly "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-frame-pointer-elim-non-leaf"="true" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }

!0 = !{!"short", !1}
!1 = !{!"omnipotent char", !2}
!2 = !{!"Simple C/C++ TBAA"}
!3 = !{!"any pointer", !1}
