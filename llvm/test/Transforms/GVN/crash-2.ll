; RUN: opt -instcombine -gvn -S %s
; PR5733
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32"
target triple = "i686-linux-gnu"

@ff_cropTbl = external global [2304 x i8]         ; <[2304 x i8]*> [#uses=1]

define void @ff_pred8x8_plane_c(i8* nocapture %src, i32 %stride) nounwind {
entry:
  br i1 true, label %bb.nph41, label %for.end


bb.nph41:                                         ; preds = %entry
  %sub.ptr.neg37 = sub i32 0, %stride             ; <i32> [#uses=1]
  %tmp55 = add i32 %sub.ptr.neg37, -1             ; <i32> [#uses=1]
  %tmp73 = add i32 %stride, -1                    ; <i32> [#uses=2]
  %sub.ptr38 = getelementptr i8* %src, i32 %tmp73 ; <i8*> [#uses=1]
  %tmp60 = load i8* %sub.ptr38                    ; <i8> [#uses=1]
  %conv61 = zext i8 %tmp60 to i32                 ; <i32> [#uses=1]
  %sub62 = sub i32 0, %conv61                     ; <i32> [#uses=1]
  %mul63 = mul i32 %sub62, 2                      ; <i32> [#uses=1]
  %add65 = add nsw i32 %mul63, 0                  ; <i32> [#uses=1]
  %tmp71.2 = mul i32 2, %stride                   ; <i32> [#uses=1]
  %tmp72.2 = sub i32 0, %tmp71.2                  ; <i32> [#uses=1]
  %tmp74.2 = add i32 %tmp72.2, %tmp73             ; <i32> [#uses=1]
  %sub.ptr38.2 = getelementptr i8* %src, i32 %tmp74.2 ; <i8*> [#uses=1]
  %tmp60.2 = load i8* %sub.ptr38.2                ; <i8> [#uses=1]
  %conv61.2 = zext i8 %tmp60.2 to i32             ; <i32> [#uses=1]
  %sub62.2 = sub i32 0, %conv61.2                 ; <i32> [#uses=1]
  %mul63.2 = mul i32 %sub62.2, 4                  ; <i32> [#uses=1]
  %add65.2 = add nsw i32 %mul63.2, %add65       ; <i32> [#uses=1]
  %scevgep = getelementptr i8* %src, i32 %tmp55   ; <i8*> [#uses=1]
  br label %for.end

for.end:                                          ; preds = %for.cond.2, %entry
  %V.0.lcssa = phi i32 [ %add65.2, %bb.nph41 ], [ 0, %entry ] ; <i32> [#uses=1]
  %src2.0.lcssa = phi i8* [ %scevgep, %bb.nph41 ], [ null, %entry ] ; <i8*> [#uses=1]
  %mul71 = mul i32 %V.0.lcssa, 17                 ; <i32> [#uses=1]
  %add72 = add nsw i32 %mul71, 16                 ; <i32> [#uses=1]
  %shr73 = ashr i32 %add72, 5                     ; <i32> [#uses=1]
  br i1 true, label %bb.nph, label %for.end181

bb.nph:                                           ; preds = %for.end
  %arrayidx79 = getelementptr inbounds i8* %src2.0.lcssa, i32 8 ; <i8*> [#uses=1]
  %tmp80 = load i8* %arrayidx79                   ; <i8> [#uses=1]
  %tmp9 = mul i32 %shr73, -3                      ; <i32> [#uses=1]
  %tmp11 = zext i8 %tmp80 to i32                  ; <i32> [#uses=1]
  %tmp12 = add i32 0, %tmp11                      ; <i32> [#uses=1]
  %tmp13 = mul i32 %tmp12, 16                     ; <i32> [#uses=1]
  %tmp38 = add i32 %tmp9, %tmp13                  ; <i32> [#uses=1]
  %tmp39 = add i32 %tmp38, 16                     ; <i32> [#uses=1]
  br label %for.body94

for.body94:                                       ; preds = %bb.nph, %for.cond90
  %add129 = add i32 0, %tmp39                     ; <i32> [#uses=1]
  %shr130 = ashr i32 %add129, 5                   ; <i32> [#uses=1]
  %.sum4 = add i32 %shr130, 1024                  ; <i32> [#uses=1]
  %arrayidx132 = getelementptr inbounds [2304 x i8]* @ff_cropTbl, i32 0, i32 %.sum4 ; <i8*> [#uses=1]
  %tmp133 = load i8* %arrayidx132                 ; <i8> [#uses=1]
  store i8 %tmp133, i8* undef
  br label %for.body94

for.end181:                                       ; preds = %for.cond90.for.end181_crit_edge, %for.end
  ret void

}
