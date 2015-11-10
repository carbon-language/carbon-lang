; RUN: opt %loadPolly -polly-code-generator=isl -polly-scops -analyze < %s | FileCheck %s
;
; Verify we introduce two ScopArrayInfo objects (or virtual arrays) for the %out variable
; as it is used as a memory base pointer (%0) but also as a scalar (%out.addr.0.lcssa).
;
; CHECK:         Arrays {
; CHECK-NEXT:        float* MemRef_out; // Element size 0
; CHECK-NEXT:        float* MemRef_out_addr_0_lcssa; // Element size 0
; CHECK-NEXT:        float MemRef_out[*]; // Element size 4
; CHECK-NEXT:    }
;
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

; Function Attrs: nounwind ssp uwtable
define void @ff_celp_lp_synthesis_filterf(float* %out) #0 {
entry:
  br label %entry.split

entry.split:                                      ; preds = %entry
  br i1 false, label %for.end.97, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %entry.split
  %arrayidx13 = getelementptr inbounds float, float* %out, i64 -3
  %0 = load float, float* %arrayidx13, align 4
  br label %for.body

for.body:                                         ; preds = %for.end, %for.body.lr.ph
  br i1 false, label %for.body.50.lr.ph, label %for.end

for.body.50.lr.ph:                                ; preds = %for.body
  br label %for.body.50

for.body.50:                                      ; preds = %for.body.50, %for.body.50.lr.ph
  br i1 false, label %for.body.50, label %for.cond.48.for.end_crit_edge

for.cond.48.for.end_crit_edge:                    ; preds = %for.body.50
  br label %for.end

for.end:                                          ; preds = %for.cond.48.for.end_crit_edge, %for.body
  %add96 = add nuw nsw i32 0, 4
  %cmp = icmp sgt i32 %add96, 0
  br i1 %cmp, label %for.cond.for.end.97_crit_edge, label %for.body

for.cond.for.end.97_crit_edge:                    ; preds = %for.end
  br label %for.end.97

for.end.97:                                       ; preds = %for.cond.for.end.97_crit_edge, %entry.split
  %out.addr.0.lcssa = phi float* [ undef, %for.cond.for.end.97_crit_edge ], [ %out, %entry.split ]
  br i1 undef, label %for.body.104.lr.ph, label %for.end.126

for.body.104.lr.ph:                               ; preds = %for.end.97
  br label %for.body.104

for.body.104:                                     ; preds = %for.inc.124, %for.body.104.lr.ph
  br i1 undef, label %for.inc.124, label %for.body.111.lr.ph

for.body.111.lr.ph:                               ; preds = %for.body.104
  br label %for.body.111

for.body.111:                                     ; preds = %for.body.111, %for.body.111.lr.ph
  br i1 undef, label %for.body.111, label %for.cond.109.for.inc.124_crit_edge

for.cond.109.for.inc.124_crit_edge:               ; preds = %for.body.111
  br label %for.inc.124

for.inc.124:                                      ; preds = %for.cond.109.for.inc.124_crit_edge, %for.body.104
  br i1 undef, label %for.body.104, label %for.cond.102.for.end.126_crit_edge

for.cond.102.for.end.126_crit_edge:               ; preds = %for.inc.124
  br label %for.end.126

for.end.126:                                      ; preds = %for.cond.102.for.end.126_crit_edge, %for.end.97
  ret void
}
