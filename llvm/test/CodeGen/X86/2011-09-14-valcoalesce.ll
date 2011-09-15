; RUN: llc < %s -march=x86
;
; Test RegistersDefinedFromSameValue. We have multiple copies of the same vreg:
; while.body85.i:
;   vreg1 = copy vreg2
;   vreg2 = add
; critical edge from land.lhs.true.i -> if.end117.i:
;   vreg27 = vreg2
; critical edge from land.lhs.true103.i -> if.end117.i:
;   vreg27 = vreg2
; if.then108.i:
;   vreg27 = vreg1
;
; Prior to fixing PR10920 401.bzip miscompile, the coalescer would
; consider vreg1 and vreg27 to be copies of the same value. It would
; then remove one of the critical edge copes, which cannot safely be removed.

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128-n8:16:32"

@.str3 = external unnamed_addr constant [59 x i8], align 1

define void @test() nounwind ssp {
entry:
  br i1 undef, label %if.then68, label %if.end85

if.then68:                                        ; preds = %entry
  br label %for.body.i.i

for.body.i.i:                                     ; preds = %for.inc.i.i, %if.then68
  br i1 undef, label %for.inc.i.i, label %if.then.i.i

if.then.i.i:                                      ; preds = %for.body.i.i
  br label %for.inc.i.i

for.inc.i.i:                                      ; preds = %if.then.i.i, %for.body.i.i
  br i1 undef, label %makeMaps_e.exit.i, label %for.body.i.i

makeMaps_e.exit.i:                                ; preds = %for.inc.i.i
  br i1 undef, label %for.cond19.preheader.i, label %for.cond.for.cond19.preheader_crit_edge.i

for.cond.for.cond19.preheader_crit_edge.i:        ; preds = %makeMaps_e.exit.i
  unreachable

for.cond19.preheader.i:                           ; preds = %makeMaps_e.exit.i
  br i1 undef, label %if.then.i, label %if.end.i

if.then.i:                                        ; preds = %for.cond19.preheader.i
  br label %if.end.i

if.end.i:                                         ; preds = %if.then.i, %for.cond19.preheader.i
  br i1 undef, label %for.inc27.us.5.i, label %for.end30.i

for.end30.i:                                      ; preds = %if.end.i
  br i1 undef, label %if.end36.i, label %if.then35.i

if.then35.i:                                      ; preds = %for.end30.i
  unreachable

if.end36.i:                                       ; preds = %for.end30.i
  br label %while.body.i188

for.cond182.preheader.i:                          ; preds = %for.cond138.preheader.i
  unreachable

while.body.i188:                                  ; preds = %for.cond138.preheader.i, %if.end36.i
  %sub.i187 = add nsw i32 0, -1
  br i1 undef, label %while.body85.i, label %if.end117.i

while.body85.i:                                   ; preds = %while.body85.i, %while.body.i188
  %ge.0519.i = phi i32 [ %inc87.i, %while.body85.i ], [ %sub.i187, %while.body.i188 ]
  %aFreq.0518.i = phi i32 [ %add93.i, %while.body85.i ], [ 0, %while.body.i188 ]
  %inc87.i = add nsw i32 %ge.0519.i, 1
  %tmp91.i = load i32* undef, align 4, !tbaa !0
  %add93.i = add nsw i32 %tmp91.i, %aFreq.0518.i
  %cmp84.i = icmp slt i32 %inc87.i, undef
  %or.cond514.i = and i1 undef, %cmp84.i
  br i1 %or.cond514.i, label %while.body85.i, label %while.end.i

while.end.i:                                      ; preds = %while.body85.i
  br i1 undef, label %land.lhs.true.i, label %if.end117.i

land.lhs.true.i:                                  ; preds = %while.end.i
  br i1 undef, label %if.end117.i, label %land.lhs.true103.i

land.lhs.true103.i:                               ; preds = %land.lhs.true.i
  br i1 undef, label %if.then108.i, label %if.end117.i

if.then108.i:                                     ; preds = %land.lhs.true103.i
  br label %if.end117.i

if.end117.i:                                      ; preds = %if.then108.i, %land.lhs.true103.i, %land.lhs.true.i, %while.end.i, %while.body.i188
  %aFreq.1.i = phi i32 [ %aFreq.0518.i, %if.then108.i ], [ %add93.i, %land.lhs.true103.i ], [ %add93.i, %while.end.i ], [ 0, %while.body.i188 ], [ %add93.i, %land.lhs.true.i ]
  %cmp121.i = icmp sgt i32 undef, 2
  br i1 %cmp121.i, label %if.then122.i, label %for.cond138.preheader.i

if.then122.i:                                     ; preds = %if.end117.i
  call void (...)* @fprintf(i32 undef, i32 0, i32 undef, i32 %aFreq.1.i, double undef) nounwind
  unreachable

for.cond138.preheader.i:                          ; preds = %if.end117.i
  %cmp73.i = icmp sgt i32 undef, 0
  br i1 %cmp73.i, label %while.body.i188, label %for.cond182.preheader.i

for.inc27.us.5.i:                                 ; preds = %if.end.i
  unreachable

if.end85:                                         ; preds = %entry
  ret void
}

declare void @fprintf(...) nounwind

!0 = metadata !{metadata !"int", metadata !1}
!1 = metadata !{metadata !"omnipotent char", metadata !2}
!2 = metadata !{metadata !"Simple C/C++ TBAA", null}
