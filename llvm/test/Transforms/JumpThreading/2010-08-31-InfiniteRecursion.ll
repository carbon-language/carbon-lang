; RUN: opt < %s -jump-threading -disable-output
; ModuleID = 'bugpoint-reduced-simplified.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.4"

define void @encode_one_macroblock_highfast() nounwind ssp {
entry:
  switch i32 undef, label %bb13 [
    i32 1, label %bb10
    i32 2, label %bb12
  ]

bb10:                                             ; preds = %entry
  unreachable

bb12:                                             ; preds = %entry
  unreachable

bb13:                                             ; preds = %entry
  br i1 undef, label %bb137, label %bb292

bb137:                                            ; preds = %bb13
  br i1 undef, label %bb150, label %bb154

bb150:                                            ; preds = %bb137
  unreachable

bb154:                                            ; preds = %bb137
  br i1 undef, label %bb292, label %bb246

bb246:                                            ; preds = %bb154
  br i1 undef, label %bb292, label %bb247

bb247:                                            ; preds = %bb246
  br i1 undef, label %bb248, label %bb292

bb248:                                            ; preds = %bb247
  br i1 undef, label %bb249, label %bb292

bb249:                                            ; preds = %bb248
  br i1 undef, label %bb254, label %bb250

bb250:                                            ; preds = %bb249
  unreachable

bb254:                                            ; preds = %bb249
  br i1 undef, label %bb292, label %bb255

bb255:                                            ; preds = %bb288.bb289.loopexit_crit_edge, %bb254
  br i1 undef, label %bb.nph.split.us, label %bb269

bb.nph.split.us:                                  ; preds = %bb255
  br i1 undef, label %bb.nph.split.us.split.us, label %bb269.us.us31

bb.nph.split.us.split.us:                         ; preds = %bb.nph.split.us
  br i1 undef, label %bb269.us.us, label %bb269.us.us.us

bb269.us.us.us:                                   ; preds = %bb287.us.us.us, %bb.nph.split.us.split.us
  %indvar = phi i64 [ %indvar.next, %bb287.us.us.us ], [ 0, %bb.nph.split.us.split.us ] ; <i64> [#uses=1]
  %0 = icmp eq i16 undef, 0                       ; <i1> [#uses=1]
  br i1 %0, label %bb287.us.us.us, label %bb286.us.us.us

bb287.us.us.us:                                   ; preds = %bb269.us.us.us
  %indvar.next = add i64 %indvar, 1               ; <i64> [#uses=2]
  %exitcond = icmp eq i64 %indvar.next, 4         ; <i1> [#uses=1]
  br i1 %exitcond, label %bb288.bb289.loopexit_crit_edge, label %bb269.us.us.us

bb286.us.us.us:                                   ; preds = %bb269.us.us.us
  unreachable

bb269.us.us:                                      ; preds = %bb287.us.us, %bb.nph.split.us.split.us
  br i1 undef, label %bb287.us.us, label %bb286.us.us

bb287.us.us:                                      ; preds = %bb269.us.us
  br i1 undef, label %bb288.bb289.loopexit_crit_edge, label %bb269.us.us

bb286.us.us:                                      ; preds = %bb269.us.us
  unreachable

bb269.us.us31:                                    ; preds = %bb.nph.split.us
  unreachable

bb269:                                            ; preds = %bb255
  unreachable

bb288.bb289.loopexit_crit_edge:                   ; preds = %bb287.us.us, %bb287.us.us.us
  br i1 undef, label %bb292, label %bb255

bb292:                                            ; preds = %bb288.bb289.loopexit_crit_edge, %bb254, %bb248, %bb247, %bb246, %bb154, %bb13
  unreachable
}
