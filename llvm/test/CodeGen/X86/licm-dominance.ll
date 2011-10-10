; RUN: llc < %s | FileCheck %s

; MachineLICM should check dominance before hoisting instructions.
; CHECK:	xorb	%cl, %cl
; CHECK-NEXT:	testb	%cl, %cl

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-macosx10.7.2"

define void @CMSColorWorldCreateParametricData() nounwind uwtable optsize ssp {
entry:
  br label %for.body.i

for.body.i:                                       ; preds = %entry
  br i1 undef, label %for.inc.i, label %land.lhs.true21.i

land.lhs.true21.i:                                ; preds = %for.body.i
  br i1 undef, label %if.then26.i, label %for.inc.i

if.then26.i:                                      ; preds = %land.lhs.true21.i
  br i1 undef, label %if.else.i.i, label %if.then.i.i

if.then.i.i:                                      ; preds = %if.then26.i
  unreachable

if.else.i.i:                                      ; preds = %if.then26.i
  br i1 undef, label %lor.lhs.false.i.i, label %if.then116.i.i

lor.lhs.false.i.i:                                ; preds = %if.else.i.i
  br i1 undef, label %lor.lhs.false104.i.i, label %if.then116.i.i

lor.lhs.false104.i.i:                             ; preds = %lor.lhs.false.i.i
  br i1 undef, label %lor.lhs.false108.i.i, label %if.then116.i.i

lor.lhs.false108.i.i:                             ; preds = %lor.lhs.false104.i.i
  br i1 undef, label %lor.lhs.false112.i.i, label %if.then116.i.i

lor.lhs.false112.i.i:                             ; preds = %lor.lhs.false108.i.i
  br i1 undef, label %if.else232.i.i, label %if.then116.i.i

if.then116.i.i:                                   ; preds = %lor.lhs.false112.i.i, %lor.lhs.false108.i.i, %lor.lhs.false104.i.i, %lor.lhs.false.i.i, %if.else.i.i
  unreachable

if.else232.i.i:                                   ; preds = %lor.lhs.false112.i.i
  br label %for.inc.i

for.inc.i:                                        ; preds = %if.else232.i.i, %land.lhs.true21.i, %for.body.i
  %cmp17.i = icmp ult i64 undef, undef
  br i1 %cmp17.i, label %for.body.i, label %if.end28.i

if.end28.i:                                       ; preds = %for.inc.i, %if.then10.i, %if.then6.i
  unreachable

createTransformParams.exit:                       ; preds = %land.lhs.true3.i, %if.then.i, %land.lhs.true.i, %entry
  ret void
}
