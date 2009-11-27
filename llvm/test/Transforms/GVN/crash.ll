; RUN: opt -gvn %s -disable-output

; PR5631

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0"

define i32* @peel_to_type(i8* %name, i32 %namelen, i32* %o, i32 %expected_type) nounwind ssp {
entry:
  br i1 undef, label %if.end13, label %while.body.preheader


if.end13:                                         ; preds = %if.then6
  br label %while.body.preheader

while.body.preheader:                             ; preds = %if.end13, %if.end
  br label %while.body

while.body:                                       ; preds = %while.body.backedge, %while.body.preheader
  %o.addr.0 = phi i32* [ undef, %while.body.preheader ], [ %o.addr.0.be, %while.body.backedge ] ; <i32*> [#uses=2]
  br i1 false, label %return.loopexit, label %lor.lhs.false

lor.lhs.false:                                    ; preds = %while.body
  %tmp20 = bitcast i32* %o.addr.0 to i32*         ; <i32*> [#uses=1]
  %tmp22 = load i32* %tmp20                       ; <i32> [#uses=0]
  br i1 undef, label %land.lhs.true24, label %if.end31

land.lhs.true24:                                  ; preds = %lor.lhs.false
  %call28 = call i32* @parse_object(i8* undef) nounwind ; <i32*> [#uses=0]
  br i1 undef, label %return.loopexit, label %if.end31

if.end31:                                         ; preds = %land.lhs.true24, %lor.lhs.false
  br i1 undef, label %return.loopexit, label %if.end41

if.end41:                                         ; preds = %if.end31
  %tmp43 = bitcast i32* %o.addr.0 to i32*         ; <i32*> [#uses=1]
  %tmp45 = load i32* %tmp43                       ; <i32> [#uses=0]
  br i1 undef, label %if.then50, label %if.else

if.then50:                                        ; preds = %if.end41
  %tmp53 = load i32** undef                       ; <i32*> [#uses=1]
  br label %while.body.backedge

if.else:                                          ; preds = %if.end41
  br i1 undef, label %if.then62, label %if.else67

if.then62:                                        ; preds = %if.else
  br label %while.body.backedge

while.body.backedge:                              ; preds = %if.then62, %if.then50
  %o.addr.0.be = phi i32* [ %tmp53, %if.then50 ], [ undef, %if.then62 ] ; <i32*> [#uses=1]
  br label %while.body

if.else67:                                        ; preds = %if.else
  ret i32* null

return.loopexit:                                  ; preds = %if.end31, %land.lhs.true24, %while.body
  ret i32* undef
}

declare i32* @parse_object(i8*)
