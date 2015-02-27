; RUN: llc %s -mcpu=core2 -o - | FileCheck %s
; This test was failing while tracking the liveness in the register scavenger
; during the branching folding pass. The allocation of the subregisters was
; incorrect.
; I.e., the faulty pattern looked like:
; CH = movb 64
; ECX = movl 3 <- CH was killed here.
; CH = subb CH, ...
;
; This reduced test case triggers the crash before the fix, but does not
; strictly speaking check that the resulting code is correct.
; To check that the code is actually correct we would need to check the
; liveness of the produced code.
;
; Currently, we check that after ECX = movl 3, we do not have subb CH,
; whereas CH could have been redefine in between and that would have been
; totally fine.
; <rdar://problem/16582185>
target datalayout = "e-m:o-p:32:32-f64:32:64-f80:128-n8:16:32-S128"
target triple = "i386-apple-macosx10.9"

%struct.A = type { %struct.B, %struct.C, %struct.D*, [1 x i8*] }
%struct.B = type { i32, [4 x i8] }
%struct.C = type { i128 }
%struct.D = type { {}*, [0 x i32] }
%union.E = type { i32 }

; CHECK-LABEL: __XXX1:
; CHECK: movl $3, %ecx
; CHECK-NOT: subb %{{[a-z]+}}, %ch
; Function Attrs: nounwind optsize ssp
define fastcc void @__XXX1(%struct.A* %ht) #0 {
entry:
  %const72 = bitcast i128 72 to i128
  %const3 = bitcast i128 3 to i128
  switch i32 undef, label %if.end196 [
    i32 1, label %sw.bb.i
    i32 3, label %sw.bb2.i
  ]

sw.bb.i:                                          ; preds = %entry
  %call.i.i.i = tail call i32 undef(%struct.A* %ht, i8 zeroext 22, i32 undef, i32 0, %struct.D* undef)
  %bf.load.i.i = load i128* undef, align 4
  %bf.lshr.i.i = lshr i128 %bf.load.i.i, %const72
  %shl1.i.i = shl nuw nsw i128 %bf.lshr.i.i, 8
  %shl.i.i = trunc i128 %shl1.i.i to i32
  br i1 undef, label %cond.false10.i.i, label %__XXX2.exit.i.i

__XXX2.exit.i.i:                    ; preds = %sw.bb.i
  %extract11.i.i.i = lshr i128 %bf.load.i.i, %const3
  %extract.t12.i.i.i = trunc i128 %extract11.i.i.i to i32
  %bf.cast7.i.i.i = and i32 %extract.t12.i.i.i, 3
  %arrayidx.i.i.i = getelementptr inbounds %struct.A, %struct.A* %ht, i32 0, i32 3, i32 %bf.cast7.i.i.i
  br label %cond.end12.i.i

cond.false10.i.i:                                 ; preds = %sw.bb.i
  %arrayidx.i6.i.i = getelementptr inbounds %struct.A, %struct.A* %ht, i32 0, i32 3, i32 0
  br label %cond.end12.i.i

cond.end12.i.i:                                   ; preds = %cond.false10.i.i, %__XXX2.exit.i.i
  %.sink.in.i.i = phi i8** [ %arrayidx.i.i.i, %__XXX2.exit.i.i ], [ %arrayidx.i6.i.i, %cond.false10.i.i ]
  %.sink.i.i = load i8** %.sink.in.i.i, align 4
  %tmp = bitcast i8* %.sink.i.i to %union.E*
  br i1 undef, label %for.body.i.i, label %if.end196

for.body.i.i:                                     ; preds = %for.body.i.i, %cond.end12.i.i
  %weak.i.i = getelementptr inbounds %union.E, %union.E* %tmp, i32 undef, i32 0
  %tmp1 = load i32* %weak.i.i, align 4
  %cmp36.i.i = icmp ne i32 %tmp1, %shl.i.i
  %or.cond = and i1 %cmp36.i.i, false
  br i1 %or.cond, label %for.body.i.i, label %if.end196

sw.bb2.i:                                         ; preds = %entry
  %bf.lshr.i85.i = lshr i128 undef, %const72
  br i1 undef, label %if.end196, label %__XXX2.exit.i95.i

__XXX2.exit.i95.i:                  ; preds = %sw.bb2.i
  %extract11.i.i91.i = lshr i128 undef, %const3
  br label %if.end196

if.end196:                                        ; preds = %__XXX2.exit.i95.i, %sw.bb2.i, %for.body.i.i, %cond.end12.i.i, %entry
  ret void
}

attributes #0 = { nounwind optsize ssp "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" }
