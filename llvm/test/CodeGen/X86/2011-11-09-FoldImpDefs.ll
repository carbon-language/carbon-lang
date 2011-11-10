; RUN: llc < %s -verify-regalloc | FileCheck %s
; PR11347
;
; This test case materializes the constant 1 in a register:
;
; %vreg19<def> = MOV32ri 1
;
; Then rematerializes the instruction for a sub-register copy:
; 1168L   %vreg14:sub_8bit<def,undef> = COPY %vreg19:sub_8bit<kill>, %vreg14<imp-def>; GR32:%vreg14,%vreg19
;        Considering merging %vreg19 with %vreg14
;                RHS = %vreg19 = [560d,656L:0)[720L,976d:0)[1088L,1168d:0)  0@560d
;                LHS = %vreg14 = [16d,160L:0)[160L,256L:2)[256L,1088L:1)[1168d,1184L:3)[1184L,1344L:2)  0@16d-phikill 1@256L-phidef-phikill 2@1184L-phidef-phikill 3@1168d-phikill
; Remat: %vreg14<def> = MOV32ri 1, %vreg14<imp-def>, %vreg14<imp-def>; GR32:%vreg14
;
; This rematerialized constant is feeding a PHI that is spilled, so the constant
; is written directly to a stack slot that gets the %esi function argument in
; another basic block:
;
; CHECK: %entry
; CHECK: movl %esi, [[FI:[0-9]+\(%rsp\)]]
; CHECK: %if.else24
; CHECK: movl $1, [[FI]]
; CHECK: %lor.end9
; CHECK: movl [[FI]],
;
; Those <imp-def> operands on the MOV32ri instruction confused the spiller
; because they were preserved by TII.foldMemoryOperand.  It is quite rare to
; see a rematerialized instruction spill, it can only happen when it is feeding
; a PHI.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.7"

@g_193 = external global i32, align 4
@g_103 = external global i32, align 4

declare i32 @func_21(i16 signext, i32) nounwind uwtable readnone ssp

define i32 @func_25(i32 %p_27, i8 signext %p_28, i32 %p_30) noreturn nounwind uwtable ssp {
entry:
  br label %for.cond

for.cond28.for.cond.loopexit_crit_edge:           ; preds = %for.cond28thread-pre-split
  store i32 0, i32* @g_103, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.cond28thread-pre-split, %for.cond28.for.cond.loopexit_crit_edge, %entry
  %l_365.0 = phi i32 [ undef, %entry ], [ %and, %for.cond28.for.cond.loopexit_crit_edge ], [ %and, %for.cond28thread-pre-split ]
  %l_288.0 = phi i32 [ undef, %entry ], [ %l_288.1.ph, %for.cond28.for.cond.loopexit_crit_edge ], [ %l_288.1.ph, %for.cond28thread-pre-split ]
  %l_349.0 = phi i32 [ undef, %entry ], [ %xor, %for.cond28.for.cond.loopexit_crit_edge ], [ %xor, %for.cond28thread-pre-split ]
  %p_28.addr.0 = phi i8 [ %p_28, %entry ], [ %p_28.addr.1.ph, %for.cond28.for.cond.loopexit_crit_edge ], [ %p_28.addr.1.ph, %for.cond28thread-pre-split ]
  br i1 undef, label %for.cond31, label %lor.end

lor.end:                                          ; preds = %for.cond
  %tobool3 = icmp eq i32 %l_349.0, 0
  br i1 %tobool3, label %for.cond31, label %if.then

if.then:                                          ; preds = %lor.end
  br i1 undef, label %lor.rhs6, label %lor.end9

lor.rhs6:                                         ; preds = %if.then
  br label %lor.end9

lor.end9:                                         ; preds = %lor.rhs6, %if.then
  %and = and i32 %l_365.0, 1
  %conv11 = sext i8 %p_28.addr.0 to i32
  %xor = xor i32 %and, %conv11
  br i1 false, label %if.else, label %if.end

if.else:                                          ; preds = %lor.end9
  br label %if.end

if.end:                                           ; preds = %if.else, %lor.end9
  %l_395.0 = phi i32 [ 0, %if.else ], [ 1, %lor.end9 ]
  %cmp14 = icmp ne i32 %and, %conv11
  %conv15 = zext i1 %cmp14 to i32
  br i1 %cmp14, label %if.then16, label %for.cond28thread-pre-split

if.then16:                                        ; preds = %if.end
  %or17 = or i32 %l_288.0, 1
  %call18 = tail call i32 @func_39(i32 0, i32 %or17, i32 0, i32 0) nounwind
  br i1 undef, label %if.else24, label %if.then20

if.then20:                                        ; preds = %if.then16
  %conv21 = trunc i32 %l_395.0 to i16
  %call22 = tail call i32 @func_21(i16 signext %conv21, i32 undef)
  br label %for.cond28thread-pre-split

if.else24:                                        ; preds = %if.then16
  store i32 %conv15, i32* @g_193, align 4
  %conv25 = trunc i32 %l_395.0 to i8
  br label %for.cond28thread-pre-split

for.cond28thread-pre-split:                       ; preds = %if.else24, %if.then20, %if.end
  %l_288.1.ph = phi i32 [ %l_288.0, %if.end ], [ %or17, %if.else24 ], [ %or17, %if.then20 ]
  %p_28.addr.1.ph = phi i8 [ %p_28.addr.0, %if.end ], [ %conv25, %if.else24 ], [ %p_28.addr.0, %if.then20 ]
  %.pr = load i32* @g_103, align 4
  %tobool2933 = icmp eq i32 %.pr, 0
  br i1 %tobool2933, label %for.cond, label %for.cond28.for.cond.loopexit_crit_edge

for.cond31:                                       ; preds = %for.cond31, %lor.end, %for.cond
  br label %for.cond31
}

declare i32 @func_39(i32, i32, i32, i32)
