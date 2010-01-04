; RUN: llc < %s
target datalayout = "E-p:64:64:64-i8:8:16-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-a0:16:16-n32:64"
target triple = "s390x-elf"

@REGISTER = external global [10 x i32]            ; <[10 x i32]*> [#uses=2]

define void @DIVR_P(i32 signext %PRINT_EFFECT) nounwind {
entry:
  %REG1 = alloca i32, align 4                     ; <i32*> [#uses=2]
  %REG2 = alloca i32, align 4                     ; <i32*> [#uses=2]
  %call = call signext i32 (...)* @FORMAT2(i32* %REG1, i32* %REG2) nounwind ; <i32> [#uses=0]
  %tmp = load i32* %REG1                          ; <i32> [#uses=1]
  %idxprom = sext i32 %tmp to i64                 ; <i64> [#uses=1]
  %arrayidx = getelementptr inbounds [10 x i32]* @REGISTER, i64 0, i64 %idxprom ; <i32*> [#uses=2]
  %tmp1 = load i32* %arrayidx                     ; <i32> [#uses=2]
  %tmp2 = load i32* %REG2                         ; <i32> [#uses=1]
  %idxprom3 = sext i32 %tmp2 to i64               ; <i64> [#uses=1]
  %arrayidx4 = getelementptr inbounds [10 x i32]* @REGISTER, i64 0, i64 %idxprom3 ; <i32*> [#uses=3]
  %tmp5 = load i32* %arrayidx4                    ; <i32> [#uses=3]
  %cmp6 = icmp sgt i32 %tmp5, 8388607             ; <i1> [#uses=1]
  %REG2_SIGN.0 = select i1 %cmp6, i32 -1, i32 1   ; <i32> [#uses=2]
  %cmp10 = icmp eq i32 %REG2_SIGN.0, 1            ; <i1> [#uses=1]
  %not.cmp = icmp slt i32 %tmp1, 8388608          ; <i1> [#uses=2]
  %or.cond = and i1 %cmp10, %not.cmp              ; <i1> [#uses=1]
  br i1 %or.cond, label %if.then13, label %if.end25

if.then13:                                        ; preds = %entry
  %div = sdiv i32 %tmp5, %tmp1                    ; <i32> [#uses=2]
  store i32 %div, i32* %arrayidx4
  br label %if.end25

if.end25:                                         ; preds = %if.then13, %entry
  %tmp35 = phi i32 [ %div, %if.then13 ], [ %tmp5, %entry ] ; <i32> [#uses=1]
  %cmp27 = icmp eq i32 %REG2_SIGN.0, -1           ; <i1> [#uses=1]
  %or.cond46 = and i1 %cmp27, %not.cmp            ; <i1> [#uses=1]
  br i1 %or.cond46, label %if.then31, label %if.end45

if.then31:                                        ; preds = %if.end25
  %sub = sub i32 16777216, %tmp35                 ; <i32> [#uses=1]
  %tmp39 = load i32* %arrayidx                    ; <i32> [#uses=1]
  %div40 = udiv i32 %sub, %tmp39                  ; <i32> [#uses=1]
  %sub41 = sub i32 16777216, %div40               ; <i32> [#uses=1]
  store i32 %sub41, i32* %arrayidx4
  ret void

if.end45:                                         ; preds = %if.end25
  ret void
}

declare signext i32 @FORMAT2(...)
