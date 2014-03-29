; RUN: llc < %s -O0 -fast-isel-abort -mtriple=arm64-apple-darwin | FileCheck %s

@fn.table = internal global [2 x i8*] [i8* blockaddress(@fn, %ZERO), i8* blockaddress(@fn, %ONE)], align 8

define i32 @fn(i32 %target) nounwind {
entry:
; CHECK: @fn
  %retval = alloca i32, align 4
  %target.addr = alloca i32, align 4
  store i32 %target, i32* %target.addr, align 4
  %0 = load i32* %target.addr, align 4
  %idxprom = zext i32 %0 to i64
  %arrayidx = getelementptr inbounds [2 x i8*]* @fn.table, i32 0, i64 %idxprom
  %1 = load i8** %arrayidx, align 8
  br label %indirectgoto

ZERO:                                             ; preds = %indirectgoto
; CHECK: LBB0_1
  store i32 0, i32* %retval
  br label %return

ONE:                                              ; preds = %indirectgoto
; CHECK: LBB0_2
  store i32 1, i32* %retval
  br label %return

return:                                           ; preds = %ONE, %ZERO
  %2 = load i32* %retval
  ret i32 %2

indirectgoto:                                     ; preds = %entry
; CHECK: ldr x0, [sp]
; CHECK: br x0
  %indirect.goto.dest = phi i8* [ %1, %entry ]
  indirectbr i8* %indirect.goto.dest, [label %ZERO, label %ONE]
}
