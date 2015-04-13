; RUN: llc -o - %s
; The selection DAG select(select()) normalisation crashed for different types
; on the condition inputs.
target datalayout = "E-m:m-p:32:32-i8:8:32-i16:16:32-i64:64-n32-S64"
target triple = "mips--"

define i64 @foobar(double %a) #0 {
entry:
  %0 = bitcast double %a to i64
  %trunc = trunc i64 %0 to i32
  %and = and i32 %trunc, 32767
  %sub = add nsw i32 %and, -16383
  %cmp = icmp ugt i32 %and, 16382
  %and5 = and i32 %trunc, 32768
  %tobool = icmp eq i32 %and5, 0
  %or = and i1 %cmp, %tobool
  %cmp6 = icmp ugt i32 %sub, 64
  %sext = sext i1 %cmp6 to i64
  %retval.0 = select i1 %or, i64 %sext, i64 0
  ret i64 %retval.0
}
