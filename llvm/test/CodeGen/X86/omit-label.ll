; RUN: llvm-as < %s | llc -asm-verbose=false -mtriple=x86_64-linux-gnu | FileCheck %s
; PR4126
; PR4732

; Don't omit these labels' definitions.

; CHECK: bux:
; CHECK: LBB1_1:

define void @bux(i32 %p_53) nounwind optsize {
entry:
	%0 = icmp eq i32 %p_53, 0		; <i1> [#uses=1]
	%1 = icmp sgt i32 %p_53, 0		; <i1> [#uses=1]
	%or.cond = and i1 %0, %1		; <i1> [#uses=1]
	br i1 %or.cond, label %bb.i, label %bb3

bb.i:		; preds = %entry
	%2 = add i32 %p_53, 1		; <i32> [#uses=1]
	%3 = icmp slt i32 %2, 0		; <i1> [#uses=0]
	br label %bb3

bb3:		; preds = %bb.i, %entry
	%4 = tail call i32 (...)* @baz(i32 0) nounwind		; <i32> [#uses=0]
	ret void
}

declare i32 @baz(...)

; Don't omit this label in the assembly output.
; CHECK: int321:
; CHECK: LBB2_1
; CHECK: LBB2_1
; CHECK: LBB2_1:

define void @int321(i8 signext %p_103, i32 %uint8p_104) nounwind readnone {
entry:
  %tobool = icmp eq i8 %p_103, 0                  ; <i1> [#uses=1]
  %cmp.i = icmp sgt i8 %p_103, 0                  ; <i1> [#uses=1]
  %or.cond = and i1 %tobool, %cmp.i               ; <i1> [#uses=1]
  br i1 %or.cond, label %land.end.i, label %for.cond.preheader

land.end.i:                                       ; preds = %entry
  %conv3.i = sext i8 %p_103 to i32                ; <i32> [#uses=1]
  %div.i = sdiv i32 1, %conv3.i                   ; <i32> [#uses=1]
  %tobool.i = icmp eq i32 %div.i, -2147483647     ; <i1> [#uses=0]
  br label %for.cond.preheader

for.cond.preheader:                               ; preds = %land.end.i, %entry
  %cmp = icmp sgt i8 %p_103, 1                    ; <i1> [#uses=1]
  br i1 %cmp, label %for.end.split, label %for.cond

for.cond:                                         ; preds = %for.cond.preheader, %for.cond
  br label %for.cond

for.end.split:                                    ; preds = %for.cond.preheader
  ret void
}
