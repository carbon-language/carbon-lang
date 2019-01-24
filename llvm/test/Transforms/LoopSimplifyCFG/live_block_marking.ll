; XFAIL: *
; REQUIRES: asserts
; RUN: opt -S -enable-loop-simplifycfg-term-folding=true -indvars -loop-simplifycfg -debug-only=loop-simplifycfg -verify-loop-info -verify-dom-info -verify-loop-lcssa 2>&1 < %s | FileCheck %s
; RUN: opt -S -enable-loop-simplifycfg-term-folding=true -passes='require<domtree>,loop(indvars,simplify-cfg)' -debug-only=loop-simplifycfg -verify-loop-info -verify-dom-info -verify-loop-lcssa 2>&1 < %s | FileCheck %s
; RUN: opt -S -enable-loop-simplifycfg-term-folding=true -indvars -loop-simplifycfg -enable-mssa-loop-dependency=true -verify-memoryssa -debug-only=loop-simplifycfg -verify-loop-info -verify-dom-info -verify-loop-lcssa 2>&1 < %s | FileCheck %s

; This test demonstrates a bug in live blocks markup that is only catchable in
; inter-pass interaction.
define void @test(i1 %c) {

; CHECK-LABEL: @test(

entry:
  br label %outer

outer:
  br i1 %c, label %to_fold, label %latch

to_fold:
  br i1 %c, label %latch, label %inner

inner:
  %iv = phi i32 [0, %to_fold], [%iv.next, %inner_latch]
  %never = icmp sgt i32 %iv, 40
  br i1 %never, label %inner_latch, label %undead

inner_latch:
  %iv.next = add i32 %iv, 1
  %cmp = icmp slt i32 %iv.next, 10
  br i1 %cmp, label %inner, label %latch

undead:
  br label %latch

latch:
  br i1 true, label %outer, label %dead_exit

dead_exit:
  ret void
}
