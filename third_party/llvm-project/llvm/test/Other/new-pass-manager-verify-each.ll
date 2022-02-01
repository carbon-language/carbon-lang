; RUN: opt -disable-output -debug-pass-manager -verify-each -passes='no-op-module,verify,cgscc(no-op-cgscc,function(no-op-function,loop(no-op-loop)))' %s 2>&1 | FileCheck %s

; Added manually by opt at beginning
; CHECK: Running pass: VerifierPass

; CHECK: Running pass: NoOpModulePass
; CHECK: Verifying module
; CHECK-NOT: Verifying module
; CHECK: Running pass: NoOpCGSCCPass
; CHECK: Verifying module
; CHECK-NOT: Verifying module
; CHECK: Running pass: NoOpFunctionPass
; CHECK: Verifying function foo
; CHECK: Running pass: LoopSimplifyPass
; CHECK: Verifying function foo
; CHECK: Running pass: LCSSAPass
; CHECK: Verifying function foo
; CHECK: Running pass: NoOpLoopPass
; CHECK: Verifying function foo
; CHECK-NOT: Verifying function
; CHECK-NOT: Verifying module

; Added manually by opt at end
; CHECK: Running pass: VerifierPass

define void @foo(i1 %x, i8* %p1, i8* %p2) {
entry:
  store i8 42, i8* %p1
  br i1 %x, label %loop, label %exit

loop:
  %tmp1 = load i8, i8* %p2
  br label %loop

exit:
  ret void
}

declare void @bar()
