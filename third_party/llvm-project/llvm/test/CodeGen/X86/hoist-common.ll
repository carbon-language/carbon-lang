; RUN: llc < %s -mtriple=x86_64-apple-macosx  | FileCheck %s
; This is supposed to be testing BranchFolding's common
; code hoisting logic, but has been erroneously passing due
; to there being a redundant xorl in the entry block
; and no common code to hoist.
; However, now that MachineSink sinks the redundant xor
; hoist-common looks at it and rejects it for hoisting,
; which causes this test to fail.
; Since it seems this test is broken, marking XFAIL for now
; until someone decides to remove it or fix what it tests.
; XFAIL: *

; Common "xorb al, al" instruction in the two successor blocks should be
; moved to the entry block above the test + je.

; rdar://9145558

define zeroext i1 @t(i32 %c) nounwind ssp {
entry:
; CHECK-LABEL: t:
; CHECK: xorl %eax, %eax
; CHECK: test
; CHECK: je
  %tobool = icmp eq i32 %c, 0
  br i1 %tobool, label %return, label %if.then

if.then:
; CHECK: callq
  %call = tail call zeroext i1 (...) @foo() nounwind
  br label %return

return:
; CHECK: ret
  %retval.0 = phi i1 [ %call, %if.then ], [ false, %entry ]
  ret i1 %retval.0
}

declare zeroext i1 @foo(...)
