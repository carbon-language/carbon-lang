; Test that we can recurse, at least a little bit.  The -time-passes flag here
; is a hack to make sure that neither echo nor the shell expands the response
; file for us.  Tokenization with quotes is tested in unittests.
; RUN: echo %s > %t.list1
; RUN: echo "-time-passes @%t.list1" > %t.list2
; RUN: llvm-as @%t.list2 -o %t.bc
; RUN: llvm-nm %t.bc 2>&1 | FileCheck %s

; CHECK: T foobar

define void @foobar() {
  ret void
}
