; RUN: llvm-as < %s | llc -mtriple=i386-mingw32-pc | FileCheck %s

declare dllimport void @foo()

define void @bar() nounwind {
; CHECK: call	*__imp__foo
  call void @foo()
  ret void
}
