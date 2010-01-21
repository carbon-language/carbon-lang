; RUN: llc < %s -march=x86-64 | FileCheck %s

; CodeGen should remat the zero instead of spilling it.

declare void @foo(i64 %p)

; CHECK: bar:
; CHECK: xorl %edi, %edi
; CHECK: xorl %edi, %edi
define void @bar() nounwind {
  call void @foo(i64 0)
  call void @foo(i64 0)
  ret void
}

; CHECK: bat:
; CHECK: movq $-1, %rdi
; CHECK: movq $-1, %rdi
define void @bat() nounwind {
  call void @foo(i64 -1)
  call void @foo(i64 -1)
  ret void
}

; CHECK: bau:
; CHECK: movl $1, %edi
; CHECK: movl $1, %edi
define void @bau() nounwind {
  call void @foo(i64 1)
  call void @foo(i64 1)
  ret void
}

