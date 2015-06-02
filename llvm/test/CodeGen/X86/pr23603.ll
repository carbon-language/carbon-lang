; RUN: llc < %s -mtriple=x86_64-apple-darwin | FileCheck %s

declare void @free_v()

define void @f(i32* %x, i32 %c32, i32* %y) {
; CHECK-LABEL: f
 entry:
  %v = load i32, i32* %x, !invariant.load !0
; CHECK: movl (%rdi), %ebx
; CHECK: free_v
; CHECK-NOT: movl (%rdi), %ebx
  call void @free_v()
  %c = icmp ne i32 %c32, 0
  br i1 %c, label %left, label %merge

 left:
  store i32 %v, i32* %y
  br label %merge

 merge:
  ret void
}

!0 = !{}
