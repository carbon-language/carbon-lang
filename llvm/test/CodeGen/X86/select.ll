; RUN: llc < %s -march=x86-64 | FileCheck %s
; PR5757

; CHECK: cmovneq %rdi, %rsi
; CHECK: movl (%rsi), %eax

%0 = type { i64, i32 }

define i32 @foo(%0* %p, %0* %q, i1 %r) nounwind {
  %t0 = load %0* %p
  %t1 = load %0* %q
  %t4 = select i1 %r, %0 %t0, %0 %t1
  %t5 = extractvalue %0 %t4, 1
  ret i32 %t5
}
