; RUN: llc < %s -mtriple=x86_64-unknown-unknown | FileCheck %s

define dso_local void @test_cancel2(i40* %p1, i40* %p2) {
; CHECK:       # %entry
; CHECK-NEXT:  movl    (%rdi), %eax
; CHECK-NEXT:  shrl    %eax
; CHECK-NEXT:  andl    $524287, %eax
; CHECK-NEXT:  movl    %eax, (%rsi)
; CHECK-NEXT:  movb    $0, 4(%rsi)
; CHECK-NEXT:  retq
entry:
  %0 = load i40, i40* %p1, align 8
  %shl414 = shl i40 %0, 19
  %unsclear415 = and i40 %shl414, 549755813887
  %shr416 = lshr i40 %unsclear415, 20
  %unsclear417 = and i40 %shr416, 549755813887
  store i40 %unsclear417, i40* %p2, align 8
  ret void
}

