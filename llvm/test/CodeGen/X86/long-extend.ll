; RUN: llc < %s -mcpu=core-avx-i -mtriple=x86_64-linux -asm-verbose=0| FileCheck %s
define void @test_long_extend(<16 x i8> %a, <16 x i32>* %p) nounwind {
; CHECK-LABEL: test_long_extend
; CHECK: vpunpcklbw	%xmm1, %xmm0, [[REG1:%xmm[0-9]+]]
; CHECK: vpunpckhwd	%xmm1, [[REG1]], [[REG2:%xmm[0-9]+]]
; CHECK: vpunpcklwd	%xmm1, [[REG1]], %x[[REG3:mm[0-9]+]]
; CHECK: vinsertf128	$1, [[REG2]], %y[[REG3]], [[REG_result0:%ymm[0-9]+]]
; CHECK: vpunpckhbw	%xmm1, %xmm0, [[REG4:%xmm[0-9]+]]
; CHECK: vpunpckhwd	%xmm1, [[REG4]], [[REG5:%xmm[0-9]+]]
; CHECK: vpunpcklwd	%xmm1, [[REG4]], %x[[REG6:mm[0-9]+]]
; CHECK: vinsertf128	$1, [[REG5]], %y[[REG6]], [[REG_result1:%ymm[0-9]+]]
; CHECK: vmovaps	[[REG_result1]], 32(%rdi)
; CHECK: vmovaps	[[REG_result0]], (%rdi)

  %tmp = zext <16 x i8> %a to <16 x i32>
  store <16 x i32> %tmp, <16 x i32>*%p
  ret void
}
