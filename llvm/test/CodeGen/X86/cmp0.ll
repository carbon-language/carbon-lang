; RUN: llc < %s -march=x86-64 | FileCheck %s

define i64 @test0(i64 %x) nounwind {
  %t = icmp eq i64 %x, 0
  %r = zext i1 %t to i64
  ret i64 %r
; CHECK: test0:
; CHECK: 	testq	%rdi, %rdi
; CHECK: 	sete	%al
; CHECK: 	movzbl	%al, %eax
; CHECK: 	ret
}

define i64 @test1(i64 %x) nounwind {
  %t = icmp slt i64 %x, 1
  %r = zext i1 %t to i64
  ret i64 %r
; CHECK: test1:
; CHECK: 	testq	%rdi, %rdi
; CHECK: 	setle	%al
; CHECK: 	movzbl	%al, %eax
; CHECK: 	ret
}

