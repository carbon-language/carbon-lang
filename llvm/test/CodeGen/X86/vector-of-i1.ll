; RUN: llc < %s -march=x86-64 | FileCheck %s

; Vectors of i1 are stored with each element having a
; different address. Since the address unit on x86 is 8 bits,
; that means each i1 value takes 8 bits of storage.

; CHECK: store:
; CHECK: movb  $1, 7(%rdi)
; CHECK: movb  $1, 6(%rdi)
; CHECK: movb  $0, 5(%rdi)
; CHECK: movb  $0, 4(%rdi)
; CHECK: movb  $1, 3(%rdi)
; CHECK: movb  $0, 2(%rdi)
; CHECK: movb  $1, 1(%rdi)
; CHECK: movb  $0, (%rdi)
define void @store(<8 x i1>* %p) nounwind {
  store <8 x i1> <i1 0, i1 1, i1 0, i1 1, i1 0, i1 0, i1 1, i1 1>, <8 x i1>* %p
  ret void
}

; CHECK: variable_extract:
; CHECK: movb  7(%rdi), 
; CHECK: movb  6(%rdi), 
; CHECK: movb  5(%rdi), 
define i32 @variable_extract(<8 x i1>* %p, i32 %n) nounwind {
  %t = load <8 x i1>* %p
  %s = extractelement <8 x i1> %t, i32 %n
  %e = zext i1 %s to i32
  ret i32 %e
}

; CHECK: constant_extract:
; CHECK: movzbl 3(%rdi), %eax
define i32 @constant_extract(<8 x i1>* %p, i32 %n) nounwind {
  %t = load <8 x i1>* %p
  %s = extractelement <8 x i1> %t, i32 3
  %e = zext i1 %s to i32
  ret i32 %e
}
