; RUN: llc < %s -o - -mcpu=generic -march=x86-64 -mattr=+sse2 | FileCheck %s
; RUN: llc < %s -o - -mcpu=generic -march=x86-64 -mattr=+avx | FileCheck %s

define <8 x i16> @foo(<8 x i16> %a, <8 x i16> %b) {
; CHECK: .short	      32
; CHECK-NEXT: .short	      32
; CHECK-NEXT: .short	      32
; CHECK-NEXT: .short	      32
; CHECK-NEXT: .short	      32
; CHECK-NEXT: .short	      32
; CHECK-NEXT: .short	      32
; CHECK-NEXT: .short	      32
; CHECK-LABEL: {{^_?foo:}}
; CHECK-NOT: psll
entry:
  %icmp = icmp eq <8 x i16> %a, %b
  %zext = zext <8 x i1> %icmp to <8 x i16>
  %shl = shl nuw nsw <8 x i16> %zext, <i16 5, i16 5, i16 5, i16 5, i16 5, i16 5, i16 5, i16 5>
  ret <8 x i16> %shl
}

; Don't fail with an assert due to an undef in the buildvector
define <8 x i16> @bar(<8 x i16> %a, <8 x i16> %b) {
; CHECK-LABEL: bar
entry:
  %icmp = icmp eq <8 x i16> %a, %b
  %zext = zext <8 x i1> %icmp to <8 x i16>
  %shl = shl nuw nsw <8 x i16> %zext, <i16 5, i16 undef, i16 5, i16 5, i16 5, i16 5, i16 5, i16 5>
  ret <8 x i16> %shl
}
