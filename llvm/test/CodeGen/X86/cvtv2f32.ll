; RUN: llc < %s -mtriple=i686-linux-pc -mcpu=corei7 | FileCheck %s

define <2 x float> @bar(<2 x i32> %in) {
  %r = uitofp <2 x i32> %in to <2 x float>
  ret <2 x float> %r
; CHECK: bar
; CHECK: or
; CHECK: subpd
; CHECK: cvtpd2ps
; CHECK: ret
}
