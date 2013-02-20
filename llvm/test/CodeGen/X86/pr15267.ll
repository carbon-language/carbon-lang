; RUN: llc < %s -mtriple=x86_64-pc-linux -mcpu=corei7-avx | FileCheck %s

define <4 x i3> @test1(<4 x i3>* %in) nounwind {
  %ret = load <4 x i3>* %in, align 1
  ret <4 x i3> %ret
}

; CHECK: test1
; CHECK: movzwl
; CHECK: shrl $3
; CHECK: andl $7
; CHECK: andl $7
; CHECK: vmovd
; CHECK: pinsrd $1
; CHECK: shrl $6
; CHECK: andl $7
; CHECK: pinsrd $2
; CHECK: shrl $9
; CHECK: andl $7
; CHECK: pinsrd $3
; CHECK: ret

define <4 x i1> @test2(<4 x i1>* %in) nounwind {
  %ret = load <4 x i1>* %in, align 1
  ret <4 x i1> %ret
}

; CHECK: test2
; CHECK: movzbl
; CHECK: shrl
; CHECK: andl $1
; CHECK: andl $1
; CHECK: vmovd
; CHECK: pinsrd $1
; CHECK: shrl $2
; CHECK: andl $1
; CHECK: pinsrd $2
; CHECK: shrl $3
; CHECK: andl $1
; CHECK: pinsrd $3
; CHECK: ret

define <4 x i64> @test3(<4 x i1>* %in) nounwind {
  %wide.load35 = load <4 x i1>* %in, align 1
  %sext = sext <4 x i1> %wide.load35 to <4 x i64>
  ret <4 x i64> %sext
}

; CHECK: test3
; CHECK: movzbl
; CHECK: shrl
; CHECK: andl $1
; CHECK: andl $1
; CHECK: vmovd
; CHECK: pinsrd $1
; CHECK: shrl $2
; CHECK: andl $1
; CHECK: pinsrd $2
; CHECK: shrl $3
; CHECK: andl $1
; CHECK: pinsrd $3
; CHECK: pslld
; CHECK: psrad
; CHECK: pmovsxdq
; CHECK: pmovsxdq
; CHECK: ret
