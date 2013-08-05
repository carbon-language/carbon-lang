; RUN: llc < %s -march=x86-64 -mcpu=knl | FileCheck %s

define i16 @mask16(i16 %x) {
  %m0 = bitcast i16 %x to <16 x i1>
  %m1 = xor <16 x i1> %m0, <i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1>
  %ret = bitcast <16 x i1> %m1 to i16
  ret i16 %ret
; CHECK: mask16
; CHECK: knotw
; CHECK: ret
}

define i8 @mask8(i8 %x) {
  %m0 = bitcast i8 %x to <8 x i1>
  %m1 = xor <8 x i1> %m0, <i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1>
  %ret = bitcast <8 x i1> %m1 to i8
  ret i8 %ret
; CHECK: mask8
; CHECK: knotw
; CHECK: ret
}

define i16 @mand16(i16 %x, i16 %y) {
  %ma = bitcast i16 %x to <16 x i1>
  %mb = bitcast i16 %y to <16 x i1>
  %mc = and <16 x i1> %ma, %mb
  %md = xor <16 x i1> %ma, %mb
  %me = or <16 x i1> %mc, %md
  %ret = bitcast <16 x i1> %me to i16
; CHECK: kxorw
; CHECK: kandw
; CHECK: korw
  ret i16 %ret
}

; CHECK: unpckbw_test
; CHECK: kunpckbw
; CHECK:ret
declare <16 x i1> @llvm.x86.kunpck.v16i1(<8 x i1>, <8 x i1>) nounwind readnone

define i16 @unpckbw_test(i8 %x, i8 %y) {
  %m0 = bitcast i8 %x to <8 x i1>
  %m1 = bitcast i8 %y to <8 x i1>
  %k = tail call <16 x i1> @llvm.x86.kunpck.v16i1(<8 x i1> %m0, <8 x i1> %m1)
  %r = bitcast <16 x i1> %k to i16
  ret i16 %r
}

; CHECK: shuf_test1
; CHECK: kshiftrw        $8
; CHECK:ret
define i8 @shuf_test1(i16 %v) nounwind {
   %v1 = bitcast i16 %v to <16 x i1>
   %mask = shufflevector <16 x i1> %v1, <16 x i1> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
   %mask1 = bitcast <8 x i1> %mask to i8
   ret i8 %mask1
}

