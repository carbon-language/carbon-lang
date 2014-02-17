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
; CHECK: kandw
; CHECK: kxorw
; CHECK: korw
  ret i16 %ret
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

; CHECK: zext_test1
; CHECK: kshiftlw
; CHECK: kshiftrw
; CHECK: kmovw
; CHECK:ret
define i32 @zext_test1(<16 x i32> %a, <16 x i32> %b) {
  %cmp_res = icmp ugt <16 x i32> %a, %b
  %cmp_res.i1 = extractelement <16 x i1> %cmp_res, i32 5
  %res = zext i1 %cmp_res.i1 to i32
  ret i32 %res
}

; CHECK: zext_test2
; CHECK: kshiftlw
; CHECK: kshiftrw
; CHECK: kmovw
; CHECK:ret
define i16 @zext_test2(<16 x i32> %a, <16 x i32> %b) {
  %cmp_res = icmp ugt <16 x i32> %a, %b
  %cmp_res.i1 = extractelement <16 x i1> %cmp_res, i32 5
  %res = zext i1 %cmp_res.i1 to i16
  ret i16 %res
}

; CHECK: zext_test3
; CHECK: kshiftlw
; CHECK: kshiftrw
; CHECK: kmovw
; CHECK:ret
define i8 @zext_test3(<16 x i32> %a, <16 x i32> %b) {
  %cmp_res = icmp ugt <16 x i32> %a, %b
  %cmp_res.i1 = extractelement <16 x i1> %cmp_res, i32 5
  %res = zext i1 %cmp_res.i1 to i8
  ret i8 %res
}
