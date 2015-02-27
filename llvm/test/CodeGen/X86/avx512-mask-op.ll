; RUN: llc < %s -march=x86-64 -mtriple=x86_64-apple-darwin -mcpu=knl | FileCheck %s --check-prefix=KNL --check-prefix=CHECK
; RUN: llc < %s -march=x86-64 -mtriple=x86_64-apple-darwin -mcpu=skx | FileCheck %s --check-prefix=SKX --check-prefix=CHECK

; CHECK-LABEL: mask16
; CHECK: kmovw
; CHECK-NEXT: knotw
; CHECK-NEXT: kmovw
define i16 @mask16(i16 %x) {
  %m0 = bitcast i16 %x to <16 x i1>
  %m1 = xor <16 x i1> %m0, <i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1>
  %ret = bitcast <16 x i1> %m1 to i16
  ret i16 %ret
}

; CHECK-LABEL: mask8
; KNL: kmovw
; KNL-NEXT: knotw
; KNL-NEXT: kmovw
; SKX: kmovb
; SKX-NEXT: knotb
; SKX-NEXT: kmovb

define i8 @mask8(i8 %x) {
  %m0 = bitcast i8 %x to <8 x i1>
  %m1 = xor <8 x i1> %m0, <i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1>
  %ret = bitcast <8 x i1> %m1 to i8
  ret i8 %ret
}

; CHECK-LABEL: mask16_mem
; CHECK: kmovw ([[ARG1:%rdi|%rcx]]), %k{{[0-7]}}
; CHECK-NEXT: knotw
; CHECK-NEXT: kmovw %k{{[0-7]}}, ([[ARG1]])
; CHECK: ret

define void @mask16_mem(i16* %ptr) {
  %x = load i16, i16* %ptr, align 4
  %m0 = bitcast i16 %x to <16 x i1>
  %m1 = xor <16 x i1> %m0, <i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1>
  %ret = bitcast <16 x i1> %m1 to i16
  store i16 %ret, i16* %ptr, align 4
  ret void
}

; CHECK-LABEL: mask8_mem
; KNL: kmovw ([[ARG1]]), %k{{[0-7]}}
; KNL-NEXT: knotw
; KNL-NEXT: kmovw %k{{[0-7]}}, ([[ARG1]])
; SKX: kmovb ([[ARG1]]), %k{{[0-7]}}
; SKX-NEXT: knotb
; SKX-NEXT: kmovb %k{{[0-7]}}, ([[ARG1]])

define void @mask8_mem(i8* %ptr) {
  %x = load i8, i8* %ptr, align 4
  %m0 = bitcast i8 %x to <8 x i1>
  %m1 = xor <8 x i1> %m0, <i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1>
  %ret = bitcast <8 x i1> %m1 to i8
  store i8 %ret, i8* %ptr, align 4
  ret void
}

; CHECK-LABEL: mand16
; CHECK: kandw
; CHECK: kxorw
; CHECK: korw
define i16 @mand16(i16 %x, i16 %y) {
  %ma = bitcast i16 %x to <16 x i1>
  %mb = bitcast i16 %y to <16 x i1>
  %mc = and <16 x i1> %ma, %mb
  %md = xor <16 x i1> %ma, %mb
  %me = or <16 x i1> %mc, %md
  %ret = bitcast <16 x i1> %me to i16
  ret i16 %ret
}

; CHECK-LABEL: shuf_test1
; CHECK: kshiftrw        $8
define i8 @shuf_test1(i16 %v) nounwind {
   %v1 = bitcast i16 %v to <16 x i1>
   %mask = shufflevector <16 x i1> %v1, <16 x i1> undef, <8 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
   %mask1 = bitcast <8 x i1> %mask to i8
   ret i8 %mask1
}

; CHECK-LABEL: zext_test1
; CHECK: kshiftlw
; CHECK: kshiftrw
; CHECK: kmovw

define i32 @zext_test1(<16 x i32> %a, <16 x i32> %b) {
  %cmp_res = icmp ugt <16 x i32> %a, %b
  %cmp_res.i1 = extractelement <16 x i1> %cmp_res, i32 5
  %res = zext i1 %cmp_res.i1 to i32
  ret i32 %res
}

; CHECK-LABEL: zext_test2
; CHECK: kshiftlw
; CHECK: kshiftrw
; CHECK: kmovw

define i16 @zext_test2(<16 x i32> %a, <16 x i32> %b) {
  %cmp_res = icmp ugt <16 x i32> %a, %b
  %cmp_res.i1 = extractelement <16 x i1> %cmp_res, i32 5
  %res = zext i1 %cmp_res.i1 to i16
  ret i16 %res
}

; CHECK-LABEL: zext_test3
; CHECK: kshiftlw
; CHECK: kshiftrw
; CHECK: kmovw

define i8 @zext_test3(<16 x i32> %a, <16 x i32> %b) {
  %cmp_res = icmp ugt <16 x i32> %a, %b
  %cmp_res.i1 = extractelement <16 x i1> %cmp_res, i32 5
  %res = zext i1 %cmp_res.i1 to i8
  ret i8 %res
}

; CHECK-LABEL: conv1
; KNL: kmovw   %k0, %eax
; KNL: movb    %al, (%rdi)
; SKX: kmovb   %k0, (%rdi)
define i8 @conv1(<8 x i1>* %R) {
entry:
  store <8 x i1> <i1 1, i1 1, i1 1, i1 1, i1 1, i1 1, i1 1, i1 1>, <8 x i1>* %R

  %maskPtr = alloca <8 x i1>
  store <8 x i1> <i1 0, i1 1, i1 1, i1 1, i1 1, i1 1, i1 1, i1 1>, <8 x i1>* %maskPtr
  %mask = load <8 x i1>, <8 x i1>* %maskPtr
  %mask_convert = bitcast <8 x i1> %mask to i8
  ret i8 %mask_convert
}