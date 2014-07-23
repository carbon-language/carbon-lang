; RUN: llc < %s -march=x86-64 -mcpu=skx | FileCheck %s

define i8 @mask8(i8 %x) {
  %m0 = bitcast i8 %x to <8 x i1>
  %m1 = xor <8 x i1> %m0, <i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1>
  %ret = bitcast <8 x i1> %m1 to i8
  ret i8 %ret
; CHECK: mask8
; CHECK: knotb
; CHECK: ret
}

define void @mask8_mem(i8* %ptr) {
  %x = load i8* %ptr, align 4
  %m0 = bitcast i8 %x to <8 x i1>
  %m1 = xor <8 x i1> %m0, <i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1, i1 -1>
  %ret = bitcast <8 x i1> %m1 to i8
  store i8 %ret, i8* %ptr, align 4
  ret void
; CHECK-LABEL: mask8_mem
; CHECK: kmovb (%rdi), %k{{[0-7]}}
; CHECK-NEXT: knotb
; CHECK-NEXT: kmovb %k{{[0-7]}}, (%rdi)
; CHECK: ret
}
