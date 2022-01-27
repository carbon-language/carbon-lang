; RUN: llc < %s -mtriple=thumbv7-apple-ios -O0 -verify-machineinstrs -fast-isel-abort=1 -relocation-model=static -mattr=+long-calls | FileCheck -check-prefix=CHECK-LONG %s
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -O0 -verify-machineinstrs -fast-isel-abort=1 -relocation-model=static -mattr=+long-calls | FileCheck -check-prefix=CHECK-LONG %s
; RUN: llc < %s -mtriple=thumbv7-apple-ios -O0 -verify-machineinstrs -fast-isel-abort=1 -relocation-model=static | FileCheck -check-prefix=CHECK-NORM %s
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -O0 -verify-machineinstrs -fast-isel-abort=1 -relocation-model=static | FileCheck -check-prefix=CHECK-NORM %s

define void @myadd(float* %sum, float* %addend) nounwind {
entry:
  %sum.addr = alloca float*, align 4
  %addend.addr = alloca float*, align 4
  store float* %sum, float** %sum.addr, align 4
  store float* %addend, float** %addend.addr, align 4
  %tmp = load float*, float** %sum.addr, align 4
  %tmp1 = load float, float* %tmp
  %tmp2 = load float*, float** %addend.addr, align 4
  %tmp3 = load float, float* %tmp2
  %add = fadd float %tmp1, %tmp3
  %tmp4 = load float*, float** %sum.addr, align 4
  store float %add, float* %tmp4
  ret void
}

define i32 @main(i32 %argc, i8** %argv) nounwind {
entry:
  %ztot = alloca float, align 4
  %z = alloca float, align 4
  store float 0.000000e+00, float* %ztot, align 4
  store float 1.000000e+00, float* %z, align 4
; CHECK-LONG: blx     r
; CHECK-NORM: bl      {{_?}}myadd
  call void @myadd(float* %ztot, float* %z)
  ret i32 0
}
