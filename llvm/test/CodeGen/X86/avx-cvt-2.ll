; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx | FileCheck %s

; Check that we generate vector conversion from float to narrower int types

%f32vec_t = type <8 x float>
%i16vec_t = type <8 x i16>
%i8vec_t =  type <8 x i8>

define void @fptoui16(%f32vec_t %a, %i16vec_t *%p) {
; CHECK-LABEL: fptoui16:
; CHECK: vcvttps2dq %ymm
; CHECK-NOT: vcvttss2si
  %b = fptoui %f32vec_t %a to %i16vec_t
  store %i16vec_t %b, %i16vec_t * %p
  ret void
}

define void @fptosi16(%f32vec_t %a, %i16vec_t *%p) {
; CHECK-LABEL: fptosi16:
; CHECK: vcvttps2dq %ymm
; CHECK-NOT: vcvttss2si
  %b = fptosi %f32vec_t %a to %i16vec_t
  store %i16vec_t %b, %i16vec_t * %p
  ret void
}

define void @fptoui8(%f32vec_t %a, %i8vec_t *%p) {
; CHECK-LABEL: fptoui8:
; CHECK: vcvttps2dq %ymm
; CHECK-NOT: vcvttss2si
  %b = fptoui %f32vec_t %a to %i8vec_t
  store %i8vec_t %b, %i8vec_t * %p
  ret void
}

define void @fptosi8(%f32vec_t %a, %i8vec_t *%p) {
; CHECK-LABEL: fptosi8:
; CHECK: vcvttps2dq %ymm
; CHECK-NOT: vcvttss2si
  %b = fptosi %f32vec_t %a to %i8vec_t
  store %i8vec_t %b, %i8vec_t * %p
  ret void
}
