; RUN: llc -march=aarch64 < %s | FileCheck %s

define i8 @f_v16i8(<16 x i8>* %arr)  {
; CHECK-LABEL: f_v16i8
; CHECK: addv {{b[0-9]+}}, {{v[0-9]+}}.16b
  %bin.rdx = load <16 x i8>, <16 x i8>* %arr
  %rdx.shuf0 = shufflevector <16 x i8> %bin.rdx, <16 x i8> undef, <16 x i32> <i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 undef, i32 undef,i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx0 = add <16 x i8> %bin.rdx, %rdx.shuf0
  %rdx.shuf = shufflevector <16 x i8> %bin.rdx0, <16 x i8> undef, <16 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef,i32 undef, i32 undef, i32 undef, i32 undef,i32 undef, i32 undef, i32 undef, i32 undef,i32 undef, i32 undef >
  %bin.rdx11 = add <16 x i8> %bin.rdx0, %rdx.shuf
  %rdx.shuf12 = shufflevector <16 x i8> %bin.rdx11, <16 x i8> undef, <16 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef,i32 undef, i32 undef, i32 undef, i32 undef,i32 undef, i32 undef>
  %bin.rdx13 = add <16 x i8> %bin.rdx11, %rdx.shuf12
  %rdx.shuf13 = shufflevector <16 x i8> %bin.rdx13, <16 x i8> undef, <16 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef,i32 undef, i32 undef, i32 undef, i32 undef,i32 undef, i32 undef>
  %bin.rdx14 = add <16 x i8> %bin.rdx13, %rdx.shuf13
  %r = extractelement <16 x i8> %bin.rdx14, i32 0
  ret i8 %r
}

define i16 @f_v8i16(<8 x i16>* %arr)  {
; CHECK-LABEL: f_v8i16
; CHECK: addv {{h[0-9]+}}, {{v[0-9]+}}.8h
  %bin.rdx = load <8 x i16>, <8 x i16>* %arr
  %rdx.shuf = shufflevector <8 x i16> %bin.rdx, <8 x i16> undef, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef,i32 undef, i32 undef>
  %bin.rdx11 = add <8 x i16> %bin.rdx, %rdx.shuf
  %rdx.shuf12 = shufflevector <8 x i16> %bin.rdx11, <8 x i16> undef, <8 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx13 = add <8 x i16> %bin.rdx11, %rdx.shuf12
  %rdx.shuf13 = shufflevector <8 x i16> %bin.rdx13, <8 x i16> undef, <8 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx14 = add <8 x i16> %bin.rdx13, %rdx.shuf13
  %r = extractelement <8 x i16> %bin.rdx14, i32 0
  ret i16 %r
}

define i32 @f_v4i32( <4 x i32>* %arr)  {
; CHECK-LABEL: f_v4i32
; CHECK: addv {{s[0-9]+}}, {{v[0-9]+}}.4s
  %bin.rdx = load <4 x i32>, <4 x i32>* %arr
  %rdx.shuf = shufflevector <4 x i32> %bin.rdx, <4 x i32> undef, <4 x i32> <i32 2, i32 3, i32 undef, i32 undef>
  %bin.rdx11 = add <4 x i32> %bin.rdx, %rdx.shuf
  %rdx.shuf12 = shufflevector <4 x i32> %bin.rdx11, <4 x i32> undef, <4 x i32> <i32 1, i32 undef, i32 undef, i32 undef>
  %bin.rdx13 = add <4 x i32> %bin.rdx11, %rdx.shuf12
  %r = extractelement <4 x i32> %bin.rdx13, i32 0
  ret i32 %r
}

define i64 @f_v2i64(<2 x i64>* %arr)  {
; CHECK-LABEL: f_v2i64
; CHECK-NOT: addv
  %bin.rdx = load <2 x i64>, <2 x i64>* %arr
  %rdx.shuf0 = shufflevector <2 x i64> %bin.rdx, <2 x i64> undef, <2 x i32> <i32 1, i32 undef>
  %bin.rdx0 = add <2 x i64> %bin.rdx, %rdx.shuf0
  %r = extractelement <2 x i64> %bin.rdx0, i32 0
  ret i64 %r
}
