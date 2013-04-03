; RUN: llc -mtriple=x86_64-apple-darwin -mcpu=core2 < %s | FileCheck --check-prefix=SSE2-CODEGEN %s
; RUN: opt -mtriple=x86_64-apple-darwin -mcpu=core2 -cost-model -analyze < %s | FileCheck --check-prefix=SSE2 %s

%shifttype = type <2 x i16>
define %shifttype @shift2i16(%shifttype %a, %shifttype %b) {
entry:
  ; SSE2: shift2i16
  ; SSE2: cost of 20 {{.*}} ashr
  ; SSE2-CODEGEN: shift2i16
  ; SSE2-CODEGEN: sarq %cl

  %0 = ashr %shifttype %a , %b
  ret %shifttype %0
}

%shifttype4i16 = type <4 x i16>
define %shifttype4i16 @shift4i16(%shifttype4i16 %a, %shifttype4i16 %b) {
entry:
  ; SSE2: shift4i16
  ; SSE2: cost of 40 {{.*}} ashr
  ; SSE2-CODEGEN: shift4i16
  ; SSE2-CODEGEN: sarl %cl

  %0 = ashr %shifttype4i16 %a , %b
  ret %shifttype4i16 %0
}

%shifttype8i16 = type <8 x i16>
define %shifttype8i16 @shift8i16(%shifttype8i16 %a, %shifttype8i16 %b) {
entry:
  ; SSE2: shift8i16
  ; SSE2: cost of 80 {{.*}} ashr
  ; SSE2-CODEGEN: shift8i16
  ; SSE2-CODEGEN: sarw %cl

  %0 = ashr %shifttype8i16 %a , %b
  ret %shifttype8i16 %0
}

%shifttype16i16 = type <16 x i16>
define %shifttype16i16 @shift16i16(%shifttype16i16 %a, %shifttype16i16 %b) {
entry:
  ; SSE2: shift16i16
  ; SSE2: cost of 160 {{.*}} ashr
  ; SSE2-CODEGEN: shift16i16
  ; SSE2-CODEGEN: sarw %cl

  %0 = ashr %shifttype16i16 %a , %b
  ret %shifttype16i16 %0
}

%shifttype32i16 = type <32 x i16>
define %shifttype32i16 @shift32i16(%shifttype32i16 %a, %shifttype32i16 %b) {
entry:
  ; SSE2: shift32i16
  ; SSE2: cost of 320 {{.*}} ashr
  ; SSE2-CODEGEN: shift32i16
  ; SSE2-CODEGEN: sarw %cl

  %0 = ashr %shifttype32i16 %a , %b
  ret %shifttype32i16 %0
}

%shifttype2i32 = type <2 x i32>
define %shifttype2i32 @shift2i32(%shifttype2i32 %a, %shifttype2i32 %b) {
entry:
  ; SSE2: shift2i32
  ; SSE2: cost of 20 {{.*}} ashr
  ; SSE2-CODEGEN: shift2i32
  ; SSE2-CODEGEN: sarq %cl

  %0 = ashr %shifttype2i32 %a , %b
  ret %shifttype2i32 %0
}

%shifttype4i32 = type <4 x i32>
define %shifttype4i32 @shift4i32(%shifttype4i32 %a, %shifttype4i32 %b) {
entry:
  ; SSE2: shift4i32
  ; SSE2: cost of 40 {{.*}} ashr
  ; SSE2-CODEGEN: shift4i32
  ; SSE2-CODEGEN: sarl %cl

  %0 = ashr %shifttype4i32 %a , %b
  ret %shifttype4i32 %0
}

%shifttype8i32 = type <8 x i32>
define %shifttype8i32 @shift8i32(%shifttype8i32 %a, %shifttype8i32 %b) {
entry:
  ; SSE2: shift8i32
  ; SSE2: cost of 80 {{.*}} ashr
  ; SSE2-CODEGEN: shift8i32
  ; SSE2-CODEGEN: sarl %cl

  %0 = ashr %shifttype8i32 %a , %b
  ret %shifttype8i32 %0
}

%shifttype16i32 = type <16 x i32>
define %shifttype16i32 @shift16i32(%shifttype16i32 %a, %shifttype16i32 %b) {
entry:
  ; SSE2: shift16i32
  ; SSE2: cost of 160 {{.*}} ashr
  ; SSE2-CODEGEN: shift16i32
  ; SSE2-CODEGEN: sarl %cl

  %0 = ashr %shifttype16i32 %a , %b
  ret %shifttype16i32 %0
}

%shifttype32i32 = type <32 x i32>
define %shifttype32i32 @shift32i32(%shifttype32i32 %a, %shifttype32i32 %b) {
entry:
  ; SSE2: shift32i32
  ; SSE2: cost of 256 {{.*}} ashr
  ; SSE2-CODEGEN: shift32i32
  ; SSE2-CODEGEN: sarl %cl

  %0 = ashr %shifttype32i32 %a , %b
  ret %shifttype32i32 %0
}

%shifttype2i64 = type <2 x i64>
define %shifttype2i64 @shift2i64(%shifttype2i64 %a, %shifttype2i64 %b) {
entry:
  ; SSE2: shift2i64
  ; SSE2: cost of 20 {{.*}} ashr
  ; SSE2-CODEGEN: shift2i64
  ; SSE2-CODEGEN: sarq %cl

  %0 = ashr %shifttype2i64 %a , %b
  ret %shifttype2i64 %0
}

%shifttype4i64 = type <4 x i64>
define %shifttype4i64 @shift4i64(%shifttype4i64 %a, %shifttype4i64 %b) {
entry:
  ; SSE2: shift4i64
  ; SSE2: cost of 40 {{.*}} ashr
  ; SSE2-CODEGEN: shift4i64
  ; SSE2-CODEGEN: sarq %cl

  %0 = ashr %shifttype4i64 %a , %b
  ret %shifttype4i64 %0
}

%shifttype8i64 = type <8 x i64>
define %shifttype8i64 @shift8i64(%shifttype8i64 %a, %shifttype8i64 %b) {
entry:
  ; SSE2: shift8i64
  ; SSE2: cost of 80 {{.*}} ashr
  ; SSE2-CODEGEN: shift8i64
  ; SSE2-CODEGEN: sarq %cl

  %0 = ashr %shifttype8i64 %a , %b
  ret %shifttype8i64 %0
}

%shifttype16i64 = type <16 x i64>
define %shifttype16i64 @shift16i64(%shifttype16i64 %a, %shifttype16i64 %b) {
entry:
  ; SSE2: shift16i64
  ; SSE2: cost of 160 {{.*}} ashr
  ; SSE2-CODEGEN: shift16i64
  ; SSE2-CODEGEN: sarq %cl

  %0 = ashr %shifttype16i64 %a , %b
  ret %shifttype16i64 %0
}

%shifttype32i64 = type <32 x i64>
define %shifttype32i64 @shift32i64(%shifttype32i64 %a, %shifttype32i64 %b) {
entry:
  ; SSE2: shift32i64
  ; SSE2: cost of 256 {{.*}} ashr
  ; SSE2-CODEGEN: shift32i64
  ; SSE2-CODEGEN: sarq %cl

  %0 = ashr %shifttype32i64 %a , %b
  ret %shifttype32i64 %0
}

%shifttype2i8 = type <2 x i8>
define %shifttype2i8 @shift2i8(%shifttype2i8 %a, %shifttype2i8 %b) {
entry:
  ; SSE2: shift2i8
  ; SSE2: cost of 20 {{.*}} ashr
  ; SSE2-CODEGEN: shift2i8
  ; SSE2-CODEGEN: sarq %cl

  %0 = ashr %shifttype2i8 %a , %b
  ret %shifttype2i8 %0
}

%shifttype4i8 = type <4 x i8>
define %shifttype4i8 @shift4i8(%shifttype4i8 %a, %shifttype4i8 %b) {
entry:
  ; SSE2: shift4i8
  ; SSE2: cost of 40 {{.*}} ashr
  ; SSE2-CODEGEN: shift4i8
  ; SSE2-CODEGEN: sarl %cl

  %0 = ashr %shifttype4i8 %a , %b
  ret %shifttype4i8 %0
}

%shifttype8i8 = type <8 x i8>
define %shifttype8i8 @shift8i8(%shifttype8i8 %a, %shifttype8i8 %b) {
entry:
  ; SSE2: shift8i8
  ; SSE2: cost of 80 {{.*}} ashr
  ; SSE2-CODEGEN: shift8i8
  ; SSE2-CODEGEN: sarw %cl

  %0 = ashr %shifttype8i8 %a , %b
  ret %shifttype8i8 %0
}

%shifttype16i8 = type <16 x i8>
define %shifttype16i8 @shift16i8(%shifttype16i8 %a, %shifttype16i8 %b) {
entry:
  ; SSE2: shift16i8
  ; SSE2: cost of 160 {{.*}} ashr
  ; SSE2-CODEGEN: shift16i8
  ; SSE2-CODEGEN: sarb %cl

  %0 = ashr %shifttype16i8 %a , %b
  ret %shifttype16i8 %0
}

%shifttype32i8 = type <32 x i8>
define %shifttype32i8 @shift32i8(%shifttype32i8 %a, %shifttype32i8 %b) {
entry:
  ; SSE2: shift32i8
  ; SSE2: cost of 320 {{.*}} ashr
  ; SSE2-CODEGEN: shift32i8
  ; SSE2-CODEGEN: sarb %cl

  %0 = ashr %shifttype32i8 %a , %b
  ret %shifttype32i8 %0
}

