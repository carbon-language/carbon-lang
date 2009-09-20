; RUN: opt < %s -gvn -S | FileCheck %s

; 32-bit little endian target.
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"

;; Trivial RLE test.
define i32 @test0(i32 %V, i32* %P) {
  store i32 %V, i32* %P

  %A = load i32* %P
  ret i32 %A
; CHECK: @test0
; CHECK: ret i32 %V
}

;;===----------------------------------------------------------------------===;;
;; Store -> Load  and  Load -> Load forwarding where src and dst are different
;; types, but where the base pointer is a must alias.
;;===----------------------------------------------------------------------===;;

;; i32 -> f32 forwarding.
define float @coerce_mustalias1(i32 %V, i32* %P) {
  store i32 %V, i32* %P
   
  %P2 = bitcast i32* %P to float*

  %A = load float* %P2
  ret float %A
; CHECK: @coerce_mustalias1
; CHECK-NOT: load
; CHECK: ret float 
}

;; i32* -> float forwarding.
define float @coerce_mustalias2(i32* %V, i32** %P) {
  store i32* %V, i32** %P
   
  %P2 = bitcast i32** %P to float*

  %A = load float* %P2
  ret float %A
; CHECK: @coerce_mustalias2
; CHECK-NOT: load
; CHECK: ret float 
}

;; float -> i32* forwarding.
define i32* @coerce_mustalias3(float %V, float* %P) {
  store float %V, float* %P
   
  %P2 = bitcast float* %P to i32**

  %A = load i32** %P2
  ret i32* %A
; CHECK: @coerce_mustalias3
; CHECK-NOT: load
; CHECK: ret i32* 
}

;; i32 -> f32 load forwarding.
define float @coerce_mustalias4(i32* %P, i1 %cond) {
  %A = load i32* %P
  
  %P2 = bitcast i32* %P to float*
  %B = load float* %P2
  br i1 %cond, label %T, label %F
T:
  ret float %B
  
F:
  %X = bitcast i32 %A to float
  ret float %X

; CHECK: @coerce_mustalias4
; CHECK: %A = load i32* %P
; CHECK: bitcast
; CHECK-NOT: load
; CHECK: ret float
; CHECK: F:
}

;; i32 -> i8 forwarding
define i8 @coerce_mustalias5(i32 %V, i32* %P) {
  store i32 %V, i32* %P
   
  %P2 = bitcast i32* %P to i8*

  %A = load i8* %P2
  ret i8 %A
; CHECK: @coerce_mustalias5
; CHECK-NOT: load
; CHECK: ret i8
}

;; i64 -> float forwarding
define float @coerce_mustalias6(i64 %V, i64* %P) {
  store i64 %V, i64* %P
   
  %P2 = bitcast i64* %P to float*

  %A = load float* %P2
  ret float %A
; CHECK: @coerce_mustalias6
; CHECK-NOT: load
; CHECK: ret float
}

;; i64 -> i8* (32-bit) forwarding
define i8* @coerce_mustalias7(i64 %V, i64* %P) {
  store i64 %V, i64* %P
   
  %P2 = bitcast i64* %P to i8**

  %A = load i8** %P2
  ret i8* %A
; CHECK: @coerce_mustalias7
; CHECK-NOT: load
; CHECK: ret i8*
}

;; non-local i32/float -> i8 load forwarding.
define i8 @coerce_mustalias_nonlocal0(i32* %P, i1 %cond) {
  %P2 = bitcast i32* %P to float*
  %P3 = bitcast i32* %P to i8*
  br i1 %cond, label %T, label %F
T:
  store i32 42, i32* %P
  br label %Cont
  
F:
  store float 1.0, float* %P2
  br label %Cont

Cont:
  %A = load i8* %P3
  ret i8 %A

; CHECK: @coerce_mustalias_nonlocal0
; CHECK: Cont:
; CHECK:   %A = phi i8 [
; CHECK-NOT: load
; CHECK: ret i8 %A
}

;; non-local i32 -> i8 partial redundancy load forwarding.
define i8 @coerce_mustalias_pre0(i32* %P, i1 %cond) {
  %P3 = bitcast i32* %P to i8*
  br i1 %cond, label %T, label %F
T:
  store i32 42, i32* %P
  br label %Cont
  
F:
  br label %Cont

Cont:
  %A = load i8* %P3
  ret i8 %A

; CHECK: @coerce_mustalias_pre0
; CHECK: F:
; CHECK:   load i8* %P3
; CHECK: Cont:
; CHECK:   %A = phi i8 [
; CHECK-NOT: load
; CHECK: ret i8 %A
}

