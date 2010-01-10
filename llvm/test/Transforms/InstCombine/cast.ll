; Tests to make sure elimination of casts is working correctly
; RUN: opt < %s -instcombine -S | FileCheck %s
target datalayout = "E-p:64:64:64-a0:0:8-f32:32:32-f64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-v64:64:64-v128:128:128-n8:16:32:64"

@inbuf = external global [32832 x i8]           ; <[32832 x i8]*> [#uses=1]

define i32 @test1(i32 %A) {
        %c1 = bitcast i32 %A to i32             ; <i32> [#uses=1]
        %c2 = bitcast i32 %c1 to i32            ; <i32> [#uses=1]
        ret i32 %c2
; CHECK: ret i32 %A
}

define i64 @test2(i8 %A) {
        %c1 = zext i8 %A to i16         ; <i16> [#uses=1]
        %c2 = zext i16 %c1 to i32               ; <i32> [#uses=1]
        %Ret = zext i32 %c2 to i64              ; <i64> [#uses=1]
        ret i64 %Ret
; CHECK: %Ret = zext i8 %A to i64
; CHECK: ret i64 %Ret
}

; This function should just use bitwise AND
define i64 @test3(i64 %A) {
        %c1 = trunc i64 %A to i8                ; <i8> [#uses=1]
        %c2 = zext i8 %c1 to i64                ; <i64> [#uses=1]
        ret i64 %c2
; CHECK: %c2 = and i64 %A, 255
; CHECK: ret i64 %c2
}

define i32 @test4(i32 %A, i32 %B) {
        %COND = icmp slt i32 %A, %B             ; <i1> [#uses=1]
        ; Booleans are unsigned integrals
        %c = zext i1 %COND to i8                ; <i8> [#uses=1]
        ; for the cast elim purpose
        %result = zext i8 %c to i32             ; <i32> [#uses=1]
        ret i32 %result
; CHECK: %COND = icmp slt i32 %A, %B
; CHECK: %result = zext i1 %COND to i32
; CHECK: ret i32 %result
}

define i32 @test5(i1 %B) {
        ; This cast should get folded into
        %c = zext i1 %B to i8           ; <i8> [#uses=1]
        ; this cast        
        %result = zext i8 %c to i32             ; <i32> [#uses=1]
        ret i32 %result
; CHECK: %result = zext i1 %B to i32
; CHECK: ret i32 %result
}

define i32 @test6(i64 %A) {
        %c1 = trunc i64 %A to i32               ; <i32> [#uses=1]
        %res = bitcast i32 %c1 to i32           ; <i32> [#uses=1]
        ret i32 %res
; CHECK:  trunc i64 %A to i32
; CHECK-NEXT: ret i32
}

define i64 @test7(i1 %A) {
        %c1 = zext i1 %A to i32         ; <i32> [#uses=1]
        %res = sext i32 %c1 to i64              ; <i64> [#uses=1]
        ret i64 %res
; CHECK: %res = zext i1 %A to i64
; CHECK: ret i64 %res
}

define i64 @test8(i8 %A) {
        %c1 = sext i8 %A to i64         ; <i64> [#uses=1]
        %res = bitcast i64 %c1 to i64           ; <i64> [#uses=1]
        ret i64 %res
; CHECK: = sext i8 %A to i64
; CHECK-NEXT: ret i64
}

define i16 @test9(i16 %A) {
        %c1 = sext i16 %A to i32                ; <i32> [#uses=1]
        %c2 = trunc i32 %c1 to i16              ; <i16> [#uses=1]
        ret i16 %c2
; CHECK: ret i16 %A
}

define i16 @test10(i16 %A) {
        %c1 = sext i16 %A to i32                ; <i32> [#uses=1]
        %c2 = trunc i32 %c1 to i16              ; <i16> [#uses=1]
        ret i16 %c2
; CHECK: ret i16 %A
}

declare void @varargs(i32, ...)

define void @test11(i32* %P) {
        %c = bitcast i32* %P to i16*            ; <i16*> [#uses=1]
        call void (i32, ...)* @varargs( i32 5, i16* %c )
        ret void
; CHECK: call void (i32, ...)* @varargs(i32 5, i32* %P)
; CHECK: ret void
}

define i32* @test12() {
        %p = malloc [4 x i8]            ; <[4 x i8]*> [#uses=1]
        %c = bitcast [4 x i8]* %p to i32*               ; <i32*> [#uses=1]
        ret i32* %c
; CHECK: %malloccall = tail call i8* @malloc(i32 4)
; CHECK: ret i32* %c
}

define i8* @test13(i64 %A) {
        %c = getelementptr [0 x i8]* bitcast ([32832 x i8]* @inbuf to [0 x i8]*), i64 0, i64 %A             ; <i8*> [#uses=1]
        ret i8* %c
; CHECK: %c = getelementptr [32832 x i8]* @inbuf, i64 0, i64 %A
; CHECK: ret i8* %c
}

define i1 @test14(i8 %A) {
        %c = bitcast i8 %A to i8                ; <i8> [#uses=1]
        %X = icmp ult i8 %c, -128               ; <i1> [#uses=1]
        ret i1 %X
; CHECK: %X = icmp sgt i8 %A, -1
; CHECK: ret i1 %X
}


; This just won't occur when there's no difference between ubyte and sbyte
;bool %test15(ubyte %A) {
;        %c = cast ubyte %A to sbyte
;        %X = setlt sbyte %c, 0   ; setgt %A, 127
;        ret bool %X
;}

define i1 @test16(i32* %P) {
        %c = icmp ne i32* %P, null              ; <i1> [#uses=1]
        ret i1 %c
; CHECK: %c = icmp ne i32* %P, null
; CHECK: ret i1 %c
}

define i16 @test17(i1 %tmp3) {
        %c = zext i1 %tmp3 to i32               ; <i32> [#uses=1]
        %t86 = trunc i32 %c to i16              ; <i16> [#uses=1]
        ret i16 %t86
; CHECK: %t86 = zext i1 %tmp3 to i16
; CHECK: ret i16 %t86
}

define i16 @test18(i8 %tmp3) {
        %c = sext i8 %tmp3 to i32               ; <i32> [#uses=1]
        %t86 = trunc i32 %c to i16              ; <i16> [#uses=1]
        ret i16 %t86
; CHECK: %t86 = sext i8 %tmp3 to i16
; CHECK: ret i16 %t86
}

define i1 @test19(i32 %X) {
        %c = sext i32 %X to i64         ; <i64> [#uses=1]
        %Z = icmp slt i64 %c, 12345             ; <i1> [#uses=1]
        ret i1 %Z
; CHECK: %Z = icmp slt i32 %X, 12345
; CHECK: ret i1 %Z
}

define i1 @test20(i1 %B) {
        %c = zext i1 %B to i32          ; <i32> [#uses=1]
        %D = icmp slt i32 %c, -1                ; <i1> [#uses=1]
        ;; false
        ret i1 %D
; CHECK: ret i1 false
}

define i32 @test21(i32 %X) {
        %c1 = trunc i32 %X to i8                ; <i8> [#uses=1]
        ;; sext -> zext -> and -> nop
        %c2 = sext i8 %c1 to i32                ; <i32> [#uses=1]
        %RV = and i32 %c2, 255          ; <i32> [#uses=1]
        ret i32 %RV
; CHECK: %c21 = and i32 %X, 255
; CHECK: ret i32 %c21
}

define i32 @test22(i32 %X) {
        %c1 = trunc i32 %X to i8                ; <i8> [#uses=1]
        ;; sext -> zext -> and -> nop
        %c2 = sext i8 %c1 to i32                ; <i32> [#uses=1]
        %RV = shl i32 %c2, 24           ; <i32> [#uses=1]
        ret i32 %RV
; CHECK: shl i32 %X, 24
; CHECK-NEXT: ret i32
}

define i32 @test23(i32 %X) {
        ;; Turn into an AND even though X
        %c1 = trunc i32 %X to i16               ; <i16> [#uses=1]
        ;; and Z are signed.
        %c2 = zext i16 %c1 to i32               ; <i32> [#uses=1]
        ret i32 %c2
; CHECK: %c2 = and i32 %X, 65535
; CHECK: ret i32 %c2
}

define i1 @test24(i1 %C) {
        %X = select i1 %C, i32 14, i32 1234             ; <i32> [#uses=1]
        ;; Fold cast into select
        %c = icmp ne i32 %X, 0          ; <i1> [#uses=1]
        ret i1 %c
; CHECK: ret i1 true
}

define void @test25(i32** %P) {
        %c = bitcast i32** %P to float**                ; <float**> [#uses=1]
        ;; Fold cast into null
        store float* null, float** %c
        ret void
; CHECK: store i32* null, i32** %P
; CHECK: ret void
}

define i32 @test26(float %F) {
        ;; no need to cast from float->double.
        %c = fpext float %F to double           ; <double> [#uses=1]
        %D = fptosi double %c to i32            ; <i32> [#uses=1]
        ret i32 %D
; CHECK: %D = fptosi float %F to i32
; CHECK: ret i32 %D
}

define [4 x float]* @test27([9 x [4 x float]]* %A) {
        %c = bitcast [9 x [4 x float]]* %A to [4 x float]*              ; <[4 x float]*> [#uses=1]
        ret [4 x float]* %c
; CHECK: %c = getelementptr inbounds [9 x [4 x float]]* %A, i64 0, i64 0
; CHECK: ret [4 x float]* %c
}

define float* @test28([4 x float]* %A) {
        %c = bitcast [4 x float]* %A to float*          ; <float*> [#uses=1]
        ret float* %c
; CHECK: %c = getelementptr inbounds [4 x float]* %A, i64 0, i64 0
; CHECK: ret float* %c
}

define i32 @test29(i32 %c1, i32 %c2) {
        %tmp1 = trunc i32 %c1 to i8             ; <i8> [#uses=1]
        %tmp4.mask = trunc i32 %c2 to i8                ; <i8> [#uses=1]
        %tmp = or i8 %tmp4.mask, %tmp1          ; <i8> [#uses=1]
        %tmp10 = zext i8 %tmp to i32            ; <i32> [#uses=1]
        ret i32 %tmp10
; CHECK: %tmp2 = or i32 %c2, %c1
; CHECK: %tmp10 = and i32 %tmp2, 255
; CHECK: ret i32 %tmp10
}

define i32 @test30(i32 %c1) {
        %c2 = trunc i32 %c1 to i8               ; <i8> [#uses=1]
        %c3 = xor i8 %c2, 1             ; <i8> [#uses=1]
        %c4 = zext i8 %c3 to i32                ; <i32> [#uses=1]
        ret i32 %c4
; CHECK: %c3 = and i32 %c1, 255
; CHECK: %c4 = xor i32 %c3, 1
; CHECK: ret i32 %c4
}

define i1 @test31(i64 %A) {
        %B = trunc i64 %A to i32                ; <i32> [#uses=1]
        %C = and i32 %B, 42             ; <i32> [#uses=1]
        %D = icmp eq i32 %C, 10         ; <i1> [#uses=1]
        ret i1 %D
; CHECK: %C1 = and i64 %A, 42
; CHECK: %D = icmp eq i64 %C1, 10
; CHECK: ret i1 %D
}

define void @test32(double** %tmp) {
        %tmp8 = malloc [16 x i8]                ; <[16 x i8]*> [#uses=1]
        %tmp8.upgrd.1 = bitcast [16 x i8]* %tmp8 to double*             ; <double*> [#uses=1]
        store double* %tmp8.upgrd.1, double** %tmp
        ret void
; CHECK: %malloccall = tail call i8* @malloc(i32 16)
; CHECK: %tmp8.upgrd.1 = bitcast i8* %malloccall to double*
; CHECK: store double* %tmp8.upgrd.1, double** %tmp
; CHECK: ret void
}

define i32 @test33(i32 %c1) {
        %x = bitcast i32 %c1 to float           ; <float> [#uses=1]
        %y = bitcast float %x to i32            ; <i32> [#uses=1]
        ret i32 %y
; CHECK: ret i32 %c1
}

define i16 @test34(i16 %a) {
        %c1 = zext i16 %a to i32                ; <i32> [#uses=1]
        %tmp21 = lshr i32 %c1, 8                ; <i32> [#uses=1]
        %c2 = trunc i32 %tmp21 to i16           ; <i16> [#uses=1]
        ret i16 %c2
; CHECK: %tmp21 = lshr i16 %a, 8
; CHECK: ret i16 %tmp21
}

define i16 @test35(i16 %a) {
        %c1 = bitcast i16 %a to i16             ; <i16> [#uses=1]
        %tmp2 = lshr i16 %c1, 8         ; <i16> [#uses=1]
        %c2 = bitcast i16 %tmp2 to i16          ; <i16> [#uses=1]
        ret i16 %c2
; CHECK: %tmp2 = lshr i16 %a, 8
; CHECK: ret i16 %tmp2
}

; icmp sgt i32 %a, -1
; rdar://6480391
define i1 @test36(i32 %a) {
        %b = lshr i32 %a, 31
        %c = trunc i32 %b to i8
        %d = icmp eq i8 %c, 0
        ret i1 %d
; CHECK: %d = icmp sgt i32 %a, -1
; CHECK: ret i1 %d
}

; ret i1 false
define i1 @test37(i32 %a) {
        %b = lshr i32 %a, 31
        %c = or i32 %b, 512
        %d = trunc i32 %c to i8
        %e = icmp eq i8 %d, 11
        ret i1 %e
; CHECK: ret i1 false
}

define i64 @test38(i32 %a) {
	%1 = icmp eq i32 %a, -2
	%2 = zext i1 %1 to i8
	%3 = xor i8 %2, 1
	%4 = zext i8 %3 to i64
        ret i64 %4
; CHECK: %1 = icmp ne i32 %a, -2
; CHECK: %2 = zext i1 %1 to i64
; CHECK: ret i64 %2
}

define i16 @test39(i16 %a) {
        %tmp = zext i16 %a to i32
        %tmp21 = lshr i32 %tmp, 8
        %tmp5 = shl i32 %tmp, 8
        %tmp.upgrd.32 = or i32 %tmp21, %tmp5
        %tmp.upgrd.3 = trunc i32 %tmp.upgrd.32 to i16
        ret i16 %tmp.upgrd.3
; CHECK: @test39
; CHECK: %tmp.upgrd.32 = call i16 @llvm.bswap.i16(i16 %a)
; CHECK: ret i16 %tmp.upgrd.32
}

define i16 @test40(i16 %a) {
        %tmp = zext i16 %a to i32
        %tmp21 = lshr i32 %tmp, 9
        %tmp5 = shl i32 %tmp, 8
        %tmp.upgrd.32 = or i32 %tmp21, %tmp5
        %tmp.upgrd.3 = trunc i32 %tmp.upgrd.32 to i16
        ret i16 %tmp.upgrd.3
; CHECK: @test40
; CHECK: %tmp21 = lshr i16 %a, 9
; CHECK: %tmp5 = shl i16 %a, 8
; CHECK: %tmp.upgrd.32 = or i16 %tmp21, %tmp5
; CHECK: ret i16 %tmp.upgrd.32
}

; PR1263
define i32* @test41(i32* %tmp1) {
        %tmp64 = bitcast i32* %tmp1 to { i32 }*
        %tmp65 = getelementptr { i32 }* %tmp64, i32 0, i32 0
        ret i32* %tmp65
; CHECK: @test41
; CHECK: ret i32* %tmp1
}

define i32 @test42(i32 %X) {
        %Y = trunc i32 %X to i8         ; <i8> [#uses=1]
        %Z = zext i8 %Y to i32          ; <i32> [#uses=1]
        ret i32 %Z
; CHECK: @test42
; CHECK: %Z = and i32 %X, 255
}

; rdar://6598839
define zeroext i64 @test43(i8 zeroext %on_off) nounwind readonly {
	%A = zext i8 %on_off to i32
	%B = add i32 %A, -1
	%C = sext i32 %B to i64
	ret i64 %C  ;; Should be (add (zext i8 -> i64), -1)
; CHECK: @test43
; CHECK-NEXT: %A = zext i8 %on_off to i64
; CHECK-NEXT: %B = add i64 %A, -1
; CHECK-NEXT: ret i64 %B
}

define i64 @test44(i8 %T) {
 %A = zext i8 %T to i16
 %B = or i16 %A, 1234
 %C = zext i16 %B to i64
 ret i64 %C
; CHECK: @test44
; CHECK-NEXT: %A = zext i8 %T to i64
; CHECK-NEXT: %B = or i64 %A, 1234
; CHECK-NEXT: ret i64 %B
}

define i64 @test45(i8 %A, i64 %Q) {
 %D = trunc i64 %Q to i32  ;; should be removed
 %B = sext i8 %A to i32
 %C = or i32 %B, %D
 %E = zext i32 %C to i64 
 ret i64 %E
; CHECK: @test45
; CHECK-NEXT: %B = sext i8 %A to i64
; CHECK-NEXT: %C = or i64 %B, %Q
; CHECK-NEXT: %E = and i64 %C, 4294967295
; CHECK-NEXT: ret i64 %E
}


define i64 @test46(i64 %A) {
 %B = trunc i64 %A to i32
 %C = and i32 %B, 42
 %D = shl i32 %C, 8
 %E = zext i32 %D to i64 
 ret i64 %E
; CHECK: @test46
; CHECK-NEXT: %C = shl i64 %A, 8
; CHECK-NEXT: %D = and i64 %C, 10752
; CHECK-NEXT: ret i64 %D
}

define i64 @test47(i8 %A) {
 %B = sext i8 %A to i32
 %C = or i32 %B, 42
 %E = zext i32 %C to i64 
 ret i64 %E
; CHECK: @test47
; CHECK-NEXT:   %B = sext i8 %A to i64
; CHECK-NEXT: %C = or i64 %B, 42
; CHECK-NEXT:  %E = and i64 %C, 4294967295
; CHECK-NEXT:  ret i64 %E
}

define i64 @test48(i8 %A, i8 %a) {
  %b = zext i8 %a to i32
  %B = zext i8 %A to i32
  %C = shl i32 %B, 8
  %D = or i32 %C, %b
  %E = zext i32 %D to i64
  ret i64 %E
; CHECK: @test48
; CHECK-NEXT: %b = zext i8 %a to i64
; CHECK-NEXT: %B = zext i8 %A to i64
; CHECK-NEXT: %C = shl i64 %B, 8
; CHECK-NEXT: %D = or i64 %C, %b
; CHECK-NEXT: ret i64 %D
}

define i64 @test49(i64 %A) {
 %B = trunc i64 %A to i32
 %C = or i32 %B, 1
 %D = sext i32 %C to i64 
 ret i64 %D
; CHECK: @test49
; CHECK-NEXT: %C = shl i64 %A, 32
; CHECK-NEXT: ashr i64 %C, 32
; CHECK-NEXT: %D = or i64 {{.*}}, 1
; CHECK-NEXT: ret i64 %D
}

define i64 @test50(i64 %A) {
  %a = lshr i64 %A, 2
  %B = trunc i64 %a to i32
  %D = add i32 %B, -1
  %E = sext i32 %D to i64
  ret i64 %E
; CHECK: @test50
; CHECK-NEXT: shl i64 %A, 30
; CHECK-NEXT: add i64 {{.*}}, -4294967296
; CHECK-NEXT: %E = ashr i64 {{.*}}, 32
; CHECK-NEXT: ret i64 %E
}

define i64 @test51(i64 %A, i1 %cond) {
  %B = trunc i64 %A to i32
  %C = and i32 %B, -2
  %D = or i32 %B, 1
  %E = select i1 %cond, i32 %C, i32 %D
  %F = sext i32 %E to i64
  ret i64 %F
; CHECK: @test51
; CHECK-NEXT: %C = and i64 %A, 4294967294
; CHECK-NEXT: %D = or i64 %A, 1
; CHECK-NEXT: %E = select i1 %cond, i64 %C, i64 %D
; CHECK-NEXT: %sext = shl i64 %E, 32
; CHECK-NEXT: %F = ashr i64 %sext, 32
; CHECK-NEXT: ret i64 %F
}

define i32 @test52(i64 %A) {
  %B = trunc i64 %A to i16
  %C = or i16 %B, -32574
  %D = and i16 %C, -25350
  %E = zext i16 %D to i32
  ret i32 %E
; CHECK: @test52
; CHECK-NEXT: %B = trunc i64 %A to i32
; CHECK-NEXT: %C = or i32 %B, 32962
; CHECK-NEXT: %D = and i32 %C, 40186
; CHECK-NEXT: ret i32 %D
}

define i64 @test53(i32 %A) {
  %B = trunc i32 %A to i16
  %C = or i16 %B, -32574
  %D = and i16 %C, -25350
  %E = zext i16 %D to i64
  ret i64 %E
; CHECK: @test53
; CHECK-NEXT: %B = zext i32 %A to i64
; CHECK-NEXT: %C = or i64 %B, 32962
; CHECK-NEXT: %D = and i64 %C, 40186
; CHECK-NEXT: ret i64 %D
}

define i32 @test54(i64 %A) {
  %B = trunc i64 %A to i16
  %C = or i16 %B, -32574
  %D = and i16 %C, -25350
  %E = sext i16 %D to i32
  ret i32 %E
; CHECK: @test54
; CHECK-NEXT: %B = trunc i64 %A to i32
; CHECK-NEXT: %C = or i32 %B, -32574
; CHECK-NEXT: %D = and i32 %C, -25350
; CHECK-NEXT: ret i32 %D
}

define i64 @test55(i32 %A) {
  %B = trunc i32 %A to i16
  %C = or i16 %B, -32574
  %D = and i16 %C, -25350
  %E = sext i16 %D to i64
  ret i64 %E
; CHECK: @test55
; CHECK-NEXT: %B = zext i32 %A to i64
; CHECK-NEXT: %C = or i64 %B, -32574
; CHECK-NEXT: %D = and i64 %C, -25350
; CHECK-NEXT: ret i64 %D
}