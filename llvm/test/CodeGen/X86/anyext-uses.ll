; RUN: llvm-as < %s | llc -march=x86-64 > %t
; RUN: grep mov %t | count 8
; RUN: not grep implicit %t

; Avoid partial register updates; don't define an i8 register and read
; the i32 super-register.

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-apple-darwin9.6"
	%struct.RC4_KEY = type { i8, i8, [256 x i8] }

define void @foo(%struct.RC4_KEY* nocapture %key, i64 %len, i8* %indata, i8* %outdata) nounwind {
entry:
	br label %bb24

bb24:		; preds = %bb24, %entry
	%0 = load i8* null, align 1		; <i8> [#uses=1]
	%1 = zext i8 %0 to i64		; <i64> [#uses=1]
	%2 = shl i64 %1, 32		; <i64> [#uses=1]
	%3 = getelementptr %struct.RC4_KEY* %key, i64 0, i32 2, i64 0		; <i8*> [#uses=1]
	%4 = load i8* %3, align 1		; <i8> [#uses=2]
	%5 = add i8 %4, 0		; <i8> [#uses=2]
	%6 = zext i8 %5 to i64		; <i64> [#uses=0]
	%7 = load i8* null, align 1		; <i8> [#uses=1]
	%8 = zext i8 %4 to i32		; <i32> [#uses=1]
	%9 = zext i8 %7 to i32		; <i32> [#uses=1]
	%10 = add i32 %9, %8		; <i32> [#uses=1]
	%11 = and i32 %10, 255		; <i32> [#uses=1]
	%12 = zext i32 %11 to i64		; <i64> [#uses=1]
	%13 = getelementptr %struct.RC4_KEY* %key, i64 0, i32 2, i64 %12		; <i8*> [#uses=1]
	%14 = load i8* %13, align 1		; <i8> [#uses=1]
	%15 = zext i8 %14 to i64		; <i64> [#uses=1]
	%16 = shl i64 %15, 48		; <i64> [#uses=1]
	%17 = getelementptr %struct.RC4_KEY* %key, i64 0, i32 2, i64 0		; <i8*> [#uses=1]
	%18 = load i8* %17, align 1		; <i8> [#uses=2]
	%19 = add i8 %18, %5		; <i8> [#uses=1]
	%20 = zext i8 %19 to i64		; <i64> [#uses=1]
	%21 = getelementptr %struct.RC4_KEY* %key, i64 0, i32 2, i64 %20		; <i8*> [#uses=1]
	store i8 %18, i8* %21, align 1
	%22 = or i64 0, %2		; <i64> [#uses=1]
	%23 = or i64 %22, 0		; <i64> [#uses=1]
	%24 = or i64 %23, %16		; <i64> [#uses=1]
	%25 = or i64 %24, 0		; <i64> [#uses=1]
	%26 = xor i64 %25, 0		; <i64> [#uses=1]
	store i64 %26, i64* null, align 8
	br label %bb24
}
