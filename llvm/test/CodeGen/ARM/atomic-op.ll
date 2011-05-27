; RUN: llc < %s -mtriple=armv7-apple-darwin10 | FileCheck %s
; RUN: llc < %s -mtriple=thumbv7-apple-darwin10 | FileCheck %s

define void @func(i32 %argc, i8** %argv) nounwind {
entry:
	%argc.addr = alloca i32		; <i32*> [#uses=1]
	%argv.addr = alloca i8**		; <i8***> [#uses=1]
	%val1 = alloca i32		; <i32*> [#uses=2]
	%val2 = alloca i32		; <i32*> [#uses=15]
	%andt = alloca i32		; <i32*> [#uses=2]
	%ort = alloca i32		; <i32*> [#uses=2]
	%xort = alloca i32		; <i32*> [#uses=2]
	%old = alloca i32		; <i32*> [#uses=18]
	%temp = alloca i32		; <i32*> [#uses=2]
	store i32 %argc, i32* %argc.addr
	store i8** %argv, i8*** %argv.addr
	store i32 0, i32* %val1
	store i32 31, i32* %val2
	store i32 3855, i32* %andt
	store i32 3855, i32* %ort
	store i32 3855, i32* %xort
	store i32 4, i32* %temp
	%tmp = load i32* %temp
  ; CHECK: ldrex
  ; CHECK: add
  ; CHECK: strex
	call i32 @llvm.atomic.load.add.i32.p0i32( i32* %val1, i32 %tmp )		; <i32>:0 [#uses=1]
	store i32 %0, i32* %old
  ; CHECK: ldrex
  ; CHECK: sub
  ; CHECK: strex
	call i32 @llvm.atomic.load.sub.i32.p0i32( i32* %val2, i32 30 )		; <i32>:1 [#uses=1]
	store i32 %1, i32* %old
  ; CHECK: ldrex
  ; CHECK: add
  ; CHECK: strex
	call i32 @llvm.atomic.load.add.i32.p0i32( i32* %val2, i32 1 )		; <i32>:2 [#uses=1]
	store i32 %2, i32* %old
  ; CHECK: ldrex
  ; CHECK: sub
  ; CHECK: strex
	call i32 @llvm.atomic.load.sub.i32.p0i32( i32* %val2, i32 1 )		; <i32>:3 [#uses=1]
	store i32 %3, i32* %old
  ; CHECK: ldrex
  ; CHECK: and
  ; CHECK: strex
	call i32 @llvm.atomic.load.and.i32.p0i32( i32* %andt, i32 4080 )		; <i32>:4 [#uses=1]
	store i32 %4, i32* %old
  ; CHECK: ldrex
  ; CHECK: or
  ; CHECK: strex
	call i32 @llvm.atomic.load.or.i32.p0i32( i32* %ort, i32 4080 )		; <i32>:5 [#uses=1]
	store i32 %5, i32* %old
  ; CHECK: ldrex
  ; CHECK: eor
  ; CHECK: strex
	call i32 @llvm.atomic.load.xor.i32.p0i32( i32* %xort, i32 4080 )		; <i32>:6 [#uses=1]
	store i32 %6, i32* %old
  ; CHECK: ldrex
  ; CHECK: cmp
  ; CHECK: strex
	call i32 @llvm.atomic.load.min.i32.p0i32( i32* %val2, i32 16 )		; <i32>:7 [#uses=1]
	store i32 %7, i32* %old
	%neg = sub i32 0, 1		; <i32> [#uses=1]
  ; CHECK: ldrex
  ; CHECK: cmp
  ; CHECK: strex
	call i32 @llvm.atomic.load.min.i32.p0i32( i32* %val2, i32 %neg )		; <i32>:8 [#uses=1]
	store i32 %8, i32* %old
  ; CHECK: ldrex
  ; CHECK: cmp
  ; CHECK: strex
	call i32 @llvm.atomic.load.max.i32.p0i32( i32* %val2, i32 1 )		; <i32>:9 [#uses=1]
	store i32 %9, i32* %old
  ; CHECK: ldrex
  ; CHECK: cmp
  ; CHECK: strex
	call i32 @llvm.atomic.load.max.i32.p0i32( i32* %val2, i32 0 )		; <i32>:10 [#uses=1]
	store i32 %10, i32* %old
	ret void
}

declare i32 @llvm.atomic.load.add.i32.p0i32(i32*, i32) nounwind 

declare i32 @llvm.atomic.load.sub.i32.p0i32(i32*, i32) nounwind 

declare i32 @llvm.atomic.load.and.i32.p0i32(i32*, i32) nounwind 

declare i32 @llvm.atomic.load.or.i32.p0i32(i32*, i32) nounwind 

declare i32 @llvm.atomic.load.xor.i32.p0i32(i32*, i32) nounwind 

declare i32 @llvm.atomic.load.min.i32.p0i32(i32*, i32) nounwind 

declare i32 @llvm.atomic.load.max.i32.p0i32(i32*, i32) nounwind 

declare i32 @llvm.atomic.load.umax.i32.p0i32(i32*, i32) nounwind 

declare i32 @llvm.atomic.load.umin.i32.p0i32(i32*, i32) nounwind 

declare i32 @llvm.atomic.swap.i32.p0i32(i32*, i32) nounwind 

declare i32 @llvm.atomic.cmp.swap.i32.p0i32(i32*, i32, i32) nounwind 
