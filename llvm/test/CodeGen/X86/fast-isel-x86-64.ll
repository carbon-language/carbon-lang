; RUN: llc < %s  -fast-isel -O0 -regalloc=fast -asm-verbose=0 -fast-isel-abort | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"

; Make sure that fast-isel folds the immediate into the binop even though it
; is non-canonical.
define i32 @test1(i32 %i) nounwind ssp {
  %and = and i32 8, %i
  ret i32 %and
}

; CHECK: test1:
; CHECK: andl	$8, 


@G = external global i32
define i64 @test3() nounwind {
  %A = ptrtoint i32* @G to i64
  ret i64 %A
; CHECK: test3:
; CHECK: movq _G@GOTPCREL(%rip), %rax
; CHECK-NEXT: ret
}



; rdar://9289558
@rtx_length = external global [153 x i8]

define i32 @test4(i64 %idxprom9) nounwind {
  %arrayidx10 = getelementptr inbounds [153 x i8]* @rtx_length, i32 0, i64 %idxprom9
  %tmp11 = load i8* %arrayidx10, align 1
  %conv = zext i8 %tmp11 to i32
  ret i32 %conv

; CHECK: test4:
; CHECK: movq	_rtx_length@GOTPCREL(%rip), %rax
; CHECK-NEXT: movzbl	(%rax,%rdi), %eax
; CHECK-NEXT: ret
}


; PR3242 - Out of range shifts should not be folded by fastisel.
define void @test5(i32 %x, i32* %p) nounwind {
  %y = ashr i32 %x, 50000
  store i32 %y, i32* %p
  ret void

; CHECK: test5:
; CHECK: movl	$50000, %ecx
; CHECK: sarl	%cl, %edi
; CHECK: ret
}

; rdar://9289501 - fast isel should fold trivial multiplies to shifts.
define i64 @test6(i64 %x) nounwind ssp {
entry:
  %mul = mul nsw i64 %x, 8
  ret i64 %mul

; CHECK: test6:
; CHECK: leaq	(,%rdi,8), %rax
}

define i32 @test7(i32 %x) nounwind ssp {
entry:
  %mul = mul nsw i32 %x, 8
  ret i32 %mul
; CHECK: test7:
; CHECK: leal	(,%rdi,8), %eax
}


; rdar://9289507 - folding of immediates into 64-bit operations.
define i64 @test8(i64 %x) nounwind ssp {
entry:
  %add = add nsw i64 %x, 7
  ret i64 %add

; CHECK: test8:
; CHECK: addq	$7, %rdi
}

define i64 @test9(i64 %x) nounwind ssp {
entry:
  %add = mul nsw i64 %x, 7
  ret i64 %add
; CHECK: test9:
; CHECK: imulq	$7, %rdi, %rax
}

; rdar://9297011 - Don't reject udiv by a power of 2.
define i32 @test10(i32 %X) nounwind {
  %Y = udiv i32 %X, 8
  ret i32 %Y
; CHECK: test10:
; CHECK: shrl	$3, 
}

define i32 @test11(i32 %X) nounwind {
  %Y = sdiv exact i32 %X, 8
  ret i32 %Y
; CHECK: test11:
; CHECK: sarl	$3, 
}


; rdar://9297006 - Trunc to bool.
define void @test12(i8 %tmp) nounwind ssp noredzone {
entry:
  %tobool = trunc i8 %tmp to i1
  br i1 %tobool, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  call void @test12(i8 0) noredzone
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
; CHECK: test12:
; CHECK: testb	$1,
; CHECK-NEXT: je L
; CHECK-NEXT: movl $0, %edi
; CHECK-NEXT: callq
}

declare void @test13f(i1 %X)

define void @test13() nounwind {
  call void @test13f(i1 0)
  ret void
; CHECK: test13:
; CHECK: movl $0, %edi
; CHECK-NEXT: callq
}



; rdar://9297003 - fast isel bails out on all functions taking bools
define void @test14(i8 %tmp) nounwind ssp noredzone {
entry:
  %tobool = trunc i8 %tmp to i1
  call void @test13f(i1 zeroext %tobool) noredzone
  ret void
; CHECK: test14:
; CHECK: andb	$1, 
; CHECK: callq
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8*, i8*, i64, i32, i1)

; rdar://9289488 - fast-isel shouldn't bail out on llvm.memcpy
define void @test15(i8* %a, i8* %b) nounwind {
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %a, i8* %b, i64 4, i32 4, i1 false)
  ret void
; CHECK: test15:
; CHECK-NEXT: movl	(%rsi), %eax
; CHECK-NEXT: movl	%eax, (%rdi)
; CHECK-NEXT: ret
}

; Handling for varargs calls
declare void @test16callee(...) nounwind
define void @test16() nounwind {
; CHECK: test16:
; CHECK: movl $1, %edi
; CHECK: movb $0, %al
; CHECK: callq _test16callee
  call void (...)* @test16callee(i32 1)
  br label %block2

block2:
; CHECK: movabsq $1
; CHECK: cvtsi2sdq {{.*}} %xmm0
; CHECK: movb $1, %al
; CHECK: callq _test16callee
  call void (...)* @test16callee(double 1.000000e+00)
  ret void
}
