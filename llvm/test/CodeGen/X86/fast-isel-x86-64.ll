; RUN: llc < %s  -fast-isel -O0 -regalloc=fast | FileCheck %s

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


; rdar://9289512 - The load should fold into the compare.
define void @test2(i64 %x) nounwind ssp {
entry:
  %x.addr = alloca i64, align 8
  store i64 %x, i64* %x.addr, align 8
  %tmp = load i64* %x.addr, align 8
  %cmp = icmp sgt i64 %tmp, 42
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

; CHECK: test2:
; CHECK: movq	%rdi, -8(%rsp)
; CHECK: cmpq	$42, -8(%rsp)



@G = external global i32
define i64 @test3() nounwind {
  %A = ptrtoint i32* @G to i64
  ret i64 %A
}

; CHECK: test3:
; CHECK: movq _G@GOTPCREL(%rip), %rax
; CHECK-NEXT: ret
