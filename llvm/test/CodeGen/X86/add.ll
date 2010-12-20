; RUN: llc < %s -march=x86 | FileCheck %s -check-prefix=X32
; RUN: llc < %s -march=x86-64 | FileCheck %s -check-prefix=X64

; The immediate can be encoded in a smaller way if the
; instruction is a sub instead of an add.

define i32 @test1(i32 inreg %a) nounwind {
  %b = add i32 %a, 128
  ret i32 %b
; X32: subl	$-128, %eax
; X64: subl $-128, 
}
define i64 @test2(i64 inreg %a) nounwind {
  %b = add i64 %a, 2147483648
  ret i64 %b
; X32: addl	$-2147483648, %eax
; X64: subq	$-2147483648,
}
define i64 @test3(i64 inreg %a) nounwind {
  %b = add i64 %a, 128
  ret i64 %b
  
; X32: addl $128, %eax
; X64: subq	$-128,
}

define i1 @test4(i32 %v1, i32 %v2, i32* %X) nounwind {
entry:
  %t = call {i32, i1} @llvm.sadd.with.overflow.i32(i32 %v1, i32 %v2)
  %sum = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  br i1 %obit, label %overflow, label %normal

normal:
  store i32 0, i32* %X
  br label %overflow

overflow:
  ret i1 false
  
; X32: test4:
; X32: addl
; X32-NEXT: jo

; X64:        test4:
; X64:          addl	%esi, %edi
; X64-NEXT:	jo
}

define i1 @test5(i32 %v1, i32 %v2, i32* %X) nounwind {
entry:
  %t = call {i32, i1} @llvm.uadd.with.overflow.i32(i32 %v1, i32 %v2)
  %sum = extractvalue {i32, i1} %t, 0
  %obit = extractvalue {i32, i1} %t, 1
  br i1 %obit, label %carry, label %normal

normal:
  store i32 0, i32* %X
  br label %carry

carry:
  ret i1 false

; X32: test5:
; X32: addl
; X32-NEXT: jb

; X64:        test5:
; X64:          addl	%esi, %edi
; X64-NEXT:	jb
}

declare {i32, i1} @llvm.sadd.with.overflow.i32(i32, i32)
declare {i32, i1} @llvm.uadd.with.overflow.i32(i32, i32)


define i64 @test6(i64 %A, i32 %B) nounwind {
        %tmp12 = zext i32 %B to i64             ; <i64> [#uses=1]
        %tmp3 = shl i64 %tmp12, 32              ; <i64> [#uses=1]
        %tmp5 = add i64 %tmp3, %A               ; <i64> [#uses=1]
        ret i64 %tmp5

; X32: test6:
; X32:	    movl 12(%esp), %edx
; X32-NEXT: addl 8(%esp), %edx
; X32-NEXT: movl 4(%esp), %eax
; X32-NEXT: ret
        
; X64: test6:
; X64:	shlq	$32, %rsi
; X64:	leaq	(%rsi,%rdi), %rax
; X64:	ret
}

define {i32, i1} @test7(i32 %v1, i32 %v2) nounwind {
   %t = call {i32, i1} @llvm.uadd.with.overflow.i32(i32 %v1, i32 %v2)
   ret {i32, i1} %t
}

; X64: test7:
; X64: addl %esi, %eax
; X64-NEXT: setb %dl
; X64-NEXT: ret
