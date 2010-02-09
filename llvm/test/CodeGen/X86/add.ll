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
