; RUN: llc < %s -march=x86 | FileCheck %s
; <rdar://problem/8449754>

define i32 @test1(i32 %sum, i32 %x) nounwind readnone ssp {
entry:
; CHECK: test1:
; CHECK:	sbbl	%ecx, %ecx
; CHECK-NOT: addl
; CHECK: subl	%ecx, %eax
  %add4 = add i32 %x, %sum
  %cmp = icmp ult i32 %add4, %x
  %inc = zext i1 %cmp to i32
  %z.0 = add i32 %add4, %inc
  ret i32 %z.0
}

; Instcombine transforms test1 into test2:
; CHECK: test2:
; CHECK: movl
; CHECK-NEXT: addl
; CHECK-NEXT: sbbl
; CHECK-NEXT: subl
; CHECK-NEXT: ret
define i32 @test2(i32 %sum, i32 %x) nounwind readnone ssp {
entry:
  %uadd = call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 %x, i32 %sum)
  %0 = extractvalue { i32, i1 } %uadd, 0
  %cmp = extractvalue { i32, i1 } %uadd, 1
  %inc = zext i1 %cmp to i32
  %z.0 = add i32 %0, %inc
  ret i32 %z.0
}

declare { i32, i1 } @llvm.uadd.with.overflow.i32(i32, i32) nounwind readnone
