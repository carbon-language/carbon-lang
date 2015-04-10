; RUN: llc < %s -march=x86 | FileCheck %s
; <rdar://problem/8449754>

define i32 @test1(i32 %sum, i32 %x) nounwind readnone ssp {
entry:
; CHECK-LABEL: test1:
; CHECK: movl
; CHECK-NEXT: addl
; CHECK-NEXT: adcl $0
; CHECK-NEXT: ret
  %add4 = add i32 %x, %sum
  %cmp = icmp ult i32 %add4, %x
  %inc = zext i1 %cmp to i32
  %z.0 = add i32 %add4, %inc
  ret i32 %z.0
}

; <rdar://problem/12579915>
define i32 @test2(i32 %x, i32 %y, i32 %res) nounwind uwtable readnone ssp {
entry:
  %cmp = icmp ugt i32 %x, %y
  %dec = sext i1 %cmp to i32
  %dec.res = add nsw i32 %dec, %res
  ret i32 %dec.res
; CHECK-LABEL: test2:
; CHECK: cmpl
; CHECK: sbbl
; CHECK: ret
}
