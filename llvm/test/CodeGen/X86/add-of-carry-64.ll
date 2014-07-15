; RUN: llc < %s -march=x86-64 | FileCheck %s

define i32 @testi32(i32 %x0, i32 %x1, i32 %y0, i32 %y1) {
entry:
  %uadd = tail call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 %x0, i32 %y0)
  %add1 = add i32 %y1, %x1
  %cmp = extractvalue { i32, i1 } %uadd, 1
  %conv2 = zext i1 %cmp to i32
  %add3 = add i32 %add1, %conv2
  ret i32 %add3
; CHECK-LABEL: testi32:
; CHECK: addl
; CHECK-NEXT: adcl
; CHECK: ret
}

define i64 @testi64(i64 %x0, i64 %x1, i64 %y0, i64 %y1) {
entry:
  %uadd = tail call { i64, i1 } @llvm.uadd.with.overflow.i64(i64 %x0, i64 %y0)
  %add1 = add i64 %y1, %x1
  %cmp = extractvalue { i64, i1 } %uadd, 1
  %conv2 = zext i1 %cmp to i64
  %add3 = add i64 %add1, %conv2
  ret i64 %add3
; CHECK-LABEL: testi64:
; CHECK: addq
; CHECK-NEXT: adcq
; CHECK: ret
}

declare { i32, i1 } @llvm.uadd.with.overflow.i32(i32, i32) nounwind readnone
declare { i64, i1 } @llvm.uadd.with.overflow.i64(i64, i64) nounwind readnone
