; RUN: opt < %s -instcombine -S | FileCheck %s

declare void @llvm.experimental.guard(i1, ...)

define void @test_guard_adjacent(i1 %A) {
; CHECK-LABEL: @test_guard_adjacent(
; CHECK-NEXT:    call void (i1, ...) @llvm.experimental.guard(i1 %A) [ "deopt"() ]
; CHECK-NEXT:    ret void
  call void(i1, ...) @llvm.experimental.guard( i1 %A )[ "deopt"() ]
  call void(i1, ...) @llvm.experimental.guard( i1 %A )[ "deopt"() ]
  call void(i1, ...) @llvm.experimental.guard( i1 %A )[ "deopt"() ]
  call void(i1, ...) @llvm.experimental.guard( i1 %A )[ "deopt"() ]
  call void(i1, ...) @llvm.experimental.guard( i1 %A )[ "deopt"() ]
  call void(i1, ...) @llvm.experimental.guard( i1 %A )[ "deopt"() ]
  call void(i1, ...) @llvm.experimental.guard( i1 %A )[ "deopt"() ]
  call void(i1, ...) @llvm.experimental.guard( i1 %A )[ "deopt"() ]
  call void(i1, ...) @llvm.experimental.guard( i1 %A )[ "deopt"() ]
  call void(i1, ...) @llvm.experimental.guard( i1 %A )[ "deopt"() ]
  ret void
}

define void @test_guard_adjacent_neg(i1 %A, i1 %B) {
; CHECK-LABEL: @test_guard_adjacent_neg(
; CHECK-NEXT:    call void (i1, ...) @llvm.experimental.guard(i1 %A) [ "deopt"() ]
; CHECK-NEXT:    call void (i1, ...) @llvm.experimental.guard(i1 %B) [ "deopt"() ]
; CHECK-NEXT:    ret void
  call void(i1, ...) @llvm.experimental.guard( i1 %A )[ "deopt"() ]
  call void(i1, ...) @llvm.experimental.guard( i1 %B )[ "deopt"() ]
  ret void
}

define void @test_guard_and(i1 %A, i1 %B) {
; CHECK-LABEL: @test_guard_and(
; CHECK-NEXT:    call void (i1, ...) @llvm.experimental.guard(i1 %A) [ "deopt"() ]
; CHECK-NEXT:    call void (i1, ...) @llvm.experimental.guard(i1 %B) [ "deopt"() ]
; CHECK-NEXT:    ret void
  %C = and i1 %A, %B
  call void(i1, ...) @llvm.experimental.guard( i1 %C )[ "deopt"() ]
  ret void
}

define void @test_guard_and_non_default_cc(i1 %A, i1 %B) {
; CHECK-LABEL: @test_guard_and_non_default_cc(
; CHECK-NEXT:    call cc99 void (i1, ...) @llvm.experimental.guard(i1 %A) [ "deopt"() ]
; CHECK-NEXT:    call cc99 void (i1, ...) @llvm.experimental.guard(i1 %B) [ "deopt"() ]
; CHECK-NEXT:    ret void
  %C = and i1 %A, %B
  call cc99 void(i1, ...) @llvm.experimental.guard( i1 %C )[ "deopt"() ]
  ret void
}
