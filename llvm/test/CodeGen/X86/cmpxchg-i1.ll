; RUN: llc -mtriple=x86_64 -o - %s | FileCheck %s

define i1 @try_cmpxchg(i32* %addr, i32 %desired, i32 %new) {
; CHECK-LABEL: try_cmpxchg:
; CHECK: cmpxchgl
; CHECK-NOT: cmp
; CHECK: sete %al
; CHECK: retq
  %old = cmpxchg i32* %addr, i32 %desired, i32 %new seq_cst seq_cst
  %success = icmp eq i32 %old, %desired
  ret i1 %success
}

define void @cmpxchg_flow(i64* %addr, i64 %desired, i64 %new) {
; CHECK-LABEL: cmpxchg_flow:
; CHECK: cmpxchgq
; CHECK-NOT: cmp
; CHECK-NOT: set
; CHECK: {{jne|jeq}}
  %old = cmpxchg i64* %addr, i64 %desired, i64 %new seq_cst seq_cst
  %success = icmp eq i64 %old, %desired
  br i1 %success, label %true, label %false

true:
  call void @foo()
  ret void

false:
  call void @bar()
  ret void
}

define i1 @cmpxchg_arithcmp(i16* %addr, i16 %desired, i16 %new) {
; CHECK-LABEL: cmpxchg_arithcmp:
; CHECK: cmpxchgw
; CHECK-NOT: cmp
; CHECK: setbe %al
; CHECK: retq
  %old = cmpxchg i16* %addr, i16 %desired, i16 %new seq_cst seq_cst
  %success = icmp uge i16 %old, %desired
  ret i1 %success
}

define i1 @cmpxchg_arithcmp_swapped(i8* %addr, i8 %desired, i8 %new) {
; CHECK-LABEL: cmpxchg_arithcmp_swapped:
; CHECK: cmpxchgb
; CHECK-NOT: cmp
; CHECK: setge %al
; CHECK: retq
  %old = cmpxchg i8* %addr, i8 %desired, i8 %new seq_cst seq_cst
  %success = icmp sge i8 %desired, %old
  ret i1 %success
}

define i64 @cmpxchg_sext(i32* %addr, i32 %desired, i32 %new) {
; CHECK-LABEL: cmpxchg_sext:
; CHECK-DAG: cmpxchgl
; CHECK-DAG: movq $-1, %rax
; CHECK-DAG: xorl %e[[ZERO:[a-z0-9]+]], %e[[ZERO]]
; CHECK-NOT: cmpl
; CHECK: cmovneq %r[[ZERO]], %rax
; CHECK: retq
  %old = cmpxchg i32* %addr, i32 %desired, i32 %new seq_cst seq_cst
  %success = icmp eq i32 %old, %desired
  %mask = sext i1 %success to i64
  ret i64 %mask
}

define i32 @cmpxchg_zext(i32* %addr, i32 %desired, i32 %new) {
; CHECK-LABEL: cmpxchg_zext:
; CHECK: cmpxchgl
; CHECK-NOT: cmp
; CHECK: sete [[BYTE:%[a-z0-9]+]]
; CHECK: movzbl [[BYTE]], %eax
  %old = cmpxchg i32* %addr, i32 %desired, i32 %new seq_cst seq_cst
  %success = icmp eq i32 %old, %desired
  %mask = zext i1 %success to i32
  ret i32 %mask
}

declare void @foo()
declare void @bar()
