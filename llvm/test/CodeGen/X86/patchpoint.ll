; RUN: llc < %s -march=x86-64 | FileCheck %s

; Trivial patchpoint codegen
;
; FIXME: We should verify that the call target is materialize after
; the label immediately before the call.
; <rdar://15187295> [JS] llvm.webkit.patchpoint call target should be
; materialized in nop slide.
define i64 @trivial_patchpoint_codegen(i64 %p1, i64 %p2, i64 %p3, i64 %p4) {
entry:
; CHECK-LABEL: _trivial_patchpoint_codegen:
; CHECK:      Ltmp
; CHECK:      callq *%rax
; CHECK-NEXT: nop
; CHECK:      movq %rax, %[[REG:r.+]]
; CHECK:      callq *%rax
; CHECK-NEXT: nop
; CHECK:      movq %[[REG]], %rax
; CHECK:      ret
  %resolveCall2 = inttoptr i64 -559038736 to i8*
  %result = tail call i64 (i32, i32, i8*, i32, ...)* @llvm.experimental.patchpoint.i64(i32 2, i32 12, i8* %resolveCall2, i32 4, i64 %p1, i64 %p2, i64 %p3, i64 %p4)
  %resolveCall3 = inttoptr i64 -559038737 to i8*
  tail call void (i32, i32, i8*, i32, ...)* @llvm.experimental.patchpoint.void(i32 3, i32 12, i8* %resolveCall3, i32 2, i64 %p1, i64 %result)
  ret i64 %result
}

; Caller frame metadata with stackmaps. This should not be optimized
; as a leaf function.
;
; CHECK-LABEL: _caller_meta_leaf
; CHECK: subq $24, %rsp
; CHECK: Ltmp
; CHECK: addq $24, %rsp
; CHECK: ret
define void @caller_meta_leaf() {
entry:
  %metadata = alloca i64, i32 3, align 8
  store i64 11, i64* %metadata
  store i64 12, i64* %metadata
  store i64 13, i64* %metadata
  call void (i32, i32, ...)* @llvm.experimental.stackmap(i32 4, i32 0, i64* %metadata)
  ret void
}


declare void @llvm.experimental.stackmap(i32, i32, ...)
declare void @llvm.experimental.patchpoint.void(i32, i32, i8*, i32, ...)
declare i64 @llvm.experimental.patchpoint.i64(i32, i32, i8*, i32, ...)
