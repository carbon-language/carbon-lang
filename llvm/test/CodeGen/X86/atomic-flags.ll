; RUN: llc < %s -mtriple=x86_64-unknown-unknown -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -mtriple=i686-unknown-unknown -verify-machineinstrs | FileCheck %s

; Make sure that flags are properly preserved despite atomic optimizations.

define i32 @atomic_and_flags(i8* %p, i32 %a, i32 %b) {
; CHECK-LABEL: atomic_and_flags:

  ; Generate flags value, and use it.
  ; CHECK:      cmpl
  ; CHECK-NEXT: jne
  %cmp = icmp eq i32 %a, %b
  br i1 %cmp, label %L1, label %L2

L1:
  ; The following pattern will get folded.
  ; CHECK: addb
  %1 = load atomic i8, i8* %p seq_cst, align 1
  %2 = add i8 %1, 2
  store atomic i8 %2, i8* %p release, align 1

  ; Use the comparison result again. We need to rematerialize the comparison
  ; somehow. This test checks that cmpl gets emitted again, but any
  ; rematerialization would work (the optimizer used to clobber the flags with
  ; the add).
  ; CHECK-NEXT: cmpl
  ; CHECK-NEXT: jne
  br i1 %cmp, label %L3, label %L4

L2:
  ret i32 2

L3:
  ret i32 3

L4:
  ret i32 4
}
