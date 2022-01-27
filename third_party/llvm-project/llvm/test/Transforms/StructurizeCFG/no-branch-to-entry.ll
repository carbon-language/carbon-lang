; XFAIL: *

; This test used to generate a region that caused it to delete the entry block,
; but it does not anymore after the changes to handling of infinite loops in the
; PostDominatorTree.
; TODO: This should be either replaced with another IR or deleted completely.

; RUN: opt -S -o - -structurizecfg -verify-dom-info < %s | FileCheck %s

; CHECK-LABEL: @no_branch_to_entry_undef(
; CHECK: entry:
; CHECK-NEXT: br label %entry.orig
define void @no_branch_to_entry_undef(i32 addrspace(1)* %out) {
entry:
  br i1 undef, label %for.end, label %for.body

for.body:                                         ; preds = %entry, %for.body
  store i32 999, i32 addrspace(1)* %out, align 4
  br label %for.body

for.end:                                          ; preds = %Flow
  ret void
}

; CHECK-LABEL: @no_branch_to_entry_true(
; CHECK: entry:
; CHECK-NEXT: br label %entry.orig
define void @no_branch_to_entry_true(i32 addrspace(1)* %out) {
entry:
  br i1 true, label %for.end, label %for.body

for.body:                                         ; preds = %entry, %for.body
  store i32 999, i32 addrspace(1)* %out, align 4
  br label %for.body

for.end:                                          ; preds = %Flow
  ret void
}
