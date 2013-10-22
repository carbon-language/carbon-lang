; RUN: llc < %s -mtriple=x86_64-apple-darwin | FileCheck %s
; PR3168

; CHECK-LABEL: append

define i32* @append() gc "ocaml" {
entry:
  switch i32 0, label %L2 [i32 0, label %L1]
L1:
  %var8 = alloca i8*
  call void @llvm.gcroot(i8** %var8,i8* null)
  br label %L3
L2:
  call ccc void @oread_runtime_casenotcovered()
  unreachable
L3:
  ret i32* null
}

declare ccc void @oread_runtime_casenotcovered()
declare void @llvm.gcroot(i8**,i8*)
