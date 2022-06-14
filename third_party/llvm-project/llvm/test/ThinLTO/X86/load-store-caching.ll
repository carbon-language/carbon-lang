; Test that instruction operands from loads are not cached when
; processing stores. Reference from @foo to @obj should not be
; readonly or writeonly

; RUN: opt -module-summary %s -o %t.bc
; RUN: llvm-dis %t.bc -o - | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.S = type { %struct.Derived* }
%struct.Derived = type { i32 }
%struct.Base = type { i32 }

@obj = dso_local local_unnamed_addr global %struct.S zeroinitializer, align 8

define dso_local %struct.Base* @foo() local_unnamed_addr {
entry:
  %0 = load %struct.Base*, %struct.Base** bitcast (%struct.S* @obj to %struct.Base**), align 8
  store %struct.Base* null, %struct.Base** bitcast (%struct.S* @obj to %struct.Base**), align 8
  ret %struct.Base* %0
}

; CHECK:       ^0 = module:
; CHECK-NEXT:  ^1 = gv: (name: "obj", summaries: (variable: (module: ^0, flags: (linkage: external, visibility: default, notEligibleToImport: 0, live: 0, dsoLocal: 1, canAutoHide: 0), varFlags: (readonly: 1, writeonly: 1, constant: 0)))) ; guid =
; CHECK-NEXT:  ^2 = gv: (name: "foo", summaries: (function: (module: ^0, flags: (linkage: external, visibility: default, notEligibleToImport: 0, live: 0, dsoLocal: 1, canAutoHide: 0), insts: 3, refs: (^1)))) ; guid =
