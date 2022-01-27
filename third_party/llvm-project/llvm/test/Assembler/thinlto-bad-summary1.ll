; Test that we get appropriate error when parsing summary with a missing
; summary type label.
; RUN: not opt %s 2>&1 | FileCheck %s

; CHECK: error: Expected 'gv', 'module', 'typeid', 'flags' or 'blockcount' at the start of summary entry

; ModuleID = 'thinlto-function-summary-callgraph.ll'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define i32 @main() #0 {
entry:
    ret i32 1
}

^0 = module: (path: "{{.*}}thinlto-bad-summary1.ll", hash: (0, 0, 0, 0, 0))
^1 = ()
