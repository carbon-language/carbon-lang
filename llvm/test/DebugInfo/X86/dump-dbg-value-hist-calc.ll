; REQUIRES: asserts
; RUN: opt -S -o - -debugify %s \
; RUN:   | %llc_dwarf -debug-only=dwarfdebug -o /dev/null 2>&1 \
; RUN:   | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx"

; CHECK: DbgValueHistoryMap:
; CHECK-NEXT:  - 1 at <unknown location> --
; CHECK-NEXT:    Begin: DBG_VALUE {{.*}} line no:1
; CHECK-NEXT:    End  : CALL64pcrel32 @h{{.*}}:2:1

define void @f() {
entry:
  %a = add i32 0, 0
  %b = call i32 @h(i32 %a)
  ret void
}

declare i32 @h(i32)
