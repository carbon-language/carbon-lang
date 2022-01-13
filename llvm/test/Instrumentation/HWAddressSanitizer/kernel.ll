; Test KHWASan instrumentation.
;
; RUN: opt < %s -hwasan -hwasan-kernel=1 -hwasan-recover=1 -S | FileCheck %s --allow-empty --check-prefixes=INIT
; RUN: opt < %s -hwasan -hwasan-kernel=1 -hwasan-recover=1 -S | FileCheck %s  --check-prefixes=CHECK,NOOFFSET,MATCH-ALL
; RUN: opt < %s -hwasan -hwasan-kernel=1 -hwasan-recover=1 -hwasan-mapping-offset=12345678 -S | FileCheck %s  --check-prefixes=CHECK,OFFSET,MATCH-ALL
; RUN: opt < %s -hwasan -hwasan-kernel=1 -hwasan-recover=1 -hwasan-match-all-tag=-1 -S | FileCheck %s  --check-prefixes=CHECK,NOOFFSET,NO-MATCH-ALL
; RUN: opt < %s -hwasan -hwasan-kernel=1 -hwasan-recover=1 -hwasan-inline-all-checks=0 -hwasan-mapping-offset=12345678 -S | FileCheck %s  --check-prefixes=OUTLINE

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-android"

define i8 @test_load(i8* %a) sanitize_hwaddress {
; CHECK-LABEL: @test_load(
; OFFSET: %[[SHADOW:[^ ]*]] = call i8* asm "", "=r,0"(i8* inttoptr (i64 12345678 to i8*))
; CHECK: %[[A:[^ ]*]] = ptrtoint i8* %a to i64
; CHECK: %[[B:[^ ]*]] = lshr i64 %[[A]], 56
; CHECK: %[[PTRTAG:[^ ]*]] = trunc i64 %[[B]] to i8
; CHECK: %[[C:[^ ]*]] = or i64 %[[A]], -72057594037927936
; CHECK: %[[D:[^ ]*]] = lshr i64 %[[C]], 4

; NOOFFSET: %[[E:[^ ]*]] = inttoptr i64 %[[D]] to i8*

; OFFSET: %[[E:[^ ]*]] = getelementptr i8, i8* %[[SHADOW]], i64 %[[D]]

; CHECK: %[[MEMTAG:[^ ]*]] = load i8, i8* %[[E]]
; CHECK: %[[F:[^ ]*]] = icmp ne i8 %[[PTRTAG]], %[[MEMTAG]]

; MATCH-ALL: %[[G:[^ ]*]] = icmp ne i8 %[[PTRTAG]], -1
; MATCH-ALL: %[[H:[^ ]*]] = and i1 %[[F]], %[[G]]
; MATCH-ALL: br i1 %[[H]], label {{.*}}, label {{.*}}, !prof {{.*}}

; NO-MATCH-ALL: br i1 %[[F]], label {{.*}}, label {{.*}}, !prof {{.*}}

; CHECK: call void asm sideeffect "brk #2336", "{x0}"(i64 %[[A]])
; CHECK: br label

; CHECK: %[[G:[^ ]*]] = load i8, i8* %a, align 4
; CHECK: ret i8 %[[G]]

; OUTLINE: %[[SHADOW:[^ ]*]] = call i8* asm "", "=r,0"(i8* inttoptr (i64 12345678 to i8*))
; OUTLINE: call void @llvm.hwasan.check.memaccess(i8* %[[SHADOW]], i8* %a, i32 67043360)
entry:
  %b = load i8, i8* %a, align 4
  ret i8 %b
}

; INIT-NOT: call void @__hwasan_init
