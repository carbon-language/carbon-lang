; Test that inline assembly is parsed by the MC layer when MC support is mature
; (even when the output is assembly).

; RUN: not llc -march=aarch64 < %s 2>&1 | FileCheck %s
; RUN: not llc -march=aarch64 -filetype=obj < %s 2>&1 | FileCheck %s
; RUN: not llc -march=arm < %s 2>&1 | FileCheck %s
; RUN: not llc -march=arm -filetype=obj < %s 2>&1 | FileCheck %s
; RUN: not llc -march=thumb < %s 2>&1 | FileCheck %s
; RUN: not llc -march=thumb -filetype=obj < %s 2>&1 | FileCheck %s
; RUN: not llc -march=x86 < %s 2>&1 | FileCheck %s
; RUN: not llc -march=x86 -filetype=obj < %s 2>&1 | FileCheck %s
; RUN: not llc -march=x86-64 < %s 2>&1 | FileCheck %s
; RUN: not llc -march=x86-64 -filetype=obj < %s 2>&1 | FileCheck %s

module asm "	.this_directive_is_very_unlikely_to_exist"

; CHECK: LLVM ERROR: Error parsing inline asm
