; Test that inline assembly is parsed by the MC layer when MC support is mature
; (even when the output is assembly).

; RUN: not llc -march=aarch64 < %s > /dev/null 2> %t1
; RUN: FileCheck %s < %t1

; RUN: not llc -march=aarch64 -filetype=obj < %s > /dev/null 2> %t2
; RUN: FileCheck %s < %t2

module asm "	.this_directive_is_very_unlikely_to_exist"

; CHECK: LLVM ERROR: Error parsing inline asm
