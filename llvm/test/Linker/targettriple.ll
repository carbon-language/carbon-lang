; RUN: llvm-link %s %S/Inputs/targettriple-a.ll -S -o - 2>%t.a.err | FileCheck %s
; RUN: cat %t.a.err | not FileCheck %s 2>&1 | FileCheck --check-prefix=WARN-A %s

; RUN: llvm-link %s %S/Inputs/targettriple-b.ll -S -o - 2>%t.b.err | FileCheck %s
; RUN: cat %t.b.err | FileCheck --check-prefix=WARN-B %s

target triple = "e"

; CHECK: target triple = "e"

; this is a hack to check that llvm-link printed no warnings.
; WARN-A: FileCheck error: '-' is empty.

; WARN-B: WARNING: Linking two modules of different target triples:
