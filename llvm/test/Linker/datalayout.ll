; RUN: llvm-link %s %S/Inputs/datalayout-a.ll -S -o - 2>%t.a.err | FileCheck %s
; RUN: cat %t.a.err | not FileCheck %s 2>&1 | FileCheck --check-prefix=WARN-A %s

; RUN: llvm-link %s %S/Inputs/datalayout-b.ll -S -o - 2>%t.b.err | FileCheck %s
; RUN: cat %t.b.err | FileCheck --check-prefix=WARN-B %s

target datalayout = "e"

; CHECK: target datalayout = "e"

; this is a hack to check that llvm-link printed no warnings.
; WARN-A: FileCheck error: '-' is empty.

; WARN-B: WARNING: Linking two modules of different data layouts:
