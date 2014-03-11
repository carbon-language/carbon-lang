; REQUIRES: shell
; RUN: llvm-link %s %S/Inputs/datalayout-a.ll -S -o - 2>%t.a.err | FileCheck %s
; RUN: (echo foo ;cat %t.a.err) | FileCheck --check-prefix=WARN-A %s

; RUN: llvm-link %s %S/Inputs/datalayout-b.ll -S -o - 2>%t.b.err | FileCheck %s
; RUN: cat %t.b.err | FileCheck --check-prefix=WARN-B %s

target datalayout = "e"

; CHECK: target datalayout = "e"

; WARN-A-NOT: WARNING

; WARN-B: WARNING: Linking two modules of different data layouts:
