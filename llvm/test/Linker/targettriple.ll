; REQUIRES: shell
; RUN: llvm-link %s %S/Inputs/targettriple-a.ll -S -o - 2>%t.a.err | FileCheck %s
; RUN: (echo foo ;cat %t.a.err) | FileCheck --check-prefix=WARN-A %s

; RUN: llvm-link %s %S/Inputs/targettriple-b.ll -S -o - 2>%t.b.err | FileCheck %s
; RUN: cat %t.b.err | FileCheck --check-prefix=WARN-B %s

; RUN: llvm-link -suppress-warnings %s %S/Inputs/targettriple-b.ll -S -o - 2>%t.no-warn.err | FileCheck %s
; RUN: (echo foo ;cat %t.no-warn.err) | FileCheck --check-prefix=WARN-A %s

target triple = "x86_64-unknown-linux-gnu"

; CHECK: target triple = "x86_64-unknown-linux-gnu"

; WARN-A-NOT: WARNING

; i386 and x86_64 map to different ArchType enums.
; WARN-B: WARNING: Linking two modules of different target triples:

; x86_64h and x86_64 map to the same ArchType enum.
; WARN-C-NOT: WARNING
