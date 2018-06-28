; RUN: llc %s -debug-pass=Structure -verify-machineinstrs \
; RUN: -enable-machine-outliner=always -mtriple arm64---- -o /dev/null 2>&1 \
; RUN: | FileCheck %s -check-prefix=ALWAYS

; RUN: llc %s -debug-pass=Structure -verify-machineinstrs \
; RUN: -enable-machine-outliner -mtriple arm64---- -o /dev/null 2>&1 \
; RUN: | FileCheck %s -check-prefix=ENABLE

; RUN: llc %s -debug-pass=Structure -verify-machineinstrs \
; RUN: -enable-machine-outliner=never -mtriple arm64---- -o /dev/null 2>&1 \
; RUN: | FileCheck %s -check-prefix=NEVER

; RUN: llc %s -debug-pass=Structure -verify-machineinstrs \
; RUN: -mtriple arm64---- -o /dev/null 2>&1 \
; RUN: | FileCheck %s -check-prefix=NOT-ADDED

; Make sure that the outliner flags all work properly. If we specify
; -enable-machine-outliner with always or no argument, it should be added to the
; pass pipeline. If we specify it with never, or don't pass the flag,
; then we shouldn't add it.

; ALWAYS: Machine Outliner
; ENABLE: Machine Outliner
; NEVER-NOT: Machine Outliner
; NOT-ADDED-NOT: Machine Outliner

define void @foo() {
  ret void;
}

