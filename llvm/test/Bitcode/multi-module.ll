; RUN: llvm-cat -o %t %s %S/Inputs/multi-module.ll
; RUN: not llvm-dis -o - %t 2>&1 | FileCheck --check-prefix=ERROR %s
; ERROR: Expected a single module

; FIXME: Introduce a tool for extracting modules from bitcode and use it here.
; For now we can at least check that the bitcode contains multiple modules.
; RUN: llvm-bcanalyzer -dump %t | FileCheck --check-prefix=BCA %s

; RUN: llvm-as -o %t1 %s
; RUN: llvm-as -o %t2 %S/Inputs/multi-module.ll
; RUN: llvm-cat -o %t %t1 %t2
; RUN: not llvm-dis -o - %t 2>&1 | FileCheck --check-prefix=ERROR %s
; RUN: llvm-bcanalyzer -dump %t | FileCheck --check-prefix=BCA %s

; RUN: llvm-cat -b -o %t %t1 %t2
; RUN: not llvm-dis -o - %t 2>&1 | FileCheck --check-prefix=ERROR %s
; RUN: llvm-bcanalyzer -dump %t | FileCheck --check-prefix=BCA %s

; RUN: llvm-cat -b -o %t3 %t %t
; RUN: not llvm-dis -o - %t3 2>&1 | FileCheck --check-prefix=ERROR %s
; RUN: llvm-bcanalyzer -dump %t3 | FileCheck --check-prefix=BCA4 %s

; BCA: <IDENTIFICATION_BLOCK
; BCA: <MODULE_BLOCK
; BCA: <IDENTIFICATION_BLOCK
; BCA: <MODULE_BLOCK

; BCA4: <IDENTIFICATION_BLOCK
; BCA4: <MODULE_BLOCK
; BCA4: <IDENTIFICATION_BLOCK
; BCA4: <MODULE_BLOCK
; BCA4: <IDENTIFICATION_BLOCK
; BCA4: <MODULE_BLOCK
; BCA4: <IDENTIFICATION_BLOCK
; BCA4: <MODULE_BLOCK

define void @f1() {
  ret void
}
