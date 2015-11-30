; RUN: not llvm-link -S %s %p/Inputs/comdat13.ll -o %t.ll 2>&1 | FileCheck %s

; In Inputs/comdat13.ll a function not in the $foo comdat (zed) references an
; internal function in the comdat $foo.
; We might want to have the verifier reject that, but for now we at least check
; that the linker produces an error.
; This is the IR equivalent of the "relocation refers to discarded section" in
; an ELF linker.

; CHECK: Declaration points to discarded value

$foo = comdat any
@foo = global i8 0, comdat
