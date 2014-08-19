; RUN: llvm-as < %s | llvm-dis | grep 1.0
; RUN: verify-uselistorder %s

define double @test() {
        ret double 1.0   ;; This should not require hex notation
}

