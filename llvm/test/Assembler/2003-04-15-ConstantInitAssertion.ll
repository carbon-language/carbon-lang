; RUN: not llvm-as < %s >/dev/null |& grep {struct initializer doesn't match struct element type}
; Test the case of a misformed constant initializer
; This should cause an assembler error, not an assertion failure!
constant { i32 } { float 1.0 }
