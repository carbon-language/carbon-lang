; This isn't really an assembly file. This test runs the ParallelJIT example
; program and ensures its output is sane.
; RUN: ParallelJIT | grep -q "Fib2 returned 267914296"
