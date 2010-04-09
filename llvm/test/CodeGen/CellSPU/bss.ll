; RUN: llc < %s -march=cellspu > %t1.s
; RUN: grep "\.section" %t1.s | grep "\.bss" | count 1
; CHECK-NOT: .lcomm

@bssVar = global i32 zeroinitializer
@localVar= internal global i32 zeroinitializer

