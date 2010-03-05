; RUN: llc < %s -march=cellspu > %t1.s
; RUN: grep "\.section" %t1.s | grep "\.bss" | count 1

@bssVar = global i32 zeroinitializer

