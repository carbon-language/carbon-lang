; Test that appending linkage works correctly when arrays are the same size.

; RUN: echo "%X = appending global [1x int] [int 8]" | as > %t.2.bc
; RUN: as < %s > %t.1.bc
; RUN: link %t.[12].bc | dis | grep 7 | grep 8

%X = appending global [1 x int] [int 7]
