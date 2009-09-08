; RUN: llc < %s -mtriple=i386-linux | FileCheck %s
	%union.x = type { i64 }

; CHECK:	.globl r
; CHECK: r:
; CHECK: .quad	((r) & 4294967295)

@r = global %union.x { i64 ptrtoint (%union.x* @r to i64) }, align 4
