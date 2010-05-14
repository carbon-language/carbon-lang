; RUN: llc -filetype=obj -o - < %s
; XFAIL: vg_leak

; ModuleID = 't.c'
target triple = "x86_64-apple-darwin10.0.0"

module asm ".desc _f0, 0x10"
