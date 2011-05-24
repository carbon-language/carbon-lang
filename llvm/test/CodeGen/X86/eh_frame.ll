; RUN: llc < %s -mtriple x86_64-unknown-linux-gnu | FileCheck %s

@__FRAME_END__ = constant [1 x i32] zeroinitializer, section ".eh_frame"

; CHECK: .section	.eh_frame,"a",@progbits
