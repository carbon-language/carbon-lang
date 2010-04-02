; RUN: opt -globalopt -S < %s | grep {, align 16}

; Globalopt should refine the alignment for global variables.

target datalayout = "e-p:64:64:64"
@a = global [4 x i32] [i32 2, i32 3, i32 4, i32 5 ]
