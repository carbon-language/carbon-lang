; RUN: llvm-link %s %p/unnamed-addr1-b.ll -S -o - | FileCheck %s

; Only in this file
@a = common global i32 0
; CHECK-DAG: @a = common global i32 0
@b = common unnamed_addr global i32 0
; CHECK-DAG: @b = common unnamed_addr global i32 0

; Other file has unnamed_addr definition
@c = common unnamed_addr global i32 0
; CHECK-DAG: @c = common unnamed_addr global i32 0
@d = external global i32
; CHECK-DAG: @d = global i32 42
@e = external unnamed_addr global i32
; CHECK-DAG: @e = unnamed_addr global i32 42
@f = weak global i32 42
; CHECK-DAG: @f = global i32 42

; Other file has non-unnamed_addr definition
@g = common unnamed_addr global i32 0
; CHECK-DAG: @g = common global i32 0
@h = external global i32
; CHECK-DAG: @h = global i32 42
@i = external unnamed_addr global i32
; CHECK-DAG: @i = global i32 42
@j = weak global i32 42
; CHECK-DAG: @j = global i32 42
