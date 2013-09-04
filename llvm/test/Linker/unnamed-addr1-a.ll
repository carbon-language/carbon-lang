; RUN: llvm-link %s %p/unnamed-addr1-b.ll -S -o - | FileCheck %s

; Only in this file
@global-a = common global i32 0
; CHECK-DAG: @global-a = common global i32 0
@global-b = common unnamed_addr global i32 0
; CHECK-DAG: @global-b = common unnamed_addr global i32 0

; Other file has unnamed_addr definition
@global-c = common unnamed_addr global i32 0
; CHECK-DAG: @global-c = common unnamed_addr global i32 0
@global-d = external global i32
; CHECK-DAG: @global-d = global i32 42
@global-e = external unnamed_addr global i32
; CHECK-DAG: @global-e = unnamed_addr global i32 42
@global-f = weak global i32 42
; CHECK-DAG: @global-f = global i32 42

; Other file has non-unnamed_addr definition
@global-g = common unnamed_addr global i32 0
; CHECK-DAG: @global-g = common global i32 0
@global-h = external global i32
; CHECK-DAG: @global-h = global i32 42
@global-i = external unnamed_addr global i32
; CHECK-DAG: @global-i = global i32 42
@global-j = weak global i32 42
; CHECK-DAG: @global-j = global i32 42
