; RUN: opt -instcombine -S < %s | FileCheck %s

target datalayout = "e-i64:64-f80:128-n8:16:32:64"
target triple = "x86_64-unknown-linux-gnu"

%A__vtbl = type { i8*, i32 (%A*)* }
%A = type { %A__vtbl* }

@A__vtblZ = constant %A__vtbl { i8* null, i32 (%A*)* @A.foo }

declare i32 @A.foo(%A* nocapture %this)

declare i8* @allocmemory(i64)

define void @structA() {
body:
  %0 = tail call i8* @allocmemory(i64 32)
  %1 = bitcast i8* %0 to %A*
; CHECK: store %A__vtbl* @A__vtblZ
  store %A { %A__vtbl* @A__vtblZ }, %A* %1, align 8
  ret void
}

define void @structOfA() {
body:
  %0 = tail call i8* @allocmemory(i64 32)
  %1 = bitcast i8* %0 to { %A }*
; CHECK: store %A__vtbl* @A__vtblZ
  store { %A } { %A { %A__vtbl* @A__vtblZ } }, { %A }* %1, align 8
  ret void
}
