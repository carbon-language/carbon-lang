; RUN: llc < %s -O0 -filetype=null -wasm-enable-eh -exception-model=wasm -mattr=+exception-handling
; RUN: llc < %s -O0 -filetype=asm -asm-verbose=false -wasm-enable-eh -exception-model=wasm -mattr=+exception-handling | FileCheck %s

target triple = "wasm32-unknown-unknown"

declare void @llvm.wasm.throw(i32, i8*)
declare void @g()

define i32 @test(i8* %p)  {
  %n = alloca i32
  call void @llvm.wasm.throw(i32 0, i8* %p)
  call void @g()
  ret i32 0
}

; CHECK-DAG: .globaltype
; CHECK-DAG: .tagtype
; CHECK-DAG: .functype
