; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s
; PR1358

; CHECK: icmp ne (i32 (...)* @test_weak, i32 (...)* null)
@G = global i1 icmp ne (i32 (...)* @test_weak, i32 (...)* null)

declare extern_weak i32 @test_weak(...)

