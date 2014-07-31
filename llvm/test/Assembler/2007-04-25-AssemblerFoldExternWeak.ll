; RUN: llvm-as < %s | llvm-dis | grep "icmp.*test_weak.*null"
; RUN: verify-uselistorder %s -preserve-bc-use-list-order
; PR1358
@G = global i1 icmp ne (i32 (...)* @test_weak, i32 (...)* null)

declare extern_weak i32 @test_weak(...)

