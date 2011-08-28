; RUN: llc < %s -mtriple=i386-linux-gnu | FileCheck %s
; PR5329

@llvm.global_ctors = appending global [3 x { i32, void ()* }] [{ i32, void ()* } { i32 2000, void ()* @construct_2 }, { i32, void ()* } { i32 3000, void ()* @construct_3 }, { i32, void ()* } { i32 1000, void ()* @construct_1 }]
; CHECK: ctors
; CHECK: construct_3
; CHECK: construct_2
; CHECK: construct_1

@llvm.global_dtors = appending global [3 x { i32, void ()* }] [{ i32, void ()* } { i32 2000, void ()* @destruct_2 }, { i32, void ()* } { i32 1000, void ()* @destruct_1 }, { i32, void ()* } { i32 3000, void ()* @destruct_3 }]
; CHECK: dtors
; CHECK: destruct_3
; CHECK: destruct_2
; CHECK: destruct_1

declare void @construct_1()
declare void @construct_2()
declare void @construct_3()
declare void @destruct_1()
declare void @destruct_2()
declare void @destruct_3()
