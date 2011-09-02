; RUN: llc < %s -mtriple=i386-linux-gnu | FileCheck %s --check-prefix=CHECK-DEFAULT
; RUN: llc < %s -mtriple=i386-apple-darwin | FileCheck %s --check-prefix=CHECK-DARWIN
; PR5329

@llvm.global_ctors = appending global [3 x { i32, void ()* }] [{ i32, void ()* } { i32 2000, void ()* @construct_2 }, { i32, void ()* } { i32 3000, void ()* @construct_3 }, { i32, void ()* } { i32 1000, void ()* @construct_1 }]
; CHECK-DEFAULT: construct_3
; CHECK-DEFAULT: construct_2
; CHECK-DEFAULT: construct_1

; CHECK-DARWIN: construct_1
; CHECK-DARWIN: construct_2
; CHECK-DARWIN: construct_3

@llvm.global_dtors = appending global [3 x { i32, void ()* }] [{ i32, void ()* } { i32 2000, void ()* @destruct_2 }, { i32, void ()* } { i32 1000, void ()* @destruct_1 }, { i32, void ()* } { i32 3000, void ()* @destruct_3 }]
; CHECK-DEFAULT: destruct_3
; CHECK-DEFAULT: destruct_2
; CHECK-DEFAULT: destruct_1

; CHECK-DARWIN: destruct_1
; CHECK-DARWIN: destruct_2
; CHECK-DARWIN: destruct_3

declare void @construct_1()
declare void @construct_2()
declare void @construct_3()
declare void @destruct_1()
declare void @destruct_2()
declare void @destruct_3()
