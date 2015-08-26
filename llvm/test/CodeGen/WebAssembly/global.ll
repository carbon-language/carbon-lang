; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test that globals assemble as expected.

target datalayout = "e-p:32:32-i64:64-v128:8:128-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-NOT: llvm.used
; CHECK-NOT: llvm.metadata
@llvm.used = appending global [1 x i32*] [i32* @g], section "llvm.metadata"

@g = private global i32 1337; ; CHECK: (global $g i32 1337)

@ud = internal global i32 undef;            ; CHECK: (global $ud i32 0)
@nil = internal global i32 zeroinitializer; ; CHECK: (global $nil i32 0)
@z = internal global i32 0;                 ; CHECK: (global $z i32 0)
@one = internal global i32 1;               ; CHECK: (global $one i32 1)
@answer = internal global i32 42;           ; CHECK: (global $answer i32 42)
@u32max = internal global i32 -1;           ; CHECK: (global $u32max i32 4294967295)

@ud64 = internal global i64 undef;            ; CHECK: (global $ud64 i64 0)
@nil64 = internal global i64 zeroinitializer; ; CHECK: (global $nil64 i64 0)
@z64 = internal global i64 0;                 ; CHECK: (global $z64 i64 0)
@twoP32 = internal global i64 4294967296;     ; CHECK: (global $twoP32 i64 4294967296)
@u64max = internal global i64 -1;             ; CHECK: (global $u64max i64 18446744073709551615)

@f32ud = internal global float undef;            ; CHECK: (global $f32ud f32 0x0p0)
@f32nil = internal global float zeroinitializer; ; CHECK: (global $f32nil f32 0x0p0)
@f32z = internal global float 0.0;               ; CHECK: (global $f32z f32 0x0p0)
@f32nz = internal global float -0.0;             ; CHECK: (global $f32nz f32 -0x0p0)
@f32two = internal global float 2.0;             ; CHECK: (global $f32two f32 0x1p1)

@f64ud = internal global double undef;            ; CHECK: (global $f64ud f64 0x0p0)
@f64nil = internal global double zeroinitializer; ; CHECK: (global $f64nil f64 0x0p0)
@f64z = internal global double 0.0;               ; CHECK: (global $f64z f64 0x0p0)
@f64nz = internal global double -0.0;             ; CHECK: (global $f64nz f64 -0x0p0)
@f64two = internal global double 2.0;             ; CHECK: (global $f64two f64 0x1p1)
