; RUN: llc < %s -march=x86 -mcpu=corei7-avx -mattr=+avx -mtriple=i686-pc-win32

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f80:128:128-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32-S32"
target triple = "i686-pc-win32"

define void @bad_cast() {
entry:
  %vext.i = shufflevector <2 x i64> undef, <2 x i64> undef, <3 x i32> <i32 0, i32 1, i32 undef>
  %vecinit8.i = shufflevector <3 x i64> zeroinitializer, <3 x i64> %vext.i, <3 x i32> <i32 0, i32 3, i32 4>
  store <3 x i64> %vecinit8.i, <3 x i64>* undef, align 32
  unreachable
}

