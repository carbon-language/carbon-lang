; RUN: llc < %s -mtriple=x86_64-unknown -mcpu=corei7

define void @autogen_SD13708(i32) {
BB:
 %Shuff7 = shufflevector <8 x i32> zeroinitializer, <8 x i32> zeroinitializer, <8 x i32> <i32 8, i32 10, i32 12, i32 14, i32 undef, i32 2, i32 4, i32 undef>
 br label %CF

CF:
 %Tr = trunc <8 x i64> zeroinitializer to <8 x i32>
 %Shuff20 = shufflevector <8 x i32> %Shuff7, <8 x i32> %Tr, <8 x i32> <i32 13, i32 15, i32 1, i32 3, i32 5, i32 7, i32 undef, i32 11>
 br i1 undef, label %CF, label %CF247

CF247:
 %I171 = insertelement <8 x i32> %Shuff20, i32 %0, i32 0
 br i1 undef, label %CF, label %CF247
}