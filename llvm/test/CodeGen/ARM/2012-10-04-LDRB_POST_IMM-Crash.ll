; RUN: llc < %s -mtriple=armv7-none-linux-gnueabi | FileCheck %s
; Check that LDRB_POST_IMM instruction emitted properly.

%my_struct_t = type { double, double, double }
@main.val = private unnamed_addr constant %my_struct_t { double 1.0, double 2.0, double 3.0 }, align 8

declare void @f(i32 %n1, %my_struct_t* byval %val);


; CHECK: main:
define i32 @main() nounwind {
entry:
  %val = alloca %my_struct_t, align 8
  %0 = bitcast %my_struct_t* %val to i8*

; CHECK: ldrb	{{(r[0-9]+)}}, {{(\[r[0-9]+\])}}, #1
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %0, i8* bitcast (%my_struct_t* @main.val to i8*), i32 24, i32 8, i1 false)

  call void @f(i32 555, %my_struct_t* byval %val)
  ret i32 0
}

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) nounwind
