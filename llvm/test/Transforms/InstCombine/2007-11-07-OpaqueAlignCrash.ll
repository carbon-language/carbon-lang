; RUN: llvm-as < %s | opt -instcombine -disable-output
; PR1780

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"
target triple = "i686-pc-linux-gnu"

%opaque_t = type opaque

@g = external global %opaque_t

define i32 @foo() {
entry:
        %x = load i8* bitcast (%opaque_t* @g to i8*)
        %y = load i32* bitcast (%opaque_t* @g to i32*)
	%z = zext i8 %x to i32
	%r = add i32 %y, %z
        ret i32 %r
}

