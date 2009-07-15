; RUN: llvm-as < %s | llc | grep {bl L_exact_log2.stub}
; PR4482
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "powerpc-apple-darwin8"

define i32 @foo(i64 %x) nounwind {
entry:
        %A = call i32 @exact_log2(i64 %x) nounwind
	ret i32 %A
}

define available_externally i32 @exact_log2(i64 %x) nounwind {
entry:
	ret i32 42
}
