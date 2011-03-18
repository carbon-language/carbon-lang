; RUN: opt < %s -indvars -S > %t
; RUN: grep {icmp ugt i8\\\*} %t | count 1
; RUN: grep {icmp sgt i8\\\*} %t | count 1

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n:32:64"

	%struct.CKenCodeCodec = type <{ i8 }>

define void @foo(i8* %str1Ptr, i8* %str2Ptr, i8* %inLastBytePtr) nounwind {
entry:
	%0 = icmp ult i8* %str2Ptr, %str1Ptr		; <i1> [#uses=1]
	%str2Ptr_addr.0 = select i1 %0, i8* %str1Ptr, i8* %str2Ptr		; <i8*> [#uses=1]
	br label %bb2

bb2:		; preds = %bb2, %entry
	%str2Ptr_addr.1 = phi i8* [ %str2Ptr_addr.0, %entry ], [ %1, %bb2 ]		; <i8*> [#uses=1]
	%1 = getelementptr i8* %str2Ptr_addr.1, i64 1		; <i8*> [#uses=2]
	%2 = icmp ult i8* %1, %inLastBytePtr		; <i1> [#uses=0]
	br i1 undef, label %bb2, label %return

return:		; preds = %bb2
	ret void
}

define void @sfoo(i8* %str1Ptr, i8* %str2Ptr, i8* %inLastBytePtr) nounwind {
entry:
	%0 = icmp slt i8* %str2Ptr, %str1Ptr		; <i1> [#uses=1]
	%str2Ptr_addr.0 = select i1 %0, i8* %str1Ptr, i8* %str2Ptr		; <i8*> [#uses=1]
	br label %bb2

bb2:		; preds = %bb2, %entry
	%str2Ptr_addr.1 = phi i8* [ %str2Ptr_addr.0, %entry ], [ %1, %bb2 ]		; <i8*> [#uses=1]
	%1 = getelementptr i8* %str2Ptr_addr.1, i64 1		; <i8*> [#uses=2]
	%2 = icmp slt i8* %1, %inLastBytePtr		; <i1> [#uses=0]
	br i1 undef, label %bb2, label %return

return:		; preds = %bb2
	ret void
}
