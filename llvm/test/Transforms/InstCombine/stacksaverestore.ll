; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep {call.*stackrestore}

;; Test that llvm.stackrestore is removed when possible.
define i32* @test1(i32 %P) {
	%tmp = call i8* @llvm.stacksave( )
	call void @llvm.stackrestore( i8* %tmp ) ;; not restoring anything
	%A = alloca i32, i32 %P		
	ret i32* %A
}

define void @test2(i8* %X) {
	call void @llvm.stackrestore( i8* %X )  ;; no allocas before return.
	ret void
}

declare i8* @llvm.stacksave()

declare void @llvm.stackrestore(i8*)

