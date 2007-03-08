; RUN: llvm-as < %s | llc -march=x86 -mattr=+mmx

;; A basic sanity check to make sure that MMX arithmetic actually compiles.

define void @foo(<2 x i32>* %A, <2 x i32>* %B) {
entry:
	%tmp1 = load <2 x i32>* %A		; <<2 x i32>> [#uses=1]
	%tmp3 = load <2 x i32>* %B		; <<2 x i32>> [#uses=1]
	%tmp4 = add <2 x i32> %tmp1, %tmp3		; <<2 x i32>> [#uses=1]
	store <2 x i32> %tmp4, <2 x i32>* %A
	tail call void @llvm.x86.mmx.emms( )
	ret void
}

define void @bar(<4 x i16>* %A, <4 x i16>* %B) {
entry:
	%tmp1 = load <4 x i16>* %A		; <<4 x i16>> [#uses=1]
	%tmp3 = load <4 x i16>* %B		; <<4 x i16>> [#uses=1]
	%tmp4 = add <4 x i16> %tmp1, %tmp3		; <<4 x i16>> [#uses=1]
	store <4 x i16> %tmp4, <4 x i16>* %A
	tail call void @llvm.x86.mmx.emms( )
	ret void
}

define void @baz(<8 x i8>* %A, <8 x i8>* %B) {
entry:
	%tmp1 = load <8 x i8>* %A		; <<8 x i8>> [#uses=1]
	%tmp3 = load <8 x i8>* %B		; <<8 x i8>> [#uses=1]
	%tmp4 = add <8 x i8> %tmp1, %tmp3		; <<8 x i8>> [#uses=1]
	store <8 x i8> %tmp4, <8 x i8>* %A
	tail call void @llvm.x86.mmx.emms( )
	ret void
}

declare void @llvm.x86.mmx.emms()
