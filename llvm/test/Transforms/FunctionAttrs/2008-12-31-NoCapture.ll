; RUN: llvm-as < %s | opt -functionattrs | llvm-dis | not grep {@c.*nocapture}
; RUN: llvm-as < %s | opt -functionattrs | llvm-dis | grep nocapture | count 3
@g = global i32* null		; <i32**> [#uses=1]

define i32* @c1(i32* %p) {
	ret i32* %p
}

define void @c2(i32* %p) {
	store i32* %p, i32** @g
	ret void
}

define void @c3(i32* %p) {
	call void @c2(i32* %p)
	ret void
}

define i32 @nc1(i32* %p) {
	%tmp = bitcast i32* %p to i32*		; <i32*> [#uses=2]
	%val = load i32* %tmp		; <i32> [#uses=1]
	store i32 0, i32* %tmp
	ret i32 %val
}

define void @nc2(i32* %p) {
	%1 = call i32 @nc1(i32* %p)		; <i32> [#uses=0]
	ret void
}

define void @nc3(void ()* %f) {
	call void %f()
	ret void
}
