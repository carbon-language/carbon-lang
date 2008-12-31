; RUN: llvm-as < %s | opt -functionattrs | llvm-dis | not grep {nocapture *%%q}
; RUN: llvm-as < %s | opt -functionattrs | llvm-dis | grep {nocapture *%%p} | count 3
@g = global i32* null		; <i32**> [#uses=1]

define i32* @c1(i32* %q) {
	ret i32* %q
}

define void @c2(i32* %q) {
	store i32* %q, i32** @g
	ret void
}

define void @c3(i32* %q) {
	call void @c2(i32* %q)
	ret void
}

define i32 @nc1(i32* %q, i32* %p, i1 %b) {
e:
	br label %l
l:
	%x = phi i32* [ %p, %e ]
	%y = phi i32* [ %q, %e ]
	%tmp = bitcast i32* %x to i32*		; <i32*> [#uses=2]
	%tmp2 = select i1 %b, i32* %tmp, i32* %y
	%val = load i32* %tmp2		; <i32> [#uses=1]
	store i32 0, i32* %tmp
	store i32* %y, i32** @g
	ret i32 %val
}

define void @nc2(i32* %p, i32* %q) {
	%1 = call i32 @nc1(i32* %q, i32* %p, i1 0)		; <i32> [#uses=0]
	ret void
}

define void @nc3(void ()* %p) {
	call void %p()
	ret void
}
