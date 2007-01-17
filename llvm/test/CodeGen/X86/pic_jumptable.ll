; RUN: llvm-upgrade < %s | llvm-as | llc -relocation-model=pic -march=x86 | not grep -F .text
target endian = little
target pointersize = 32
target triple = "i386-linux-gnu"

implementation   ; Functions:

declare void %_Z3bari( int  )

linkonce void %_Z3fooILi1EEvi(int %Y) {
entry:
	%Y_addr = alloca int		; <int*> [#uses=2]
	"alloca point" = cast int 0 to int		; <int> [#uses=0]
	store int %Y, int* %Y_addr
	%tmp = load int* %Y_addr		; <int> [#uses=1]
	switch int %tmp, label %bb10 [
		 int 0, label %bb3
		 int 1, label %bb
		 int 2, label %bb
		 int 3, label %bb
		 int 4, label %bb
		 int 5, label %bb
		 int 6, label %bb
		 int 7, label %bb
		 int 8, label %bb
		 int 9, label %bb
		 int 10, label %bb
		 int 12, label %bb1
		 int 13, label %bb5
		 int 14, label %bb6
		 int 16, label %bb2
		 int 17, label %bb4
		 int 23, label %bb8
		 int 27, label %bb7
		 int 34, label %bb9
	]

bb:		; preds = %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry
	br label %bb1

bb1:		; preds = %bb, %entry
	br label %bb2

bb2:		; preds = %bb1, %entry
	call void %_Z3bari( int 1 )
	br label %bb11

bb3:		; preds = %entry
	br label %bb4

bb4:		; preds = %bb3, %entry
	br label %bb5

bb5:		; preds = %bb4, %entry
	br label %bb6

bb6:		; preds = %bb5, %entry
	call void %_Z3bari( int 2 )
	br label %bb11

bb7:		; preds = %entry
	br label %bb8

bb8:		; preds = %bb7, %entry
	br label %bb9

bb9:		; preds = %bb8, %entry
	call void %_Z3bari( int 3 )
	br label %bb11

bb10:		; preds = %entry
	br label %bb11

bb11:		; preds = %bb10, %bb9, %bb6, %bb2
	br label %return

return:		; preds = %bb11
	ret void
}
