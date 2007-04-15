; try to check that we have the most important instructions, which shouldn't 
; appear otherwise
; RUN: llvm-upgrade < %s | llvm-as | llc -march=alpha | grep jmp
; RUN: llvm-upgrade < %s | llvm-as | llc -march=alpha | grep gprel32
; RUN: llvm-upgrade < %s | llvm-as | llc -march=alpha | grep ldl
; RUN: llvm-upgrade < %s | llvm-as | llc -march=alpha | grep rodata
; END.

target endian = little
target pointersize = 64
target triple = "alphaev67-unknown-linux-gnu"
%str = internal constant [2 x sbyte] c"1\00"		; <[2 x sbyte]*> [#uses=1]
%str1 = internal constant [2 x sbyte] c"2\00"		; <[2 x sbyte]*> [#uses=1]
%str2 = internal constant [2 x sbyte] c"3\00"		; <[2 x sbyte]*> [#uses=1]
%str3 = internal constant [2 x sbyte] c"4\00"		; <[2 x sbyte]*> [#uses=1]
%str4 = internal constant [2 x sbyte] c"5\00"		; <[2 x sbyte]*> [#uses=1]
%str5 = internal constant [2 x sbyte] c"6\00"		; <[2 x sbyte]*> [#uses=1]
%str6 = internal constant [2 x sbyte] c"7\00"		; <[2 x sbyte]*> [#uses=1]
%str7 = internal constant [2 x sbyte] c"8\00"		; <[2 x sbyte]*> [#uses=1]

implementation   ; Functions:

int %main(int %x, sbyte** %y) {
entry:
	%x_addr = alloca int		; <int*> [#uses=2]
	%y_addr = alloca sbyte**		; <sbyte***> [#uses=1]
	%retval = alloca int, align 4		; <int*> [#uses=2]
	%tmp = alloca int, align 4		; <int*> [#uses=2]
	%foo = alloca sbyte*, align 8		; <sbyte**> [#uses=9]
	"alloca point" = cast int 0 to int		; <int> [#uses=0]
	store int %x, int* %x_addr
	store sbyte** %y, sbyte*** %y_addr
	%tmp = load int* %x_addr		; <int> [#uses=1]
	switch int %tmp, label %bb15 [
		 int 1, label %bb
		 int 2, label %bb1
		 int 3, label %bb3
		 int 4, label %bb5
		 int 5, label %bb7
		 int 6, label %bb9
		 int 7, label %bb11
		 int 8, label %bb13
	]

bb:		; preds = %entry
	%tmp = getelementptr [2 x sbyte]* %str, int 0, ulong 0		; <sbyte*> [#uses=1]
	store sbyte* %tmp, sbyte** %foo
	br label %bb16

bb1:		; preds = %entry
	%tmp2 = getelementptr [2 x sbyte]* %str1, int 0, ulong 0		; <sbyte*> [#uses=1]
	store sbyte* %tmp2, sbyte** %foo
	br label %bb16

bb3:		; preds = %entry
	%tmp4 = getelementptr [2 x sbyte]* %str2, int 0, ulong 0		; <sbyte*> [#uses=1]
	store sbyte* %tmp4, sbyte** %foo
	br label %bb16

bb5:		; preds = %entry
	%tmp6 = getelementptr [2 x sbyte]* %str3, int 0, ulong 0		; <sbyte*> [#uses=1]
	store sbyte* %tmp6, sbyte** %foo
	br label %bb16

bb7:		; preds = %entry
	%tmp8 = getelementptr [2 x sbyte]* %str4, int 0, ulong 0		; <sbyte*> [#uses=1]
	store sbyte* %tmp8, sbyte** %foo
	br label %bb16

bb9:		; preds = %entry
	%tmp10 = getelementptr [2 x sbyte]* %str5, int 0, ulong 0		; <sbyte*> [#uses=1]
	store sbyte* %tmp10, sbyte** %foo
	br label %bb16

bb11:		; preds = %entry
	%tmp12 = getelementptr [2 x sbyte]* %str6, int 0, ulong 0		; <sbyte*> [#uses=1]
	store sbyte* %tmp12, sbyte** %foo
	br label %bb16

bb13:		; preds = %entry
	%tmp14 = getelementptr [2 x sbyte]* %str7, int 0, ulong 0		; <sbyte*> [#uses=1]
	store sbyte* %tmp14, sbyte** %foo
	br label %bb16

bb15:		; preds = %entry
	br label %bb16

bb16:		; preds = %bb15, %bb13, %bb11, %bb9, %bb7, %bb5, %bb3, %bb1, %bb
	%tmp17 = load sbyte** %foo		; <sbyte*> [#uses=1]
	%tmp18 = call int (...)* %print( sbyte* %tmp17 )		; <int> [#uses=0]
	store int 0, int* %tmp
	%tmp19 = load int* %tmp		; <int> [#uses=1]
	store int %tmp19, int* %retval
	br label %return

return:		; preds = %bb16
	%retval = load int* %retval		; <int> [#uses=1]
	ret int %retval
}

declare int %print(...)
