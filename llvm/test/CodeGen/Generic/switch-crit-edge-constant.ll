; PR925
; RUN: llvm-upgrade < %s | llvm-as | llc -march=x86 | \
; RUN:   grep mov.*str1 | wc -l | grep 1

target endian = little
target pointersize = 32
target triple = "i686-apple-darwin8.7.2"
%str1 = internal constant [5 x sbyte] c"bonk\00"		; <[5 x sbyte]*> [#uses=1]
%str2 = internal constant [5 x sbyte] c"bork\00"		; <[5 x sbyte]*> [#uses=1]
%str = internal constant [8 x sbyte] c"perfwap\00"		; <[8 x sbyte]*> [#uses=1]

implementation   ; Functions:

void %foo(int %C) {
entry:
	switch int %C, label %bb2 [
		 int 1, label %blahaha
		 int 2, label %blahaha
		 int 3, label %blahaha
		 int 4, label %blahaha
		 int 5, label %blahaha
		 int 6, label %blahaha
		 int 7, label %blahaha
		 int 8, label %blahaha
		 int 9, label %blahaha
		 int 10, label %blahaha
	]

bb2:		; preds = %entry
	%tmp5 = and int %C, 123		; <int> [#uses=1]
	%tmp = seteq int %tmp5, 0		; <bool> [#uses=1]
	br bool %tmp, label %blahaha, label %cond_true

cond_true:		; preds = %bb2
	br label %blahaha

blahaha:		; preds = %cond_true, %bb2, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry, %entry
	%s.0 = phi sbyte* [ getelementptr ([8 x sbyte]* %str, int 0, uint 0), %cond_true ], [ getelementptr ([5 x sbyte]* %str1, int 0, uint 0), %entry ], [ getelementptr ([5 x sbyte]* %str1, int 0, uint 0), %entry ], [ getelementptr ([5 x sbyte]* %str1, int 0, uint 0), %entry ], [ getelementptr ([5 x sbyte]* %str1, int 0, uint 0), %entry ], [ getelementptr ([5 x sbyte]* %str1, int 0, uint 0), %entry ], [ getelementptr ([5 x sbyte]* %str1, int 0, uint 0), %entry ], [ getelementptr ([5 x sbyte]* %str1, int 0, uint 0), %entry ], [ getelementptr ([5 x sbyte]* %str1, int 0, uint 0), %entry ], [ getelementptr ([5 x sbyte]* %str1, int 0, uint 0), %entry ], [ getelementptr ([5 x sbyte]* %str1, int 0, uint 0), %entry ], [ getelementptr ([5 x sbyte]* %str2, int 0, uint 0), %bb2 ]		; <sbyte*> [#uses=13]
	%tmp8 = tail call int (sbyte*, ...)* %printf( sbyte* %s.0 )		; <int> [#uses=0]
	%tmp10 = tail call int (sbyte*, ...)* %printf( sbyte* %s.0 )		; <int> [#uses=0]
	%tmp12 = tail call int (sbyte*, ...)* %printf( sbyte* %s.0 )		; <int> [#uses=0]
	%tmp14 = tail call int (sbyte*, ...)* %printf( sbyte* %s.0 )		; <int> [#uses=0]
	%tmp16 = tail call int (sbyte*, ...)* %printf( sbyte* %s.0 )		; <int> [#uses=0]
	%tmp18 = tail call int (sbyte*, ...)* %printf( sbyte* %s.0 )		; <int> [#uses=0]
	%tmp20 = tail call int (sbyte*, ...)* %printf( sbyte* %s.0 )		; <int> [#uses=0]
	%tmp22 = tail call int (sbyte*, ...)* %printf( sbyte* %s.0 )		; <int> [#uses=0]
	%tmp24 = tail call int (sbyte*, ...)* %printf( sbyte* %s.0 )		; <int> [#uses=0]
	%tmp26 = tail call int (sbyte*, ...)* %printf( sbyte* %s.0 )		; <int> [#uses=0]
	%tmp28 = tail call int (sbyte*, ...)* %printf( sbyte* %s.0 )		; <int> [#uses=0]
	%tmp30 = tail call int (sbyte*, ...)* %printf( sbyte* %s.0 )		; <int> [#uses=0]
	%tmp32 = tail call int (sbyte*, ...)* %printf( sbyte* %s.0 )		; <int> [#uses=0]
	ret void
}

declare int %printf(sbyte*, ...)
