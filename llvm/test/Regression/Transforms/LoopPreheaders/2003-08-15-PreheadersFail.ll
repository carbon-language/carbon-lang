; RUN: llvm-as < %s | opt -tailduplicate -instcombine -simplifycfg -licm -disable-output

target endian = little
target pointersize = 32
%yy_base = external global [787 x short]		; <[787 x short]*> [#uses=1]
%yy_state_ptr = external global int*		; <int**> [#uses=3]
%yy_state_buf = external global [16386 x int]		; <[16386 x int]*> [#uses=1]
%yy_lp = external global int		; <int*> [#uses=1]

implementation   ; Functions:

int %_yylex() {		; No predecessors!
	br label %loopentry.0

loopentry.0:		; preds = %0, %else.26
	store int* getelementptr ([16386 x int]* %yy_state_buf, long 0, long 0), int** %yy_state_ptr
	%tmp.35 = load int** %yy_state_ptr		; <int*> [#uses=2]
	%inc.0 = getelementptr int* %tmp.35, long 1		; <int*> [#uses=1]
	store int* %inc.0, int** %yy_state_ptr
	%tmp.36 = load int* null		; <int> [#uses=1]
	store int %tmp.36, int* %tmp.35
	br label %loopexit.2

loopexit.2:		; preds = %loopentry.0, %else.26, %loopexit.2
	store sbyte* null, sbyte** null
	%tmp.91 = load int* null		; <int> [#uses=1]
	%tmp.92 = cast int %tmp.91 to long		; <long> [#uses=1]
	%tmp.93 = getelementptr [787 x short]* %yy_base, long 0, long %tmp.92		; <short*> [#uses=1]
	%tmp.94 = load short* %tmp.93		; <short> [#uses=1]
	%tmp.95 = setne short %tmp.94, 4394		; <bool> [#uses=1]
	br bool %tmp.95, label %loopexit.2, label %yy_find_action

yy_find_action:		; preds = %loopexit.2, %else.26
	br label %loopentry.3

loopentry.3:		; preds = %yy_find_action, %shortcirc_done.0, %then.9
	%tmp.105 = load int* %yy_lp		; <int> [#uses=1]
	%tmp.106 = setne int %tmp.105, 0		; <bool> [#uses=1]
	br bool %tmp.106, label %shortcirc_next.0, label %shortcirc_done.0

shortcirc_next.0:		; preds = %loopentry.3
	%tmp.114 = load short* null		; <short> [#uses=1]
	%tmp.115 = cast short %tmp.114 to int		; <int> [#uses=1]
	%tmp.116 = setlt int 0, %tmp.115		; <bool> [#uses=1]
	br label %shortcirc_done.0

shortcirc_done.0:		; preds = %loopentry.3, %shortcirc_next.0
	%shortcirc_val.0 = phi bool [ false, %loopentry.3 ], [ %tmp.116, %shortcirc_next.0 ]		; <bool> [#uses=1]
	br bool %shortcirc_val.0, label %else.0, label %loopentry.3

else.0:		; preds = %shortcirc_done.0
	%tmp.144 = load int* null		; <int> [#uses=1]
	%tmp.145 = and int %tmp.144, 8192		; <int> [#uses=1]
	%tmp.146 = setne int %tmp.145, 0		; <bool> [#uses=1]
	br bool %tmp.146, label %then.9, label %else.26

then.9:		; preds = %else.0
	br label %loopentry.3

else.26:		; preds = %else.1
	switch uint 0, label %loopentry.0 [
		 uint 2, label %yy_find_action
		 uint 0, label %loopexit.2
	]
}
