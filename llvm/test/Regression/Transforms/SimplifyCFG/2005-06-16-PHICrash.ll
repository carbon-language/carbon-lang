; RUN: llvm-as < %s | opt -simplifycfg -disable-output
; PR584

%g_38098584 = external global uint		; <uint*> [#uses=1]
%g_60187400 = external global uint		; <uint*> [#uses=1]
%g_59182229 = external global uint		; <uint*> [#uses=2]

implementation   ; Functions:

int %_Z13func_26556482h(ubyte %l_88173906) {
entry:
	%tmp.1 = cast ubyte %l_88173906 to sbyte		; <sbyte> [#uses=2]
	%tmp.3 = seteq ubyte %l_88173906, 0		; <bool> [#uses=1]
	br bool %tmp.3, label %else.0, label %then.0

then.0:		; preds = %entry
	%tmp.5 = seteq ubyte %l_88173906, 0		; <bool> [#uses=1]
	br bool %tmp.5, label %else.1, label %then.1

then.1:		; preds = %then.0
	br label %return

else.1:		; preds = %then.0
	br label %loopentry.0

loopentry.0:		; preds = %no_exit.0, %else.1
	%i.0.1 = phi int [ 0, %else.1 ], [ %inc.0, %no_exit.0 ]		; <int> [#uses=2]
	%tmp.9 = setgt int %i.0.1, 99		; <bool> [#uses=1]
	br bool %tmp.9, label %endif.0, label %no_exit.0

no_exit.0:		; preds = %loopentry.0
	%inc.0 = add int %i.0.1, 1		; <int> [#uses=1]
	br label %loopentry.0

else.0:		; preds = %entry
	%tmp.12 = cast sbyte %tmp.1 to int		; <int> [#uses=1]
	br label %return

endif.0:		; preds = %loopentry.0
	%tmp.14 = cast sbyte %tmp.1 to int		; <int> [#uses=1]
	%tmp.16 = cast ubyte %l_88173906 to int		; <int> [#uses=1]
	%tmp.17 = setgt int %tmp.14, %tmp.16		; <bool> [#uses=1]
	%tmp.19 = load uint* %g_59182229		; <uint> [#uses=2]
	br bool %tmp.17, label %cond_true, label %cond_false

cond_true:		; preds = %endif.0
	%tmp.20 = setne uint %tmp.19, 1		; <bool> [#uses=1]
	br label %cond_continue

cond_false:		; preds = %endif.0
	%tmp.22 = setne uint %tmp.19, 0		; <bool> [#uses=1]
	br label %cond_continue

cond_continue:		; preds = %cond_false, %cond_true
	%mem_tmp.0 = phi bool [ %tmp.20, %cond_true ], [ %tmp.22, %cond_false ]		; <bool> [#uses=1]
	br bool %mem_tmp.0, label %then.2, label %else.2

then.2:		; preds = %cond_continue
	%tmp.25 = cast ubyte %l_88173906 to int		; <int> [#uses=1]
	br label %return

else.2:		; preds = %cond_continue
	br label %loopentry.1

loopentry.1:		; preds = %endif.3, %else.2
	%i.1.1 = phi int [ 0, %else.2 ], [ %inc.3, %endif.3 ]		; <int> [#uses=2]
	%i.3.2 = phi int [ undef, %else.2 ], [ %i.3.0, %endif.3 ]		; <int> [#uses=2]
	%l_88173906_addr.1 = phi ubyte [ %l_88173906, %else.2 ], [ %l_88173906_addr.0, %endif.3 ]		; <ubyte> [#uses=3]
	%tmp.29 = setgt int %i.1.1, 99		; <bool> [#uses=1]
	br bool %tmp.29, label %endif.2, label %no_exit.1

no_exit.1:		; preds = %loopentry.1
	%tmp.30 = load uint* %g_38098584		; <uint> [#uses=1]
	%tmp.31 = seteq uint %tmp.30, 0		; <bool> [#uses=1]
	br bool %tmp.31, label %else.3, label %then.3

then.3:		; preds = %no_exit.1
	br label %endif.3

else.3:		; preds = %no_exit.1
	br bool false, label %else.4, label %then.4

then.4:		; preds = %else.3
	br label %endif.3

else.4:		; preds = %else.3
	br bool false, label %else.5, label %then.5

then.5:		; preds = %else.4
	store uint 3290648471, uint* %g_59182229
	br label %return

else.5:		; preds = %else.4
	br label %loopentry.3

loopentry.3:		; preds = %then.7, %else.5
	%i.3.3 = phi int [ 0, %else.5 ], [ %inc.2, %then.7 ]		; <int> [#uses=3]
	%tmp.55 = setgt int %i.3.3, 99		; <bool> [#uses=1]
	br bool %tmp.55, label %endif.3, label %no_exit.3

no_exit.3:		; preds = %loopentry.3
	%tmp.57 = seteq ubyte %l_88173906_addr.1, 0		; <bool> [#uses=1]
	br bool %tmp.57, label %else.7, label %then.7

then.7:		; preds = %no_exit.3
	store uint 16239, uint* %g_60187400
	%inc.2 = add int %i.3.3, 1		; <int> [#uses=1]
	br label %loopentry.3

else.7:		; preds = %no_exit.3
	br label %return

endif.3:		; preds = %loopentry.3, %then.4, %then.3
	%i.3.0 = phi int [ %i.3.2, %then.3 ], [ %i.3.2, %then.4 ], [ %i.3.3, %loopentry.3 ]		; <int> [#uses=1]
	%l_88173906_addr.0 = phi ubyte [ 100, %then.3 ], [ %l_88173906_addr.1, %then.4 ], [ %l_88173906_addr.1, %loopentry.3 ]		; <ubyte> [#uses=1]
	%inc.3 = add int %i.1.1, 1		; <int> [#uses=1]
	br label %loopentry.1

endif.2:		; preds = %loopentry.1
	br label %return

return:		; preds = %endif.2, %else.7, %then.5, %then.2, %else.0, %then.1
	%result.0 = phi int [ 1624650671, %then.1 ], [ %tmp.25, %then.2 ], [ 3379, %then.5 ], [ 52410, %else.7 ], [ -1526438411, %endif.2 ], [ %tmp.12, %else.0 ]		; <int> [#uses=1]
	ret int %result.0
}
