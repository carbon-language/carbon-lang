; RUN: llvm-upgrade < %s | llvm-as | opt -predsimplify -disable-output

; ModuleID = 'b.bc'
target datalayout = "e-p:32:32"
target endian = little
target pointersize = 32
target triple = "i686-pc-linux-gnu"
deplibs = [ "c", "crtend" ]
	%struct.VDIR_ST = type { int, int, int, %struct.acl*, %struct.pfile*, %struct.vlink*, %struct.vlink*, %struct.vlink*, %struct.VDIR_ST*, %struct.VDIR_ST* }
	%struct.acl = type { int, sbyte*, sbyte*, sbyte*, %struct.restrict*, %struct.acl*, %struct.acl* }
	%struct.avalue = type { sbyte* }
	%struct.pattrib = type { sbyte, sbyte*, sbyte*, %struct.avalue, %struct.pattrib*, %struct.pattrib* }
	%struct.pfile = type { int, int, int, int, int, %struct.vlink*, %struct.vlink*, %struct.pattrib*, %struct.pfile*, %struct.pfile* }
	%struct.restrict = type { %struct.acl*, %struct.acl* }
	%struct.vlink = type { int, sbyte*, sbyte, int, sbyte*, %struct.vlink*, %struct.vlink*, sbyte*, sbyte*, sbyte*, sbyte*, int, int, %struct.acl*, int, int, sbyte*, %struct.pattrib*, %struct.pfile*, %struct.vlink*, %struct.vlink* }

implementation   ; Functions:

void %vl_insert(%struct.vlink* %vl) {
entry:
	%tmp91 = call int %vl_comp( )		; <int> [#uses=2]
	%tmp93 = setgt int %tmp91, 0		; <bool> [#uses=1]
	br bool %tmp93, label %cond_next84, label %bb94

cond_next84:		; preds = %entry
	ret void

bb94:		; preds = %entry
	%tmp96 = seteq int %tmp91, 0		; <bool> [#uses=1]
	br bool %tmp96, label %cond_true97, label %cond_next203

cond_true97:		; preds = %bb94
	br bool false, label %cond_next105, label %cond_true102

cond_true102:		; preds = %cond_true97
	ret void

cond_next105:		; preds = %cond_true97
	%tmp110 = getelementptr %struct.vlink* %vl, int 0, uint 12		; <int*> [#uses=1]
	%tmp111 = load int* %tmp110		; <int> [#uses=1]
	%tmp129 = seteq int %tmp111, 0		; <bool> [#uses=1]
	br bool %tmp129, label %cond_true130, label %cond_next133

cond_true130:		; preds = %cond_next105
	ret void

cond_next133:		; preds = %cond_next105
	ret void

cond_next203:		; preds = %bb94
	ret void
}

declare int %vl_comp()
