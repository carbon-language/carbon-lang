; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine | llvm-dis | \
; RUN:   grep {icmp sgt}
; END.

; ModuleID = 'visible.bc'
target datalayout = "e-p:32:32"
target endian = little
target pointersize = 32
target triple = "i686-pc-linux-gnu"
	%struct.point = type { int, int }

implementation   ; Functions:

int %visible(int %direction, long %p1.0, long %p2.0, long %p3.0) {
entry:
	%p1_addr = alloca %struct.point		; <%struct.point*> [#uses=2]
	%p2_addr = alloca %struct.point		; <%struct.point*> [#uses=2]
	%p3_addr = alloca %struct.point		; <%struct.point*> [#uses=2]
	"alloca point" = bitcast int 0 to int		; <int> [#uses=0]
	%tmp = bitcast %struct.point* %p1_addr to { long }*		; <{ long }*> [#uses=1]
	%tmp = getelementptr { long }* %tmp, uint 0, uint 0		; <long*> [#uses=1]
	store long %p1.0, long* %tmp
	%tmp1 = bitcast %struct.point* %p2_addr to { long }*		; <{ long }*> [#uses=1]
	%tmp2 = getelementptr { long }* %tmp1, uint 0, uint 0		; <long*> [#uses=1]
	store long %p2.0, long* %tmp2
	%tmp3 = bitcast %struct.point* %p3_addr to { long }*		; <{ long }*> [#uses=1]
	%tmp4 = getelementptr { long }* %tmp3, uint 0, uint 0		; <long*> [#uses=1]
	store long %p3.0, long* %tmp4
	%tmp = seteq int %direction, 0		; <bool> [#uses=1]
	%tmp5 = bitcast %struct.point* %p1_addr to { long }*		; <{ long }*> [#uses=1]
	%tmp6 = getelementptr { long }* %tmp5, uint 0, uint 0		; <long*> [#uses=1]
	%tmp = load long* %tmp6		; <long> [#uses=1]
	%tmp7 = bitcast %struct.point* %p2_addr to { long }*		; <{ long }*> [#uses=1]
	%tmp8 = getelementptr { long }* %tmp7, uint 0, uint 0		; <long*> [#uses=1]
	%tmp9 = load long* %tmp8		; <long> [#uses=1]
	%tmp10 = bitcast %struct.point* %p3_addr to { long }*		; <{ long }*> [#uses=1]
	%tmp11 = getelementptr { long }* %tmp10, uint 0, uint 0		; <long*> [#uses=1]
	%tmp12 = load long* %tmp11		; <long> [#uses=1]
	%tmp13 = call int %determinant( long %tmp, long %tmp9, long %tmp12 )		; <int> [#uses=2]
	br bool %tmp, label %cond_true, label %cond_false

cond_true:		; preds = %entry
	%tmp14 = setlt int %tmp13, 0		; <bool> [#uses=1]
	%tmp14 = zext bool %tmp14 to int		; <int> [#uses=1]
	br label %return

cond_false:		; preds = %entry
	%tmp26 = setgt int %tmp13, 0		; <bool> [#uses=1]
	%tmp26 = zext bool %tmp26 to int		; <int> [#uses=1]
	br label %return

return:		; preds = %cond_false, %cond_true
	%retval.0 = phi int [ %tmp14, %cond_true ], [ %tmp26, %cond_false ]		; <int> [#uses=1]
	ret int %retval.0
}

declare int %determinant(long, long, long)
