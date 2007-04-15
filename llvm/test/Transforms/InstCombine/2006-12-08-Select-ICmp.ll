; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine | llvm-dis | grep select
; END.

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
	%tmp = bitcast %struct.point* %p1_addr to { long }*		; <{ long }*> [#uses=1]
	%tmp = getelementptr { long }* %tmp, int 0, uint 0		; <long*> [#uses=1]
	store long %p1.0, long* %tmp
	%tmp1 = bitcast %struct.point* %p2_addr to { long }*		; <{ long }*> [#uses=1]
	%tmp2 = getelementptr { long }* %tmp1, int 0, uint 0		; <long*> [#uses=1]
	store long %p2.0, long* %tmp2
	%tmp3 = bitcast %struct.point* %p3_addr to { long }*		; <{ long }*> [#uses=1]
	%tmp4 = getelementptr { long }* %tmp3, int 0, uint 0		; <long*> [#uses=1]
	store long %p3.0, long* %tmp4
	%tmp = seteq int %direction, 0		; <bool> [#uses=1]
	%tmp5 = bitcast %struct.point* %p1_addr to { long }*		; <{ long }*> [#uses=1]
	%tmp6 = getelementptr { long }* %tmp5, int 0, uint 0		; <long*> [#uses=1]
	%tmp = load long* %tmp6		; <long> [#uses=1]
	%tmp7 = bitcast %struct.point* %p2_addr to { long }*		; <{ long }*> [#uses=1]
	%tmp8 = getelementptr { long }* %tmp7, int 0, uint 0		; <long*> [#uses=1]
	%tmp9 = load long* %tmp8		; <long> [#uses=1]
	%tmp10 = bitcast %struct.point* %p3_addr to { long }*		; <{ long }*> [#uses=1]
	%tmp11 = getelementptr { long }* %tmp10, int 0, uint 0		; <long*> [#uses=1]
	%tmp12 = load long* %tmp11		; <long> [#uses=1]
	%tmp13 = call int %determinant( long %tmp, long %tmp9, long %tmp12 )		; <int> [#uses=2]
	%tmp14 = setlt int %tmp13, 0		; <bool> [#uses=1]
	%tmp26 = setgt int %tmp13, 0		; <bool> [#uses=1]
	%retval.0.in = select bool %tmp, bool %tmp14, bool %tmp26		; <bool> [#uses=1]
	%retval.0 = zext bool %retval.0.in to int		; <int> [#uses=1]
	ret int %retval.0
}

declare int %determinant(long, long, long)
