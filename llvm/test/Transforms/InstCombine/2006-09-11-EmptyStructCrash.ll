; PR905
; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine -disable-output
; END.

	%RPYTHON_EXCEPTION = type { %RPYTHON_EXCEPTION_VTABLE* }
	%RPYTHON_EXCEPTION_VTABLE = type { %RPYTHON_EXCEPTION_VTABLE*, int, int, %RPyOpaque_RuntimeTypeInfo*, %arraytype_Char*, %functiontype_12* }
	%RPyOpaque_RuntimeTypeInfo = type opaque*
	%arraytype_Char = type { int, [0 x sbyte] }
	%fixarray_array1019 = type [1019 x sbyte*]
	%functiontype_12 = type %RPYTHON_EXCEPTION* ()
	%functiontype_14 = type void (%structtype_pypy.rpython.memory.gc.MarkSweepGC*)
	%structtype_AddressLinkedListChunk = type { %structtype_AddressLinkedListChunk*, int, %fixarray_array1019 }
	%structtype_exceptions.Exception = type { %RPYTHON_EXCEPTION }
	%structtype_gc_pool = type {  }
	%structtype_gc_pool_node = type { %structtype_header*, %structtype_gc_pool_node* }
	%structtype_header = type { int, %structtype_header* }
	%structtype_pypy.rpython.memory.gc.MarkSweepGC = type { %structtype_exceptions.Exception, int, int, bool, %structtype_gc_pool*, int, %structtype_header*, %structtype_header*, %structtype_gc_pool_node*, double, double }

implementation   ; Functions:

fastcc void %pypy_MarkSweepGC.collect() {
block0:
	%v1221 = load %structtype_AddressLinkedListChunk** null		; <%structtype_AddressLinkedListChunk*> [#uses=1]
	%v1222 = setne %structtype_AddressLinkedListChunk* %v1221, null		; <bool> [#uses=1]
	br bool %v1222, label %block79, label %block4

block4:		; preds = %block0
	ret void

block22:		; preds = %block79
	ret void

block67:		; preds = %block79
	%v1459 = load %structtype_gc_pool** null		; <%structtype_gc_pool*> [#uses=1]
	%v1460 = cast %structtype_gc_pool* %v1459 to sbyte*		; <sbyte*> [#uses=1]
	%tmp_873 = cast sbyte* %v1460 to int		; <int> [#uses=1]
	%tmp_874 = sub int %tmp_873, 0		; <int> [#uses=1]
	%v1461 = cast int %tmp_874 to sbyte*		; <sbyte*> [#uses=1]
	%v1462 = cast sbyte* %v1461 to %structtype_header*		; <%structtype_header*> [#uses=1]
	%tmp_876 = getelementptr %structtype_header* %v1462, int 0, uint 0		; <int*> [#uses=1]
	store int 0, int* %tmp_876
	ret void

block79:		; preds = %block0
	%v1291 = load %structtype_gc_pool** null		; <%structtype_gc_pool*> [#uses=1]
	%v1292 = setne %structtype_gc_pool* %v1291, null		; <bool> [#uses=1]
	br bool %v1292, label %block67, label %block22
}
