; This shouldn't crash
; RUN: llvm-as < %s | llc -march=alpha 

; ModuleID = 'simp.bc'
target endian = little
target pointersize = 64
target triple = "alphaev6-unknown-linux-gnu"
deplibs = [ "c", "crtend", "stdc++" ]
	%struct.__va_list_tag = type { sbyte*, int }

implementation   ; Functions:

uint %emit_library_call_value(int %nargs, ...) {
entry:
	%tmp.223 = va_arg %struct.__va_list_tag* null, uint		; <uint> [#uses=0]
	ret uint %tmp.223
}
