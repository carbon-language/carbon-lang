; RUN: llc -mtriple=arm-eabi %s -o /dev/null
; PR3795

define fastcc void @_D3foo3fooFAriZv({ i32, { double, double }* } %d_arg, i32 %x_arg) {
entry:
	%d = alloca { i32, { double, double }* }		; <{ i32, { double, double }* }*> [#uses=2]
	%x = alloca i32		; <i32*> [#uses=2]
	%b = alloca { double, double }		; <{ double, double }*> [#uses=1]
	store { i32, { double, double }* } %d_arg, { i32, { double, double }* }* %d
	store i32 %x_arg, i32* %x
	%tmp = load i32, i32* %x		; <i32> [#uses=1]
	%tmp1 = getelementptr { i32, { double, double }* }, { i32, { double, double }* }* %d, i32 0, i32 1		; <{ double, double }**> [#uses=1]
	%.ptr = load { double, double }*, { double, double }** %tmp1		; <{ double, double }*> [#uses=1]
	%tmp2 = getelementptr { double, double }, { double, double }* %.ptr, i32 %tmp		; <{ double, double }*> [#uses=1]
	%tmp3 = load { double, double }, { double, double }* %tmp2		; <{ double, double }> [#uses=1]
	store { double, double } %tmp3, { double, double }* %b
	ret void
}
