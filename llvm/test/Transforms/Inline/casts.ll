; RUN: llvm-upgrade < %s | llvm-as | opt -inline | llvm-dis | grep {ret i32 1}
; ModuleID = 'short.opt.bc'

implementation   ; Functions:

int %testBool(bool %X) {
	%tmp = zext bool %X to int		; <int> [#uses=1]
	ret int %tmp
}

int %testByte(sbyte %X) {
	%tmp = setne sbyte %X, 0		; <bool> [#uses=1]
	%tmp.i = zext bool %tmp to int		; <int> [#uses=1]
	ret int %tmp.i
}

int %main() {
        %rslt = call int %testByte( sbyte 123)
	ret int %rslt
}
