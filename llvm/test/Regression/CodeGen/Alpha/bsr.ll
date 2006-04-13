; Make sure this testcase codegens the bsr instruction
; RUN: llvm-as < %s | llc -march=alpha | grep 'bsr'

implementation

internal long %abc(int %x) {
	%tmp.2 = add int %x, -1		; <int> [#uses=1]
	%tmp.0 = call long %abc( int %tmp.2 )		; <long> [#uses=1]
	%tmp.5 = add int %x, -2		; <int> [#uses=1]
	%tmp.3 = call long %abc( int %tmp.5 )		; <long> [#uses=1]
	%tmp.6 = add long %tmp.0, %tmp.3		; <long> [#uses=1]
	ret long %tmp.6
}
