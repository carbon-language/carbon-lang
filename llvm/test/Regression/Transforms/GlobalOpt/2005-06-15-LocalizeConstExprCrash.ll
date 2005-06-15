; RUN: llvm-as < %s | opt -globalopt -disable-output
; PR579

%g_40507551 = internal global short 31038		; <short*> [#uses=1]

void %main() {
	%tmp.4.i.1 = load ubyte* getelementptr (ubyte* cast (short* %g_40507551 to ubyte*), int 1)
	ret void
}
