; RUN: llvm-as < %s | opt -raise -disable-output

%T = type { [0 x ubyte] }

void %test(%T* %tmp.22) {
	%tmp.23 = getelementptr %T* %tmp.22, long 0, ubyte 0
	%tmp.24 = cast [0 x ubyte]* %tmp.23 to sbyte**
	%tmp.25 = load sbyte** %tmp.24
	ret void
}
