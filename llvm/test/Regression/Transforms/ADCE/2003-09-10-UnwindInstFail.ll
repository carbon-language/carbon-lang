; RUN: llvm-as < %s | opt -adce -disable-output

implementation   ; Functions:

void %test() {
	br bool false, label %then, label %endif

then:
	invoke void null( sbyte* null )
			to label %invoke_cont except label %invoke_catch

invoke_catch:
	unwind

invoke_cont:
	ret void

endif:
	ret void
}
