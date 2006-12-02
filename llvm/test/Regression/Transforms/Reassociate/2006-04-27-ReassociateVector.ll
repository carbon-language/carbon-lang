; RUN: llvm-upgrade < %s | llvm-as | opt -reassociate -disable-output

void %foo() {
	%tmp162 = sub <4 x float> zeroinitializer, zeroinitializer
	%tmp164 = mul <4 x float> zeroinitializer, %tmp162
	ret void
}
