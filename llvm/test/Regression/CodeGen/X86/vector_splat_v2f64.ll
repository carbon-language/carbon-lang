; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep movlhps

void %test(<2 x double>* %P, <2 x double>* %Q, double %X) {
entry:
	%tmp = insertelement <2 x double> zeroinitializer, double %X, uint 0		; <<2 x double>> [#uses=1]
	%tmp2 = insertelement <2 x double> %tmp, double %X, uint 1		; <<2 x double>> [#uses=1]
	%tmp4 = load <2 x double>* %Q		; <<2 x double>> [#uses=1]
	%tmp6 = mul <2 x double> %tmp4, %tmp2		; <<2 x double>> [#uses=1]
	store <2 x double> %tmp6, <2 x double>* %P
	ret void
}
