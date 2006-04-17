; RUN: llvm-as < %s | llc -march=x86 -mcpu=yonah

<2 x double> %test() {
	ret <2 x double> <double 0.0, double 0.0>
}
