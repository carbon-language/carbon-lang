; RUN: analyze %s -datastructure-gc  -dsgc-check-flags=G:GIM -dsgc-dspass=bu


%S = type { double, int }

%G = external global %S

void %main() {
	%b = getelementptr %S* %G, long 0, ubyte 0
	store double 0.1, double* %b
	ret void
}
