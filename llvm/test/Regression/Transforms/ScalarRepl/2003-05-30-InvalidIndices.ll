; RUN: as < %s | opt -scalarrepl

void %main() {
	%E = alloca { { int, float, double, long }, { int, float, double, long } }		; <{ { int, float, double, long }, { int, float, double, long } }*> [#uses=1]
	%tmp.151 = getelementptr { { int, float, double, long }, { int, float, double, long } }* %E, long 0, ubyte 1, ubyte 3		; <long*> [#uses=0]
	ret void
}
