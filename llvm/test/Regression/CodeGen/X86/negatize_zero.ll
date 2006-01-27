; RUN: llvm-as < %s | llc -march=x86 -mattr=-sse2 | grep fchs


double %T() {
	ret double -1.0   ;; codegen as fld1/fchs, not as a load from cst pool
}
