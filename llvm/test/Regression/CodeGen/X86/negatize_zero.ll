; RUN: llvm-as < %s | llc -march=x86 | grep fchs


double %T() {
	ret double -1.0   ;; codegen as fld1/fchs, not as a load from cst pool
}
