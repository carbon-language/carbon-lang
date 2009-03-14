; RUN: llvm-as < %s | llc -march=x86-64 -relocation-model=static | grep {a(%rip)}

@a = internal global double 3.4
define double @foo() nounwind {
  %a = load double* @a
  ret double %a
}
