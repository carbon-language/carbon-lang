; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep movsd
; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 | grep mov | count 1

define fastcc double @doload64(i64 %x) nounwind  {
	%tmp717 = bitcast i64 %x to double
	ret double %tmp717
}
