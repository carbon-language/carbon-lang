; RUN: llvm-as < %s | llc -march=bfin | grep cycles
; XFAIL: *
; ExpandIntegerResult #0: 0x181a60c: i64,ch = ReadCycleCounter 0x1104b08
; Do not know how to expand the result of this operator!

declare i64 @llvm.readcyclecounter()

define i64 @foo() {
	%tmp.1 = call i64 @llvm.readcyclecounter()
	ret i64 %tmp.1
}
