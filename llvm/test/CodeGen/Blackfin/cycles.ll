; RUN: llvm-as < %s | llc -march=bfin | FileCheck %s

declare i64 @llvm.readcyclecounter()

; CHECK: cycles
; CHECK: cycles2
define i64 @cyc64() {
	%tmp.1 = call i64 @llvm.readcyclecounter()
	ret i64 %tmp.1
}

; CHECK: cycles
define i32@cyc32() {
	%tmp.1 = call i64 @llvm.readcyclecounter()
        %s = trunc i64 %tmp.1 to i32
	ret i32 %s
}
