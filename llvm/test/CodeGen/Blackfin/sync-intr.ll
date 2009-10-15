; RUN: llc < %s -march=bfin -verify-machineinstrs | FileCheck %s

define void @f() nounwind {
entry:
        ; CHECK-NOT: llvm.bfin
        ; CHECK: csync;
        call void @llvm.bfin.csync()

        ; CHECK-NOT: llvm.bfin
        ; CHECK: ssync;
        call void @llvm.bfin.ssync()
	ret void
}

declare void @llvm.bfin.csync() nounwind
declare void @llvm.bfin.ssync() nounwind
