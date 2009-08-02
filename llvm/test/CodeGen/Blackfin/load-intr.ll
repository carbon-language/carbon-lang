; RUN: llvm-as < %s | llc -march=bfin -verify-machineinstrs | FileCheck %s
; XFAIL: *
; Assertion failed: (isUsed(Reg) && "Using an undefined register!"),
; function forward, file lib/CodeGen/RegisterScavenging.cpp, line 221.

define i16 @f(i32* %p) nounwind {
entry:
        ; CHECK: disalignexcpt || r0 = [i0];
        %b = call i32 @llvm.bfin.loadbytes(i32* %p)
        ; CHECK: r0.l = ones r0;
        %c = call i16 @llvm.bfin.ones(i32 %b)
	ret i16 %c
}

declare void @llvm.bfin.ones() nounwind
declare void @llvm.bfin.loadbytes() nounwind
