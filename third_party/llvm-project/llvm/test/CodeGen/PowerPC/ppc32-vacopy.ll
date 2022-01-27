; RUN: llc -verify-machineinstrs -mtriple="powerpc-unknown-linux-gnu" -mcpu=ppc64 < %s | FileCheck %s
; PR15286

%va_list = type {i8, i8, i16, i8*, i8*}
declare void @llvm.va_copy(i8*, i8*)

define void @test_vacopy() nounwind {
entry:
	%0 = alloca %va_list
	%1 = alloca %va_list
	%2 = bitcast %va_list* %0 to i8*
	%3 = bitcast %va_list* %1 to i8*

	call void @llvm.va_copy(i8* %3, i8* %2)

	ret void
}
; CHECK: test_vacopy:
; CHECK-DAG: lwz [[REG1:[0-9]+]], {{.*}}
; CHECK-DAG: lwz [[REG2:[0-9]+]], {{.*}}
; CHECK-DAG: lwz [[REG3:[0-9]+]], {{.*}}
; CHECK-DAG: stw [[REG1]], {{.*}}
; CHECK-DAG: stw [[REG2]], {{.*}}
; CHECK-DAG: stw [[REG3]], {{.*}}
; CHECK: blr
