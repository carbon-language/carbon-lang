; RUN: llc < %s -march=x86-64 | FileCheck %s

; CHECK: stack_end_address
; CHECK: {{movq.+rbp.*$}}
; CHECK: {{movq.+rbp.*$}}
; CHECK: ret

define i64* @stack_end_address() nounwind  {
entry:
	tail call i8* @llvm.frameaddress( i32 0 )
	bitcast i8* %0 to i64*
	ret i64* %1
}

declare i8* @llvm.frameaddress(i32) nounwind readnone 
