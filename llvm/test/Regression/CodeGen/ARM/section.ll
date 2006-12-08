; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm | grep "__DTOR_END__:" &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm | grep ".section .dtors,\"aw\",@progbits"

%__DTOR_END__ = internal global [1 x int] zeroinitializer, section ".dtors"
