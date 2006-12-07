; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm | grep "__DTOR_END__:"

%__DTOR_END__ = internal global [1 x int] zeroinitializer, section ".dtors"
