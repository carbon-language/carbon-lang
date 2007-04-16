; RUN: llvm-upgrade < %s | llvm-as | llc -mtriple=arm-linux | \
; RUN:   grep {__DTOR_END__:}
; RUN: llvm-upgrade < %s | llvm-as | llc -mtriple=arm-linux | \
; RUN:   grep {.section .dtors,"aw",.progbits}

%__DTOR_END__ = internal global [1 x int] zeroinitializer, section ".dtors"
