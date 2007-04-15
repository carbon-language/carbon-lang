; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm | grep .weak.*f
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm | grep .weak.*h

implementation   ; Functions:

weak uint %f() {
entry:
	unreachable
}

void %g() {
entry:
        tail call void %h( )
        ret void
}

declare extern_weak void %h()
