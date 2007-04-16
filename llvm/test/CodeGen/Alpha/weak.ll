; RUN: llvm-upgrade < %s | llvm-as | llc -march=alpha | grep .weak.*f
; RUN: llvm-upgrade < %s | llvm-as | llc -march=alpha | grep .weak.*h

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
