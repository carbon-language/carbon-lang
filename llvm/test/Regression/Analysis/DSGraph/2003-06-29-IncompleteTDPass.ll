; RUN: analyze %s -datastructure-gc -dsgc-dspass=td -dsgc-check-flags=Ptr:HR



int %main() {
	call void %A()
	call void %B()
	ret int 0
} 

internal void %A() {
	%V = malloc int
	call void %Callee(int* %V)
	ret void
}

internal void %B() {
	%V = malloc int
	call void %Callee(int* %V)
	ret void
}

internal void %Callee(int* %Ptr) {
	load int* %Ptr
	ret void
}
