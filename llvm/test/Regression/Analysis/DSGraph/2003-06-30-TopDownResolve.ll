; RUN: analyze %s -datastructure-gc -dsgc-dspass=td -dsgc-abort-if-incomplete=X

%G = internal global int 5

implementation

internal void %leaf(int *%X) {
	store int 0, int* %X
	ret void
}

internal void %intermediate(void(int*)* %Fn, int* %Ptr) {
	call void %Fn(int* %Ptr)
	ret void
}

int %main() {
	call void %intermediate(void(int*)* %leaf, int* %G)
	ret int 0
}
