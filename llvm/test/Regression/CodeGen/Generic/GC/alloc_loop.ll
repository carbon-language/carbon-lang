implementation

declare sbyte* %llvm_gc_allocate(uint)
declare void %llvm_gc_initialize(uint)

declare void %llvm.gcroot(sbyte**, sbyte*)
declare void %llvm.gcwrite(sbyte*, sbyte**)

int %main() {
entry:
	%A = alloca sbyte*
	%B = alloca sbyte**

	call void %llvm_gc_initialize(uint 1048576)  ; Start with 1MB heap

        ;; void *A;
	call void %llvm.gcroot(sbyte** %A, sbyte* null)

        ;; A = gcalloc(10);
	%Aptr = call sbyte* %llvm_gc_allocate(uint 10)
	store sbyte* %Aptr, sbyte** %A

        ;; void **B;
	%tmp.1 = cast sbyte*** %B to sbyte **
	call void %llvm.gcroot(sbyte** %tmp.1, sbyte* null)

	;; B = gcalloc(4);
	%B = call sbyte* %llvm_gc_allocate(uint 8)
	%tmp.2 = cast sbyte* %B to sbyte**
	store sbyte** %tmp.2, sbyte*** %B

	;; *B = A;
	%B.1 = load sbyte*** %B
	%A.1 = load sbyte** %A
	call void %llvm.gcwrite(sbyte* %A.1, sbyte** %B.1)
	
	br label %AllocLoop

AllocLoop:
	%i = phi uint [ 0, %entry ], [ %indvar.next, %AllocLoop ]
        ;; Allocated mem: allocated memory is immediately dead.
	call sbyte* %llvm_gc_allocate(uint 100)
	
	%indvar.next = add uint %i, 1
	%exitcond = seteq uint %indvar.next, 10000000
	br bool %exitcond, label %Exit, label %AllocLoop

Exit:
	ret int 0
}

declare void %__main()
