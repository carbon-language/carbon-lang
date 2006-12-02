; RUN: llvm-upgrade < %s | llvm-as | llc -mtriple=i686-apple-darwin -relocation-model=pic | not grep lea
;
; Make sure the PIC label flags2-"L1$pb" is not moved up to the preheader.

%flags2 = internal global [8193 x sbyte] zeroinitializer, align 32

void %test(int %k, int %i) {
entry:
	%i = bitcast int %i to uint
	%k_addr.012 = shl int %i, ubyte 1
	%tmp14 = setgt int %k_addr.012, 8192
	br bool %tmp14, label %return, label %bb

bb:
	%indvar = phi uint [ 0, %entry ], [ %indvar.next, %bb ]
	%tmp. = shl uint %i, ubyte 1
	%tmp.15 = mul uint %indvar, %i
	%tmp.16 = add uint %tmp.15, %tmp.
	%k_addr.0.0 = bitcast uint %tmp.16 to int
	%tmp = getelementptr [8193 x sbyte]* %flags2, int 0, uint %tmp.16
	store sbyte 0, sbyte* %tmp
	%k_addr.0 = add int %k_addr.0.0, %i
	%tmp = setgt int %k_addr.0, 8192
	%indvar.next = add uint %indvar, 1
	br bool %tmp, label %return, label %bb

return:
	ret void
}
