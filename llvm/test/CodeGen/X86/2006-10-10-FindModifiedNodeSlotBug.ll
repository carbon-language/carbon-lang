; RUN: llvm-upgrade < %s | llvm-as | llc -march=x86 | grep shrl
; Bug in FindModifiedNodeSlot cause tmp14 load to become a zextload and shr 31
; is then optimized away.

%tree_code_type = external global [0 x uint]

void %copy_if_shared_r() {
	%tmp = load uint* null
	%tmp56 = and uint %tmp, 255
	%tmp8 = getelementptr [0 x uint]* %tree_code_type, int 0, uint %tmp56
	%tmp9 = load uint* %tmp8
	%tmp10 = add uint %tmp9, 4294967295
	%tmp = setgt uint %tmp10, 2
	%tmp14 = load uint* null
	%tmp15 = shr uint %tmp14, ubyte 31
	%tmp15 = cast uint %tmp15 to ubyte
	%tmp16 = setne ubyte %tmp15, 0
	br bool %tmp, label %cond_false25, label %cond_true

cond_true:
	br bool %tmp16, label %cond_true17, label %cond_false

cond_true17:
	ret void

cond_false:
	ret void

cond_false25:
	ret void
}
