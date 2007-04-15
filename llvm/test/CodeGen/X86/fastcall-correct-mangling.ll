; RUN: llvm-upgrade < %s | llvm-as | llc -march=x86 -mtriple=mingw32 | \
; RUN:   grep {@12}

; Check that a fastcall function gets correct mangling

x86_fastcallcc void %func(long %X, ubyte %Y, ubyte %G, ushort %Z) {
	ret void
}
