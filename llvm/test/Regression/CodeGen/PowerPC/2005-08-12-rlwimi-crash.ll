; this should not crash the ppc backend

; RUN: llvm-as < %s | llc -march=ppc32

uint %test( int %j.0.0.i) {
  %tmp.85.i = and int %j.0.0.i, 7
  %tmp.161278.i = cast int %tmp.85.i to uint
  %tmp.5.i77.i = shr uint %tmp.161278.i, ubyte 3
  ret uint %tmp.5.i77.i
}


