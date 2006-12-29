; RUN: llvm-upgrade < %s | llvm-as | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

void "NewCasts" (short %x) {
  %a = zext short %x to int
  %b = sext short %x to uint
  %c = trunc short %x to ubyte
  %d = uitofp short %x to float
  %e = sitofp short %x to double
  %f = fptoui float %d to short
  %g = fptosi double %e to short
  %i = fpext float %d to double
  %j = fptrunc double %i to float
  %k = bitcast int %a to float
  %l = inttoptr short %x to int*
  %m = ptrtoint int* %l to long
  ret void
}


ushort "ZExtConst" () {
  ret ushort trunc ( uint zext ( short 42 to uint) to ushort )
}

short "SExtConst" () {
  ret short trunc (int sext (ushort 42 to int) to short )
}
