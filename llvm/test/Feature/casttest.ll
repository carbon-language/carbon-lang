; RUN: llvm-as %s -o - | llvm-dis > %t1.ll
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

short "FunFunc"(long %x, sbyte %z)
begin
bb0:            ;;<label>
        %cast110 = cast sbyte %z to short       ;;<short>:(signed operands)
        %cast10 = cast long %x to short         ;;<short>
        %reg109 = add short %cast110, %cast10   ;;<short>
        ret short %reg109                       ;;<void>
end

