; RUN: llvm-as %s -o - | llvm-dis > %t1.ll
; RUN: llvm-as %t1.ll -o - | llvm-dis > %t2.ll
; RUN: diff %t1.ll %t2.ll

implementation

short "FunFunc"(long %x, sbyte %z)
begin
bb0:            ;;<label>
        %cast110 = cast sbyte %z to short       ;;<short>:(signed operands)
        %cast10 = cast long %x to short         ;;<short>
        %reg109 = add short %cast110, %cast10   ;;<short>
        ret short %reg109                       ;;<void>
end

