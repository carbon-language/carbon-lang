implementation

short "FunFunc"(long %x, sbyte %z)
begin
bb0:            ;;<label>
        %cast110 = cast sbyte %z to short       ;;<short>:(signed operands)
        %cast10 = cast long %x to short         ;;<short>
        %reg109 = add short %cast110, %cast10   ;;<short>
        ret short %reg109                       ;;<void>
end

