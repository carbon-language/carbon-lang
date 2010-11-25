-- RUN: %llvmgcc -S %s
procedure Array_Range_Ref is
   A : String (1 .. 3);
   B : String := A (A'RANGE)(1 .. 3);
begin
   null;
end;
