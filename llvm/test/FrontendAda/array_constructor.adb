-- RUN: %llvmgcc -c %s -o /dev/null
procedure Array_Constructor is
   A : array (Integer range <>) of Boolean := (True, False);
begin
   null;
end;
