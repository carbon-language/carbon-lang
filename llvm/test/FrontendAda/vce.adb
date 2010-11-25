-- RUN: %llvmgcc -S %s
procedure VCE is
  S : String (1 .. 2);
  B : Character := 'B';
begin
  S := 'A' & B;
end;
