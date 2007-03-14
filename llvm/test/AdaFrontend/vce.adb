-- RUN: %llvmgcc -c %s -o /dev/null
procedure VCE is
  S : String (1 .. 2);
  B : Character := 'B';
begin
  S := 'A' & B;
end;
