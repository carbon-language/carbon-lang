-- RUN: %llvmgcc -c %s -o /dev/null
with System.Machine_Code;
procedure Asm is
begin
   System.Machine_Code.Asm ("");
end;
