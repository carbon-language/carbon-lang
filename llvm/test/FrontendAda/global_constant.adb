-- RUN: %llvmgcc -c %s -o /dev/null
package body Global_Constant is
begin
   raise An_Error;
end;
