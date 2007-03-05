-- RUN: %llvmgcc -c %s -o /dev/null
-- RUN: %llvmgcc -c %s -O2 -o /dev/null
package body Fat_Fields is
   procedure Proc is
   begin
      if P = null then
         null;
      end if;
   end;
end;
