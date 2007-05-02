-- RUN: %llvmgcc -c %s -o /dev/null
package body Unc_Constructor is
   procedure P (X : A) is
   begin
      if X = A0 then
         null;
      end if;
   end;
end;
