-- RUN: %llvmgcc -S %s -I%p/Support
package body Var_Offset is
   function F (X : T) return Character is
   begin
      return X.Bad_Field;
   end;
end;
