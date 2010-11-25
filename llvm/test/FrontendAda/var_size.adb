-- RUN: %llvmgcc -S %s -I%p/Support
package body Var_Size is
   function A (X : T) return String is
   begin
      return X.A;
   end;
end;
