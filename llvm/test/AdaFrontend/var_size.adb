-- RUN: %llvmgcc -c %s -o /dev/null
package body Var_Size is
   function A (X : T) return String is
   begin
      return X.A;
   end;
end;
