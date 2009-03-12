-- RUN: %llvmgcc -S -O2 %s -I%p/Support -o - | grep 6899714
package body Element_Copy is
   function F return VariableSizedField is
      X : VariableSizedField;
   begin
      return X;
   end;
end;
