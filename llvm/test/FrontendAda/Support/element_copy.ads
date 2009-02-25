package Element_Copy is
   type SmallInt is range 1 .. 4;
   type SmallStr is array (SmallInt range <>) of Character;
   type VariableSizedField (D : SmallInt := 2) is record
      S : SmallStr (1 .. D) := "Hi";
   end record;
   function F return VariableSizedField;
end;
