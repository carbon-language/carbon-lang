-- RUN: %llvmgcc -c %s
with System;
procedure Negative_Field_Offset (N : Integer) is
   type String_Pointer is access String;
   --  Force use of a thin pointer.
   for String_Pointer'Size use System.Word_Size;
   P : String_Pointer;

   procedure Q (P : String_Pointer) is
   begin
      P (1) := 'Z';
   end;
begin
   P := new String (1 .. N);
   Q (P);
end;
