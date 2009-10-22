-- RUN: %llvmgcc -c %s
with System;
procedure Negative_Field_Offset (N : Integer) is
   type String_Pointer is access String;
   --  Force use of a thin pointer.
   for String_Pointer'Size use System.Word_Size;
   P : String_Pointer;
begin
   P := new String (1 .. N);
end;
