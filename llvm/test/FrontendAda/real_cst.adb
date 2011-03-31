-- RUN: %llvmgcc -S -O2 -gnatn %s -I%p/Support
package body Real_Cst is
   Cst : constant Float := 0.0;
   procedure Write (Stream : access Ada.Streams.Root_Stream_Type'Class) is
   begin
      Float'Write (Stream, Cst);
   end;
end;
