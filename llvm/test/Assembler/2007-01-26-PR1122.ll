; PR1122
; Make sure llvm-upgrade can upgrade this.
; RUN: llvm-upgrade < %s > /dev/null 
%arraytype_Char_1 = type { int, [0 x sbyte] }
%structtype_rpy_string = type { int, %arraytype_Char_1 }

%RPyString = type %structtype_rpy_string      ;**doesn't work
;%RPyString = type { int, %arraytype_Char_1 } ;**works 

sbyte* %RPyString_AsString(%RPyString* %structstring) {
 %source1ptr = getelementptr %RPyString* %structstring, int 0, uint 1, uint 1
 %source1 = cast [0 x sbyte]* %source1ptr to sbyte*
 ret sbyte* %source1
}
