

int %foo() {
  ret int 0
}

int %main() {
  ; cast bool to ...
  cast bool true to bool
  cast bool true to int
  cast bool true to long
  cast bool true to ulong
  cast bool true to float
  cast bool true to double

  ; cast sbyte to ...
  cast sbyte 0 to sbyte
  cast sbyte 4 to short
  cast sbyte 4 to long
  cast sbyte 4 to ulong
  cast sbyte 4 to double

  ; cast ubyte to ...
  cast ubyte 0 to float
  cast ubyte 0 to double

  ; cast short to ...
  cast short 0 to short
  cast short 0 to long
  cast short 0 to ulong
  cast short 0 to double

  ; cast ushort to ...
  cast ushort 0 to float
  cast ushort 0 to double

  ; cast int to ...
  cast int 6 to bool
  cast int 6 to short
  cast int 0 to int
  cast int 0 to long
  cast int 0 to ulong
  cast int 0 to float
  cast int 0 to double

  ; cast uint to ...
  cast uint 0 to long
  cast uint 0 to ulong
  cast uint 0 to float
  cast uint 0 to double

  ; cast long to ...
  cast long 0 to bool
  cast long 0 to sbyte
  cast long 0 to ubyte
  cast long 0 to short
  cast long 0 to ushort
  cast long 0 to int
  cast long 0 to uint
  cast long 0 to long
  cast long 0 to ulong
  cast long 0 to float
  cast long 0 to double

  cast ulong 0 to bool
  
  ; cast float to ...
  ;cast float 0.0 to bool
  cast float 0.0 to float
  cast float 0.0 to double

  ; cast double to ...
  ;cast double 0.0 to bool
  cast double 0.0 to sbyte
  cast double 0.0 to ubyte
  cast double 0.0 to short
  cast double 0.0 to ushort
  cast double 0.0 to int
  cast double 0.0 to uint
  cast double 0.0 to long
  ;cast double 0.0 to ulong
  cast double 0.0 to float
  cast double 0.0 to double

  ret int 0
}
