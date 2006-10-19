; The optimizer should be able to remove cast operation here.
; RUN: llvm-as %s -o - | opt -instcombine | llvm-dis | not grep 'cast.*int'

bool %eq_signed_to_small_unsigned(sbyte %SB) {
   %Y = cast sbyte %SB to uint         ; <uint> [#uses=1]
   %C = seteq uint %Y, 17              ; <bool> [#uses=1]
   ret bool %C
 }

