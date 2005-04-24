; This test case tests the InstructionCombining optimization that
; reduces things like:
;   %Y = cast sbyte %X to uint
;   %C = setlt uint %Y, 1024
; to
;   %C = bool true
; It includes test cases for different constant values, signedness of the
; cast operands, and types of setCC operators. In all cases, the cast should
; be eliminated. In many cases the setCC is also eliminated based on the
; constant value and the range of the casted value.
;
; RUN: llvm-as %s -o - | opt -instcombine | llvm-dis | not grep 'cast.*int'

implementation   ; Functions:

bool %lt_signed_to_large_unsigned(sbyte %SB) {
  %Y = cast sbyte %SB to uint		; <uint> [#uses=1]
  %C = setlt uint %Y, 1024		; <bool> [#uses=1]
  ret bool %C
}

bool %lt_signed_to_large_signed(sbyte %SB) {
  %Y = cast sbyte %SB to int
  %C = setlt int %Y, 1024
  ret bool %C
}

bool %lt_signed_to_large_negative(sbyte %SB) {
  %Y = cast sbyte %SB to int
  %C = setlt int %Y, -1024
  ret bool %C
}

bool %lt_signed_to_small_unsigned(sbyte %SB) {
  %Y = cast sbyte %SB to uint		; <uint> [#uses=1]
  %C = setlt uint %Y, 17		; <bool> [#uses=1]
  ret bool %C
}

bool %lt_signed_to_small_signed(sbyte %SB) {
  %Y = cast sbyte %SB to int
  %C = setlt int %Y, 17
  ret bool %C
}

bool %lt_signed_to_small_negative(sbyte %SB) {
  %Y = cast sbyte %SB to int
  %C = setlt int %Y, -17
  ret bool %C
}

bool %lt_unsigned_to_large_unsigned(ubyte %SB) {
  %Y = cast ubyte %SB to uint		; <uint> [#uses=1]
  %C = setlt uint %Y, 1024		; <bool> [#uses=1]
  ret bool %C
}

bool %lt_unsigned_to_large_signed(ubyte %SB) {
  %Y = cast ubyte %SB to int
  %C = setlt int %Y, 1024
  ret bool %C
}

bool %lt_unsigned_to_large_negative(ubyte %SB) {
  %Y = cast ubyte %SB to int
  %C = setlt int %Y, -1024
  ret bool %C
}

bool %lt_unsigned_to_small_unsigned(ubyte %SB) {
  %Y = cast ubyte %SB to uint		; <uint> [#uses=1]
  %C = setlt uint %Y, 17		; <bool> [#uses=1]
  ret bool %C
}

bool %lt_unsigned_to_small_signed(ubyte %SB) {
  %Y = cast ubyte %SB to int
  %C = setlt int %Y, 17
  ret bool %C
}

bool %lt_unsigned_to_small_negative(ubyte %SB) {
  %Y = cast ubyte %SB to int
  %C = setlt int %Y, -17
  ret bool %C
}

bool %gt_signed_to_large_unsigned(sbyte %SB) {
  %Y = cast sbyte %SB to uint		; <uint> [#uses=1]
  %C = setgt uint %Y, 1024		; <bool> [#uses=1]
  ret bool %C
}

bool %gt_signed_to_large_signed(sbyte %SB) {
  %Y = cast sbyte %SB to int
  %C = setgt int %Y, 1024
  ret bool %C
}

bool %gt_signed_to_large_negative(sbyte %SB) {
  %Y = cast sbyte %SB to int
  %C = setgt int %Y, -1024
  ret bool %C
}

bool %gt_signed_to_small_unsigned(sbyte %SB) {
  %Y = cast sbyte %SB to uint		; <uint> [#uses=1]
  %C = setgt uint %Y, 17		; <bool> [#uses=1]
  ret bool %C
}

bool %gt_signed_to_small_signed(sbyte %SB) {
  %Y = cast sbyte %SB to int
  %C = setgt int %Y, 17
  ret bool %C
}

bool %gt_signed_to_small_negative(sbyte %SB) {
  %Y = cast sbyte %SB to int
  %C = setgt int %Y, -17
  ret bool %C
}

bool %gt_unsigned_to_large_unsigned(ubyte %SB) {
  %Y = cast ubyte %SB to uint		; <uint> [#uses=1]
  %C = setgt uint %Y, 1024		; <bool> [#uses=1]
  ret bool %C
}

bool %gt_unsigned_to_large_signed(ubyte %SB) {
  %Y = cast ubyte %SB to int
  %C = setgt int %Y, 1024
  ret bool %C
}

bool %gt_unsigned_to_large_negative(ubyte %SB) {
  %Y = cast ubyte %SB to int
  %C = setgt int %Y, -1024
  ret bool %C
}

bool %gt_unsigned_to_small_unsigned(ubyte %SB) {
  %Y = cast ubyte %SB to uint		; <uint> [#uses=1]
  %C = setgt uint %Y, 17		; <bool> [#uses=1]
  ret bool %C
}

bool %gt_unsigned_to_small_signed(ubyte %SB) {
  %Y = cast ubyte %SB to int
  %C = setgt int %Y, 17
  ret bool %C
}

bool %gt_unsigned_to_small_negative(ubyte %SB) {
  %Y = cast ubyte %SB to int
  %C = setgt int %Y, -17
  ret bool %C
}
