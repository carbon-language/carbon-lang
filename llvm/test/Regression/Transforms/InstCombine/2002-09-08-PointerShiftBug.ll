; This tests for a bug exposed by the Olden health benchmark where a shift of a
; pointer was thrown away, causing this whole example to be "optimized" away, 
; which was quite bogus.  Check that this is fixed.
; 
; RUN: as < %s | opt -instcombine | dis | grep reg162

implementation   ; Functions:

int %test(uint %cann-indvar) {
        %reg189-scale = mul uint %cann-indvar, 4294967295               ; <uint> [#uses=1]
        %reg189 = add uint %reg189-scale, 3             ; <uint> [#uses=1]
        %cast362 = cast uint %reg189 to int             ; <int> [#uses=1]
        %cast363 = cast int %cast362 to sbyte*          ; <sbyte*> [#uses=2]
        %reg160 = shl sbyte* %cast363, ubyte 1          ; <sbyte*> [#uses=1]
        %reg161 = add sbyte* %reg160, %cast363          ; <sbyte*> [#uses=1]
        %reg162 = shl sbyte* %reg161, ubyte 2           ; <sbyte*> [#uses=1]
        %RV = cast sbyte* %reg162 to int                ; <int> [#uses=1]
        ret int %RV
}

