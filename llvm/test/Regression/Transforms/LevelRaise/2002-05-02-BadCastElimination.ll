; This test contains two cast instructions that cannot be eliminated.  If the
; input of the "test" function is negative, it should be correctly converted
; to a 32 bit version of the number with all upper 16 bits clear (ushort->uint
; involves no sign extension).  Optimizing this to a single cast is invalid!
;
; RUN: llvm-as < %s | opt -raise -q | lli -abort-on-exception
;
implementation

uint "test"(short %argc)
begin
	%cast223 = cast short %argc to ushort		; <ushort> [#uses=1]
	%cast114 = cast ushort %cast223 to uint		; <uint> [#uses=1]
	ret uint %cast114
end

int "main"()
begin
	%Ret = call uint %test(short -1)
	%test = cast uint %Ret to int
	%Res = seteq int %test, -1        ; If it returns -1 as int, it's a failure
	%Res = cast bool %Res to int
	ret int %Res
end
