; Tests to make sure elimination of casts is working correctly

; RUN: if as < %s | opt -instcombine -dce | grep '%c'
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi

implementation

int "test1"(int %A)
begin
	%c1 = cast int %A to uint
	%c2 = cast uint %c1 to int
	ret int %c2
end

ulong "test2"(ubyte %A)
begin
	%c1 = cast ubyte %A to ushort
	%c2 = cast ushort %c1 to uint
	%Ret = cast uint %c2 to ulong
	ret ulong %Ret
end
