; Compiling this file produces:
; Sparc.cpp:91: failed assertion `(offset - OFFSET) % getStackFrameSizeAlignment() == 0'
;
implementation

declare int "SIM"(sbyte* %A, sbyte* %B, int %M, int %N, int %K, [256 x int]* %V, int %Q, int %R, int %nseq)

void "foo"()
begin
bb0:					;[#uses=0]
	%V = alloca [256 x int], uint 256		; <[256 x int]*> [#uses=1]
	call int %SIM( sbyte* null, sbyte* null, int 0, int 0, int 0, [256 x int]* %V, int 0, int 0, int 2 )		; <int>:0 [#uses=0]
	ret void
end


