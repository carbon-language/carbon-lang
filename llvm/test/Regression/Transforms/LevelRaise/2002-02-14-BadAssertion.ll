; An invalid assertion killed the level raiser.  Fixed.
;
; RUN: llvm-as < %s | opt -raise

implementation

declare int "connect_left"()

int "do_merge"()
begin
	%reg108 = call int %connect_left( )	
	%cast1002 = cast ulong 8 to sbyte *		
        %reg108-idxcast = cast int %reg108 to long
        %reg1000 = getelementptr sbyte * %cast1002, long %reg108-idxcast
	%cast1003 = cast sbyte * %reg1000 to sbyte * *	
	%reg112 = load sbyte * * %cast1003		
	%cast111 = cast sbyte * %reg112 to int
	ret int %cast111
end
