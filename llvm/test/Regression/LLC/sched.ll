implementation
declare int "printf"(sbyte*, int, float)


int "testissue"(int %i, float %x, float %y)
begin
	br label %bb1
bb1:
	%x1 = mul float %x, %y			;; x1
	%y1 = mul float %y, 0.75		;; y1
	%z1 = add float %x1, %y1		;; z1 = x1 + y1
	
	%x2 = mul float %x, 0.5			;; x2
	%y2 = mul float %y, 0.9			;; y2
	%z2 = add float %x2, %y2		;; z2 = x2 + y2
	
	%z3 = add float %z1, %z2		;; z3 = z1 + z2
	    
	%i1 = shl int   %i, ubyte 3		;; i1
	%j1 = add int   %i, 7			;; j1
	%m1 = add int   %i1, %j1		;; k1 = i1 + j1
;;	%m1 = div int   %k1, 99			;; m1 = k1 / 99
	
	%b  = setle int %m1, 6			;; (m1 <= 6)?
	br bool %b, label %bb1, label %bb2

bb2:
	%Msg = cast ulong 0 to sbyte *
	call int %printf(sbyte* %Msg, int %m1, float %z3)
	ret int 0
end
