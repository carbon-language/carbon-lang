implementation

;; Emitting: void combinations(unsigned int, unsigned int*)
void "combinations"(uint %n, uint* %A)
begin
bb1:		;;<label>
	%reg111 = shl uint %n, ubyte 2		;;<uint>
	%cast112 = cast uint %reg111 to uint*		;;<uint*>
	%reg113 = add uint* %A, %cast112		;;<uint*>
	store uint 1, uint* %reg113		;;<void>
	store uint 1, uint* %A		;;<void>
	%reg128 = shr uint %n, ubyte 1		;;<uint>:(unsigned operands)
	%cond105 = setgt uint 1, %reg128		;;<bool>:(unsigned operands)
	br bool %cond105, label %bb3, label %bb2		;;<void>

bb2:		;;<label>
	%reg129 = phi uint [ %reg131, %bb2 ], [ 1, %bb1 ]		;;<uint>
	%reg130 = phi uint [ %reg132, %bb2 ], [ 1, %bb1 ]		;;<uint>
	%reg117 = sub uint %n, %reg130		;;<uint>
	%reg118 = add uint %reg117, 1		;;<uint>
	%reg119 = mul uint %reg129, %reg118		;;<uint>
	%reg131 = div uint %reg119, %reg130		;;<uint>:(unsigned operands)
	%reg120 = shl uint %reg130, ubyte 2		;;<uint>
	%cast121 = cast uint %reg120 to uint*		;;<uint*>
	%reg122 = add uint* %A, %cast121		;;<uint*>
	%reg124 = shl uint %reg117, ubyte 2		;;<uint>
	%cast125 = cast uint %reg124 to uint*		;;<uint*>
	%reg126 = add uint* %A, %cast125		;;<uint*>
	store uint %reg131, uint* %reg126		;;<void>
	store uint %reg131, uint* %reg122		;;<void>
	%reg132 = add uint %reg130, 1		;;<uint>
	%cond47 = setle uint %reg132, %reg128		;;<bool>:(unsigned operands)
	br bool %cond47, label %bb2, label %bb3		;;<void>

bb3:		;;<label>
	ret void 		;;<void>
end

declare void "_Z12combinationsjPj"	(uint, uint*)	;; Prototype for: void combinations(unsigned int, unsigned int*)
