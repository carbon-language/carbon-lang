
implementation

declare void "_Z12combinationsjPj"      (uint, uint*)   ;; Prototype for: void combinations(unsigned int, unsigned int*)

;; Emitting: void UseAllocaFunction(unsigned int)
void "_Z17UseAllocaFunctionj"(uint %n)
begin
bb1:            ;;<label>
        %reg110 = shl uint %n, ubyte 2          ;;<uint>
        %reg108 = alloca [ubyte], uint %reg110            ;;<ubyte*>
        %cast1000 = cast [ubyte]* %reg108 to uint*                ;;<uint*>
        call void(uint, uint*) %_Z12combinationsjPj(uint %n, uint* %cast1000)                ;;<void>
        %cast113 = cast uint %reg110 to ulong*          ;;<ulong*>
	cast uint 7 to ulong *
        %reg114 = add ulong* %cast113, %0                ;;<ulong*>
        %reg115 = shr ulong* %reg114, ubyte 3           ;;<ulong*>:(uns ops)
        %reg117 = shl ulong* %reg115, ubyte 3           ;;<ulong*>
        %cast1001 = cast ulong* %reg117 to uint         ;;<uint>
        %reg118 = alloca [ubyte], uint %cast1001          ;;<ubyte*>
        %cast1002 = cast [ubyte]* %reg118 to uint*                ;;<uint*>
        call void(uint, uint*) %_Z12combinationsjPj(uint %n, uint* %cast1002)                ;;<void>
        ret void                ;;<void>
end

