	%Flat_struct = type { sbyte, float }
	%Flat_t = type { sbyte, float }
	%Mixed_struct = type { int, [10 x double], [10 x [10 x double]], [10 x { sbyte, float }] }
	%Mixed_t = type { int, [10 x double], [10 x [10 x double]], [10 x { sbyte, float }] }

%trstr = internal constant [34 x sbyte] c"ENTERING METHOD:  int () * %%ain\0A\00"

declare int "printf"(sbyte *, ...)

declare int "ArrayRef"([100 x int] * %Array, uint %I, uint %J)

implementation

void "InitializeMixed"(%Mixed_struct * %M, int %base)
begin
bb0:					;[#uses=2]
	%reg112 = add int %base, 1		; <int> [#uses=1]
	%reg164-idxcast = cast int %reg112 to uint		; <uint> [#uses=1]
	store sbyte 81, %Mixed_struct * %M, uint 0, ubyte 3, uint %reg164-idxcast, ubyte 0

	store double 2.17, %Mixed_struct * %M, uint 0, ubyte 1, uint %reg164-idxcast
	ret void
end


int "main"()
begin
bb0:					;[#uses=1]
	%Array = alloca [100 x [100 x int]]
	%ArraySlice = getelementptr [100 x [100 x int]]* %Array, uint 0, uint 0
  	%trstrP = getelementptr [34 x sbyte] * %trstr, uint 0, uint 0

    	%trace  = call int (sbyte *, ...) * %printf( sbyte * %trstrP )

	%eltVal = call int %ArrayRef([100 x int]* %ArraySlice, uint 8, uint 12)
;;	ret int %eltVal

	ret int 0
end

int "ArrayRef"([100 x int] * %Array, uint %I, uint %J)
begin
bb0:					;[#uses=3]
	%reg120 = getelementptr [100 x int] * %Array, uint %I		; <[100 x int] *> [#uses=1]
	%reg121 = load [100 x int] * %reg120, uint 0, uint %J		; <int> [#uses=1]
	ret int %reg121;
end
