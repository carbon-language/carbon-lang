; RUN: as < %s | opt -f -lowerrefs -o Output/%s.bc
; 
	%Flat_struct = type { sbyte, float }
	%Flat_t = type { sbyte, float }
	%Mixed_struct = type { int, [10 x double], [10 x [10 x double]], [10 x { sbyte, float }] }
	%Mixed_t = type { int, [10 x double], [10 x [10 x double]], [10 x { sbyte, float }] }

%trstr = internal constant [34 x sbyte] c"ENTERING METHOD:  int () * %%ain\0A\00"

declare int "printf"(sbyte *, ...)

declare int "ArrayRef"([100 x int] * %Array, uint %I, uint %J)

implementation

int "test"([20 x [10 x [5 x int]]] * %A)
begin
	%i = load [20 x [10 x [5 x int]]] * %A, uint 1, uint 2, uint 3, uint 4

	;; same as above but via a GEP
	%iptr = getelementptr [20 x [10 x [5 x int]]] * %A, uint 1, uint 2, uint 3, uint 4
	%ii= load int* %iptr

	;; EXPECTED RESULT: decomposed indices for above LOAD or GEP
	;; %ptr1 = getelementptr [20 x [10 x [5 x int]]] * %A,            uint 1
	;; %ptr2 = getelementptr [20 x [10 x [5 x int]]] * %ptr1, uint 0, uint 2
	;; %ptr3 = getelementptr       [10 x [5 x int]]  * %ptr2, uint 0, uint 3
	;; %iii  = load                      [5 x int]   * %ptr3, uint 0, uint 4

	ret int %i
end

void "InitializeMixed"(%Mixed_struct * %M, int %base)
begin
bb0:					;[#uses=2]
	%reg112 = add int %base, 1		; <int> [#uses=1]
	%reg164-idxcast = cast int %reg112 to uint		; <uint> [#uses=1]

	;; Store to a structure field
	store sbyte 81, %Mixed_struct * %M, uint 0, ubyte 3, uint %reg164-idxcast, ubyte 0

	;; EXPECTED RESULT: decomposed indices for above STORE
	;; %ptr1 = getelementptr %Mixed_struct * %M,             uint 0, ubyte 3
	;; %ptr2 = getelementptr [10 x { sbyte, float }]* %ptr1, uint 0, uint %reg164-idxcast 
	;; store sbyte 81, {sbyte,float}* %ptr2, uint 0, ubyte 0

	;; Store to an array field within a structure
	store double 2.17, %Mixed_struct * %M, uint 0, ubyte 1, uint %reg164-idxcast

	;; EXPECTED RESULT: decomposed indices for above STORE
	;; %ptr1 = getelementptr %Mixed_struct * %M, uint 0, ubyte 1
	;; store double 2.17, [10 x double]* %ptr1, uint 0, uint %reg164-idxcast

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
	ret int %eltVal

;;	ret int 0
end

int "ArrayRef"([100 x int] * %Array, uint %I, uint %J)
begin
bb0:					;[#uses=3]
        %reg121 = load [100 x int]* %Array, uint %I, uint %J	; <int> [#uses=1]			
	ret int %reg121;
end

sbyte "PtrRef"(sbyte** %argv, uint %I, uint %J)
begin
bb0:					;[#uses=3]
	%reg222 = load sbyte** %argv, uint %I, uint %J		; <sbyte> [#uses=1]
	ret sbyte %reg222;
end
