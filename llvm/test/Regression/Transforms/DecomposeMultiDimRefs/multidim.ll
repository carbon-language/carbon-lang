; RUN: as < %s | opt -f -lowerrefs -o Output/%s.bc
; 
	%Flat_struct = type { sbyte, float }
	%Flat_t = type { sbyte, float }
	%Mixed_struct = type { int, [10 x double], [10 x [10 x double]], [10 x { sbyte, float }] }
	%Mixed_t = type { int, [10 x double], [10 x [10 x double]], [10 x { sbyte, float }] }

%trstr = internal constant [34 x sbyte] c"ENTERING METHOD:  int () * %%ain\0A\00"

declare int "printf"(sbyte *, ...)

declare int "ArrayRef"([100 x int] * %Array, long %I, long %J)

implementation

int "test"([20 x [10 x [5 x int]]] * %A)
begin
	%idx = getelementptr [20 x [10 x [5 x int]]] * %A, long 1, long 2, long 3, long 4
	%i = load int* %idx

	;; same as above but via a GEP
	%iptr = getelementptr [20 x [10 x [5 x int]]] * %A, long 1, long 2, long 3, long 4
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
	%reg164-idxcast = cast int %reg112 to long		; <uint> [#uses=1]

	;; Store to a structure field
	%idx1 = getelementptr %Mixed_struct * %M, long 0, ubyte 3, long %reg164-idxcast, ubyte 0
	store sbyte 81, sbyte* %idx1

	;; EXPECTED RESULT: decomposed indices for above STORE
	;; %ptr1 = getelementptr %Mixed_struct * %M,             uint 0, ubyte 3
	;; %ptr2 = getelementptr [10 x { sbyte, float }]* %ptr1, uint 0, uint %reg164-idxcast 
	;; store sbyte 81, {sbyte,float}* %ptr2, uint 0, ubyte 0

	;; Store to an array field within a structure
	%idx2 = getelementptr %Mixed_struct * %M, long 0, ubyte 1, long %reg164-idxcast
	store double 2.17, double* %idx2

	;; EXPECTED RESULT: decomposed indices for above STORE
	;; %ptr1 = getelementptr %Mixed_struct * %M, uint 0, ubyte 1
	;; store double 2.17, [10 x double]* %ptr1, uint 0, uint %reg164-idxcast

	ret void
end


int "main"()
begin
bb0:					;[#uses=1]
	%Array = alloca [100 x [100 x int]]
	%ArraySlice = getelementptr [100 x [100 x int]]* %Array, long 0, long 0
  	%trstrP = getelementptr [34 x sbyte] * %trstr, long 0, long 0

    	%trace  = call int (sbyte *, ...) * %printf( sbyte * %trstrP )

	%eltVal = call int %ArrayRef([100 x int]* %ArraySlice, long 8, long 12)
	ret int %eltVal

;;	ret int 0
end

int "ArrayRef"([100 x int] * %Array, long %I, long %J)
begin
bb0:					;[#uses=3]
	%idx = getelementptr [100 x int]* %Array, long %I, long %J	; <int> [#uses=1]			
        %reg121 = load int* %idx
	ret int %reg121
end

sbyte "PtrRef"(sbyte** %argv, long %I, long %J)
begin
bb0:					;[#uses=3]
	%idx = getelementptr sbyte** %argv, long %I
	%reg222 = load sbyte** %idx
	%tmp = load sbyte* %reg222
	ret sbyte %tmp
end
