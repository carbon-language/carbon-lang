; Various test cases to ensure basic functionality is working for GCSE

; RUN: llvm-as < %s | opt -gcse

implementation

void "testinsts"(int %i, int %j, int* %p)
begin
	%A = cast int %i to uint
	%B = cast int %i to uint
	
	%C = shl int %i, ubyte 1
	%D = shl int %i, ubyte 1

	%E = getelementptr int* %p, long 12
	%F = getelementptr int* %p, long 12
	%G = getelementptr int* %p, long 13
	ret void
end


; Test different combinations of domination properties...
void "sameBBtest"(int %i, int %j)
begin
	%A = add int %i, %j
	%B = add int %i, %j

	%C = xor int %A, -1
	%D = xor int %B, -1
	%E = xor int %j, -1

	ret void
end

int "dominates"(int %i, int %j)
begin
	%A = add int %i, %j
	br label %BB2

BB2:
	%B = add int %i, %j
	ret int %B
end

int "hascommondominator"(int %i, int %j)
begin
	br bool true, label %BB1, label %BB2

BB1:
	%A = add int %i, %j
	ret int %A

BB2:
	%B = add int %i, %j
	ret int %B
end

