; Test that ds-aa can be used for queries that require field sensitive AA.
; RUN: llvm-as < %s | opt -no-aa -ds-aa -load-vn -gcse | llvm-dis | not grep load

%Pair = type { int, int }

implementation

%Pair* %id(%Pair* %P) { ret %Pair *%P }

int %foo() {
	%X = alloca %Pair
	%XP = call %Pair* %id(%Pair* %X)

	%F1 = getelementptr %Pair* %X, int 0, uint 0
	%F2 = getelementptr %Pair* %XP, int 0, uint 1
	store int 14, int* %F1
	store int 0, int* %F2     ; no alias F1
	%B = load int* %F1        ; Should eliminate load!
	ret int %B
}

