;
; RUN: analyze %s -tddatastructure

int* %test1(int *%A) {
	%R = getelementptr int* %A, long 1
	ret int* %R
}

int* %test2(uint %Num) {
	%A = malloc sbyte, uint %Num
	%B = cast sbyte* %A to int*
	ret int* %B
}

int* %test3(uint %Num) {
	%A = malloc sbyte, uint %Num
	%B = cast sbyte* %A to int*
	%C = getelementptr int* %B, long 5
	ret int* %C
}

int* %test4(bool %C, uint %Num) {
	br bool %C, label %L1, label %L2
L1:
	%A = malloc sbyte, uint %Num
	%B = cast sbyte* %A to int*
	br label %L3
L2:
	%C = malloc int, uint %Num
	br label %L3
L3:
	%D = phi int* [%B, %L1], [%C, %L2]
	%E = getelementptr int* %D, long 5
	ret int* %E
}

int* %test5(bool %C, uint %Num) {
	br bool %C, label %L1, label %L2
L1:
	%C = malloc int, uint %Num
	br label %L3
L2:
	%A = malloc sbyte, uint %Num
	%B = cast sbyte* %A to int*
	br label %L3
L3:
	%D = phi int* [%C, %L1], [%B, %L2]
	%E = getelementptr int* %D, long 5
	ret int* %E
}

int %test6({int, int}* %A) {
	%B = getelementptr {int, int}* %A, long 0, ubyte 0
	%b = load int* %B
	%C = getelementptr {int, int}* %A, long 0, ubyte 1
	%c = load int* %C
	%d = add int %b, %c
	ret int %d
}

sbyte* %test7(uint %Num) {
	%X = malloc sbyte, uint %Num
	%Y = getelementptr sbyte* %X, long 1
	store sbyte 0, sbyte* %Y
	ret sbyte* %X
}

