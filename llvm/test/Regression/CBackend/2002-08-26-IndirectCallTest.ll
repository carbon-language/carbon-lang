; Indirect function call test... found by Joel & Brian
;

%taskArray = uninitialized global int*

void %test(int %X) {
	%Y = add int %X, -1          ; <int>:1 [#uses=3]
        %cast100 = cast int %Y to long          ; <uint> [#uses=1]
        %gep100 = getelementptr int** %taskArray, long %cast100         ; <int**> [#uses=1]
        %fooPtr = load int** %gep100            ; <int*> [#uses=1]
        %cast101 = cast int* %fooPtr to void (int)*             ; <void (int)*> [#uses=1]
        call void %cast101( int 1000 )
	ret void
}
