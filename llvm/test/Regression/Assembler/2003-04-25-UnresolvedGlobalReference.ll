; There should be absolutely no problem with this testcase.

implementation

int %test(int %arg1, int %arg2) {  
	ret int cast (int (int, int)* %test to int)
}
