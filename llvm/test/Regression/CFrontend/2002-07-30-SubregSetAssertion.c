
union X {
  //char C;
  //int A;
  void *B;
};

union X foo(union X A) {
	//A.C = 123;
	//A.A = 39249;
	A.B = (void*)123040123321;
	return A;
}
