
union X {
  void *B;
};

union X foo() {
	union X A;
	A.B = (void*)123;
	return A;
}
