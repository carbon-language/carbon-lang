struct C {
        int A, B;
        ~C() {}

	void operator^(C b) const { }
};

void test(C *P) {
        *P ^ *P;
}

