struct B { int i(); int j(); };

void foo(int (B::*Fn)());

void test() {
	foo(&B::i);
	foo(&B::j);
}
