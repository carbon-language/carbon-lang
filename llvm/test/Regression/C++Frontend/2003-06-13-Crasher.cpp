void bar();

void foo() {
	  struct TEST {
		  ~TEST() { bar(); }
	  } TESTOBJ;

}
