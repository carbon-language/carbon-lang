void bar(){}

void foo() {
	  struct TEST {
		  ~TEST() { bar(); }
	  } TESTOBJ;

}

int main() { return 0; }
