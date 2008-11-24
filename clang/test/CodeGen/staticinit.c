// RUN: clang -emit-llvm < %s | grep "g.b = internal global i8. getelementptr"

struct AStruct { 
	int i;
	char *s;
	double d;
};

void f() {
  static int i = 42;
  static int is[] = { 1, 2, 3, 4 };
  static char* str = "forty-two";
  static char* strs[] = { "one", "two", "three", "four" };
  static struct AStruct myStruct = { 1, "two", 3.0 };
}

void g() {
  static char a[10];
  static char *b = a;
}

struct s { void *p; };

void foo(void) {
  static struct s var = {((void*)&((char*)0)[0])};
}
