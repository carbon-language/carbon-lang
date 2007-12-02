// RUN: clang -emit-llvm %s

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
