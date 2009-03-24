// RUN: clang-cc -emit-llvm %s -o - | not grep opaque

enum teste1 test1f(void), (*test1)(void) = test1f;
struct tests2 test2f(), (*test2)() = test2f;
struct tests3;
void test3f(struct tests3), (*test3)(struct tests3) = test3f;
enum teste1 { TEST1 };
struct tests2 { int x,y,z,a,b,c,d,e,f,g; };
struct tests3 { float x; };

