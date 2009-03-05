// RUN: clang -emit-llvm %s -o - | not grep opaque

enum teste1 (*test1)(void);
struct tests2 (*test2)();
struct tests3;
void (*test3)(struct tests3);
enum teste1 { TEST1 };
struct tests2 { int x,y,z,a,b,c,d,e,f,g; };
struct tests3 { float x; };

