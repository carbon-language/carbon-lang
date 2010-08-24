// The line and column layout of this test is significant. Run lines
// are at the end.
typedef void (^block_t)(float f, double d);
void f(int (^block)(int x, int y));
void g(block_t b);

void test_f() {

}

@interface A
- method:(int (^)(int x, int y))b;
- method2:(block_t)b;
@end

void test_A(A *a) {
  [a method:0];
}
// RUN: c-index-test -code-completion-at=%s:8:1 %s | FileCheck -check-prefix=CHECK-CC1 %s
// CHECK-CC1: FunctionDecl:{ResultType void}{TypedText f}{LeftParen (}{Placeholder int (^)(int x, int y)}{RightParen )} (45)
// CHECK-CC1: FunctionDecl:{ResultType void}{TypedText g}{LeftParen (}{Placeholder void (^)(float f, double d)}{RightParen )} (45)
// RUN: c-index-test -code-completion-at=%s:17:6 %s | FileCheck -check-prefix=CHECK-CC2 %s
// CHECK-CC2: ObjCInstanceMethodDecl:{ResultType id}{TypedText method2:}{Placeholder void (^)(float f, double d)} (20)
// CHECK-CC2: ObjCInstanceMethodDecl:{ResultType id}{TypedText method:}{Placeholder int (^)(int x, int y)} (20)
