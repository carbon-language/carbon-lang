void f() __attribute__((unavailable));
void g();
#define g() f()
#define h() f()
#define x x
