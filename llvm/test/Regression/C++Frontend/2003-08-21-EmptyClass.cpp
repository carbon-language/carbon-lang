
struct empty {};

void foo(empty E);

void bar() { foo(empty()); }
