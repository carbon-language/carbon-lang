// This tests compilation of EMPTY_CLASS_EXPR's

struct empty {};

void foo(empty E);

void bar() { foo(empty()); }
