int array[] = { 1, 2, 3, 4 };

struct foo {
  foo() throw();
} Constructor1;     // Global with ctor to be called before main

foo Constructor2;

struct bar {
  ~bar() throw();
} Destructor1;     // Global with dtor
