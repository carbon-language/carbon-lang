// clang %s -E | grep 'bar foo (2)'

#define foo(x) bar x
foo(foo) (2)

