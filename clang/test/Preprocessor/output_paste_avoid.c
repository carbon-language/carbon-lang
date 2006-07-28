// RUN: clang -E %s | grep '+ + - - + + = = =' &&
// RUN: clang -E %s | not grep -F '...'

// This should print as ".. ." to avoid turning into ...
#define y(a) ..a
y(.)

#define PLUS +
#define EMPTY
#define f(x) =x=
+PLUS -EMPTY- PLUS+ f(=)

