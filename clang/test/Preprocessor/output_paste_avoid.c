// RUN: clang -E %s | grep '+ + - - + + = = =' &&
// RUN: clang -E %s | not grep -F '...' &&
// RUN: clang -E %s | not grep -F 'L"str"'

// This should print as ".. ." to avoid turning into ...
#define y(a) ..a
y(.)

#define PLUS +
#define EMPTY
#define f(x) =x=
+PLUS -EMPTY- PLUS+ f(=)


// Should expand to L "str" not L"str"
#define test(x) L#x
test(str)

