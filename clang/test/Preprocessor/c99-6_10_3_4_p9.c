// Example from C99 6.10.3.4p9

// RUN: clang-cc -E %s | grep -F 'fprintf(stderr, "Flag");' &&
// RUN: clang-cc -E %s | grep -F 'fprintf(stderr, "X = %d\n", x);' &&
// RUN: clang-cc -E %s | grep -F 'puts("The first, second, and third items.");' &&
// RUN: clang-cc -E %s | grep -F '((x>y)?puts("x>y"): printf("x is %d but y is %d", x, y));'

#define debug(...) fprintf(stderr, __VA_ARGS__) 
#define showlist(...) puts(#__VA_ARGS__) 
#define report(test, ...) ((test)?puts(#test):\
                           printf(__VA_ARGS__)) 
debug("Flag"); 
debug("X = %d\n", x); 
showlist(The first, second, and third items.); 
report(x>y, "x is %d but y is %d", x, y); 

