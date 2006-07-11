// RUN: clang -E %s | grep 'noexp: foo y' &&
// RUN: clang -E %s | grep 'expand: abc' &&
// RUN: clang -E %s | grep 'noexp2: foo nonexp' &&
// RUN: clang -E %s | grep 'expand2: abc'

#define A foo
#define foo() abc
#define X A y

// This should not expand to abc, because the foo macro isn't followed by (.
noexp: X


// This should expand to abc.
#undef X
#define X A ()
expand: X


// This should be 'foo nonexp'
noexp2: A nonexp

// This should expand
expand2: A (
)


