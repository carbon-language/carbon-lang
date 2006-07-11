// RUN: clang -E %s | grep 'noexp: foo y' &&
// RUN: clang -E %s | grep 'expand: abc'

#define A foo
#define foo() abc
#define X A y

// This should not expand to abc, because the foo macro isn't followed by (.
noexp: X

#undef X

// This should expand to abc.
#define X A ()
expand: X

