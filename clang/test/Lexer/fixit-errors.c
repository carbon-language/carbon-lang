// RUN: clang-cc -fsyntax-only -pedantic -fixit %s -o - | clang-cc -pedantic -Werror -x c -

/* This is a test of the various code modification hints that are
   provided as part of warning or extension diagnostics. Eventually,
   we would like to actually try to perform the suggested
   modifications and compile the result to test that no warnings
   remain. */
// FIXME: If you put a space at the end of the line, it doesn't work yet!
char *s = "hi\
there";

// The following line isn't terminated, don't fix it.
int i; // expected-error{{no newline at end of file}}
