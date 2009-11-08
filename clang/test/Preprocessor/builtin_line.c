// RUN: clang-cc %s -E | grep "^  4"
#define FOO __LINE__

  FOO

// PR3579 - This should expand to the __LINE__ of the ')' not of the X.
// RUN: clang-cc %s -E | grep "^A 13"

#define X() __LINE__

A X(

)
