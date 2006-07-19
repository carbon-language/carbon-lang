// RUN: clang -Eonly %s 2>&1 | grep error
// pasting ""x"" and ""+"" does not give a valid preprocessing token
#define XYZ  x ## +
XYZ

