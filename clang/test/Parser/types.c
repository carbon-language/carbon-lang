// RUN: clang-cc %s -fsyntax-only

// Test the X can be overloaded inside the struct.
typedef int X; 
struct Y { short X; };

