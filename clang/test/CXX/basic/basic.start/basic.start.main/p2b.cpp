// RUN: %clang_cc1 -fsyntax-only -verify %s 

typedef int Int;
typedef char Char;
typedef Char* Carp;

Int main(Int argc, Carp argv[], Char *env[]) {
}
