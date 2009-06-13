// RUN: clang-cc -verify %s

typedef struct s { int x; } s;
typedef int I;
typedef int I2;
typedef I2 I; // expected-note {{previous definition is here}}

typedef char I; // expected-error {{typedef redefinition with different types}}
