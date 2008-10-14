// RUN: clang %s -fsyntax-only -fpascal-strings

char array[1024/(sizeof (long))];

int x['\xBb' == (char) 187 ? 1: -1];

// PR1992
void func(int x)
{
 switch (x) {
 case sizeof("abc"): break;
 case sizeof("loooong"): func(4);
 case sizeof("\ploooong"): func(4);
 }
}


// rdar://4213768
int expr;
char y[__builtin_constant_p(expr) ? -1 : 1];
char z[__builtin_constant_p(4) ? 1 : -1];

