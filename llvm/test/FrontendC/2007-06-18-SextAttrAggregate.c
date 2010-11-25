// RUN: %llvmgcc %s -o - -S -O3 | grep {i8 signext}
// PR1513

struct s{
long a;
long b;
};

void f(struct s a, char *b, signed char C) {

}
