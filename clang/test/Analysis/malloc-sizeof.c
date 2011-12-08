// RUN: %clang_cc1 -analyze -analyzer-checker=experimental.unix.MallocSizeof -verify %s

#include <stddef.h>

void *malloc(size_t size);
void *calloc(size_t nmemb, size_t size);
void *realloc(void *ptr, size_t size);

struct A {};
struct B {};

void foo() {
  int *ip1 = malloc(sizeof(1));
  int *ip2 = malloc(4 * sizeof(int));

  long *lp1 = malloc(sizeof(short)); // expected-warning {{Result of 'malloc' is converted to type 'long *', whose pointee type 'long' is incompatible with sizeof operand type 'short'}}
  long *lp2 = malloc(5 * sizeof(double)); // expected-warning {{Result of 'malloc' is converted to type 'long *', whose pointee type 'long' is incompatible with sizeof operand type 'double'}}
  long *lp3 = malloc(5 * sizeof(char) + 2); // expected-warning {{Result of 'malloc' is converted to type 'long *', whose pointee type 'long' is incompatible with sizeof operand type 'char'}}

  struct A *ap1 = calloc(1, sizeof(struct A));
  struct A *ap2 = calloc(2, sizeof(*ap1));
  struct A *ap3 = calloc(2, sizeof(ap1)); // expected-warning {{Result of 'calloc' is converted to type 'struct A *', whose pointee type 'struct A' is incompatible with sizeof operand type 'struct A *'}}
  struct A *ap4 = calloc(3, sizeof(struct A*)); // expected-warning {{Result of 'calloc' is converted to type 'struct A *', whose pointee type 'struct A' is incompatible with sizeof operand type 'struct A *'}}
  struct A *ap5 = calloc(4, sizeof(struct B)); // expected-warning {{Result of 'calloc' is converted to type 'struct A *', whose pointee type 'struct A' is incompatible with sizeof operand type 'struct B'}}
  struct A *ap6 = realloc(ap5, sizeof(struct A));
  struct A *ap7 = realloc(ap5, sizeof(struct B)); // expected-warning {{Result of 'realloc' is converted to type 'struct A *', whose pointee type 'struct A' is incompatible with sizeof operand type 'struct B'}}
}
