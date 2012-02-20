// RUN: %clang_cc1 -analyze -analyzer-checker=unix.Malloc -analyzer-inline-call -analyzer-store=region -verify %s

#include "system-header-simulator.h"

typedef __typeof(sizeof(int)) size_t;
void *malloc(size_t);
void *valloc(size_t);
void free(void *);
void *realloc(void *ptr, size_t size);
void *reallocf(void *ptr, size_t size);
void *calloc(size_t nmemb, size_t size);
extern void exit(int) __attribute__ ((__noreturn__));

static void my_malloc1(void **d, size_t size) {
  *d = malloc(size);
}

static void *my_malloc2(int elevel, size_t size) {
  void     *data;
  data = malloc(size);
  if (data == 0)
    exit(0);
  return data;
}

static void my_free1(void *p) {
  free(p);
}

static void test1() {
  void *data = 0;
  my_malloc1(&data, 4); // expected-warning {{Memory is never released; potential memory leak}}
}

static void test2() {
  void * data = my_malloc2(1, 4);
  data = my_malloc2(1, 4);// expected-warning {{Memory is never released; potential memory leak}}
}

static void test3() {
  void *data = my_malloc2(1, 4);
  free(data);
  data = my_malloc2(1, 4);
  free(data);
}

int test4() {
  int *data = (int*)my_malloc2(1, 4);
  my_free1(data);
  data = (int *)my_malloc2(1, 4);
  my_free1(data);
  return *data; // expected-warning {{Use of memory after it is freed}}
}

