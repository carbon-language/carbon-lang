// RUN: %clang_cc1 -analyze -analyzer-check-objc-mem -analyzer-experimental-checks -analyzer-store region -verify %s

typedef __typeof__(sizeof(int)) size_t;
typedef struct _IO_FILE FILE;
FILE *fopen(const char *path, const char *mode);
size_t fread(void *ptr, size_t size, size_t nmemb, FILE *stream);

void f1(void) {
  FILE *p = fopen("foo", "r");
  char buf[1024];
  fread(buf, 1, 1, p); // expected-warning {{Stream pointer might be NULL.}}
}
