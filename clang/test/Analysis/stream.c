// RUN: %clang_cc1 -analyze -analyzer-check-objc-mem -analyzer-experimental-checks -analyzer-store region -verify %s

typedef __typeof__(sizeof(int)) size_t;
typedef struct _IO_FILE FILE;
#define SEEK_SET	0	/* Seek from beginning of file.  */
#define SEEK_CUR	1	/* Seek from current position.  */
#define SEEK_END	2	/* Seek from end of file.  */
extern FILE *fopen(const char *path, const char *mode);
extern size_t fread(void *ptr, size_t size, size_t nmemb, FILE *stream);
extern int fseek (FILE *__stream, long int __off, int __whence);
extern long int ftell (FILE *__stream);
extern void rewind (FILE *__stream);

void f1(void) {
  FILE *p = fopen("foo", "r");
  char buf[1024];
  fread(buf, 1, 1, p); // expected-warning {{Stream pointer might be NULL.}}
}

void f2(void) {
  FILE *p = fopen("foo", "r");
  fseek(p, 1, SEEK_SET); // expected-warning {{Stream pointer might be NULL.}}
}

void f3(void) {
  FILE *p = fopen("foo", "r");
  ftell(p); // expected-warning {{Stream pointer might be NULL.}}
}

void f4(void) {
  FILE *p = fopen("foo", "r");
  rewind(p); // expected-warning {{Stream pointer might be NULL.}}
}

