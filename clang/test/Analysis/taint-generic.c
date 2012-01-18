// RUN: %clang_cc1  -analyze -analyzer-checker=experimental.security.taint,experimental.security.ArrayBoundV2 -Wno-format-security -verify %s

int scanf(const char *restrict format, ...);
int getchar(void);

typedef struct _FILE FILE;
extern FILE *stdin;
int fscanf(FILE *restrict stream, const char *restrict format, ...);
int sprintf(char *str, const char *format, ...);
void setproctitle(const char *fmt, ...);
typedef __typeof(sizeof(int)) size_t;

// Define string functions. Use builtin for some of them. They all default to
// the processing in the taint checker.
#define strcpy(dest, src) \
  ((__builtin_object_size(dest, 0) != -1ULL) \
   ? __builtin___strcpy_chk (dest, src, __builtin_object_size(dest, 1)) \
   : __inline_strcpy_chk(dest, src))

static char *__inline_strcpy_chk (char *dest, const char *src) {
  return __builtin___strcpy_chk(dest, src, __builtin_object_size(dest, 1));
}
char *stpcpy(char *restrict s1, const char *restrict s2);
char *strncpy( char * destination, const char * source, size_t num );
char *strndup(const char *s, size_t n);
char *strncat(char *restrict s1, const char *restrict s2, size_t n);

void *malloc(size_t);
void *calloc(size_t nmemb, size_t size);
void bcopy(void *s1, void *s2, size_t n);

#define BUFSIZE 10

int Buffer[BUFSIZE];
void bufferScanfDirect(void)
{
  int n;
  scanf("%d", &n);
  Buffer[n] = 1; // expected-warning {{Out of bound memory access }}
}

void bufferScanfArithmetic1(int x) {
  int n;
  scanf("%d", &n);
  int m = (n - 3);
  Buffer[m] = 1; // expected-warning {{Out of bound memory access }}
}

void bufferScanfArithmetic2(int x) {
  int n;
  scanf("%d", &n);
  int m = 100 / (n + 3) * x;
  Buffer[m] = 1; // expected-warning {{Out of bound memory access }}
}

void bufferScanfAssignment(int x) {
  int n;
  scanf("%d", &n);
  int m;
  if (x > 0) {
    m = n;
    Buffer[m] = 1; // expected-warning {{Out of bound memory access }}
  }
}

void scanfArg() {
  int t;
  scanf("%d", t); // expected-warning {{conversion specifies type 'int *' but the argument has type 'int'}}
}

void bufferGetchar(int x) {
  int m = getchar();
  Buffer[m] = 1;  //expected-warning {{Out of bound memory access }}
}

void testUncontrolledFormatString(char **p) {
  char s[80];
  fscanf(stdin, "%s", s);
  char buf[128];
  sprintf(buf,s); // expected-warning {{Uncontrolled Format String}}
  setproctitle(s, 3); // expected-warning {{Uncontrolled Format String}}

  // Test taint propagation through strcpy and family.
  char scpy[80];
  strcpy(scpy, s);
  sprintf(buf,scpy); // expected-warning {{Uncontrolled Format String}}

  stpcpy(*(++p), s); // this generates __inline.
  setproctitle(*(p), 3); // expected-warning {{Uncontrolled Format String}}

  char spcpy[80];
  stpcpy(spcpy, s);
  setproctitle(spcpy, 3); // expected-warning {{Uncontrolled Format String}}

  char *spcpyret;
  spcpyret = stpcpy(spcpy, s);
  setproctitle(spcpyret, 3); // expected-warning {{Uncontrolled Format String}}

  char sncpy[80];
  strncpy(sncpy, s, 20);
  setproctitle(sncpy, 3); // expected-warning {{Uncontrolled Format String}}

  char *dup;
  dup = strndup(s, 20);
  setproctitle(dup, 3); // expected-warning {{Uncontrolled Format String}}

}

int system(const char *command);
void testTaintSystemCall() {
  char buffer[156];
  char addr[128];
  scanf("%s", addr);
  system(addr); // expected-warning {{Tainted data passed to a system call}}

  // Test that spintf transfers taint.
  sprintf(buffer, "/bin/mail %s < /tmp/email", addr);
  system(buffer); // expected-warning {{Tainted data passed to a system call}}
}

void testTaintSystemCall2() {
  // Test that snpintf transfers taint.
  char buffern[156];
  char addr[128];
  scanf("%s", addr);
  __builtin_snprintf(buffern, 10, "/bin/mail %s < /tmp/email", addr);
  system(buffern); // expected-warning {{Tainted data passed to a system call}}
}

void testTaintSystemCall3() {
  char buffern2[156];
  int numt;
  char addr[128];
  scanf("%s %d", addr, &numt);
  __builtin_snprintf(buffern2, numt, "/bin/mail %s < /tmp/email", "abcd");
  system(buffern2); // expected-warning {{Tainted data passed to a system call}}
}

void testTaintedBufferSize() {
  size_t ts;
  scanf("%zd", &ts);

  int *buf1 = (int*)malloc(ts*sizeof(int)); // expected-warning {{Tainted data is used to specify the buffer size}}
  char *dst = (char*)calloc(ts, sizeof(char)); //expected-warning {{Tainted data is used to specify the buffer size}}
  bcopy(buf1, dst, ts); // expected-warning {{Tainted data is used to specify the buffer size}}
  __builtin_memcpy(dst, buf1, (ts + 4)*sizeof(char)); // expected-warning {{Tainted data is used to specify the buffer size}}

  // If both buffers are trusted, do not issue a warning.
  char *dst2 = (char*)malloc(ts*sizeof(char)); // expected-warning {{Tainted data is used to specify the buffer size}}
  strncat(dst2, dst, ts); // no-warning

}
