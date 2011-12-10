// RUN: %clang_cc1  -analyze -analyzer-checker=experimental.security.taint,experimental.security.ArrayBoundV2 -verify %s

int scanf(const char *restrict format, ...);
int getchar(void);

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
  scanf("%d", t); // expected-warning {{Pointer argument is expected}} \
                  // expected-warning {{conversion specifies type 'int *' but the argument has type 'int'}}
}

void bufferGetchar(int x) {
  int m = getchar();
  Buffer[m] = 1;  //expected-warning {{Out of bound memory access }}
}
