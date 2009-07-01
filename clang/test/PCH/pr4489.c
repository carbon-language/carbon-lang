// RUN: clang -x c-header -o %t.pch %s &&
// RUN: clang -include %t -x c /dev/null -emit-llvm -S -o -
// PR 4489: Crash with PCH

typedef struct _IO_FILE FILE;
extern int fprintf (struct _IO_FILE *__restrict __stream,
                    __const char *__restrict __format, ...);

int x(void)
{
  switch (1) {
    case 2: ;
      int y = 0;
  }
}

void y(void) {
  extern char z;
  fprintf (0, "a");
}