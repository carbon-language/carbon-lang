// RUN: %clang_cc1 -analyze -analyzer-checker=core,alpha.unix.SimpleStream -verify %s

typedef struct __sFILE {
  unsigned char *_p;
} FILE;
FILE *fopen(const char * restrict, const char * restrict) __asm("_" "fopen" );
int fputc(int, FILE *);
int fputs(const char * restrict, FILE * restrict) __asm("_" "fputs" );
int fclose(FILE *);
void exit(int);

void checkDoubleFClose(int *Data) {
  FILE *F = fopen("myfile.txt", "w");
  if (F != 0) {
    fputs ("fopen example", F);
    if (!Data)
      fclose(F);
    else
      fputc(*Data, F);
    fclose(F); // expected-warning {{Closing a previously closed file stream}}
  }
}

int checkLeak(int *Data) {
  FILE *F = fopen("myfile.txt", "w");
  if (F != 0) {
    fputs ("fopen example", F);
  }

  if (Data) // expected-warning {{Opened file is never closed; potential resource leak}}
    return *Data;
  else
    return 0;
}

void checkLeakFollowedByAssert(int *Data) {
  FILE *F = fopen("myfile.txt", "w");
  if (F != 0) {
    fputs ("fopen example", F);
    if (!Data)
      exit(0);
    fclose(F);
  }
}

void CloseOnlyOnValidFileHandle() {
  FILE *F = fopen("myfile.txt", "w");
  if (F)
    fclose(F);
  int x = 0; // no warning
}
