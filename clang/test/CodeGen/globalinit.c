// RUN: clang -emit-llvm %s

int A[10] = { 1,2,3,4,5 };


extern int x[];
void foo() { x[0] = 1; }
int x[10];
void bar() { x[0] = 1; }


extern int y[];
//void *g = y;

int latin_ptr2len (char *p);
int (*mb_ptr2len) (char *p) = latin_ptr2len;


char string[8] = "string";   // extend init
char string2[4] = "string";  // truncate init

