#include <stdio.h>
#include <stdarg.h>


#undef LLVM_CAN_PASS_STRUCTS_BY_VALUE
#ifdef LLVM_CAN_PASS_STRUCTS_BY_VALUE
typedef struct SmallStruct_struct {
  char c1, c2, c3, c4;
  int  n; 
} SmallStruct;


typedef struct BigStruct_struct {
  char       c1, c2, c3, c4;
  double     d1, d2;                    /* Note: d1 will need padding */
  int        n; 
  struct BigStruct_struct* next;        /* Note: next will need padding */
} BigStruct;


SmallStruct
printStructArgs(SmallStruct s1,              /* Could fit in reg */
                int a1, float a2, char a3, double a4, char* a5,
                BigStruct s2,                /* Must go on stack */
                int a6, float a7, char a8, double a9, char* a10,
                SmallStruct  s3,             /* Probably no available regs */
                int a11, float a12, char a13, double a14, char* a15)
{
  SmallStruct result;
  
  printf("\nprintStructArgs with 13 arguments:\n");
  printf("\tArg  1    : %c %c %c %c %d\n",  s1.c1, s1.c2, s1.c3, s1.c4, s1.n);
  printf("\tArgs 2-6  : %d %f %c %lf %c\n", a1, a2, a3, a4, *a5);
  printf("\tArg  7    : %c %c %c %c %lf %lf %d %p\n",
                       s2.c1, s2.c2, s2.c3, s2.c4, s2.d1, s2.d2, s2.n,s2.next);
  printf("\tArg  8    : %c %c %c %c %d\n",  s3.c1, s3.c2, s3.c3, s3.c4, s3.n);
  printf("\tArgs 9-13 : %d %f %c %lf %c\n", a6, a7, a8, a9, *a10);
  printf("\tArgs 14-18: %d %f %c %lf %c\n", a11, a12, a13, a14, *a15);
  printf("\n");
  
  result.c1 = s2.c1;
  result.c2 = s2.c2;
  result.c3 = s2.c3;
  result.c4 = s2.c4;
  result.n  = s2.n;
  
  return result;
}
#endif  /* LLVM_CAN_PASS_STRUCTS_BY_VALUE */

#undef LLC_SUPPORTS_VARARGS_FUNCTIONS
#ifdef LLC_SUPPORTS_VARARGS_FUNCTIONS
void
printVarArgs(int a1, ...)
{
  double a2, a7,  a12;                  /* float is promoted to double! */
  int    a3, a8,  a13;                  /* char is promoted to int! */
  double a4, a9,  a14;
  char  *a5, *a10, *a15;
  int    a6, a11;
  
  va_list ap;
  va_start(ap, a1);
  a2  = va_arg(ap, double);
  a3  = va_arg(ap, int);
  a4  = va_arg(ap, double);
  a5  = va_arg(ap, char*);
  
  a6  = va_arg(ap, int);
  a7  = va_arg(ap, double);
  a8  = va_arg(ap, int);
  a9  = va_arg(ap, double);
  a10 = va_arg(ap, char*);
  
  a11 = va_arg(ap, int);
  a12 = va_arg(ap, double);
  a13 = va_arg(ap, int);
  a14 = va_arg(ap, double);
  a15 = va_arg(ap, char*);
  
  printf("\nprintVarArgs with 15 arguments:\n");
  printf("\tArgs 1-5  : %d %f %c %lf %c\n", a1,  a2,  a3,  a4,  *a5);
  printf("\tArgs 6-10 : %d %f %c %lf %c\n", a6,  a7,  a8,  a9,  *a10);
  printf("\tArgs 11-14: %d %f %c %lf %c\n", a11, a12, a13, a14, *a15);
  printf("\n");
  return;
}
#endif /* LLC_SUPPORTS_VARARGS_FUNCTIONS */


void
printArgsNoRet(int a1,  float a2,  char a3,  double a4,  char* a5,
               int a6,  float a7,  char a8,  double a9,  char* a10,
               int a11, float a12, char a13, double a14, char* a15)
{
  printf("\nprintArgsNoRet with 15 arguments:\n");
  printf("\tArgs 1-5  : %d %f %c %lf %c\n", a1,  a2,  a3,  a4,  *a5);
  printf("\tArgs 6-10 : %d %f %c %lf %c\n", a6,  a7,  a8,  a9,  *a10);
  printf("\tArgs 11-14: %d %f %c %lf %c\n", a11, a12, a13, a14, *a15);
  printf("\n");
  return;
}


int
main(int argc, char** argv)
{
#ifdef LLVM_CAN_PASS_STRUCTS_BY_VALUE
  SmallStruct s1, s3, result;
  BigStruct   s2;
#endif /* LLVM_CAN_PASS_STRUCTS_BY_VALUE */
  
  printArgsNoRet(1,  2.1,  'c', 4.1,  "e",
                 6,  7.1,  'h', 9.1,  "j",
                 11, 12.1, 'm', 14.1, "o");
  
#ifdef LLC_SUPPORTS_VARARGS_FUNCTIONS
  printVarArgs(1,  2.2,  'c', 4.2,  "e",
               6,  7.2,  'h', 9.2,  "j",
               11, 12.2, 'm', 14.2, "o");
#endif /* LLC_SUPPORTS_VARARGS_FUNCTIONS */
  
#ifdef LLVM_CAN_PASS_STRUCTS_BY_VALUE
  s1.c1 = 'a'; 
  s1.c2 = 'b'; 
  s1.c3 = 'c'; 
  s1.c4 = 'd'; 
  s1.n  = 111;
  
  s2.c1 = 'h'; 
  s2.c2 = 'i'; 
  s2.c3 = 'j'; 
  s2.c4 = 'k'; 
  s2.d1 = 1.1;
  s2.d2 = 2.2;
  s2.n  = 222;
  s2.next = &s2;
  
  s3.c1 = 'w'; 
  s3.c2 = 'x'; 
  s3.c3 = 'y'; 
  s3.c4 = 'z'; 
  s3.n  = 333;
  
  result = printStructArgs(s1,
                           1, 2.0, 'c', 4.0, "e",
                           s2,
                           6, 7.0, 'h', 9.0, "j",
                           s3);
  
  printf("\nprintStructArgs returns:\n\t%c %c %c %c %d\n\n",
         result.c1, result.c2, result.c3, result.c4, result.n);
#endif /* LLVM_CAN_PASS_STRUCTS_BY_VALUE */

  return 0;
}
