/* Provide Declarations */

#ifndef NULL
#define NULL 0
#endif

#ifndef __cplusplus
typedef unsigned char bool;
#endif


/* Support for floating point constants */
typedef unsigned long long ConstantDoubleTy;
typedef unsigned int        ConstantFloatTy;


/* Global Declarations */

/* External Global Variable Declarations */

/* Function Declarations */
void __main();
int printf(signed char *, ...);
void testfunc(short l5_s, float l11_X, signed char l3_C, signed long long l9_LL, int l7_I, double l12_D);
void main();

/* Malloc to make sun happy */
extern void * malloc(size_t);



/* Global Variable Declerations */
extern signed char l27_d_LC0[26];


/* Global Variable Definitions and Initialization */
static signed char l27_d_LC0[26] = "%d, %f, %d, %lld, %d, %f\n";


/* Function Bodies */
void testfunc(short l5_s, float l11_X, signed char l3_C, signed long long l9_LL, int l7_I, double l12_D) {
  int l7_reg226;


  l7_reg226 = printf((&(l27_d_LC0[0ll])), ((unsigned )l5_s), ((double )l11_X), ((unsigned )l3_C), l9_LL, l7_I, l12_D);
  return;
}

void main() {

  const ConstantFloatTy FloatConstant0 = 0x3f9f5c29;    /* 1.245 */
  const ConstantDoubleTy FloatConstant1 = 0x432ff973cafa8000;    /* 4.5e+15 */

  __main();
  testfunc(12, (*(float*)&FloatConstant0), 120, 123456677890ll, -10, (*(double*)&FloatConstant1));
  return;
}

