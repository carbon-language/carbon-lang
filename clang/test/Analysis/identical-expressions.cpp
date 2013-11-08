// RUN: %clang_cc1 -analyze -analyzer-checker=core,alpha.core.IdenticalExpr -verify %s

/* Only one expected warning per function allowed at the very end. */

/* '!=' operator*/

/* '!=' with float */
int checkNotEqualFloatLiteralCompare1(void) {
  return (5.14F != 5.14F); // no warning
}

int checkNotEqualFloatLiteralCompare2(void) {
  return (6.14F != 7.14F); // no warning
}

int checkNotEqualFloatDeclCompare1(void) {
  float f = 7.1F;
  float g = 7.1F;
  return (f != g); // no warning
}

int checkNotEqualFloatDeclCompare12(void) {
  float f = 7.1F;
  return (f != f); // no warning
}

int checkNotEqualFloatDeclCompare3(void) {
  float f = 7.1F;
  return (f != 7.1F); // no warning
}

int checkNotEqualFloatDeclCompare4(void) {
  float f = 7.1F;
  return (7.1F != f); // no warning
}

int checkNotEqualFloatDeclCompare5(void) {
  float f = 7.1F;
  int t = 7;
  return (t != f); // no warning
}

int checkNotEqualFloatDeclCompare6(void) {
  float f = 7.1F;
  int t = 7;
  return (f != t); // no warning
}



int checkNotEqualCastFloatDeclCompare11(void) {
  float f = 7.1F;
  return ((int)f != (int)f); // expected-warning {{comparison of identical expressions always evaluates to false}}
}
int checkNotEqualCastFloatDeclCompare12(void) {
  float f = 7.1F;
  return ((char)f != (int)f); // no warning
}
int checkNotEqualBinaryOpFloatCompare1(void) {
  int res;
  float f= 3.14F;
  res = (f + 3.14F != f + 3.14F);  // no warning
  return (0);
}
int checkNotEqualBinaryOpFloatCompare2(void) {
  float f = 7.1F;
  float g = 7.1F;
  return (f + 3.14F != g + 3.14F); // no warning
}
int checkNotEqualBinaryOpFloatCompare3(void) {
  int res;
  float f= 3.14F;
  res = ((int)f + 3.14F != (int)f + 3.14F);  // no warning
  return (0);
}
int checkNotEqualBinaryOpFloatCompare4(void) {
  int res;
  float f= 3.14F;
  res = ((int)f + 3.14F != (char)f + 3.14F);  // no warning
  return (0);
}

int checkNotEqualNestedBinaryOpFloatCompare1(void) {
  int res;
  int t= 1;
  int u= 2;
  float f= 3.14F;
  res = (((int)f + (3.14F - u)*t) != ((int)f + (3.14F - u)*t));  // no warning
  return (0);
}

int checkNotEqualNestedBinaryOpFloatCompare2(void) {
  int res;
  int t= 1;
  int u= 2;
  float f= 3.14F;
  res = (((int)f + (u - 3.14F)*t) != ((int)f + (3.14F - u)*t));  // no warning
  return (0);
}

int checkNotEqualNestedBinaryOpFloatCompare3(void) {
  int res;
  int t= 1;
  int u= 2;
  float f= 3.14F;
  res = (((int)f + (u - 3.14F)*t) != ((int)f + (3.14F - u)*(f + t != f + t)));  // no warning
  return (0);
}




/* end '!=' with float*/

/* '!=' with int*/

int checkNotEqualIntLiteralCompare1(void) {
  return (5 != 5); // expected-warning {{comparison of identical expressions always evaluates to false}}
}

int checkNotEqualIntLiteralCompare2(void) {
  return (6 != 7); // no warning
}

int checkNotEqualIntDeclCompare1(void) {
  int f = 7;
  int g = 7;
  return (f != g); // no warning
}

int checkNotEqualIntDeclCompare3(void) {
  int f = 7;
  return (f != 7); // no warning
}

int checkNotEqualIntDeclCompare4(void) {
  int f = 7;
  return (7 != f); // no warning
}

int checkNotEqualCastIntDeclCompare11(void) {
  int f = 7;
  return ((int)f != (int)f); // expected-warning {{comparison of identical expressions always evaluates to false}}
}
int checkNotEqualCastIntDeclCompare12(void) {
  int f = 7;
  return ((char)f != (int)f); // no warning
}
int checkNotEqualBinaryOpIntCompare1(void) {
  int res;
  int t= 1;
  int u= 2;
  int f= 4;
  res = (f + 4 != f + 4);  // expected-warning {{comparison of identical expressions always evaluates to false}}
  return (0);
}
int checkNotEqualBinaryOpIntCompare2(void) {
  int f = 7;
  int g = 7;
  return (f + 4 != g + 4); // no warning
}


int checkNotEqualBinaryOpIntCompare3(void) {
  int res;
  int t= 1;
  int u= 2;
  int f= 4;
  res = ((int)f + 4 != (int)f + 4);  // expected-warning {{comparison of identical expressions always evaluates to false}}
  return (0);
}
int checkNotEqualBinaryOpIntCompare4(void) {
  int res;
  int t= 1;
  int u= 2;
  int f= 4;
  res = ((int)f + 4 != (char)f + 4);  // no warning
  return (0);
}
int checkNotEqualBinaryOpIntCompare5(void) {
  int res;
  int t= 1;
  int u= 2;
  res = (u + t != u + t);  // expected-warning {{comparison of identical expressions always evaluates to false}}
  return (0);
}

int checkNotEqualNestedBinaryOpIntCompare1(void) {
  int res;
  int t= 1;
  int u= 2;
  int f= 3;
  res = (((int)f + (3 - u)*t) != ((int)f + (3 - u)*t));  // expected-warning {{comparison of identical expressions always evaluates to false}}
  return (0);
}

int checkNotEqualNestedBinaryOpIntCompare2(void) {
  int res;
  int t= 1;
  int u= 2;
  int f= 3;
  res = (((int)f + (u - 3)*t) != ((int)f + (3 - u)*t));  // no warning
  return (0);
}

int checkNotEqualNestedBinaryOpIntCompare3(void) {
  int res;
  int t= 1;
  int u= 2;
  int f= 3;
  res = (((int)f + (u - 3)*t) != ((int)f + (3 - u)*(t + 1 != t + 1)));  // expected-warning {{comparison of identical expressions always evaluates to false}}
  return (0);
}

/*   end '!=' int          */



/* '!=' with int pointer */

int checkNotEqualIntPointerLiteralCompare1(void) {
  int* p = 0;
  return (p != 0); // no warning
}

int checkNotEqualIntPointerLiteralCompare2(void) {
  return (6 != 7); // no warning
}

int checkNotEqualIntPointerDeclCompare1(void) {
  int k = 3;
  int* f = &k;
  int* g = &k;
  return (f != g); // no warning
}

int checkNotEqualCastIntPointerDeclCompare11(void) {
  int k = 7;
  int* f = &k;
  return ((int*)f != (int*)f); // expected-warning {{comparison of identical expressions always evaluates to false}}
}
int checkNotEqualCastIntPointerDeclCompare12(void) {
  int k = 7;
  int* f = &k;
  return ((int*)((char*)f) != (int*)f); // no warning
}
int checkNotEqualBinaryOpIntPointerCompare1(void) {
  int k = 7;
  int res;
  int* f= &k;
  res = (f + 4 != f + 4);  // expected-warning {{comparison of identical expressions always evaluates to false}}
  return (0);
}
int checkNotEqualBinaryOpIntPointerCompare2(void) {
  int k = 7;
  int* f = &k;
  int* g = &k;
  return (f + 4 != g + 4); // no warning
}


int checkNotEqualBinaryOpIntPointerCompare3(void) {
  int k = 7;
  int res;
  int* f= &k;
  res = ((int*)f + 4 != (int*)f + 4);  // expected-warning {{comparison of identical expressions always evaluates to false}}
  return (0);
}
int checkNotEqualBinaryOpIntPointerCompare4(void) {
  int k = 7;
  int res;
  int* f= &k;
  res = ((int*)f + 4 != (int*)((char*)f) + 4);  // no warning
  return (0);
}

int checkNotEqualNestedBinaryOpIntPointerCompare1(void) {
  int res;
  int k = 7;
  int t= 1;
  int* u= &k+2;
  int* f= &k+3;
  res = ((f + (3)*t) != (f + (3)*t));  // expected-warning {{comparison of identical expressions always evaluates to false}}
  return (0);
}

int checkNotEqualNestedBinaryOpIntPointerCompare2(void) {
  int res;
  int k = 7;
  int t= 1;
  int* u= &k+2;
  int* f= &k+3;
  res = (((3)*t + f) != (f + (3)*t));  // no warning
  return (0);
}
/*   end '!=' int*          */

/*   end '!=' */



/* EQ operator           */

int checkEqualIntPointerDeclCompare(void) {
  int k = 3;
  int* f = &k;
  int* g = &k;
  return (f == g); // no warning
}

int checkEqualIntPointerDeclCompare0(void) {
  int k = 3;
  int* f = &k;
  return (f+1 == f+1); // expected-warning {{comparison of identical expressions always evaluates to true}}
}

/* EQ with float*/

int checkEqualFloatLiteralCompare1(void) {
  return (5.14F == 5.14F); // no warning
}

int checkEqualFloatLiteralCompare2(void) {
  return (6.14F == 7.14F); // no warning
}

int checkEqualFloatDeclCompare1(void) {
  float f = 7.1F;
  float g = 7.1F;
  return (f == g); // no warning
}

int checkEqualFloatDeclCompare12(void) {
  float f = 7.1F;
  return (f == f); // no warning
}


int checkEqualFloatDeclCompare3(void) {
  float f = 7.1F;
  return (f == 7.1F); // no warning
}

int checkEqualFloatDeclCompare4(void) {
  float f = 7.1F;
  return (7.1F == f); // no warning
}

int checkEqualFloatDeclCompare5(void) {
  float f = 7.1F;
  int t = 7;
  return (t == f); // no warning
}

int checkEqualFloatDeclCompare6(void) {
  float f = 7.1F;
  int t = 7;
  return (f == t); // no warning
}




int checkEqualCastFloatDeclCompare11(void) {
  float f = 7.1F;
  return ((int)f == (int)f); // expected-warning {{comparison of identical expressions always evaluates to true}}
}
int checkEqualCastFloatDeclCompare12(void) {
  float f = 7.1F;
  return ((char)f == (int)f); // no warning
}
int checkEqualBinaryOpFloatCompare1(void) {
  int res;
  float f= 3.14F;
  res = (f + 3.14F == f + 3.14F);  // no warning
  return (0);
}
int checkEqualBinaryOpFloatCompare2(void) {
  float f = 7.1F;
  float g = 7.1F;
  return (f + 3.14F == g + 3.14F); // no warning
}
int checkEqualBinaryOpFloatCompare3(void) {
  int res;
  float f= 3.14F;
  res = ((int)f + 3.14F == (int)f + 3.14F);  // no warning
  return (0);
}
int checkEqualBinaryOpFloatCompare4(void) {
  int res;
  float f= 3.14F;
  res = ((int)f + 3.14F == (char)f + 3.14F);  // no warning
  return (0);
}

int checkEqualNestedBinaryOpFloatCompare1(void) {
  int res;
  int t= 1;
  int u= 2;
  float f= 3.14F;
  res = (((int)f + (3.14F - u)*t) == ((int)f + (3.14F - u)*t));  // no warning
  return (0);
}

int checkEqualNestedBinaryOpFloatCompare2(void) {
  int res;
  int t= 1;
  int u= 2;
  float f= 3.14F;
  res = (((int)f + (u - 3.14F)*t) == ((int)f + (3.14F - u)*t));  // no warning
  return (0);
}

int checkEqualNestedBinaryOpFloatCompare3(void) {
  int res;
  int t= 1;
  int u= 2;
  float f= 3.14F;
  res = (((int)f + (u - 3.14F)*t) == ((int)f + (3.14F - u)*(f + t == f + t)));  // no warning
  return (0);
}





/* Equal with int*/

int checkEqualIntLiteralCompare1(void) {
  return (5 == 5); // expected-warning {{comparison of identical expressions always evaluates to true}}
}

int checkEqualIntLiteralCompare2(void) {
  return (6 == 7); // no warning
}

int checkEqualIntDeclCompare1(void) {
  int f = 7;
  int g = 7;
  return (f == g); // no warning
}

int checkEqualCastIntDeclCompare11(void) {
  int f = 7;
  return ((int)f == (int)f); // expected-warning {{comparison of identical expressions always evaluates to true}}
}
int checkEqualCastIntDeclCompare12(void) {
  int f = 7;
  return ((char)f == (int)f); // no warning
}

int checkEqualIntDeclCompare3(void) {
  int f = 7;
  return (f == 7); // no warning
}

int checkEqualIntDeclCompare4(void) {
  int f = 7;
  return (7 == f); // no warning
}

int checkEqualBinaryOpIntCompare1(void) {
  int res;
  int t= 1;
  int u= 2;
  int f= 4;
  res = (f + 4 == f + 4);  // expected-warning {{comparison of identical expressions always evaluates to true}}
  return (0);
}
int checkEqualBinaryOpIntCompare2(void) {
  int f = 7;
  int g = 7;
  return (f + 4 == g + 4); // no warning
}


int checkEqualBinaryOpIntCompare3(void) {
  int res;
  int t= 1;
  int u= 2;
  int f= 4;
  res = ((int)f + 4 == (int)f + 4);  // expected-warning {{comparison of identical expressions always evaluates to true}}
  return (0);

}
int checkEqualBinaryOpIntCompare4(void) {
  int res;
  int t= 1;
  int u= 2;
  int f= 4;
  res = ((int)f + 4 == (char)f + 4);  // no warning
  return (0);
}
int checkEqualBinaryOpIntCompare5(void) {
  int res;
  int t= 1;
  int u= 2;
  res = (u + t == u + t);  // expected-warning {{comparison of identical expressions always evaluates to true}}
  return (0);
}

int checkEqualNestedBinaryOpIntCompare1(void) {
  int res;
  int t= 1;
  int u= 2;
  int f= 3;
  res = (((int)f + (3 - u)*t) == ((int)f + (3 - u)*t));  // expected-warning {{comparison of identical expressions always evaluates to true}}
  return (0);
}

int checkEqualNestedBinaryOpIntCompare2(void) {
  int res;
  int t= 1;
  int u= 2;
  int f= 3;
  res = (((int)f + (u - 3)*t) == ((int)f + (3 - u)*t));  // no warning
  return (0);
}

int checkEqualNestedBinaryOpIntCompare3(void) {
  int res;
  int t= 1;
  int u= 2;
  int f= 3;
  res = (((int)f + (u - 3)*t) == ((int)f + (3 - u)*(t + 1 == t + 1)));  // expected-warning {{comparison of identical expressions always evaluates to true}}
  return (0);
}


/*   end EQ int          */

/* end EQ */


/*  LT */

/*  LT with float */

int checkLessThanFloatLiteralCompare1(void) {
  return (5.14F < 5.14F); // expected-warning {{comparison of identical expressions always evaluates to false}}
}

int checkLessThanFloatLiteralCompare2(void) {
  return (6.14F < 7.14F); // no warning
}

int checkLessThanFloatDeclCompare1(void) {
  float f = 7.1F;
  float g = 7.1F;
  return (f < g); // no warning
}

int checkLessThanFloatDeclCompare12(void) {
  float f = 7.1F;
  return (f < f); // expected-warning {{comparison of identical expressions always evaluates to false}}
}

int checkLessThanFloatDeclCompare3(void) {
  float f = 7.1F;
  return (f < 7.1F); // no warning
}

int checkLessThanFloatDeclCompare4(void) {
  float f = 7.1F;
  return (7.1F < f); // no warning
}

int checkLessThanFloatDeclCompare5(void) {
  float f = 7.1F;
  int t = 7;
  return (t < f); // no warning
}

int checkLessThanFloatDeclCompare6(void) {
  float f = 7.1F;
  int t = 7;
  return (f < t); // no warning
}


int checkLessThanCastFloatDeclCompare11(void) {
  float f = 7.1F;
  return ((int)f < (int)f); // expected-warning {{comparison of identical expressions always evaluates to false}}
}
int checkLessThanCastFloatDeclCompare12(void) {
  float f = 7.1F;
  return ((char)f < (int)f); // no warning
}
int checkLessThanBinaryOpFloatCompare1(void) {
  int res;
  float f= 3.14F;
  res = (f + 3.14F < f + 3.14F);  // no warning
  return (0);
}
int checkLessThanBinaryOpFloatCompare2(void) {
  float f = 7.1F;
  float g = 7.1F;
  return (f + 3.14F < g + 3.14F); // no warning
}
int checkLessThanBinaryOpFloatCompare3(void) {
  int res;
  float f= 3.14F;
  res = ((int)f + 3.14F < (int)f + 3.14F);  // no warning
  return (0);
}
int checkLessThanBinaryOpFloatCompare4(void) {
  int res;
  float f= 3.14F;
  res = ((int)f + 3.14F < (char)f + 3.14F);  // no warning
  return (0);
}

int checkLessThanNestedBinaryOpFloatCompare1(void) {
  int res;
  int t= 1;
  int u= 2;
  float f= 3.14F;
  res = (((int)f + (3.14F - u)*t) < ((int)f + (3.14F - u)*t));  // no warning
  return (0);
}

int checkLessThanNestedBinaryOpFloatCompare2(void) {
  int res;
  int t= 1;
  int u= 2;
  float f= 3.14F;
  res = (((int)f + (u - 3.14F)*t) < ((int)f + (3.14F - u)*t));  // no warning
  return (0);
}

int checkLessThanNestedBinaryOpFloatCompare3(void) {
  int res;
  int t= 1;
  int u= 2;
  float f= 3.14F;
  res = (((int)f + (u - 3.14F)*t) < ((int)f + (3.14F - u)*(f + t < f + t)));  // no warning
  return (0);
}

/*  end LT with float */

/*  LT with int */


int checkLessThanIntLiteralCompare1(void) {
  return (5 < 5); // expected-warning {{comparison of identical expressions always evaluates to false}}
}

int checkLessThanIntLiteralCompare2(void) {
  return (6 < 7); // no warning
}

int checkLessThanIntDeclCompare1(void) {
  int f = 7;
  int g = 7;
  return (f < g); // no warning
}

int checkLessThanIntDeclCompare3(void) {
  int f = 7;
  return (f < 7); // no warning
}

int checkLessThanIntDeclCompare4(void) {
  int f = 7;
  return (7 < f); // no warning
}

int checkLessThanIntDeclCompare5(void) {
  int f = 7;
  int t = 7;
  return (t < f); // no warning
}

int checkLessThanIntDeclCompare6(void) {
  int f = 7;
  int t = 7;
  return (f < t); // no warning
}

int checkLessThanCastIntDeclCompare11(void) {
  int f = 7;
  return ((int)f < (int)f); // expected-warning {{comparison of identical expressions always evaluates to false}}
}
int checkLessThanCastIntDeclCompare12(void) {
  int f = 7;
  return ((char)f < (int)f); // no warning
}
int checkLessThanBinaryOpIntCompare1(void) {
  int res;
  int f= 3;
  res = (f + 3 < f + 3);  // expected-warning {{comparison of identical expressions always evaluates to false}}
  return (0);
}
int checkLessThanBinaryOpIntCompare2(void) {
  int f = 7;
  int g = 7;
  return (f + 3 < g + 3); // no warning
}
int checkLessThanBinaryOpIntCompare3(void) {
  int res;
  int f= 3;
  res = ((int)f + 3 < (int)f + 3);  // expected-warning {{comparison of identical expressions always evaluates to false}}
  return (0);
}
int checkLessThanBinaryOpIntCompare4(void) {
  int res;
  int f= 3;
  res = ((int)f + 3 < (char)f + 3);  // no warning
  return (0);
}

int checkLessThanNestedBinaryOpIntCompare1(void) {
  int res;
  int t= 1;
  int u= 2;
  int f= 3;
  res = (((int)f + (3 - u)*t) < ((int)f + (3 - u)*t));  // expected-warning {{comparison of identical expressions always evaluates to false}}
  return (0);
}

int checkLessThanNestedBinaryOpIntCompare2(void) {
  int res;
  int t= 1;
  int u= 2;
  int f= 3;
  res = (((int)f + (u - 3)*t) < ((int)f + (3 - u)*t));  // no warning
  return (0);
}

int checkLessThanNestedBinaryOpIntCompare3(void) {
  int res;
  int t= 1;
  int u= 2;
  int f= 3;
  res = (((int)f + (u - 3)*t) < ((int)f + (3 - u)*(t + u < t + u)));  // expected-warning {{comparison of identical expressions always evaluates to false}}
  return (0);
}

/* end LT with int */

/* end LT */


/* GT */

/* GT with float */

int checkGreaterThanFloatLiteralCompare1(void) {
  return (5.14F > 5.14F); // expected-warning {{comparison of identical expressions always evaluates to false}}
}

int checkGreaterThanFloatLiteralCompare2(void) {
  return (6.14F > 7.14F); // no warning
}

int checkGreaterThanFloatDeclCompare1(void) {
  float f = 7.1F;
  float g = 7.1F;

  return (f > g); // no warning
}

int checkGreaterThanFloatDeclCompare12(void) {
  float f = 7.1F;
  return (f > f); // expected-warning {{comparison of identical expressions always evaluates to false}}
}


int checkGreaterThanFloatDeclCompare3(void) {
  float f = 7.1F;
  return (f > 7.1F); // no warning
}

int checkGreaterThanFloatDeclCompare4(void) {
  float f = 7.1F;
  return (7.1F > f); // no warning
}

int checkGreaterThanFloatDeclCompare5(void) {
  float f = 7.1F;
  int t = 7;
  return (t > f); // no warning
}

int checkGreaterThanFloatDeclCompare6(void) {
  float f = 7.1F;
  int t = 7;
  return (f > t); // no warning
}

int checkGreaterThanCastFloatDeclCompare11(void) {
  float f = 7.1F;
  return ((int)f > (int)f); // expected-warning {{comparison of identical expressions always evaluates to false}}
}
int checkGreaterThanCastFloatDeclCompare12(void) {
  float f = 7.1F;
  return ((char)f > (int)f); // no warning
}
int checkGreaterThanBinaryOpFloatCompare1(void) {
  int res;
  float f= 3.14F;
  res = (f + 3.14F > f + 3.14F);  // no warning
  return (0);
}
int checkGreaterThanBinaryOpFloatCompare2(void) {
  float f = 7.1F;
  float g = 7.1F;
  return (f + 3.14F > g + 3.14F); // no warning
}
int checkGreaterThanBinaryOpFloatCompare3(void) {
  int res;
  float f= 3.14F;
  res = ((int)f + 3.14F > (int)f + 3.14F);  // no warning
  return (0);
}
int checkGreaterThanBinaryOpFloatCompare4(void) {
  int res;
  float f= 3.14F;
  res = ((int)f + 3.14F > (char)f + 3.14F);  // no warning
  return (0);
}

int checkGreaterThanNestedBinaryOpFloatCompare1(void) {
  int res;
  int t= 1;
  int u= 2;
  float f= 3.14F;
  res = (((int)f + (3.14F - u)*t) > ((int)f + (3.14F - u)*t));  // no warning
  return (0);
}

int checkGreaterThanNestedBinaryOpFloatCompare2(void) {
  int res;
  int t= 1;
  int u= 2;
  float f= 3.14F;
  res = (((int)f + (u - 3.14F)*t) > ((int)f + (3.14F - u)*t));  // no warning
  return (0);
}

int checkGreaterThanNestedBinaryOpFloatCompare3(void) {
  int res;
  int t= 1;
  int u= 2;
  float f= 3.14F;
  res = (((int)f + (u - 3.14F)*t) > ((int)f + (3.14F - u)*(f + t > f + t)));  // no warning
  return (0);
}

/*  end GT with float */

/*  GT with int */


int checkGreaterThanIntLiteralCompare1(void) {
  return (5 > 5); // expected-warning {{comparison of identical expressions always evaluates to false}}
}

int checkGreaterThanIntLiteralCompare2(void) {
  return (6 > 7); // no warning
}

int checkGreaterThanIntDeclCompare1(void) {
  int f = 7;
  int g = 7;

  return (f > g); // no warning
}

int checkGreaterThanIntDeclCompare3(void) {
  int f = 7;
  return (f > 7); // no warning
}

int checkGreaterThanIntDeclCompare4(void) {
  int f = 7;
  return (7 > f); // no warning
}

int checkGreaterThanCastIntDeclCompare11(void) {
  int f = 7;
  return ((int)f > (int)f); // expected-warning {{comparison of identical expressions always evaluates to false}}
}
int checkGreaterThanCastIntDeclCompare12(void) {
  int f = 7;
  return ((char)f > (int)f); // no warning
}
int checkGreaterThanBinaryOpIntCompare1(void) {
  int res;
  int f= 3;
  res = (f + 3 > f + 3);  // expected-warning {{comparison of identical expressions always evaluates to false}}
  return (0);
}
int checkGreaterThanBinaryOpIntCompare2(void) {
  int f = 7;
  int g = 7;
  return (f + 3 > g + 3); // no warning
}
int checkGreaterThanBinaryOpIntCompare3(void) {
  int res;
  int f= 3;
  res = ((int)f + 3 > (int)f + 3);  // expected-warning {{comparison of identical expressions always evaluates to false}}
  return (0);
}
int checkGreaterThanBinaryOpIntCompare4(void) {
  int res;
  int f= 3;
  res = ((int)f + 3 > (char)f + 3);  // no warning
  return (0);
}

int checkGreaterThanNestedBinaryOpIntCompare1(void) {
  int res;
  int t= 1;
  int u= 2;
  int f= 3;
  res = (((int)f + (3 - u)*t) > ((int)f + (3 - u)*t));  // expected-warning {{comparison of identical expressions always evaluates to false}}
  return (0);
}

int checkGreaterThanNestedBinaryOpIntCompare2(void) {
  int res;
  int t= 1;
  int u= 2;
  int f= 3;
  res = (((int)f + (u - 3)*t) > ((int)f + (3 - u)*t));  // no warning
  return (0);
}

int checkGreaterThanNestedBinaryOpIntCompare3(void) {
  int res;
  int t= 1;
  int u= 2;
  int f= 3;
  res = (((int)f + (u - 3)*t) > ((int)f + (3 - u)*(t + u > t + u)));  // expected-warning {{comparison of identical expressions always evaluates to false}}
  return (0);
}

/* end GT with int */

/* end GT */
