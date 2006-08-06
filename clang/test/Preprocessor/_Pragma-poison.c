// RUN: clang -Eonly %s 2>&1 | grep error | wc -l | grep 1 &&
// RUN: clang -Eonly %s 2>&1 | grep 7:4 | wc -l | grep 1

#define BAR _Pragma ("GCC poison XYZW")  XYZW /*NO ERROR*/
XYZW   // NO ERROR
BAR
   XYZW   // ERROR

