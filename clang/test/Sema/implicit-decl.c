// RUN: %clang_cc1 %s -verify -fsyntax-only

typedef int int32_t;
typedef unsigned char Boolean;

extern int printf(__const char *__restrict __format, ...); // expected-note{{'printf' declared here}}

void func() {
   int32_t *vector[16];
   const char compDesc[16 + 1];
   int32_t compCount = 0;
   if (_CFCalendarDecomposeAbsoluteTimeV(compDesc, vector, compCount)) { // expected-note {{previous implicit declaration is here}} \
         expected-warning {{implicit declaration of function '_CFCalendarDecomposeAbsoluteTimeV' is invalid in C99}}
   }

   printg("Hello, World!\n"); // expected-warning{{implicit declaration of function 'printg' is invalid in C99}} \
                              // expected-note{{did you mean 'printf'?}}

  __builtin_is_les(1, 3); // expected-error{{use of unknown builtin '__builtin_is_les'}} \
                          // expected-note{did you mean '__builtin_is_less'?}}
}
Boolean _CFCalendarDecomposeAbsoluteTimeV(const char *componentDesc, int32_t **vector, int32_t count) { // expected-error{{conflicting types for '_CFCalendarDecomposeAbsoluteTimeV'}}
 return 0;
}
