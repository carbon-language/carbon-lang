// RUN: clang-cc %s -verify -fsyntax-only

typedef int int32_t;
typedef unsigned char Boolean;

void func() {
   int32_t *vector[16];
   const char compDesc[16 + 1];
   int32_t compCount = 0;
   if (_CFCalendarDecomposeAbsoluteTimeV(compDesc, vector, compCount)) { // expected-note {{previous implicit declaration is here}}
   }
   return ((void *)0); // expected-warning {{void function 'func' should not return a value}}
}
Boolean _CFCalendarDecomposeAbsoluteTimeV(const char *componentDesc, int32_t **vector, int32_t count) { // expected-error{{conflicting types for '_CFCalendarDecomposeAbsoluteTimeV'}}
 return 0;
}

