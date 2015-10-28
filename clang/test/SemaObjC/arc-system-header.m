// RUN: %clang_cc1 -fobjc-arc -isystem %S/Inputs %s -DNO_USE
// RUN: %clang_cc1 -fobjc-arc -isystem %S/Inputs %s -verify

#include <arc-system-header.h>

#ifndef NO_USE
void test(id op, void *cp) {
  cp = test0(op); // expected-error {{'test0' is unavailable: converts between Objective-C and C pointers in -fobjc-arc}}
  cp = *test1(&op); // expected-error {{'test1' is unavailable: converts between Objective-C and C pointers in -fobjc-arc}}
// expected-note@arc-system-header.h:1 {{'test0' has been explicitly marked unavailable here}}
// expected-note@arc-system-header.h:5 {{'test1' has been explicitly marked unavailable here}}
}

void test3(struct Test3 *p) {
  p->field = 0; // expected-error {{'field' is unavailable: this system declaration uses an unsupported type}}
                // expected-note@arc-system-header.h:14 {{unsupported declaration here}}
}

void test4(Test4 *p) {
  p->field1 = 0; // expected-error {{'field1' is unavailable: this system declaration uses an unsupported type}}
                 // expected-note@arc-system-header.h:19 {{unsupported declaration here}}
  p->field2 = 0;
}

void test5(struct Test5 *p) {
  p->field = 0; // expected-error {{'field' is unavailable: this system field has retaining ownership}}
                // expected-note@arc-system-header.h:25 {{'field' has been explicitly marked unavailable here}}
}

id test6() {
  // This is actually okay to use if declared in a system header.
  id x;
  x = (id) kMagicConstant;
  x = (id) (x ? kMagicConstant : kMagicConstant);
  x = (id) (x ? kMagicConstant : (void*) 0);

  extern void test6_helper();
  x = (id) (test6_helper(), kMagicConstant);
}

void test7(Test7 *p) {
  *p.prop = 0; // expected-error {{'prop' is unavailable: this system declaration uses an unsupported type}}
  p.prop = 0; // expected-error {{'prop' is unavailable: this system declaration uses an unsupported type}}
  *[p prop] = 0; // expected-error {{'prop' is unavailable: this system declaration uses an unsupported type}}
  [p setProp: 0]; // expected-error {{'setProp:' is unavailable: this system declaration uses an unsupported type}}
// expected-note@arc-system-header.h:41 4 {{unsupported declaration here}}
// expected-note@arc-system-header.h:41 2 {{property 'prop' is declared unavailable here}}
}
#endif

// test8 in header
