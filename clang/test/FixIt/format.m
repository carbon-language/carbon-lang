// RUN: %clang_cc1 -fsyntax-only -fblocks -verify %s
// RUN: %clang_cc1 -fdiagnostics-parseable-fixits -fblocks %s 2>&1 | FileCheck %s

@class NSString;
extern void NSLog(NSString *, ...);
int printf(const char * restrict, ...) ;

void test_integer_correction (int x) {
  printf("%d", x); // no-warning
  printf("%s", x); // expected-warning{{format specifies type 'char *' but the argument has type 'int'}}
  printf("%lf", x); // expected-warning{{format specifies type 'double' but the argument has type 'int'}}
  // CHECK: fix-it:"{{.*}}":{10:11-10:13}:"%d"
  // CHECK: fix-it:"{{.*}}":{11:11-11:14}:"%d"

  NSLog(@"%d", x); // no-warning
  NSLog(@"%s", x); // expected-warning{{format specifies type 'char *' but the argument has type 'int'}}
  NSLog(@"%lf", x); // expected-warning{{format specifies type 'double' but the argument has type 'int'}}
  NSLog(@"%@", x); // expected-warning{{format specifies type 'id' but the argument has type 'int'}}
  // CHECK: fix-it:"{{.*}}":{16:11-16:13}:"%d"
  // CHECK: fix-it:"{{.*}}":{17:11-17:14}:"%d"
  // CHECK: fix-it:"{{.*}}":{18:11-18:13}:"%d"
}

void test_string_correction (char *x) {
  printf("%d", x); // expected-warning{{format specifies type 'int' but the argument has type 'char *'}}
  printf("%s", x); // no-warning
  printf("%lf", x); // expected-warning{{format specifies type 'double' but the argument has type 'char *'}}
  // CHECK: fix-it:"{{.*}}":{25:11-25:13}:"%s"
  // CHECK: fix-it:"{{.*}}":{27:11-27:14}:"%s"

  NSLog(@"%d", x); // expected-warning{{format specifies type 'int' but the argument has type 'char *'}}
  NSLog(@"%s", x); // no-warning
  NSLog(@"%lf", x); // expected-warning{{format specifies type 'double' but the argument has type 'char *'}}
  NSLog(@"%@", x); // expected-warning{{format specifies type 'id' but the argument has type 'char *'}}
  // CHECK: fix-it:"{{.*}}":{31:11-31:13}:"%s"
  // CHECK: fix-it:"{{.*}}":{33:11-33:14}:"%s"
  // CHECK: fix-it:"{{.*}}":{34:11-34:13}:"%s"
}

void test_object_correction (id x) {  
  NSLog(@"%d", x); // expected-warning{{format specifies type 'int' but the argument has type 'id'}}
  NSLog(@"%s", x); // expected-warning{{format specifies type 'char *' but the argument has type 'id'}}
  NSLog(@"%lf", x); // expected-warning{{format specifies type 'double' but the argument has type 'id'}}
  NSLog(@"%@", x); // no-warning
  // CHECK: fix-it:"{{.*}}":{41:11-41:13}:"%@"
  // CHECK: fix-it:"{{.*}}":{42:11-42:13}:"%@"
  // CHECK: fix-it:"{{.*}}":{43:11-43:14}:"%@"
}

typedef const struct __CFString * __attribute__((NSObject)) CFStringRef;
void test_cf_object_correction (CFStringRef x) {
  NSLog(@"%d", x); // expected-warning{{format specifies type 'int' but the argument has type 'CFStringRef'}}
  NSLog(@"%s", x); // expected-warning{{format specifies type 'char *' but the argument has type 'CFStringRef'}}
  NSLog(@"%lf", x); // expected-warning{{format specifies type 'double' but the argument has type 'CFStringRef'}}
  NSLog(@"%@", x); // no-warning
  // CHECK: fix-it:"{{.*}}":{52:11-52:13}:"%@"
  // CHECK: fix-it:"{{.*}}":{53:11-53:13}:"%@"
  // CHECK: fix-it:"{{.*}}":{54:11-54:14}:"%@"
}

typedef void (^block_t)(void);
void test_block_correction (block_t x) {
  NSLog(@"%d", x); // expected-warning{{format specifies type 'int' but the argument has type 'block_t'}}
  NSLog(@"%s", x); // expected-warning{{format specifies type 'char *' but the argument has type 'block_t'}}
  NSLog(@"%lf", x); // expected-warning{{format specifies type 'double' but the argument has type 'block_t'}}
  NSLog(@"%@", x); // no-warning
  // CHECK: fix-it:"{{.*}}":{63:11-63:13}:"%@"
  // CHECK: fix-it:"{{.*}}":{64:11-64:13}:"%@"
  // CHECK: fix-it:"{{.*}}":{65:11-65:14}:"%@"
}
