// RUN: %clang_cc1 -verify -fsyntax-only -fblocks %s

@interface TestAttrMallocOnMethods {}
- (id) test1 __attribute((malloc)); //  expected-warning {{attribute only applies to functions}}
- (int) test2 __attribute((malloc)); //  expected-warning {{attribute only applies to functions}}
@end

id bar(void) __attribute((malloc)); // no-warning

typedef void (^bptr)(void);
bptr baz(void) __attribute((malloc)); // no-warning

__attribute((malloc)) id (*f)(); //  expected-warning {{attribute only applies to functions}}
__attribute((malloc)) bptr (*g)(); //  expected-warning {{attribute only applies to functions}}
__attribute((malloc)) void *(^h)(); //  expected-warning {{attribute only applies to functions}}

