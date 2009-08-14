// RUN: clang-cc -verify -fsyntax-only -fblocks %s

@interface TestAttrMallocOnMethods {}
- (id) test1 __attribute((malloc)); // expected-warning{{'malloc' attribute only applies to function types}}
- (int) test2 __attribute((malloc)); // expected-warning{{'malloc' attribute only applies to function types}}
@end

id bar(void) __attribute((malloc)); // no-warning

typedef void (^bptr)(void);
bptr baz(void) __attribute((malloc)); // no-warning

__attribute((malloc)) id (*f)(); // no-warning
__attribute((malloc)) bptr (*g)(); // no-warning

