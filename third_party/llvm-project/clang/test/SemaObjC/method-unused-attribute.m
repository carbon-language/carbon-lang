// RUN: %clang_cc1  -fsyntax-only -Wunused-parameter -verify -Wno-objc-root-class %s

// -Wunused-parameter ignores ObjC method parameters that are unused.

// expected-no-diagnostics

@interface INTF
- (void) correct_use_of_unused: (void *) notice : (id)another_arg;
- (void) will_warn_unused_arg: (void *) notice : (id)warn_unused;
- (void) unused_attr_on_decl_ignored: (void *)  __attribute__((unused)) will_warn;
@end

@implementation INTF
- (void) correct_use_of_unused: (void *)  __attribute__((unused)) notice : (id) __attribute__((unused)) newarg{
}
- (void) will_warn_unused_arg: (void *) __attribute__((unused))  notice : (id)warn_unused {}
- (void) unused_attr_on_decl_ignored: (void *)  will_warn{}
@end

