// RUN: %clang_cc1  -fsyntax-only -Wunused-value -verify %s

@interface INTF
// Currently this is rejected by both GCC and Clang (and Clang was crashing on it).
- (id) foo __attribute__((warn_unused_result)); // expected-warning{{warning: 'warn_unused_result' attribute only applies to function types}}
@end


