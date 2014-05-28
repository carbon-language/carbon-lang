// RUN: %clang_cc1  -fsyntax-only -verify -Wno-objc-root-class %s
// rdar://10414277

@protocol P
void p_foo() {} // expected-warning {{function definition inside an Objective-C container is deprecated}}
@end

@interface I
void foo() {} // expected-warning {{function definition inside an Objective-C container is deprecated}}
inline void v_foo() {} // expected-warning {{function definition inside an Objective-C container is deprecated}}
static int s_foo() {return 0; } // expected-warning {{function definition inside an Objective-C container is deprecated}}
static inline int si_val() { return 1; } // expected-warning {{function definition inside an Objective-C container is deprecated}}
@end

@interface I(CAT)
void cat_foo() {} // expected-warning {{function definition inside an Objective-C container is deprecated}}
@end

@implementation I
inline void v_imp_foo() {} 
@end

@implementation I(CAT)
void cat_imp_foo() {} 
@end
