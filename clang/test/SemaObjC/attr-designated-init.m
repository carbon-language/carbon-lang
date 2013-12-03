// RUN: %clang_cc1 -fsyntax-only -verify %s

#define NS_DESIGNATED_INITIALIZER __attribute__((objc_designated_initializer))

void fnfoo(void) NS_DESIGNATED_INITIALIZER; // expected-error {{only applies to methods}}

@protocol P1
-(id)init NS_DESIGNATED_INITIALIZER; // expected-error {{only applies to methods of interface declarations}}
@end

__attribute__((objc_root_class))
@interface I1
-(void)meth NS_DESIGNATED_INITIALIZER; // expected-error {{only applies to methods of the init family}}
-(id)init NS_DESIGNATED_INITIALIZER;
+(id)init NS_DESIGNATED_INITIALIZER; // expected-error {{only applies to methods of the init family}}
@end

@interface I1(cat)
-(id)init2 NS_DESIGNATED_INITIALIZER; // expected-error {{only applies to methods of interface declarations}}
@end

@interface I1()
-(id)init3 NS_DESIGNATED_INITIALIZER; // expected-error {{only applies to methods of interface declarations}}
@end

@implementation I1
-(void)meth {}
-(id)init NS_DESIGNATED_INITIALIZER { return 0; } // expected-error {{only applies to methods of interface declarations}}
+(id)init { return 0; }
-(id)init3 { return 0; }
-(id)init4 NS_DESIGNATED_INITIALIZER { return 0; } // expected-error {{only applies to methods of interface declarations}}
@end
