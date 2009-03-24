// RUN: clang-cc -fsyntax-only -verify %s

@class SUPER, Y;

@interface INTF :SUPER  // expected-error {{cannot find interface declaration for 'SUPER', superclass of 'INTF'}}
@end

@interface SUPER @end

@interface INTF1 : SUPER  // expected-note {{previous definition is here}}
@end

@interface INTF2 : INTF1
@end

@interface INTF3 : Y // expected-error {{cannot find interface declaration for 'Y', superclass of 'INTF3'}}
@end

@interface INTF1  // expected-error {{duplicate interface definition for class 'INTF1'}}
@end

@implementation SUPER
- (void)dealloc {
    [super dealloc]; // expected-error {{no super class declared in @interface for 'SUPER'}}
}
@end
