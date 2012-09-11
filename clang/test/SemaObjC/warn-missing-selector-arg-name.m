// RUN: %clang_cc1  -fsyntax-only -verify -Wno-objc-root-class -Wmissing-argument-name-in-selector %s
// RUN: %clang_cc1 -x objective-c++ -fsyntax-only -verify -Wno-objc-root-class -Wmissing-argument-name-in-selector %s
// rdar://12263549

@interface Super @end
@interface INTF : Super
-(void) Name1:(id)Arg1 Name2:(id)Arg2; // Name1:Name2:
-(void) Name1:(id) Name2:(id)Arg2; // expected-warning {{no parameter name in the middle of a selector may result in incomplete selector name}} \
                                   // expected-note {{did you mean Name1:: as the selector name}}
-(void) Name1:(id)Arg1 Name2:(id)Arg2 Name3:(id)Arg3; // Name1:Name2:Name3:
-(void) Name1:(id)Arg1 Name2:(id) Name3:(id)Arg3; // expected-warning {{no parameter name in the middle of a selector may result in incomplete selector name}} \
                                                  // expected-note {{did you mean Name1:Name2:: as the selector name}}
@end

@implementation INTF
-(void) Name1:(id)Arg1 Name2:(id)Arg2{}
-(void) Name1:(id) Name2:(id)Arg2 {} // expected-warning {{no parameter name in the middle of a selector may result in incomplete selector name}} \
  				     // expected-note {{did you mean Name1:: as the selector name}}
-(void) Name1:(id)Arg1 Name2:(id)Arg2 Name3:(id)Arg3 {}
-(void) Name1:(id)Arg1 Name2:(id) Name3:(id)Arg3 {} // expected-warning {{no parameter name in the middle of a selector may result in incomplete selector name}} \
						    // expected-note {{did you mean Name1:Name2:: as the selector name}}
@end
