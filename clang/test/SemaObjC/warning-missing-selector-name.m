// RUN: %clang_cc1  -fsyntax-only -verify -Wno-objc-root-class %s
// RUN: %clang_cc1 -x objective-c++ -fsyntax-only -verify -Wno-objc-root-class -Wmissing-argument-name-in-selector %s
// rdar://12263549

@interface Super @end
@interface INTF : Super
-(void) Name1:(id)Arg1 Name2:(id)Arg2; // Name1:Name2:
-(void) Name1:(id) Name2:(id)Arg2;
-(void) Name1:(id)Arg1 Name2:(id)Arg2 Name3:(id)Arg3; // Name1:Name2:Name3:
-(void) Name1:(id)Arg1 Name2:(id) Name3:(id)Arg3;
@end

@implementation INTF
-(void) Name1:(id)Arg1 Name2:(id)Arg2{}
-(void) Name1:(id) Name2:(id)Arg2 {} // expected-warning {{parameter name used as selector may result in incomplete method selector name}} \
  				     // expected-note {{did you mean to use Name1:Name2: as the selector name instead of Name1::}}
-(void) Name1:(id)Arg1 Name2:(id)Arg2 Name3:(id)Arg3 {}
-(void) Name1:(id)Arg1 Name2:(id) Name3:(id)Arg3 {} // expected-warning {{parameter name used as selector may result in incomplete method selector name}} \
						    // expected-note {{did you mean to use Name1:Name2:Name3: as the selector name instead of Name1:Name2::}}
@end
