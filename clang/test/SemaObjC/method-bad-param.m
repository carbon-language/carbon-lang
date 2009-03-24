// RUN: clang-cc -fsyntax-only -verify %s

@interface foo
@end

@implementation foo
@end

@interface bar
-(void) my_method:(foo) my_param; // expected-error {{Objective-C type cannot be passed by value}}
- (foo)cccccc:(long)ddddd;  // expected-error {{Objective-C type cannot be returned by value}}
@end

@implementation bar
-(void) my_method:(foo) my_param  // expected-error {{Objective-C type cannot be passed by value}}
{
}
- (foo)cccccc:(long)ddddd // expected-error {{Objective-C type cannot be returned by value}}
{
}
@end

