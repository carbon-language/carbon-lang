// RUN: clang -fsyntax-only -verify %s

@interface foo
@end

@implementation foo
@end

@interface bar
-(void) my_method:(foo) my_param; // expected-error {{can not use an object as parameter to a method}}
@end

@implementation bar
-(void) my_method:(foo) my_param  // expected-error {{can not use an object as parameter to a method}}
{
}
@end

