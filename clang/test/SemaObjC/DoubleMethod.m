// RUN: clang-cc -fsyntax-only -verify %s

@interface Subclass
{
    int ivar;
}

- (void) method;
- (void) method;
@end

@implementation Subclass
- (void) method {;} // expected-note {{previous declaration is here}}
- (void) method {;} // expected-error {{duplicate declaration of method 'method'}}
@end

int main (void) {
    return 0;
}
