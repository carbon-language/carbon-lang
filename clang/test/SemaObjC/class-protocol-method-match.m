// RUN: %clang_cc1  -fsyntax-only -verify %s
// rdar://9352731

@protocol Bar 
@required
- (unsigned char) baz; // expected-note {{previous definition is here}}
- (char) ok;
- (void) also_ok;
@end

@protocol Bar1 
@required
- (unsigned char) baz;
- (unsigned char) also_ok;
- (void) ban : (int) arg, ...; // expected-note {{previous declaration is here}}
@end

@protocol Baz <Bar, Bar1>
- (void) bar : (unsigned char)arg; // expected-note {{previous definition is here}}
- (void) ok;
- (char) bak; // expected-note {{previous definition is here}}
@end

@interface Foo <Baz>
- (void) baz;  // expected-warning {{conflicting return type in declaration of 'baz': 'unsigned char' vs 'void'}}
- (void) bar : (unsigned char*)arg; // expected-warning {{conflicting parameter types in declaration of 'bar:': 'unsigned char' vs 'unsigned char *'}}
- (void) ok;
- (void) also_ok;
- (void) still_ok;
- (void) ban : (int) arg; // expected-warning {{conflicting variadic declaration of method and its declaration}}
@end

@interface Foo()
- (void) bak; // expected-warning {{conflicting return type in declaration of 'bak': 'char' vs 'void'}}
@end

@implementation Foo
- (void) baz {}
- (void) bar : (unsigned char*)arg {}
- (void) ok {}
- (void) also_ok {}
- (void) still_ok {}
- (void) ban : (int) arg {}
- (void) bak {}
@end

