// RUN: clang-cc -triple i386-apple-darwin9 -fobjc-gc -fsyntax-only -verify %s
@interface INTF @end

extern INTF* p2;
extern __strong INTF* p2;

extern __strong id p1;
extern id p1;

extern id CFRunLoopGetMain();
extern __strong id CFRunLoopGetMain();

extern __weak id WLoopGetMain(); // expected-note {{previous declaration is here}}
extern id WLoopGetMain();	// expected-error {{conflicting types for 'WLoopGetMain'}}

extern id p3;	// expected-note {{previous definition is here}}
extern __weak id p3;	// expected-error {{redefinition of 'p3' with a different type}}

