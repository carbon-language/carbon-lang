// RUN: clang-cc -triple i386-apple-darwin9 -fobjc-gc -fsyntax-only -verify %s
// This is really dangerous!  Disabling for now until we work out what's
// supposed to happen here.
// XFAIL
@interface INTF @end

extern INTF* p2;
extern __strong INTF* p2;

extern __strong id p1;
extern id p1;

extern id CFRunLoopGetMain();
extern __strong id CFRunLoopGetMain();

extern __strong INTF* p3;
extern id p3;

