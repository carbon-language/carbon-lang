// RUN: clang -fsyntax-only -verify %s

typedef struct { int y; } Abstract;

typedef struct { int x; } Alternate;

#define INTERFERE_TYPE Alternate*

@protocol A
@property Abstract *x; // expected-warning{{using}}
@end

@interface B
@property Abstract *y; // expected-warning{{using}}
@end

@interface B (Category)
@property Abstract *z; // expected-warning{{using}}
@end

@interface InterferencePre
-(void) x; // expected-warning{{also found}}
-(void) y; // expected-warning{{also found}}
-(void) z; // expected-warning{{also found}}
-(void) setX: (INTERFERE_TYPE) arg; // expected-warning{{also found}}
-(void) setY: (INTERFERE_TYPE) arg; // expected-warning{{also found}}
-(void) setZ: (INTERFERE_TYPE) arg; // expected-warning{{also found}}
@end

void f0(id a0) {
  Abstract *l = [a0 x]; // expected-warning {{multiple methods named 'x' found}} 
}

void f1(id a0) {
  Abstract *l = [a0 y]; // expected-warning {{multiple methods named 'y' found}}
}

void f2(id a0) {
  Abstract *l = [a0 z]; // expected-warning {{multiple methods named 'z' found}}
}

void f3(id a0, Abstract *a1) { 
  [ a0 setX: a1]; // expected-warning {{multiple methods named 'setX:' found}}
}

void f4(id a0, Abstract *a1) { 
  [ a0 setY: a1]; // expected-warning {{multiple methods named 'setY:' found}}
}

void f5(id a0, Abstract *a1) { 
  [ a0 setZ: a1]; // expected-warning {{multiple methods named 'setZ:' found}}
}
