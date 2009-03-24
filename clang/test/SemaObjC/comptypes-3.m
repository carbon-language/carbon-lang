// RUN: clang-cc -fsyntax-only -verify %s

#define nil (void *)0;

extern void foo();

@protocol MyProtocolA
- (void) methodA;
@end

@protocol MyProtocolB
- (void) methodB;
@end

@protocol MyProtocolAB <MyProtocolA, MyProtocolB>
@end

@protocol MyProtocolAC <MyProtocolA>
- (void) methodC;
@end

int main()
{
  id<MyProtocolA> obj_a = nil;
  id<MyProtocolB> obj_b = nil;
  id<MyProtocolAB> obj_ab = nil;
  id<MyProtocolAC> obj_ac = nil;

  obj_a = obj_b;  // expected-warning {{incompatible type assigning 'id<MyProtocolB>', expected 'id<MyProtocolA>'}}
  obj_a = obj_ab; /* Ok */
  obj_a = obj_ac; /* Ok */
  
  obj_b = obj_a;  // expected-warning {{incompatible type assigning 'id<MyProtocolA>', expected 'id<MyProtocolB>'}}
  obj_b = obj_ab; /* Ok */
  obj_b = obj_ac; // expected-warning {{incompatible type assigning 'id<MyProtocolAC>', expected 'id<MyProtocolB>'}}
  
  obj_ab = obj_a;  // expected-warning {{incompatible type assigning 'id<MyProtocolA>', expected 'id<MyProtocolAB>'}}
  obj_ab = obj_b;  // expected-warning {{incompatible type assigning 'id<MyProtocolB>', expected 'id<MyProtocolAB>'}}
  obj_ab = obj_ac; // expected-warning {{incompatible type assigning 'id<MyProtocolAC>', expected 'id<MyProtocolAB>'}}
  
  obj_ac = obj_a;  // expected-warning {{incompatible type assigning 'id<MyProtocolA>', expected 'id<MyProtocolAC>'}}
  obj_ac = obj_b;  // expected-warning {{incompatible type assigning 'id<MyProtocolB>', expected 'id<MyProtocolAC>'}}
  obj_ac = obj_ab; // expected-warning {{incompatible type assigning 'id<MyProtocolAB>', expected 'id<MyProtocolAC>'}}

  if (obj_a == obj_b) foo (); // expected-warning {{invalid operands to binary expression ('id<MyProtocolA>' and 'id<MyProtocolB>')}}
  if (obj_b == obj_a) foo (); // expected-warning {{invalid operands to binary expression ('id<MyProtocolB>' and 'id<MyProtocolA>')}}

  if (obj_a == obj_ab) foo (); /* Ok */
  if (obj_ab == obj_a) foo (); /* Ok */ 

  if (obj_a == obj_ac) foo (); /* Ok */ 
  if (obj_ac == obj_a) foo (); /* Ok */ 

  if (obj_b == obj_ab) foo (); /* Ok */ 
  if (obj_ab == obj_b) foo (); /* Ok */ 

  if (obj_b == obj_ac) foo (); // expected-warning {{invalid operands to binary expression ('id<MyProtocolB>' and 'id<MyProtocolAC>')}} 
  if (obj_ac == obj_b) foo (); // expected-warning {{invalid operands to binary expression ('id<MyProtocolAC>' and 'id<MyProtocolB>')}} 

  if (obj_ab == obj_ac) foo (); // expected-warning {{invalid operands to binary expression ('id<MyProtocolAB>' and 'id<MyProtocolAC>')}} 
  if (obj_ac == obj_ab) foo (); // expected-warning {{invalid operands to binary expression ('id<MyProtocolAC>' and 'id<MyProtocolAB>')}} 

  return 0;
}
