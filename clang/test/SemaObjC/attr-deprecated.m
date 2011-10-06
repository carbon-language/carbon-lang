// RUN: %clang_cc1 %s -fsyntax-only -verify

@interface A {
  int X __attribute__((deprecated));
}
+ (void)F __attribute__((deprecated));
- (void)f __attribute__((deprecated));
@end

@implementation A
+ (void)F __attribute__((deprecated))
{	// expected-warning {{method attribute can only be specified on method declarations}}
  [self F]; // no warning, since the caller is also deprecated.
}

- (void)g
{
  X++;        // expected-warning{{'X' is deprecated}}
  self->X++;  // expected-warning{{'X' is deprecated}}
  [self f]; // expected-warning{{'f' is deprecated}}
}

- (void)f
{
  [self f]; // no warning, the caller is deprecated in its interface.
}
@end

@interface B: A
@end
  
@implementation B
+ (void)G
{
  [super F]; // expected-warning{{'F' is deprecated}}
}

- (void)g
{
  [super f]; // // expected-warning{{'f' is deprecated}}
}
@end

@protocol P
- (void)p __attribute__((deprecated));
@end

void t1(A *a)
{
  [A F]; // expected-warning{{'F' is deprecated}}
  [a f]; // expected-warning{{'f' is deprecated}}
}

void t2(id a)
{
  [a f];
}

void t3(A<P>* a)
{
  [a f]; // expected-warning{{'f' is deprecated}}
  [a p]; // expected-warning{{'p' is deprecated}}
} 

void t4(Class c)
{
  [c F];
}



@interface Bar 

@property (assign, setter = MySetter:) int FooBar __attribute__ ((deprecated));
- (void) MySetter : (int) value;
@end

int t5() {
  Bar *f;
  f.FooBar = 1;	   // expected-warning {{warning: 'FooBar' is deprecated}}
  return f.FooBar; // expected-warning {{warning: 'FooBar' is deprecated}}
}


__attribute ((deprecated))  
@interface DEPRECATED {
  @public int ivar; 
  DEPRECATED *ivar2; // no warning.
} 
- (int) instancemethod;
- (DEPRECATED *) meth; // no warning.
@property  int prop; 
@end

@interface DEPRECATED (Category) // no warning.
- (DEPRECATED *) meth2; // no warning.
@end

@interface DEPRECATED (Category2) // no warning.
@end

@implementation DEPRECATED (Category2) // expected-warning {{warning: 'DEPRECATED' is deprecated}}
@end

@interface NS : DEPRECATED  // expected-warning {{warning: 'DEPRECATED' is deprecated}}
@end


@interface Test2
@property int test2 __attribute__((deprecated));
@end

void test(Test2 *foo) {
  int x;
  x = foo.test2; // expected-warning {{'test2' is deprecated}}
  x = [foo test2]; // expected-warning {{'test2' is deprecated}}
  foo.test2 = x; // expected-warning {{'test2' is deprecated}}
  [foo setTest2: x]; // expected-warning {{'setTest2:' is deprecated}}
}

__attribute__((deprecated))
@interface A(Blah) // expected-error{{attributes may not be specified on a category}}
@end
