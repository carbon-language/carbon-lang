// RUN: clang-cc %s -fsyntax-only -verify

@interface A {
  int X __attribute__((deprecated));
}
+ (void)F __attribute__((deprecated));
- (void)f __attribute__((deprecated));
@end

@implementation A
+ (void)F __attribute__((deprecated))
{
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
} 
- (int) instancemethod;
@property  int prop; 
@end

@interface DEPRECATED (Category) // expected-warning {{warning: 'DEPRECATED' is deprecated}}
@end

@interface NS : DEPRECATED  // expected-warning {{warning: 'DEPRECATED' is deprecated}}
@end


