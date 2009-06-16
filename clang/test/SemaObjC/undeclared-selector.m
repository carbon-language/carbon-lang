// RUN: clang-cc  -fsyntax-only -Wundeclared-selector -verify %s

typedef struct objc_selector *SEL;

@interface MyClass

+ (void) methodA;
- (void) methodB;
+ (void) methodD;
- (void) methodF;

@end

@implementation MyClass

+ (void) methodA {}
- (void) methodB {}
+ (void) methodD
{
  SEL d = @selector(methodD); /* Ok */
  SEL e = @selector(methodE);  // expected-warning {{undeclared selector 'methodE'}}
}

- (void) methodE
{
  SEL e = @selector(methodE); /* Ok */
}

- (void) methodF
{
  SEL e = @selector(methodE); /* Ok */
}

@end

int main (void)
{
  SEL a = @selector(methodA); /* Ok */
  SEL b = @selector(methodB); /* Ok */
  SEL c = @selector(methodC);  // expected-warning {{undeclared selector 'methodC'}}
  SEL d = @selector(methodD); /* Ok */
  SEL e = @selector(methodE); /* Ok */
  return 0;
  
}
