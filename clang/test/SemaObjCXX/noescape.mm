// RUN: %clang_cc1 -fsyntax-only -verify -fblocks -std=c++11 %s
// RUN: %clang_cc1 -fsyntax-only -verify -fblocks -std=c++1z %s

typedef void (^BlockTy)();

struct S {
  int i;
  void m();
};

void escapingFunc0(BlockTy);
void noescapeFunc0(id, __attribute__((noescape)) BlockTy);
void noescapeFunc1(id, [[clang::noescape]] BlockTy);
void noescapeFunc2(__attribute__((noescape)) int *); // expected-note {{previous declaration is here}}
void noescapeFunc3(__attribute__((noescape)) id);
void noescapeFunc4(__attribute__((noescape)) int &);
void noescapeFunc2(int *); // expected-error {{conflicting types for 'noescapeFunc2'}}

template <class T>
void noescapeFunc5(__attribute__((noescape)) T); // expected-warning {{'noescape' attribute only applies to pointer arguments}}
template <class T>
void noescapeFunc6(__attribute__((noescape)) const T &);
template <class T>
void noescapeFunc7(__attribute__((noescape)) T &&);

void invalidFunc0(int __attribute__((noescape))); // expected-warning {{'noescape' attribute only applies to pointer arguments}}
void invalidFunc1(int __attribute__((noescape(0)))); // expected-error {{'noescape' attribute takes no arguments}}
void invalidFunc2(int0 *__attribute__((noescape))); // expected-error {{use of undeclared identifier 'int0'; did you mean 'int'?}}
void invalidFunc3(__attribute__((noescape)) int (S::*Ty)); // expected-warning {{'noescape' attribute only applies to pointer arguments}}
void invalidFunc4(__attribute__((noescape)) void (S::*Ty)()); // expected-warning {{'noescape' attribute only applies to pointer arguments}}
int __attribute__((noescape)) g; // expected-warning {{'noescape' attribute only applies to parameters}}

struct S1 {
  virtual void m0(int *__attribute__((noescape))); // expected-note {{parameter of overridden method is annotated with __attribute__((noescape))}}
};

struct S2 : S1 {
  void m0(int *__attribute__((noescape))) override;
};

struct S3 : S1 {
  void m0(int *) override; // expected-warning {{parameter of overriding method should be annotated with __attribute__((noescape))}}
};

__attribute__((objc_root_class))
@interface C0
-(void) m0:(int*)__attribute__((noescape)) p; // expected-note {{parameter of overridden method is annotated with __attribute__((noescape))}}
- (void)noescapeLValRefParam:(const BlockTy &)__attribute__((noescape))p;
- (void)noescapeRValRefParam:(BlockTy &&)__attribute__((noescape))p;
@end

@implementation C0
-(void) m0:(int*)__attribute__((noescape)) p {}
- (void)noescapeLValRefParam:(const BlockTy &)__attribute__((noescape))p {
}
- (void)noescapeRValRefParam:(BlockTy &&)__attribute__((noescape))p {
}
@end

@interface C1 : C0
-(void) m0:(int*)__attribute__((noescape)) p;
@end

@implementation C1 : C0
-(void) m0:(int*)__attribute__((noescape)) p {}
@end

@interface C2 : C0
-(void) m0:(int*) p; // expected-warning {{parameter of overriding method should be annotated with __attribute__((noescape))}}
@end

@implementation C2 : C0
-(void) m0:(int*) p {}
@end

void func0(int *);
void (*fnptr0)(int *);
void (*fnptr1)(__attribute__((noescape)) int *);
template<void (*fn)(int*)> struct S4 {};
template<void (*fn)(int* __attribute__((noescape)))> struct S5 {};

#if __cplusplus < 201406
  // expected-note@-4 {{template parameter is declared here}}
  // expected-note@-4 {{template parameter is declared here}}
#endif

void test0() {
  fnptr0 = &func0;
  fnptr0 = &noescapeFunc2;
  fnptr1 = &func0; // expected-error {{incompatible function pointer types assigning to 'void (*)(__attribute__((noescape)) int *)' from 'void (*)(int *)'}}
  fnptr1 = &noescapeFunc2;
  S4<&func0> e0;
  S4<&noescapeFunc2> e1;
  S5<&func0> ne0;

#if __cplusplus < 201406
  // expected-error@-4 {{non-type template argument of type 'void (*)(__attribute__((noescape)) int *)' cannot be converted to a value of type 'void (*)(int *)'}}
  // expected-error@-4 {{non-type template argument of type 'void (*)(int *)' cannot be converted to a value of type 'void (*)(__attribute__((noescape)) int *)'}}
#else
  // expected-error@-6 {{value of type 'void (*)(int *)' is not implicitly convertible to 'void (*)(__attribute__((noescape)) int *)'}}
#endif

  S5<&noescapeFunc2> ne1;
}

@protocol NoescapeProt
-(void) m0:(int*)__attribute__((noescape)) p; // expected-note 2 {{parameter of overridden method is annotated with __attribute__((noescape))}}
+(void) m1:(int*)__attribute__((noescape)) p;
-(void) m1:(int*) p;
@end

__attribute__((objc_root_class))
@interface C3
-(void) m0:(int*) p;
+(void) m1:(int*)__attribute__((noescape)) p;
-(void) m1:(int*) p;
@end

@interface C3 () <NoescapeProt> // expected-note {{class extension conforms to protocol 'NoescapeProt' which defines method 'm0:'}}
@end

@implementation C3
-(void) m0:(int*) p { // expected-warning {{parameter of overriding method should be annotated with __attribute__((noescape))}}
}
+(void) m1:(int*)__attribute__((noescape)) p {
}
-(void) m1:(int*) p {
}
@end

__attribute__((objc_root_class))
@interface C4 <NoescapeProt>
-(void) m0:(int*) p; // expected-warning {{parameter of overriding method should be annotated with __attribute__((noescape))}}
@end

@implementation C4
-(void) m0:(int*) p {
}
+(void) m1:(int*)__attribute__((noescape)) p {
}
-(void) m1:(int*) p {
}
@end

struct S6 {
  S6();
  S6(const S6 &) = delete; // expected-note 12 {{'S6' has been explicitly marked deleted here}}
  int f;
};

void test1(C0 *c0) {
  id a;
  // __block variables that are not captured by escaping blocks don't
  // necessitate having accessible copy constructors.
  __block S6 b0;
  __block S6 b1; // expected-error {{call to deleted constructor of 'S6'}}
  __block S6 b2; // expected-error {{call to deleted constructor of 'S6'}}
  __block S6 b3; // expected-error {{call to deleted constructor of 'S6'}}
  __block S6 b4; // expected-error {{call to deleted constructor of 'S6'}}
  __block S6 b5; // expected-error {{call to deleted constructor of 'S6'}}
  __block S6 b6; // expected-error {{call to deleted constructor of 'S6'}}
  __block S6 b7; // expected-error {{call to deleted constructor of 'S6'}}
  __block S6 b8; // expected-error {{call to deleted constructor of 'S6'}}
  __block S6 b9; // expected-error {{call to deleted constructor of 'S6'}}
  __block S6 b10; // expected-error {{call to deleted constructor of 'S6'}}
  __block S6 b11; // expected-error {{call to deleted constructor of 'S6'}}
  __block S6 b12;
  __block S6 b13;
  __block S6 b14; // expected-error {{call to deleted constructor of 'S6'}}

  noescapeFunc0(a, ^{ (void)b0; });
  escapingFunc0(^{ (void)b1; });
  {
    noescapeFunc0(a, ^{ (void)b0; (void)b1; });
  }
  noescapeFunc0(a, ^{ escapingFunc0(^{ (void)b2; }); });
  escapingFunc0(^{ noescapeFunc0(a, ^{ (void)b3; }); });

  [c0 noescapeLValRefParam:^{
    (void)b4;
  }];

  [c0 noescapeRValRefParam:^{
    (void)b5;
  }];

  void noescape_id(__attribute__((noescape)) id);
  noescape_id(^{
    (void)b6;
  });

  void noescapeLValRefParam(__attribute__((noescape)) const BlockTy &);
  noescapeLValRefParam(^{
    (void)b7;
  });

  void noescapeRValRefParam(__attribute__((noescape)) BlockTy &&);
  noescapeRValRefParam(^{
    (void)b8;
  });

  // FIXME: clang shouldn't reject this.
  noescapeFunc5(^{
    (void)b9;
  });

  noescapeFunc6(^{
    (void)b10;
  });

  noescapeFunc7(^{
    (void)b11;
  });

  struct NoescapeCtor {
    NoescapeCtor(__attribute__((noescape)) void (^)());
  };
  struct EscapeCtor {
    EscapeCtor(void (^)());
  };

  void helper1(NoescapeCtor a);
  helper1(^{
    (void)b12;
  });

  void helper2(NoescapeCtor && a);
  helper2(^{
    (void)b13;
  });

  void helper3(__attribute__((noescape)) EscapeCtor && a);
  helper3(^{
    (void)b14;
  });
}
