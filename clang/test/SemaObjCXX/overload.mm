// RUN: clang -fsyntax-only -verify %s
@interface Foo
@end

@implementation Foo

void func(id);

+ zone {
 func(self);
 return self;
}

@protocol P0
@end

@protocol P1
@end

@interface A <P0>
@end

@interface B : A
@end

@interface C <P1>
@end

int& f(A*);
float& f(B*);
void g(A*);

int& h(A*);
float& h(id);

void test(A* a, B* b, id val) {
  int& i1 = f(a);
  float& f1 = f(b);
  float& f2 = f(val);
  g(a);
  g(b);
  g(val);
  int& i2 = h(a);
  float& f3 = h(val);
  //  int& i3 = h(b); FIXME: we match GCC here, but shouldn't this work?
}

int& cv(A*);
float& cv(const A*);
int& cv2(void*);
float& cv2(const void*);

void cv_test(A* a, B* b, const A* ac, const B* bc) {
  int &i1 = cv(a);
  int &i2 = cv(b);
  float &f1 = cv(ac);
  float &f2 = cv(bc);
  int& i3 = cv2(a);
  float& f3 = cv2(ac);
}


int& qualid(id<P0>);
float& qualid(id<P1>); // FIXME: GCC complains that this isn't an overload. Is it?

void qualid_test(A *a, B *b, C *c) {
  int& i1 = qualid(a);
  int& i2 = qualid(b);
  float& f1 = qualid(c);
}
