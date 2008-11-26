// RUN clang -fsyntax-only -verify %s
@interface A 
@end

@interface B : A
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
