struct S {
  int i;
  short s1, s2;
};

struct S func_returning_struct(void);

void loop(void) {
  func_returning_struct();
}
