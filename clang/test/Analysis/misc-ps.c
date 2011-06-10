// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -analyze -disable-free -analyzer-eagerly-assume -analyzer-checker=core -analyzer-checker=deadcode -verify %s

unsigned long strlen(const char *);

int size_rdar9373039 = 1;
int rdar9373039() {
  int x;
  int j = 0;

  for (int i = 0 ; i < size_rdar9373039 ; ++i)
    x = 1;
    
  // strlen doesn't invalidate the value of 'size_rdar9373039'.
  int extra = (2 + strlen ("Clang") + ((4 - ((unsigned int) (2 + strlen ("Clang")) % 4)) % 4)) + (2 + strlen ("1.0") + ((4 - ((unsigned int) (2 + strlen ("1.0")) % 4)) % 4));

  for (int i = 0 ; i < size_rdar9373039 ; ++i)
    j += x; // no-warning

  return j;
}

int foo_rdar9373039(const char *);

int rdar93730392() {
  int x;
  int j = 0;

  for (int i = 0 ; i < size_rdar9373039 ; ++i)
    x = 1;
    
  int extra = (2 + foo_rdar9373039 ("Clang") + ((4 - ((unsigned int) (2 + foo_rdar9373039 ("Clang")) % 4)) % 4)) + (2 + foo_rdar9373039 ("1.0") + ((4 - ((unsigned int) (2 + foo_rdar9373039 ("1.0")) % 4)) % 4)); // expected-warning {{never read}}

  for (int i = 0 ; i < size_rdar9373039 ; ++i)
    j += x; // expected-warning {{garbage}}

  return j;
}


int PR8962 (int *t) {
  // This should look through the __extension__ no-op.
  if (__extension__ (t)) return 0;
  return *t; // expected-warning {{null pointer}}
}

int PR8962_b (int *t) {
  // This should still ignore the nested casts
  // which aren't handled by a single IgnoreParens()
  if (((int)((int)t))) return 0;
  return *t; // expected-warning {{null pointer}}
}

int PR8962_c (int *t) {
  // If the last element in a StmtExpr was a ParenExpr, it's still live
  if (({ (t ? (_Bool)0 : (_Bool)1); })) return 0;
  return *t; // no-warning
}

int PR8962_d (int *t) {
  // If the last element in a StmtExpr is an __extension__, it's still live
  if (({ __extension__(t ? (_Bool)0 : (_Bool)1); })) return 0;
  return *t; // no-warning
}

int PR8962_e (int *t) {
  // Redundant casts can mess things up!
  // Environment used to skip through NoOp casts, but LiveVariables didn't!
  if (({ (t ? (int)(int)0L : (int)(int)1L); })) return 0;
  return *t; // no-warning
}

int PR8962_f (int *t) {
  // The StmtExpr isn't a block-level expression here,
  // the __extension__ is. But the value should be attached to the StmtExpr
  // anyway. Make sure the block-level check is /before/ IgnoreParens.
  if ( __extension__({
    _Bool r;
    if (t) r = 0;
    else r = 1;
    r;
  }) ) return 0;
  return *t; // no-warning
}
