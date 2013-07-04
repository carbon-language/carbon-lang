// RUN: %clang_cc1 -fsyntax-only -Wlogical-not-parentheses -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wlogical-not-parentheses -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

bool getBool();
int getInt();

bool test1(int i1, int i2, bool b1, bool b2) {
  bool ret;

  ret = !i1 == i2;
  // expected-warning@-1 {{logical not is only applied to the left hand side of this comparison}}
  // expected-note@-2 {{add parentheses after the '!' to evaluate the comparison first}}
  // expected-note@-3 {{add parentheses around left hand side expression to silence this warning}}
  // CHECK: to evaluate the comparison first
  // CHECK: fix-it:"{{.*}}":{10:10-10:10}:"("
  // CHECK: fix-it:"{{.*}}":{10:18-10:18}:")"
  // CHECK: to silence this warning
  // CHECK: fix-it:"{{.*}}":{10:9-10:9}:"("
  // CHECK: fix-it:"{{.*}}":{10:12-10:12}:")"

  ret = !i1 != i2;
  //expected-warning@-1 {{logical not is only applied to the left hand side of this comparison}}
  // expected-note@-2 {{add parentheses after the '!' to evaluate the comparison first}}
  // expected-note@-3 {{add parentheses around left hand side expression to silence this warning}}
  // CHECK: to evaluate the comparison first
  // CHECK: fix-it:"{{.*}}":{21:10-21:10}:"("
  // CHECK: fix-it:"{{.*}}":{21:18-21:18}:")"
  // CHECK: to silence this warning
  // CHECK: fix-it:"{{.*}}":{21:9-21:9}:"("
  // CHECK: fix-it:"{{.*}}":{21:12-21:12}:")"

  ret = !i1 < i2;
  //expected-warning@-1 {{logical not is only applied to the left hand side of this comparison}}
  // expected-note@-2 {{add parentheses after the '!' to evaluate the comparison first}}
  // expected-note@-3 {{add parentheses around left hand side expression to silence this warning}}
  // CHECK: to evaluate the comparison first
  // CHECK: fix-it:"{{.*}}":{32:10-32:10}:"("
  // CHECK: fix-it:"{{.*}}":{32:17-32:17}:")"
  // CHECK: to silence this warning
  // CHECK: fix-it:"{{.*}}":{32:9-32:9}:"("
  // CHECK: fix-it:"{{.*}}":{32:12-32:12}:")"

  ret = !i1 > i2;
  //expected-warning@-1 {{logical not is only applied to the left hand side of this comparison}}
  // expected-note@-2 {{add parentheses after the '!' to evaluate the comparison first}}
  // expected-note@-3 {{add parentheses around left hand side expression to silence this warning}}
  // CHECK: to evaluate the comparison first
  // CHECK: fix-it:"{{.*}}":{43:10-43:10}:"("
  // CHECK: fix-it:"{{.*}}":{43:17-43:17}:")"
  // CHECK: to silence this warning
  // CHECK: fix-it:"{{.*}}":{43:9-43:9}:"("
  // CHECK: fix-it:"{{.*}}":{43:12-43:12}:")"

  ret = !i1 <= i2;
  //expected-warning@-1 {{logical not is only applied to the left hand side of this comparison}}
  // expected-note@-2 {{add parentheses after the '!' to evaluate the comparison first}}
  // expected-note@-3 {{add parentheses around left hand side expression to silence this warning}}
  // CHECK: to evaluate the comparison first
  // CHECK: fix-it:"{{.*}}":{54:10-54:10}:"("
  // CHECK: fix-it:"{{.*}}":{54:18-54:18}:")"
  // CHECK: to silence this warning
  // CHECK: fix-it:"{{.*}}":{54:9-54:9}:"("
  // CHECK: fix-it:"{{.*}}":{54:12-54:12}:")"

  ret = !i1 >= i2;
  //expected-warning@-1 {{logical not is only applied to the left hand side of this comparison}}
  // expected-note@-2 {{add parentheses after the '!' to evaluate the comparison first}}
  // expected-note@-3 {{add parentheses around left hand side expression to silence this warning}}
  // CHECK: to evaluate the comparison first
  // CHECK: fix-it:"{{.*}}":{65:10-65:10}:"("
  // CHECK: fix-it:"{{.*}}":{65:18-65:18}:")"
  // CHECK: to silence this warning
  // CHECK: fix-it:"{{.*}}":{65:9-65:9}:"("
  // CHECK: fix-it:"{{.*}}":{65:12-65:12}:")"

  ret = i1 == i2;
  ret = i1 != i2;
  ret = i1 < i2;
  ret = i1 > i2;
  ret = i1 <= i2;
  ret = i1 >= i2;

  // Warning silenced by parens.
  ret = (!i1) == i2;
  ret = (!i1) != i2;
  ret = (!i1) < i2;
  ret = (!i1) > i2;
  ret = (!i1) <= i2;
  ret = (!i1) >= i2;

  ret = !b1 == b2;
  ret = !b1 != b2;
  ret = !b1 < b2;
  ret = !b1 > b2;
  ret = !b1 <= b2;
  ret = !b1 >= b2;

  ret = !getInt() == i1;
  // expected-warning@-1 {{logical not is only applied to the left hand side of this comparison}}
  // expected-note@-2 {{add parentheses after the '!' to evaluate the comparison first}}
  // expected-note@-3 {{add parentheses around left hand side expression to silence this warning}}
  // CHECK: to evaluate the comparison first
  // CHECK: fix-it:"{{.*}}":{98:10-98:10}:"("
  // CHECK: fix-it:"{{.*}}":{98:24-98:24}:")"
  // CHECK: to silence this warning
  // CHECK: fix-it:"{{.*}}":{98:9-98:9}:"("
  // CHECK: fix-it:"{{.*}}":{98:18-98:18}:")"

  ret = (!getInt()) == i1;
  ret = !getBool() == b1;
  return ret;
}

enum E {e1, e2};
E getE();

bool test2 (E e) {
  bool ret;
  ret = e == e1;
  ret = e == getE();
  ret = getE() == e1;
  ret = getE() == getE();

  ret = !e == e1;
  // expected-warning@-1 {{logical not is only applied to the left hand side of this comparison}}
  // expected-note@-2 {{add parentheses after the '!' to evaluate the comparison first}}
  // expected-note@-3 {{add parentheses around left hand side expression to silence this warning}}
  // CHECK: to evaluate the comparison first
  // CHECK: fix-it:"{{.*}}":{124:10-124:10}:"("
  // CHECK: fix-it:"{{.*}}":{124:17-124:17}:")"
  // CHECK: to silence this warning
  // CHECK: fix-it:"{{.*}}":{124:9-124:9}:"("
  // CHECK: fix-it:"{{.*}}":{124:11-124:11}:")"

  ret = !e == getE();
  // expected-warning@-1 {{logical not is only applied to the left hand side of this comparison}}
  // expected-note@-2 {{add parentheses after the '!' to evaluate the comparison first}}
  // expected-note@-3 {{add parentheses around left hand side expression to silence this warning}}
  // CHECK: to evaluate the comparison first
  // CHECK: fix-it:"{{.*}}":{135:10-135:10}:"("
  // CHECK: fix-it:"{{.*}}":{135:21-135:21}:")"
  // CHECK: to silence this warning
  // CHECK: fix-it:"{{.*}}":{135:9-135:9}:"("
  // CHECK: fix-it:"{{.*}}":{135:11-135:11}:")"

  ret = !getE() == e1;
  // expected-warning@-1 {{logical not is only applied to the left hand side of this comparison}}
  // expected-note@-2 {{add parentheses after the '!' to evaluate the comparison first}}
  // expected-note@-3 {{add parentheses around left hand side expression to silence this warning}}
  // CHECK: to evaluate the comparison first
  // CHECK: fix-it:"{{.*}}":{146:10-146:10}:"("
  // CHECK: fix-it:"{{.*}}":{146:22-146:22}:")"
  // CHECK: to silence this warning
  // CHECK: fix-it:"{{.*}}":{146:9-146:9}:"("
  // CHECK: fix-it:"{{.*}}":{146:16-146:16}:")"

  ret = !getE() == getE();
  // expected-warning@-1 {{logical not is only applied to the left hand side of this comparison}}
  // expected-note@-2 {{add parentheses after the '!' to evaluate the comparison first}}
  // expected-note@-3 {{add parentheses around left hand side expression to silence this warning}}
  // CHECK: to evaluate the comparison first
  // CHECK: fix-it:"{{.*}}":{157:10-157:10}:"("
  // CHECK: fix-it:"{{.*}}":{157:26-157:26}:")"
  // CHECK: to silence this warning
  // CHECK: fix-it:"{{.*}}":{157:9-157:9}:"("
  // CHECK: fix-it:"{{.*}}":{157:16-157:16}:")"

  ret = !(e == e1);
  ret = !(e == getE());
  ret = !(getE() == e1);
  ret = !(getE() == getE());

  ret = (!e) == e1;
  ret = (!e) == getE();
  ret = (!getE()) == e1;
  ret = (!getE()) == getE();

  return ret;
}
