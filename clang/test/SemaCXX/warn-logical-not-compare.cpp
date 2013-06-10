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
