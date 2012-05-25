// RUN: %clang_cc1 -std=gnu++11 -Wsometimes-uninitialized -verify %s

bool maybe();

int test_if_false(bool b) {
  int x; // expected-note {{variable}}
  if (b) x = 1; // expected-note {{whenever 'if' condition is false}}
  return x; // expected-warning {{sometimes uninit}}
}

int test_if_true(bool b) {
  int x; // expected-note {{variable}}
  if (b) {} // expected-note {{whenever 'if' condition is true}}
  else x = 1;
  return x; // expected-warning {{sometimes uninit}}
}

int test_while_false(bool b) {
  int x; // expected-note {{variable}}
  while (b) { // expected-note {{whenever 'while' loop exits because its condition is false}}
    if (maybe()) {
      x = 1;
      break;
    }
  };
  return x; // expected-warning {{sometimes uninit}}
}

int test_while_true(bool b) {
  int x; // expected-note {{variable}}
  while (b) { // expected-note {{whenever 'while' loop is entered}}
label:
    return x; // expected-warning {{sometimes uninit}}
  }
  x = 0;
  goto label;
}

int test_do_while_false(bool b) {
  int x; // expected-note {{variable}}
  do {
    if (maybe()) {
      x = 1;
      break;
    }
  } while (b); // expected-note {{whenever 'do' loop exits because its condition is false}}
  return x; // expected-warning {{sometimes uninit}}
}

int test_do_while_true(bool b) {
  int x; // expected-note {{variable}}
goto label2;
  do {
label1:
    return x; // expected-warning {{sometimes uninit}}
label2: ;
  } while (b); // expected-note {{whenever 'do' loop condition is true}}
  x = 0;
  goto label1;
}

int test_for_false(int k) {
  int x; // expected-note {{variable}}
  for (int n = 0;
       n < k; // expected-note {{whenever 'for' loop exits because its condition is false}}
       ++n) {
    if (maybe()) {
      x = n;
      break;
    }
  }
  return x; // expected-warning {{sometimes uninit}}
}

int test_for_true(int k) {
  int x; // expected-note {{variable}}
  int n = 0;
  for (;
       n < k; // expected-note {{whenever 'for' loop is entered}}
       ++n) {
label:
    return x; // expected-warning {{sometimes uninit}}
  }
  x = 1;
  goto label;
}

int test_for_range_false(int k) {
  int arr[3] = { 1, 2, 3 };
  int x; // expected-note {{variable}}
  for (int &a : arr) { // expected-note {{whenever 'for' loop exits because its condition is false}}
    if (a == k) {
      x = &a - arr;
      break;
    }
  }
  return x; // expected-warning {{sometimes uninit}}
}

int test_for_range_true(int k) {
  int arr[3] = { 1, 2, 3 };
  int x; // expected-note {{variable}}
  for (int &a : arr) { // expected-note {{whenever 'for' loop is entered}}
    goto label;
  }
  x = 0;
label:
  return x; // expected-warning {{sometimes uninit}}
}

int test_conditional_false(int k) {
  int x; // expected-note {{variable}}
  (void)(
      maybe() // expected-note {{whenever '?:' condition is false}}
      ? x = 1 : 0);
  return x; // expected-warning {{sometimes uninit}}
}

int test_conditional_true(int k) {
  int x; // expected-note {{variable}}
  (void)(
      maybe() // expected-note {{whenever '?:' condition is true}}
      ? 0 : x = 1);
  return x; // expected-warning {{sometimes uninit}}
}

int test_logical_and_false(int k) {
  int x; // expected-note {{variable}}
  maybe() // expected-note {{whenever '&&' condition is false}}
      && (x = 1);
  return x; // expected-warning {{sometimes uninit}}
}

int test_logical_and_true(int k) {
  int x; // expected-note {{variable}}
  maybe() // expected-note {{whenever '&&' condition is true}}
      && ({ goto skip_init; 0; });
  x = 1;
skip_init:
  return x; // expected-warning {{sometimes uninit}}
}

int test_logical_or_false(int k) {
  int x; // expected-note {{variable}}
  maybe() // expected-note {{whenever '||' condition is false}}
      || ({ goto skip_init; 0; });
  x = 1;
skip_init:
  return x; // expected-warning {{sometimes uninit}}
}

int test_logical_or_true(int k) {
  int x; // expected-note {{variable}}
  maybe() // expected-note {{whenever '||' condition is true}}
      || (x = 1);
  return x; // expected-warning {{sometimes uninit}}
}

int test_switch_case(int k) {
  int x; // expected-note {{variable}}
  switch (k) {
  case 0:
    x = 0;
    break;
  case 1: // expected-note {{whenever switch case is taken}}
    break;
  }
  return x; // expected-warning {{sometimes uninit}}
}

int test_switch_default(int k) {
  int x; // expected-note {{variable}}
  switch (k) {
  case 0:
    x = 0;
    break;
  case 1:
    x = 1;
    break;
  default: // expected-note {{whenever switch default is taken}}
    break;
  }
  return x; // expected-warning {{sometimes uninit}}
}

int test_switch_suppress_1(int k) {
  int x;
  switch (k) {
  case 0:
    x = 0;
    break;
  case 1:
    x = 1;
    break;
  }
  return x; // no-warning
}

int test_switch_suppress_2(int k) {
  int x;
  switch (k) {
  case 0:
  case 1:
    switch (k) {
    case 0:
      return 0;
    case 1:
      return 1;
    }
  case 2:
  case 3:
    x = 1;
  }
  return x; // no-warning
}

int test_multiple_notes(int k) {
  int x; // expected-note {{variable}}
  if (k > 0) {
    if (k == 5)
      x = 1;
    else if (k == 2) // expected-note {{whenever 'if' condition is false}}
      x = 2;
  } else {
    if (k == -5)
      x = 3;
    else if (k == -2) // expected-note {{whenever 'if' condition is false}}
      x = 4;
  }
  return x; // expected-warning {{sometimes uninit}}
}

int test_no_false_positive_1(int k) {
  int x;
  if (k)
    x = 5;
  while (!k)
    maybe();
  return x;
}

int test_no_false_positive_2() {
  int x;
  bool b = false;
  if (maybe()) {
    x = 5;
    b = true;
  }
  return b ? x : 0;
}

void test_null_pred_succ() {
  int x; // expected-note {{variable}}
  if (0) // expected-note {{whenever}}
    foo: x = 0;
  if (x) // expected-warning {{sometimes uninit}}
    goto foo;
}
