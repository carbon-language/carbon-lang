// RUN: %clang_analyze_cc1 -analyzer-checker=debug.ExprInspection -Wno-pointer-to-int-cast -verify %s

void clang_analyzer_eval(int);
void clang_analyzer_warnOnDeadSymbol(int);
void clang_analyzer_numTimesReached(void);
void clang_analyzer_warnIfReached(void);

void exit(int);

int conjure_index(void);

void test_that_expr_inspection_works(void) {
  do {
    int x = conjure_index();
    clang_analyzer_warnOnDeadSymbol(x);
  } while(0); // expected-warning{{SYMBOL DEAD}}

  // Make sure we don't accidentally split state in ExprInspection.
  clang_analyzer_numTimesReached(); // expected-warning{{1}}
}

// These tests verify the reaping of symbols that are only referenced as
// index values in element regions. Most of the time, depending on where
// the element region, as Loc value, is stored, it is possible to
// recover the index symbol in checker code, which is also demonstrated
// in the return_ptr_range.c test file.

int arr[3];

int *test_element_index_lifetime_in_environment_values(void) {
  int *ptr;
  do {
    int x = conjure_index();
    clang_analyzer_warnOnDeadSymbol(x);
    ptr = arr + x;
  } while (0);
  return ptr;
}

void test_element_index_lifetime_in_store_keys(void) {
  do {
    int x = conjure_index();
    clang_analyzer_warnOnDeadSymbol(x);
    arr[x] = 1;
    if (x) {}
  } while (0); // no-warning
}

int *ptr;
void test_element_index_lifetime_in_store_values(void) {
  do {
    int x = conjure_index();
    clang_analyzer_warnOnDeadSymbol(x);
    ptr = arr + x;
  } while (0); // no-warning
}

struct S1 {
  int field;
};
struct S2 {
  struct S1 array[5];
} s2;
struct S3 {
  void *field;
};

struct S1 *conjure_S1(void);
struct S3 *conjure_S3(void);

void test_element_index_lifetime_with_complicated_hierarchy_of_regions(void) {
  do {
    int x = conjure_index();
    clang_analyzer_warnOnDeadSymbol(x);
    s2.array[x].field = 1;
    if (x) {}
  } while (0); // no-warning
}

void test_loc_as_integer_element_index_lifetime(void) {
  do {
    int x;
    struct S3 *s = conjure_S3();
    clang_analyzer_warnOnDeadSymbol((int)s);
    x = (int)&(s->field);
    ptr = &arr[x];
    if (s) {}
  } while (0);
}

// Test below checks lifetime of SymbolRegionValue in certain conditions.

int **ptrptr;
void test_region_lifetime_as_store_value(int *x) {
  clang_analyzer_warnOnDeadSymbol((int) x);
  *x = 1;
  ptrptr = &x;
  (void)0; // No-op; make sure the environment forgets things and the GC runs.
  clang_analyzer_eval(**ptrptr); // expected-warning{{TRUE}}
} // no-warning

int *produce_region_referenced_only_through_field_in_environment_value(void) {
  struct S1 *s = conjure_S1();
  clang_analyzer_warnOnDeadSymbol((int) s);
  int *x = &s->field;
  return x;
}

void test_region_referenced_only_through_field_in_environment_value(void) {
  produce_region_referenced_only_through_field_in_environment_value();
} // expected-warning{{SYMBOL DEAD}}

void test_region_referenced_only_through_field_in_store_value(void) {
  struct S1 *s = conjure_S1();
  clang_analyzer_warnOnDeadSymbol((int) s);
  ptr = &s->field; // Write the symbol into a global. It should live forever.
  if (!s) {
    exit(0); // no-warning (symbol should not die here)
    // exit() is noreturn.
    clang_analyzer_warnIfReached(); // no-warning
  }
  if (!ptr) { // no-warning (symbol should not die here)
    // We exit()ed under these constraints earlier.
    clang_analyzer_warnIfReached(); // no-warning
  }
  // The exit() call invalidates globals. The symbol will die here because
  // the exit() statement itself is already over and there's no better statement
  // to put the diagnostic on.
} // expected-warning{{SYMBOL DEAD}}

void test_zombie_referenced_only_through_field_in_store_value(void) {
  struct S1 *s = conjure_S1();
  clang_analyzer_warnOnDeadSymbol((int) s);
  int *x = &s->field;
} // expected-warning{{SYMBOL DEAD}}

void double_dereference_of_implicit_value_aux1(int *p) {
  *p = 0;
}

void double_dereference_of_implicit_value_aux2(int *p) {
  if (*p != 0)
    clang_analyzer_warnIfReached(); // no-warning
}

void test_double_dereference_of_implicit_value(int **x) {
  clang_analyzer_warnOnDeadSymbol(**x);
  int **y = x;
  {
    double_dereference_of_implicit_value_aux1(*y);
    // Give time for symbol reaping to happen.
    ((void)0);
    // The symbol for **y was cleaned up from the Store at this point,
    // even though it was not perceived as dead when asked explicitly.
    // For that reason the SYMBOL DEAD warning never appeared at this point.
    double_dereference_of_implicit_value_aux2(*y);
  }
  // The symbol is generally reaped here regardless.
  ((void)0); // expected-warning{{SYMBOL DEAD}}
}
