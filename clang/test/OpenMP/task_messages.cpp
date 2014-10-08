// RUN: %clang_cc1 -verify -fopenmp=libiomp5 -ferror-limit 100 -std=c++11 -o - %s

void foo() {
}

#pragma omp task // expected-error {{unexpected OpenMP directive '#pragma omp task'}}

class S { // expected-note 6 {{'S' declared here}}
  S(const S &s) { a = s.a + 12; }
  int a;

public:
  S() : a(0) {}
  S(int a) : a(a) {}
  operator int() { return a; }
  S &operator++() { return *this; }
  S operator+(const S &) { return *this; }
};

template <class T>
int foo() {
  T a; // expected-note 3 {{'a' defined here}}
  T &b = a; // expected-note 4 {{'b' defined here}}
  int r;
#pragma omp task default(none)
#pragma omp task default(shared)
  ++a;
// expected-error@+2 {{predetermined as a firstprivate in a task construct variable must have an accessible, unambiguous copy constructor}}
#pragma omp task default(none)
#pragma omp task
// expected-note@+1 {{used here}}
  ++a;
#pragma omp task
// expected-error@+1 {{predetermined as a firstprivate in a task construct variable must have an accessible, unambiguous copy constructor}}
#pragma omp task
  // expected-note@+1 {{used here}}
  ++a;
#pragma omp task default(shared)
#pragma omp task
  ++a;
#pragma omp task
#pragma omp parallel
  ++a;
// expected-error@+2 {{predetermined as a firstprivate in a task construct variable cannot be of reference type 'int &'}}
// expected-error@+1 {{predetermined as a firstprivate in a task construct variable cannot be of reference type 'S &'}}
#pragma omp task
  // expected-note@+1 2 {{used here}}
  ++b;
// expected-error@+3 {{predetermined as a firstprivate in a task construct variable cannot be of reference type 'int &'}}
// expected-error@+2 {{predetermined as a firstprivate in a task construct variable cannot be of reference type 'S &'}}
// expected-error@+1 {{predetermined as a firstprivate in a task construct variable must have an accessible, unambiguous copy constructor}}
#pragma omp task
// expected-note@+1 3 {{used here}}
#pragma omp parallel shared(a, b)
  ++a, ++b;
// expected-note@+1 3 {{defined as reduction}}
#pragma omp parallel reduction(+ : r)
// expected-error@+1 {{argument of a reduction clause of a parallel construct must not appear in a firstprivate clause on a task construct}}
#pragma omp task firstprivate(r)
  // expected-error@+1 2 {{reduction variables may not be accessed in an explicit task}}
  ++r;
// expected-note@+1 2 {{defined as reduction}}
#pragma omp parallel reduction(+ : r)
#pragma omp task default(shared)
  // expected-error@+1 2 {{reduction variables may not be accessed in an explicit task}}
  ++r;
// expected-note@+1 2 {{defined as reduction}}
#pragma omp parallel reduction(+ : r)
#pragma omp task
  // expected-error@+1 2 {{reduction variables may not be accessed in an explicit task}}
  ++r;
#pragma omp parallel
// expected-note@+1 3 {{defined as reduction}}
#pragma omp for reduction(+ : r)
  for (int i = 0; i < 10; ++i)
// expected-error@+1 {{argument of a reduction clause of a for construct must not appear in a firstprivate clause on a task construct}}
#pragma omp task firstprivate(r)
    // expected-error@+1 2 {{reduction variables may not be accessed in an explicit task}}
    ++r;
#pragma omp parallel
// expected-note@+1 2 {{defined as reduction}}
#pragma omp for reduction(+ : r)
  for (int i = 0; i < 10; ++i)
#pragma omp task default(shared)
    // expected-error@+1 2 {{reduction variables may not be accessed in an explicit task}}
    ++r;
#pragma omp parallel
// expected-note@+1 2 {{defined as reduction}}
#pragma omp for reduction(+ : r)
  for (int i = 0; i < 10; ++i)
#pragma omp task
    // expected-error@+1 2 {{reduction variables may not be accessed in an explicit task}}
    ++r;
// expected-note@+1 {{non-shared variable in a task construct is predetermined as firstprivate}}
#pragma omp task
// expected-error@+2 {{reduction variable must be shared}}
// expected-error@+1 {{region cannot be closely nested inside 'task' region; perhaps you forget to enclose 'omp for' directive into a parallel region?}}
#pragma omp for reduction(+ : r)
  ++r;
// expected-error@+1 {{directive '#pragma omp task' cannot contain more than one 'untied' clause}}
#pragma omp task untied untied
  ++r;
// expected-error@+1 {{directive '#pragma omp task' cannot contain more than one 'mergeable' clause}}
#pragma omp task mergeable mergeable
  ++r;
  return a + b;
}

int main(int argc, char **argv) {
  int a;
  int &b = a; // expected-note 2 {{'b' defined here}}
  S sa;       // expected-note 3 {{'sa' defined here}}
  S &sb = sa; // expected-note 2 {{'sb' defined here}}
  int r;
#pragma omp task { // expected-warning {{extra tokens at the end of '#pragma omp task' are ignored}}
  foo();
#pragma omp task( // expected-warning {{extra tokens at the end of '#pragma omp task' are ignored}}
  foo();
#pragma omp task[ // expected-warning {{extra tokens at the end of '#pragma omp task' are ignored}}
  foo();
#pragma omp task] // expected-warning {{extra tokens at the end of '#pragma omp task' are ignored}}
  foo();
#pragma omp task) // expected-warning {{extra tokens at the end of '#pragma omp task' are ignored}}
  foo();
#pragma omp task } // expected-warning {{extra tokens at the end of '#pragma omp task' are ignored}}
  foo();
#pragma omp task
// expected-warning@+1 {{extra tokens at the end of '#pragma omp task' are ignored}}
#pragma omp task unknown()
  foo();
L1:
  foo();
#pragma omp task
  ;
#pragma omp task
  {
    goto L1; // expected-error {{use of undeclared label 'L1'}}
    argc++;
  }

  for (int i = 0; i < 10; ++i) {
    switch (argc) {
    case (0):
#pragma omp task
    {
      foo();
      break;    // expected-error {{'break' statement not in loop or switch statement}}
      continue; // expected-error {{'continue' statement not in loop statement}}
    }
    default:
      break;
    }
  }
#pragma omp task default(none)
  ++argc; // expected-error {{variable 'argc' must have explicitly specified data sharing attributes}}

  goto L2; // expected-error {{use of undeclared label 'L2'}}
#pragma omp task
L2:
  foo();
#pragma omp task
  {
    return 1; // expected-error {{cannot return from OpenMP region}}
  }

  [[]] // expected-error {{an attribute list cannot appear here}}
#pragma omp task
      for (int n = 0; n < 100; ++n) {
  }

#pragma omp task default(none)
#pragma omp task default(shared)
  ++a;
#pragma omp task default(none)
#pragma omp task
  ++a;
#pragma omp task default(shared)
#pragma omp task
  ++a;
#pragma omp task
#pragma omp parallel
  ++a;
// expected-error@+1 {{predetermined as a firstprivate in a task construct variable cannot be of reference type 'int &'}}
#pragma omp task
  // expected-note@+1 {{used here}}
  ++b;
// expected-error@+1 {{predetermined as a firstprivate in a task construct variable cannot be of reference type 'int &'}}
#pragma omp task
// expected-note@+1 {{used here}}
#pragma omp parallel shared(a, b)
  ++a, ++b;
#pragma omp task default(none)
#pragma omp task default(shared)
  ++sa;
#pragma omp task default(none)
// expected-error@+1 {{predetermined as a firstprivate in a task construct variable must have an accessible, unambiguous copy constructor}}
#pragma omp task
// expected-note@+1 {{used here}}
  ++sa;
#pragma omp task
// expected-error@+1 {{predetermined as a firstprivate in a task construct variable must have an accessible, unambiguous copy constructor}}
#pragma omp task
// expected-note@+1 {{used here}}
  ++sa;
#pragma omp task default(shared)
#pragma omp task
  ++sa;
#pragma omp task
#pragma omp parallel
  ++sa;
// expected-error@+1 {{predetermined as a firstprivate in a task construct variable cannot be of reference type 'S &'}}
#pragma omp task
  // expected-note@+1 {{used here}}
  ++sb;
// expected-error@+2 {{predetermined as a firstprivate in a task construct variable cannot be of reference type 'S &'}}
// expected-error@+1 {{predetermined as a firstprivate in a task construct variable must have an accessible, unambiguous copy constructor}}
#pragma omp task
// expected-note@+1 2 {{used here}}
#pragma omp parallel shared(sa, sb)
  ++sa, ++sb;
// expected-note@+1 2 {{defined as reduction}}
#pragma omp parallel reduction(+ : r)
// expected-error@+1 {{argument of a reduction clause of a parallel construct must not appear in a firstprivate clause on a task construct}}
#pragma omp task firstprivate(r)
  // expected-error@+1 {{reduction variables may not be accessed in an explicit task}}
  ++r;
// expected-note@+1 {{defined as reduction}}
#pragma omp parallel reduction(+ : r)
#pragma omp task default(shared)
  // expected-error@+1 {{reduction variables may not be accessed in an explicit task}}
  ++r;
// expected-note@+1 {{defined as reduction}}
#pragma omp parallel reduction(+ : r)
#pragma omp task
  // expected-error@+1 {{reduction variables may not be accessed in an explicit task}}
  ++r;
#pragma omp parallel
// expected-note@+1 2 {{defined as reduction}}
#pragma omp for reduction(+ : r)
  for (int i = 0; i < 10; ++i)
// expected-error@+1 {{argument of a reduction clause of a for construct must not appear in a firstprivate clause on a task construct}}
#pragma omp task firstprivate(r)
    // expected-error@+1 {{reduction variables may not be accessed in an explicit task}}
    ++r;
#pragma omp parallel
// expected-note@+1 {{defined as reduction}}
#pragma omp for reduction(+ : r)
  for (int i = 0; i < 10; ++i)
#pragma omp task default(shared)
    // expected-error@+1 {{reduction variables may not be accessed in an explicit task}}
    ++r;
#pragma omp parallel
// expected-note@+1 {{defined as reduction}}
#pragma omp for reduction(+ : r)
  for (int i = 0; i < 10; ++i)
#pragma omp task
    // expected-error@+1 {{reduction variables may not be accessed in an explicit task}}
    ++r;
// expected-note@+1 {{non-shared variable in a task construct is predetermined as firstprivate}}
#pragma omp task
// expected-error@+2 {{reduction variable must be shared}}
// expected-error@+1 {{region cannot be closely nested inside 'task' region; perhaps you forget to enclose 'omp for' directive into a parallel region?}}
#pragma omp for reduction(+ : r)
  ++r;
// expected-error@+1 {{directive '#pragma omp task' cannot contain more than one 'untied' clause}}
#pragma omp task untied untied
  ++r;
// expected-error@+1 {{directive '#pragma omp task' cannot contain more than one 'mergeable' clause}}
#pragma omp task mergeable mergeable
  ++r;
  // expected-note@+2 {{in instantiation of function template specialization 'foo<int>' requested here}}
  // expected-note@+1 {{in instantiation of function template specialization 'foo<S>' requested here}}
  return foo<int>() + foo<S>();
}

