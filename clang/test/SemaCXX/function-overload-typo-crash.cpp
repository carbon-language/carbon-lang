// RUN: %clang_cc1 -fsyntax-only -verify %s

// PR10283
void min(); //expected-note {{'min' declared here}}
void min(int);

template <typename T> void max(T); //expected-note {{'max' declared here}}

void f() {
  fin(); //expected-error {{use of undeclared identifier 'fin'; did you mean 'min'}}
  fax(0); //expected-error {{use of undeclared identifier 'fax'; did you mean 'max'}}
}

// TODO: Add proper function overloading resolution for template functions
template <typename T> void somefunc(T*, T*);
template <typename T> void somefunc(const T[]);
template <typename T1, typename T2> void somefunc(T1*, T2*);
template <typename T1, typename T2> void somefunc(T1*, const T2[]); //expected-note 5 {{'somefunc' declared here}} \
                                                                    //expected-note {{candidate function template not viable: requires 2 arguments, but 1 was provided}} TODO this shouldn't happen

void c() {
  int *i = 0, *j = 0;
  const int x[] = {1, 2, 3};
  long *l = 0;
  somefun(i, j); //expected-error {{use of undeclared identifier 'somefun'; did you mean 'somefunc'?}}
  somefun(x); //expected-error {{use of undeclared identifier 'somefun'; did you mean 'somefunc'?}} \
              //expected-error {{no matching function for call to 'somefunc'}} TODO this shouldn't happen
  somefun(i, l); //expected-error {{use of undeclared identifier 'somefun'; did you mean 'somefunc'?}}
  somefun(l, x); //expected-error {{use of undeclared identifier 'somefun'; did you mean 'somefunc'?}}
  somefun(i, x); //expected-error {{use of undeclared identifier 'somefun'; did you mean 'somefunc'?}}
}
