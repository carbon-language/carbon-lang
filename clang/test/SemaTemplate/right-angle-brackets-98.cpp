// RUN: clang -fsyntax-only -std=c++98 -verify %s
template<typename T> struct X;
template<int I> struct Y;

X<X<int> > *x1;
X<X<int>> *x2; // expected-error{{a space is required between consecutive right angle brackets (use '> >')}}

X<X<X<X<int>> // expected-error{{a space is required between consecutive right angle brackets (use '> >')}}
    >> *x3;   // expected-error{{a space is required between consecutive right angle brackets (use '> >')}}

Y<(1 >> 2)> *y1;
Y<1 >> 2> *y2;
// FIXME: when we get a -Wc++0x mode, warn about the use above
