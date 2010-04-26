// RUN: %clang_cc1 -fsyntax-only -verify %s

@interface A {
@public
  int ivar;
}
@end

typedef struct objc_object {
    Class isa;
} *id;

// Test instantiation of value-dependent ObjCIvarRefExpr and
// ObjCIsaRefExpr nodes.
A *get_an_A(unsigned);
id get_an_id(unsigned);

template<unsigned N, typename T, typename U>
void f(U value) {
  get_an_A(N)->ivar = value; // expected-error{{assigning to 'int' from incompatible type 'int *'}}
  T c = get_an_id(N)->isa; // expected-error{{cannot initialize a variable of type 'int' with an lvalue of type 'Class'}}
}

template void f<6, Class>(int);
template void f<7, Class>(int*); // expected-note{{in instantiation of}}
template void f<8, int>(int); // expected-note{{in instantiation of}}

// Test instantiation of unresolved member reference expressions to an
// ivar reference.
template<typename T, typename U>
void f2(T ptr, U value) {
  ptr->ivar = value; // expected-error{{assigning to 'int' from incompatible type 'int *'}}
}

template void f2(A*, int);
template void f2(A*, int*); // expected-note{{instantiation of}}

// Test instantiation of unresolved member referfence expressions to
// an isa.
template<typename T, typename U>
void f3(U ptr) {
  T c = ptr->isa; // expected-error{{cannot initialize a variable of type 'int' with an lvalue of type 'Class'}}
}

template void f3<Class>(id);
template void f3<int>(id); // expected-note{{instantiation of}}
