// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: cpp11-migrate -loop-convert %t.cpp -- -I %S/Inputs -std=c++11
// RUN: FileCheck -input-file=%t.cpp %s
// RUN: cpp11-migrate -loop-convert %t.cpp -risk=risky -- -I %S/Inputs
// RUN: FileCheck -check-prefix=RISKY -input-file=%t.cpp %s

#include "structures.h"

void f() {
  /// begin()/end() - based for loops here:
  T t;
  for (T::iterator it = t.begin(), e = t.end(); it != e; ++it) {
    printf("I found %d\n", *it);
  }
  // CHECK: for ({{[a-zA-Z_ ]+&? ?}}[[VAR:[a-z_]+]] : t)
  // CHECK-NEXT: printf("I found %d\n", [[VAR]]);

  T *pt;
  for (T::iterator it = pt->begin(), e = pt->end(); it != e; ++it) {
    printf("I found %d\n", *it);
  }
  // CHECK: for ({{[a-zA-Z_ ]+&? ?}}[[VAR:[a-z_]+]] : *pt)
  // CHECK-NEXT: printf("I found %d\n", [[VAR]]);

  S s;
  for (S::const_iterator it = s.begin(), e = s.end(); it != e; ++it) {
    printf("s has value %d\n", (*it).x);
  }
  // CHECK: for ({{[a-zA-Z_ ]*&? ?}}[[VAR:[a-z_]+]] : s)
  // CHECK-NEXT: printf("s has value %d\n", ([[VAR]]).x);

  S *ps;
  for (S::const_iterator it = ps->begin(), e = ps->end(); it != e; ++it) {
    printf("s has value %d\n", (*it).x);
  }
  // CHECK: for ({{[a-zA-Z_ ]*&? ?}}[[VAR:[a-z_]+]] : *ps)
  // CHECK-NEXT: printf("s has value %d\n", ([[VAR]]).x);

  for (S::const_iterator it = s.begin(), e = s.end(); it != e; ++it) {
    printf("s has value %d\n", it->x);
  }
  // CHECK: for ({{[a-zA-Z_ ]*&? ?}}[[VAR:[a-z_]+]] : s)
  // CHECK-NEXT: printf("s has value %d\n", [[VAR]].x);

  for (S::iterator it = s.begin(), e = s.end(); it != e; ++it) {
    it->x = 3;
  }
  // CHECK: for ({{[a-zA-Z_ ]*&? ?}}[[VAR:[a-z_]+]] : s)
  // CHECK-NEXT: [[VAR]].x = 3;

  for (S::iterator it = s.begin(), e = s.end(); it != e; ++it) {
    (*it).x = 3;
  }
  // CHECK: for ({{[a-zA-Z_ ]*&? ?}}[[VAR:[a-z_]+]] : s)
  // CHECK-NEXT: ([[VAR]]).x = 3;

  for (S::iterator it = s.begin(), e = s.end(); it != e; ++it) {
    it->nonConstFun(4, 5);
  }
  // CHECK: for ({{[a-zA-Z_ ]*&? ?}}[[VAR:[a-z_]+]] : s)
  // CHECK-NEXT: [[VAR]].nonConstFun(4, 5);

  U u;
  for (U::iterator it = u.begin(), e = u.end(); it != e; ++it) {
    printf("s has value %d\n", it->x);
  }
  // CHECK: for ({{[a-zA-Z_ ]*&? ?}}[[VAR:[a-z_]+]] : u)
  // CHECK-NEXT: printf("s has value %d\n", [[VAR]].x);

  for (U::iterator it = u.begin(), e = u.end(); it != e; ++it) {
    printf("s has value %d\n", (*it).x);
  }
  // CHECK: for ({{[a-zA-Z_ ]*&? ?}}[[VAR:[a-z_]+]] : u)
  // CHECK-NEXT: printf("s has value %d\n", ([[VAR]]).x);

  U::iterator A;
  for (U::iterator i = u.begin(), e = u.end(); i != e; ++i)
    int k = A->x + i->x;
  // CHECK: for ({{[a-zA-Z_ ]*&? ?}}[[VAR:[a-z_]+]] : u)
  // CHECK-NEXT: int k = A->x + [[VAR]].x;

  dependent<int> v;
  for (dependent<int>::const_iterator it = v.begin(), e = v.end();
       it != e; ++it) {
    printf("Fibonacci number is %d\n", *it);
  }
  // CHECK: for ({{[a-zA-Z_ ]*&? ?}}[[VAR:[a-z_]+]] : v)
  // CHECK-NEXT: printf("Fibonacci number is %d\n", [[VAR]]);

  for (dependent<int>::const_iterator it(v.begin()), e = v.end();
       it != e; ++it) {
    printf("Fibonacci number is %d\n", *it);
  }
  // CHECK: for ({{[a-zA-Z_ ]*&? ?}}[[VAR:[a-z_]+]] : v)
  // CHECK-NEXT: printf("Fibonacci number is %d\n", [[VAR]]);

  doublyDependent<int,int> intmap;
  for (doublyDependent<int,int>::iterator it = intmap.begin(), e = intmap.end();
       it != e; ++it) {
    printf("intmap[%d] = %d", it->first, it->second);
  }
  // CHECK: for ({{[a-zA-Z_ ]*&? ?}}[[VAR:[a-z_]+]] : intmap)
  // CHECK-NEXT: printf("intmap[%d] = %d", [[VAR]].first, [[VAR]].second);

  // PtrSet's iterator dereferences by value so auto & can't be used.
  {
    PtrSet<int*> int_ptrs;
    for (PtrSet<int*>::iterator I = int_ptrs.begin(),
         E = int_ptrs.end(); I != E; ++I) {
      // CHECK: for (auto && int_ptr : int_ptrs) {
    }
  }

  // This container uses an iterator where the derefence type is a typedef of
  // a reference type. Make sure non-const auto & is still used. A failure here
  // means canonical types aren't being tested.
  {
    TypedefDerefContainer<int> int_ptrs;
    for (TypedefDerefContainer<int>::iterator I = int_ptrs.begin(),
         E = int_ptrs.end(); I != E; ++I) {
      // CHECK: for (auto & int_ptr : int_ptrs) {
    }
  }

  {
    // Iterators returning an rvalue reference should disqualify the loop from
    // transformation.
    RValueDerefContainer<int> container;
    for (RValueDerefContainer<int>::iterator I = container.begin(),
         E = container.end(); I != E; ++I) {
      // CHECK: for (RValueDerefContainer<int>::iterator I = container.begin(),
      // CHECK-NEXT: E = container.end(); I != E; ++I) {
    }
  }
}

// Tests to ensure that an implicit 'this' is picked up as the container.
// If member calls are made to 'this' within the loop, the transform becomes
// risky as these calls may affect state that affects the loop.
class C {
public:
  typedef MutableVal *iterator;
  typedef const MutableVal *const_iterator;

  iterator begin();
  iterator end();
  const_iterator begin() const;
  const_iterator end() const;

  void doSomething();
  void doSomething() const;

  void doLoop() {
    for (iterator I = begin(), E = end(); I != E; ++I) {
      // CHECK: for (auto & elem : *this) {
    }
    for (iterator I = C::begin(), E = C::end(); I != E; ++I) {
      // CHECK: for (auto & elem : *this) {
    }
    for (iterator I = begin(), E = end(); I != E; ++I) {
      // CHECK: for (iterator I = begin(), E = end(); I != E; ++I) {
      // RISKY: for (auto & elem : *this) {
      doSomething();
    }
    for (iterator I = begin(); I != end(); ++I) {
      // CHECK: for (auto & elem : *this) {
    }
    for (iterator I = begin(); I != end(); ++I) {
      // CHECK: for (iterator I = begin(); I != end(); ++I) {
      // RISKY: for (auto & elem : *this) {
      doSomething();
    }
  }

  void doLoop() const {
    for (const_iterator I = begin(), E = end(); I != E; ++I) {
      // CHECK: for (auto & elem : *this) {
    }
    for (const_iterator I = C::begin(), E = C::end(); I != E; ++I) {
      // CHECK: for (auto & elem : *this) {
    }
    for (const_iterator I = begin(), E = end(); I != E; ++I) {
      // CHECK: for (const_iterator I = begin(), E = end(); I != E; ++I) {
      // RISKY: for (auto & elem : *this) {
      doSomething();
    }
  }
};

class C2 {
public:
  typedef MutableVal *iterator;

  iterator begin() const;
  iterator end() const;

  void doLoop() {
    // The implicit 'this' will have an Implicit cast to const C2* wrapped
    // around it. Make sure the replacement still happens.
    for (iterator I = begin(), E = end(); I != E; ++I) {
      // CHECK: for (auto & elem : *this) {
    }
  }
};
