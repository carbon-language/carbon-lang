// RUN: %python %S/check_clang_tidy.py %s modernize-loop-convert %t -- -std=c++11 -I %S/Inputs/modernize-loop-convert

#include "structures.h"

namespace Array {

const int N = 6;
const int NMinusOne = N - 1;
int arr[N] = {1, 2, 3, 4, 5, 6};
int (*pArr)[N] = &arr;

void f() {
  int sum = 0;

  for (int i = 0; i < N; ++i) {
    sum += arr[i];
    int k;
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead [modernize-loop-convert]
  // CHECK-FIXES: for (auto & elem : arr) {
  // CHECK-FIXES-NEXT: sum += elem;
  // CHECK-FIXES-NEXT: int k;
  // CHECK-FIXES-NEXT: }

  for (int i = 0; i < N; ++i) {
    printf("Fibonacci number is %d\n", arr[i]);
    sum += arr[i] + 2;
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & elem : arr)
  // CHECK-FIXES-NEXT: printf("Fibonacci number is %d\n", elem);
  // CHECK-FIXES-NEXT: sum += elem + 2;

  for (int i = 0; i < N; ++i) {
    int x = arr[i];
    int y = arr[i] + 2;
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & elem : arr)
  // CHECK-FIXES-NEXT: int x = elem;
  // CHECK-FIXES-NEXT: int y = elem + 2;

  for (int i = 0; i < N; ++i) {
    int x = N;
    x = arr[i];
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & elem : arr)
  // CHECK-FIXES-NEXT: int x = N;
  // CHECK-FIXES-NEXT: x = elem;

  for (int i = 0; i < N; ++i) {
    arr[i] += 1;
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & elem : arr) {
  // CHECK-FIXES-NEXT: elem += 1;
  // CHECK-FIXES-NEXT: }

  for (int i = 0; i < N; ++i) {
    int x = arr[i] + 2;
    arr[i]++;
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & elem : arr)
  // CHECK-FIXES-NEXT: int x = elem + 2;
  // CHECK-FIXES-NEXT: elem++;

  for (int i = 0; i < N; ++i) {
    arr[i] = 4 + arr[i];
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & elem : arr)
  // CHECK-FIXES-NEXT: elem = 4 + elem;

  for (int i = 0; i < NMinusOne + 1; ++i) {
    sum += arr[i];
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & elem : arr) {
  // CHECK-FIXES-NEXT: sum += elem;
  // CHECK-FIXES-NEXT: }

  for (int i = 0; i < N; ++i) {
    printf("Fibonacci number %d has address %p\n", arr[i], &arr[i]);
    sum += arr[i] + 2;
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & elem : arr)
  // CHECK-FIXES-NEXT: printf("Fibonacci number %d has address %p\n", elem, &elem);
  // CHECK-FIXES-NEXT: sum += elem + 2;

  Val teas[N];
  for (int i = 0; i < N; ++i) {
    teas[i].g();
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & tea : teas) {
  // CHECK-FIXES-NEXT: tea.g();
  // CHECK-FIXES-NEXT: }
}

struct HasArr {
  int Arr[N];
  Val ValArr[N];
  void implicitThis() {
    for (int i = 0; i < N; ++i) {
      printf("%d", Arr[i]);
    }
    // CHECK-MESSAGES: :[[@LINE-3]]:5: warning: use range-based for loop instead
    // CHECK-FIXES: for (auto & elem : Arr) {
    // CHECK-FIXES-NEXT: printf("%d", elem);
    // CHECK-FIXES-NEXT: }

    for (int i = 0; i < N; ++i) {
      printf("%d", ValArr[i].x);
    }
    // CHECK-MESSAGES: :[[@LINE-3]]:5: warning: use range-based for loop instead
    // CHECK-FIXES: for (auto & elem : ValArr) {
    // CHECK-FIXES-NEXT: printf("%d", elem.x);
    // CHECK-FIXES-NEXT: }
  }

  void explicitThis() {
    for (int i = 0; i < N; ++i) {
      printf("%d", this->Arr[i]);
    }
    // CHECK-MESSAGES: :[[@LINE-3]]:5: warning: use range-based for loop instead
    // CHECK-FIXES: for (auto & elem : this->Arr) {
    // CHECK-FIXES-NEXT: printf("%d", elem);
    // CHECK-FIXES-NEXT: }

    for (int i = 0; i < N; ++i) {
      printf("%d", this->ValArr[i].x);
    }
    // CHECK-MESSAGES: :[[@LINE-3]]:5: warning: use range-based for loop instead
    // CHECK-FIXES: for (auto & elem : this->ValArr) {
    // CHECK-FIXES-NEXT: printf("%d", elem.x);
    // CHECK-FIXES-NEXT: }
  }
};

// Loops whose bounds are value-dependent shold not be converted.
template <int N>
void dependentExprBound() {
  for (int i = 0; i < N; ++i)
    arr[i] = 0;
}
template void dependentExprBound<20>();

void memberFunctionPointer() {
  Val v;
  void (Val::*mfpArr[N])(void) = {&Val::g};
  for (int i = 0; i < N; ++i)
    (v.*mfpArr[i])();
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & elem : mfpArr)
  // CHECK-FIXES-NEXT: (v.*elem)();

  struct Foo {
    int (Val::*f)();
  } foo[N];

  for (int i = 0; i < N; ++i)
    int r = (v.*(foo[i].f))();
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & elem : foo)
  // CHECK-FIXES-NEXT: int r = (v.*(elem.f))();

}

} // namespace Array

namespace Iterator {

void f() {
  /// begin()/end() - based for loops here:
  T t;
  for (T::iterator it = t.begin(), e = t.end(); it != e; ++it) {
    printf("I found %d\n", *it);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & elem : t)
  // CHECK-FIXES-NEXT: printf("I found %d\n", elem);

  T *pt;
  for (T::iterator it = pt->begin(), e = pt->end(); it != e; ++it) {
    printf("I found %d\n", *it);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & elem : *pt)
  // CHECK-FIXES-NEXT: printf("I found %d\n", elem);

  S s;
  for (S::iterator it = s.begin(), e = s.end(); it != e; ++it) {
    printf("s has value %d\n", (*it).x);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & elem : s)
  // CHECK-FIXES-NEXT: printf("s has value %d\n", (elem).x);

  S *ps;
  for (S::iterator it = ps->begin(), e = ps->end(); it != e; ++it) {
    printf("s has value %d\n", (*it).x);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & p : *ps)
  // CHECK-FIXES-NEXT: printf("s has value %d\n", (p).x);

  for (S::iterator it = s.begin(), e = s.end(); it != e; ++it) {
    printf("s has value %d\n", it->x);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & elem : s)
  // CHECK-FIXES-NEXT: printf("s has value %d\n", elem.x);

  for (S::iterator it = s.begin(), e = s.end(); it != e; ++it) {
    it->x = 3;
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & elem : s)
  // CHECK-FIXES-NEXT: elem.x = 3;

  for (S::iterator it = s.begin(), e = s.end(); it != e; ++it) {
    (*it).x = 3;
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & elem : s)
  // CHECK-FIXES-NEXT: (elem).x = 3;

  for (S::iterator it = s.begin(), e = s.end(); it != e; ++it) {
    it->nonConstFun(4, 5);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & elem : s)
  // CHECK-FIXES-NEXT: elem.nonConstFun(4, 5);

  U u;
  for (U::iterator it = u.begin(), e = u.end(); it != e; ++it) {
    printf("s has value %d\n", it->x);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & elem : u)
  // CHECK-FIXES-NEXT: printf("s has value %d\n", elem.x);

  for (U::iterator it = u.begin(), e = u.end(); it != e; ++it) {
    printf("s has value %d\n", (*it).x);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & elem : u)
  // CHECK-FIXES-NEXT: printf("s has value %d\n", (elem).x);

  U::iterator A;
  for (U::iterator i = u.begin(), e = u.end(); i != e; ++i)
    int k = A->x + i->x;
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & elem : u)
  // CHECK-FIXES-NEXT: int k = A->x + elem.x;

  dependent<int> v;
  for (dependent<int>::iterator it = v.begin(), e = v.end();
       it != e; ++it) {
    printf("Fibonacci number is %d\n", *it);
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & elem : v) {
  // CHECK-FIXES-NEXT: printf("Fibonacci number is %d\n", elem);

  for (dependent<int>::iterator it(v.begin()), e = v.end();
       it != e; ++it) {
    printf("Fibonacci number is %d\n", *it);
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & elem : v) {
  // CHECK-FIXES-NEXT: printf("Fibonacci number is %d\n", elem);

  doublyDependent<int, int> intmap;
  for (doublyDependent<int, int>::iterator it = intmap.begin(), e = intmap.end();
       it != e; ++it) {
    printf("intmap[%d] = %d", it->first, it->second);
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & elem : intmap)
  // CHECK-FIXES: printf("intmap[%d] = %d", elem.first, elem.second);

  // PtrSet's iterator dereferences by value so auto & can't be used.
  {
    PtrSet<int *> int_ptrs;
    for (PtrSet<int *>::iterator I = int_ptrs.begin(),
                                 E = int_ptrs.end();
         I != E; ++I) {
    }
    // CHECK-MESSAGES: :[[@LINE-4]]:5: warning: use range-based for loop instead
    // CHECK-FIXES: for (auto && int_ptr : int_ptrs) {
  }

  // This container uses an iterator where the derefence type is a typedef of
  // a reference type. Make sure non-const auto & is still used. A failure here
  // means canonical types aren't being tested.
  {
    TypedefDerefContainer<int> int_ptrs;
    for (TypedefDerefContainer<int>::iterator I = int_ptrs.begin(),
                                              E = int_ptrs.end();
         I != E; ++I) {
    }
    // CHECK-MESSAGES: :[[@LINE-4]]:5: warning: use range-based for loop instead
    // CHECK-FIXES: for (auto & int_ptr : int_ptrs) {
  }

  {
    // Iterators returning an rvalue reference should disqualify the loop from
    // transformation.
    RValueDerefContainer<int> container;
    for (RValueDerefContainer<int>::iterator I = container.begin(),
                                             E = container.end();
         I != E; ++I) {
    }
    // CHECK-FIXES: for (RValueDerefContainer<int>::iterator I = container.begin(),
    // CHECK-FIXES-NEXT: E = container.end();
    // CHECK-FIXES-NEXT: I != E; ++I) {
  }
}

// Tests to verify the proper use of auto where the init variable type and the
// initializer type differ or are mostly the same except for const qualifiers.
void different_type() {
  // s.begin() returns a type 'iterator' which is just a non-const pointer and
  // differs from const_iterator only on the const qualification.
  S s;
  for (S::const_iterator it = s.begin(), e = s.end(); it != e; ++it) {
    printf("s has value %d\n", (*it).x);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (const auto & elem : s)
  // CHECK-FIXES-NEXT: printf("s has value %d\n", (elem).x);

  S *ps;
  for (S::const_iterator it = ps->begin(), e = ps->end(); it != e; ++it) {
    printf("s has value %d\n", (*it).x);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (const auto & p : *ps)
  // CHECK-FIXES-NEXT: printf("s has value %d\n", (p).x);

  // v.begin() returns a user-defined type 'iterator' which, since it's
  // different from const_iterator, disqualifies these loops from
  // transformation.
  dependent<int> v;
  for (dependent<int>::const_iterator it = v.begin(), e = v.end();
       it != e; ++it) {
    printf("Fibonacci number is %d\n", *it);
  }
  // CHECK-FIXES: for (dependent<int>::const_iterator it = v.begin(), e = v.end();
  // CHECK-FIXES-NEXT: it != e; ++it) {
  // CHECK-FIXES-NEXT: printf("Fibonacci number is %d\n", *it);

  for (dependent<int>::const_iterator it(v.begin()), e = v.end();
       it != e; ++it) {
    printf("Fibonacci number is %d\n", *it);
  }
  // CHECK-FIXES: for (dependent<int>::const_iterator it(v.begin()), e = v.end();
  // CHECK-FIXES-NEXT: it != e; ++it) {
  // CHECK-FIXES-NEXT: printf("Fibonacci number is %d\n", *it);
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
    }
    // CHECK-MESSAGES: :[[@LINE-2]]:5: warning: use range-based for loop instead
    // CHECK-FIXES: for (auto & elem : *this) {

    for (iterator I = C::begin(), E = C::end(); I != E; ++I) {
    }
    // CHECK-MESSAGES: :[[@LINE-2]]:5: warning: use range-based for loop instead
    // CHECK-FIXES: for (auto & elem : *this) {

    for (iterator I = begin(), E = end(); I != E; ++I) {
      doSomething();
    }

    for (iterator I = begin(); I != end(); ++I) {
    }
    // CHECK-MESSAGES: :[[@LINE-2]]:5: warning: use range-based for loop instead
    // CHECK-FIXES: for (auto & elem : *this) {

    for (iterator I = begin(); I != end(); ++I) {
      doSomething();
    }
  }

  void doLoop() const {
    for (const_iterator I = begin(), E = end(); I != E; ++I) {
    }
    // CHECK-MESSAGES: :[[@LINE-2]]:5: warning: use range-based for loop instead
    // CHECK-FIXES: for (auto & elem : *this) {

    for (const_iterator I = C::begin(), E = C::end(); I != E; ++I) {
    }
    // CHECK-MESSAGES: :[[@LINE-2]]:5: warning: use range-based for loop instead
    // CHECK-FIXES: for (auto & elem : *this) {

    for (const_iterator I = begin(), E = end(); I != E; ++I) {
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
    }
    // CHECK-MESSAGES: :[[@LINE-2]]:5: warning: use range-based for loop instead
    // CHECK-FIXES: for (auto & elem : *this) {
  }
};

} // namespace Iterator

namespace PseudoArray {

const int N = 6;
dependent<int> v;
dependent<int> *pv;

transparent<dependent<int>> cv;

void f() {
  int sum = 0;
  for (int i = 0, e = v.size(); i < e; ++i) {
    printf("Fibonacci number is %d\n", v[i]);
    sum += v[i] + 2;
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & elem : v)
  // CHECK-FIXES-NEXT: printf("Fibonacci number is %d\n", elem);
  // CHECK-FIXES-NEXT: sum += elem + 2;

  for (int i = 0, e = v.size(); i < e; ++i) {
    printf("Fibonacci number is %d\n", v.at(i));
    sum += v.at(i) + 2;
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & elem : v)
  // CHECK-FIXES-NEXT: printf("Fibonacci number is %d\n", elem);
  // CHECK-FIXES-NEXT: sum += elem + 2;

  for (int i = 0, e = pv->size(); i < e; ++i) {
    printf("Fibonacci number is %d\n", pv->at(i));
    sum += pv->at(i) + 2;
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & elem : *pv)
  // CHECK-FIXES-NEXT: printf("Fibonacci number is %d\n", elem);
  // CHECK-FIXES-NEXT: sum += elem + 2;

  // This test will fail if size() isn't called repeatedly, since it
  // returns unsigned int, and 0 is deduced to be signed int.
  // FIXME: Insert the necessary explicit conversion, or write out the types
  // explicitly.
  for (int i = 0; i < pv->size(); ++i) {
    printf("Fibonacci number is %d\n", (*pv).at(i));
    sum += (*pv)[i] + 2;
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & elem : *pv)
  // CHECK-FIXES-NEXT: printf("Fibonacci number is %d\n", elem);
  // CHECK-FIXES-NEXT: sum += elem + 2;

  for (int i = 0; i < cv->size(); ++i) {
    printf("Fibonacci number is %d\n", cv->at(i));
    sum += cv->at(i) + 2;
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & elem : *cv)
  // CHECK-FIXES-NEXT: printf("Fibonacci number is %d\n", elem);
  // CHECK-FIXES-NEXT: sum += elem + 2;
}

// Check for loops that don't mention containers.
void noContainer() {
  for (auto i = 0; i < v.size(); ++i) {
  }
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (const auto & elem : v) {

  for (auto i = 0; i < v.size(); ++i)
    ;
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (const auto & elem : v)
}

struct NoBeginEnd {
  unsigned size() const;
};

struct NoConstBeginEnd {
  NoConstBeginEnd();
  unsigned size() const;
  unsigned* begin();
  unsigned* end();
};

struct ConstBeginEnd {
  ConstBeginEnd();
  unsigned size() const;
  unsigned* begin() const;
  unsigned* end() const;
};

// Shouldn't transform pseudo-array uses if the container doesn't provide
// begin() and end() of the right const-ness.
void NoBeginEndTest() {
  NoBeginEnd NBE;
  for (unsigned i = 0, e = NBE.size(); i < e; ++i) {
  }

  const NoConstBeginEnd const_NCBE;
  for (unsigned i = 0, e = const_NCBE.size(); i < e; ++i) {
  }

  ConstBeginEnd CBE;
  for (unsigned i = 0, e = CBE.size(); i < e; ++i) {
  }
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (const auto & elem : CBE) {

  const ConstBeginEnd const_CBE;
  for (unsigned i = 0, e = const_CBE.size(); i < e; ++i) {
  }
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (const auto & elem : const_CBE) {
}

struct DerefByValue {
  DerefByValue();
  struct iter { unsigned operator*(); };
  unsigned size() const;
  iter begin();
  iter end();
  unsigned operator[](int);
};

void DerefByValueTest() {
  DerefByValue DBV;
  for (unsigned i = 0, e = DBV.size(); i < e; ++i) {
    printf("%d\n", DBV[i]);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto && elem : DBV) {

}

} // namespace PseudoArray
