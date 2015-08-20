// RUN: %python %S/check_clang_tidy.py %s modernize-loop-convert %t -- -std=c++11 -I %S/Inputs/modernize-loop-convert

#include "structures.h"

// CHECK-FIXES-NOT: for ({{.*[^:]:[^:].*}})

namespace Negative {

const int N = 6;
int arr[N] = {1, 2, 3, 4, 5, 6};
int (*pArr)[N] = &arr;
int sum = 0;

// Checks for the index start and end:
void indexStartAndEnd() {
  for (int i = 0; i < N + 1; ++i)
    sum += arr[i];

  for (int i = 0; i < N - 1; ++i)
    sum += arr[i];

  for (int i = 1; i < N; ++i)
    sum += arr[i];

  for (int i = 1; i < N; ++i)
    sum += arr[i];

  for (int i = 0;; ++i)
    sum += (*pArr)[i];
}

// Checks for invalid increment steps:
void increment() {
  for (int i = 0; i < N; --i)
    sum += arr[i];

  for (int i = 0; i < N; i)
    sum += arr[i];

  for (int i = 0; i < N;)
    sum += arr[i];

  for (int i = 0; i < N; i += 2)
    sum++;
}

// Checks to make sure that the index isn't used outside of the array:
void indexUse() {
  for (int i = 0; i < N; ++i)
    arr[i] += 1 + i;
}

// Check for loops that don't mention arrays
void noArray() {
  for (int i = 0; i < N; ++i)
    sum += i;

  for (int i = 0; i < N; ++i) {
  }

  for (int i = 0; i < N; ++i)
    ;
}

// Checks for incorrect loop variables.
void mixedVariables() {
  int badIndex;
  for (int i = 0; badIndex < N; ++i)
    sum += arr[i];

  for (int i = 0; i < N; ++badIndex)
    sum += arr[i];

  for (int i = 0; badIndex < N; ++badIndex)
    sum += arr[i];

  for (int i = 0; badIndex < N; ++badIndex)
    sum += arr[badIndex];
}

// Checks for multiple arrays indexed.
void multipleArrays() {
  int badArr[N];

  for (int i = 0; i < N; ++i)
    sum += arr[i] + badArr[i];

  for (int i = 0; i < N; ++i) {
    int k = badArr[i];
    sum += arr[i] + k;
  }
}

struct HasArr {
  int Arr[N];
  Val ValArr[N];
};

struct HasIndirectArr {
  HasArr HA;
  void implicitThis() {
    for (int i = 0; i < N; ++i) {
      printf("%d", HA.Arr[i]);
    }

    for (int i = 0; i < N; ++i) {
      printf("%d", HA.ValArr[i].x);
    }
  }

  void explicitThis() {
    for (int i = 0; i < N; ++i) {
      printf("%d", this->HA.Arr[i]);
    }

    for (int i = 0; i < N; ++i) {
      printf("%d", this->HA.ValArr[i].x);
    }
  }
};
}

namespace NegativeIterator {

S s;
T t;
U u;

struct BadBeginEnd : T {
  iterator notBegin();
  iterator notEnd();
};

void notBeginOrEnd() {
  BadBeginEnd Bad;
  for (T::iterator i = Bad.notBegin(), e = Bad.end(); i != e; ++i)
    int k = *i;

  for (T::iterator i = Bad.begin(), e = Bad.notEnd(); i != e; ++i)
    int k = *i;
}

void badLoopShapes() {
  for (T::iterator i = t.begin(), e = t.end(), f = e; i != e; ++i)
    int k = *i;

  for (T::iterator i = t.begin(), e = t.end(); i != e;)
    int k = *i;

  for (T::iterator i = t.begin(), e = t.end();; ++i)
    int k = *i;

  T::iterator outsideI;
  T::iterator outsideE;

  for (; outsideI != outsideE; ++outsideI)
    int k = *outsideI;
}

void iteratorArrayMix() {
  int lower;
  const int N = 6;
  for (T::iterator i = t.begin(), e = t.end(); lower < N; ++i)
    int k = *i;

  for (T::iterator i = t.begin(), e = t.end(); lower < N; ++lower)
    int k = *i;
}

struct ExtraConstructor : T::iterator {
  ExtraConstructor(T::iterator, int);
  explicit ExtraConstructor(T::iterator);
};

void badConstructor() {
  for (T::iterator i = ExtraConstructor(t.begin(), 0), e = t.end();
       i != e; ++i)
    int k = *i;
  for (T::iterator i = ExtraConstructor(t.begin()), e = t.end(); i != e; ++i)
    int k = *i;
}

void iteratorMemberUsed() {
  for (T::iterator i = t.begin(), e = t.end(); i != e; ++i)
    i.x = *i;

  for (T::iterator i = t.begin(), e = t.end(); i != e; ++i)
    int k = i.x + *i;

  for (T::iterator i = t.begin(), e = t.end(); i != e; ++i)
    int k = e.x + *i;
}

void iteratorMethodCalled() {
  for (T::iterator i = t.begin(), e = t.end(); i != e; ++i)
    i.insert(3);

  for (T::iterator i = t.begin(), e = t.end(); i != e; ++i)
    if (i != i)
      int k = 3;
}

void iteratorOperatorCalled() {
  for (T::iterator i = t.begin(), e = t.end(); i != e; ++i)
    int k = *(++i);

  for (S::iterator i = s.begin(), e = s.end(); i != e; ++i)
    MutableVal k = *(++i);
}

void differentContainers() {
  T other;
  for (T::iterator i = t.begin(), e = other.end(); i != e; ++i)
    int k = *i;

  for (T::iterator i = other.begin(), e = t.end(); i != e; ++i)
    int k = *i;

  S otherS;
  for (S::iterator i = s.begin(), e = otherS.end(); i != e; ++i)
    MutableVal k = *i;

  for (S::iterator i = otherS.begin(), e = s.end(); i != e; ++i)
    MutableVal k = *i;
}

void wrongIterators() {
  T::iterator other;
  for (T::iterator i = t.begin(), e = t.end(); i != other; ++i)
    int k = *i;
}

struct EvilArrow : U {
  // Please, no one ever write code like this.
  U *operator->();
};

void differentMemberAccessTypes() {
  EvilArrow A;
  for (EvilArrow::iterator i = A.begin(), e = A->end(); i != e; ++i)
    Val k = *i;
  for (EvilArrow::iterator i = A->begin(), e = A.end(); i != e; ++i)
    Val k = *i;
}

void f(const T::iterator &it, int);
void f(const T &it, int);
void g(T &it, int);

void iteratorPassedToFunction() {
  for (T::iterator i = t.begin(), e = t.end(); i != e; ++i)
    f(i, *i);
}

// FIXME: These tests can be removed if this tool ever does enough analysis to
// decide that this is a safe transformation. Until then, we don't want it
// applied.
void iteratorDefinedOutside() {
  T::iterator theEnd = t.end();
  for (T::iterator i = t.begin(); i != theEnd; ++i)
    int k = *i;

  T::iterator theBegin = t.begin();
  for (T::iterator e = t.end(); theBegin != e; ++theBegin)
    int k = *theBegin;
}

} // namespace NegativeIterator

namespace NegativePseudoArray {

const int N = 6;
dependent<int> v;
dependent<int> *pv;

transparent<dependent<int>> cv;
int sum = 0;

// Checks for the index start and end:
void indexStartAndEnd() {
  for (int i = 0; i < v.size() + 1; ++i)
    sum += v[i];

  for (int i = 0; i < v.size() - 1; ++i)
    sum += v[i];

  for (int i = 1; i < v.size(); ++i)
    sum += v[i];

  for (int i = 1; i < v.size(); ++i)
    sum += v[i];

  for (int i = 0;; ++i)
    sum += (*pv)[i];
}

// Checks for invalid increment steps:
void increment() {
  for (int i = 0; i < v.size(); --i)
    sum += v[i];

  for (int i = 0; i < v.size(); i)
    sum += v[i];

  for (int i = 0; i < v.size();)
    sum += v[i];

  for (int i = 0; i < v.size(); i += 2)
    sum++;
}

// Checks to make sure that the index isn't used outside of the container:
void indexUse() {
  for (int i = 0; i < v.size(); ++i)
    v[i] += 1 + i;
}

// Checks for incorrect loop variables.
void mixedVariables() {
  int badIndex;
  for (int i = 0; badIndex < v.size(); ++i)
    sum += v[i];

  for (int i = 0; i < v.size(); ++badIndex)
    sum += v[i];

  for (int i = 0; badIndex < v.size(); ++badIndex)
    sum += v[i];

  for (int i = 0; badIndex < v.size(); ++badIndex)
    sum += v[badIndex];
}

// Checks for an array indexed in addition to the container.
void multipleArrays() {
  int badArr[N];

  for (int i = 0; i < v.size(); ++i)
    sum += v[i] + badArr[i];

  for (int i = 0; i < v.size(); ++i)
    sum += badArr[i];

  for (int i = 0; i < v.size(); ++i) {
    int k = badArr[i];
    sum += k + 2;
  }

  for (int i = 0; i < v.size(); ++i) {
    int k = badArr[i];
    sum += v[i] + k;
  }
}

// Checks for multiple containers being indexed container.
void multipleContainers() {
  dependent<int> badArr;

  for (int i = 0; i < v.size(); ++i)
    sum += v[i] + badArr[i];

  for (int i = 0; i < v.size(); ++i)
    sum += badArr[i];

  for (int i = 0; i < v.size(); ++i) {
    int k = badArr[i];
    sum += k + 2;
  }

  for (int i = 0; i < v.size(); ++i) {
    int k = badArr[i];
    sum += v[i] + k;
  }
}

// Check to make sure that dereferenced pointers-to-containers behave nicely.
void derefContainer() {
  // Note the dependent<T>::operator*() returns another dependent<T>.
  // This test makes sure that we don't allow an arbitrary number of *'s.
  for (int i = 0; i < pv->size(); ++i)
    sum += (**pv).at(i);

  for (int i = 0; i < pv->size(); ++i)
    sum += (**pv)[i];
}

void wrongEnd() {
  int bad;
  for (int i = 0, e = v.size(); i < bad; ++i)
    sum += v[i];
}

// Checks to see that non-const member functions are not called on the container
// object.
// These could be conceivably allowed with a lower required confidence level.
void memberFunctionCalled() {
  for (int i = 0; i < v.size(); ++i) {
    sum += v[i];
    v.foo();
  }

  for (int i = 0; i < v.size(); ++i) {
    sum += v[i];
    dependent<int>::iterator it = v.begin();
  }
}

} // namespace NegativePseudoArray

namespace NegativeMultiEndCall {

S s;
T t;
U u;

void f(X);
void f(S);
void f(T);

void complexContainer() {
  X x;
  for (S::iterator i = x.s.begin(), e = x.s.end(); i != e; ++i) {
    f(x);
    MutableVal k = *i;
  }

  for (T::iterator i = x.t.begin(), e = x.t.end(); i != e; ++i) {
    f(x);
    int k = *i;
  }

  for (S::iterator i = x.s.begin(), e = x.s.end(); i != e; ++i) {
    f(x.s);
    MutableVal k = *i;
  }

  for (T::iterator i = x.t.begin(), e = x.t.end(); i != e; ++i) {
    f(x.t);
    int k = *i;
  }

  for (S::iterator i = x.getS().begin(), e = x.getS().end(); i != e; ++i) {
    f(x.getS());
    MutableVal k = *i;
  }

  X exes[5];
  int index = 0;

  for (S::iterator i = exes[index].getS().begin(),
                   e = exes[index].getS().end();
       i != e; ++i) {
    index++;
    MutableVal k = *i;
  }
}

} // namespace NegativeMultiEndCall
