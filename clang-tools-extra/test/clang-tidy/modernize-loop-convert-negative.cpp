// RUN: %python %S/check_clang_tidy.py %s modernize-loop-convert %t -- -std=c++11 -I %S/Inputs/modernize-loop-convert

#include "structures.h"

// CHECK-FIXES-NOT: for ({{.*[^:]:[^:].*}})
// CHECK-MESSAGES-NOT: modernize-loop-convert

namespace Negative {

const int N = 6;
int arr[N] = {1, 2, 3, 4, 5, 6};
int (*pArr)[N] = &arr;
int Sum = 0;

// Checks for the Index start and end:
void IndexStartAndEnd() {
  for (int I = 0; I < N + 1; ++I)
    Sum += arr[I];

  for (int I = 0; I < N - 1; ++I)
    Sum += arr[I];

  for (int I = 1; I < N; ++I)
    Sum += arr[I];

  for (int I = 1; I < N; ++I)
    Sum += arr[I];

  for (int I = 0;; ++I)
    Sum += (*pArr)[I];
}

// Checks for invalid increment steps:
void increment() {
  for (int I = 0; I < N; --I)
    Sum += arr[I];

  for (int I = 0; I < N; I)
    Sum += arr[I];

  for (int I = 0; I < N;)
    Sum += arr[I];

  for (int I = 0; I < N; I += 2)
    Sum++;
}

// Checks to make sure that the Index isn't used outside of the array:
void IndexUse() {
  for (int I = 0; I < N; ++I)
    arr[I] += 1 + I;
}

// Check for loops that don't mention arrays
void noArray() {
  for (int I = 0; I < N; ++I)
    Sum += I;

  for (int I = 0; I < N; ++I) {
  }

  for (int I = 0; I < N; ++I)
    ;
}

// Checks for incorrect loop variables.
void mixedVariables() {
  int badIndex;
  for (int I = 0; badIndex < N; ++I)
    Sum += arr[I];

  for (int I = 0; I < N; ++badIndex)
    Sum += arr[I];

  for (int I = 0; badIndex < N; ++badIndex)
    Sum += arr[I];

  for (int I = 0; badIndex < N; ++badIndex)
    Sum += arr[badIndex];
}

// Checks for multiple arrays Indexed.
void multipleArrays() {
  int badArr[N];

  for (int I = 0; I < N; ++I)
    Sum += arr[I] + badArr[I];

  for (int I = 0; I < N; ++I) {
    int K = badArr[I];
    Sum += arr[I] + K;
  }
}

struct HasArr {
  int Arr[N];
  Val ValArr[N];
};

struct HasIndirectArr {
  HasArr HA;
  void implicitThis() {
    for (int I = 0; I < N; ++I) {
      printf("%d", HA.Arr[I]);
    }

    for (int I = 0; I < N; ++I) {
      printf("%d", HA.ValArr[I].x);
    }
  }

  void explicitThis() {
    for (int I = 0; I < N; ++I) {
      printf("%d", this->HA.Arr[I]);
    }

    for (int I = 0; I < N; ++I) {
      printf("%d", this->HA.ValArr[I].x);
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
  for (T::iterator I = Bad.notBegin(), E = Bad.end();  I != E; ++I)
    int K = *I;

  for (T::iterator I = Bad.begin(), E = Bad.notEnd();  I != E; ++I)
    int K = *I;
}

void badLoopShapes() {
  for (T::iterator I = t.begin(), E = t.end(), F = E;  I != E; ++I)
    int K = *I;

  for (T::iterator I = t.begin(), E = t.end();  I != E;)
    int K = *I;

  for (T::iterator I = t.begin(), E = t.end();; ++I)
    int K = *I;

  T::iterator outsideI;
  T::iterator outsideE;

  for (; outsideI != outsideE; ++outsideI)
    int K = *outsideI;
}

void iteratorArrayMix() {
  int lower;
  const int N = 6;
  for (T::iterator I = t.begin(), E = t.end(); lower < N; ++I)
    int K = *I;

  for (T::iterator I = t.begin(), E = t.end(); lower < N; ++lower)
    int K = *I;
}

struct ExtraConstructor : T::iterator {
  ExtraConstructor(T::iterator, int);
  explicit ExtraConstructor(T::iterator);
};

void badConstructor() {
  for (T::iterator I = ExtraConstructor(t.begin(), 0), E = t.end();
        I != E; ++I)
    int K = *I;
  for (T::iterator I = ExtraConstructor(t.begin()), E = t.end();  I != E; ++I)
    int K = *I;
}

void foo(S::iterator It) {}
class Foo {public: void bar(S::iterator It); };
Foo fo;

void iteratorUsed() {
  for (S::iterator I = s.begin(), E = s.end();  I != E; ++I)
    foo(I);

  for (S::iterator I = s.begin(), E = s.end();  I != E; ++I)
    fo.bar(I);

  S::iterator Ret;
  for (S::iterator I = s.begin(), E = s.end();  I != E; ++I)
    Ret = I;
}

void iteratorMemberUsed() {
  for (T::iterator I = t.begin(), E = t.end();  I != E; ++I)
    I.x = *I;

  for (T::iterator I = t.begin(), E = t.end();  I != E; ++I)
    int K = I.x + *I;

  for (T::iterator I = t.begin(), E = t.end();  I != E; ++I)
    int K = E.x + *I;
}

void iteratorMethodCalled() {
  for (T::iterator I = t.begin(), E = t.end();  I != E; ++I)
    I.insert(3);

  for (T::iterator I = t.begin(), E = t.end();  I != E; ++I)
    if (I != I)
      int K = 3;
}

void iteratorOperatorCalled() {
  for (T::iterator I = t.begin(), E = t.end();  I != E; ++I)
    int K = *(++I);

  for (S::iterator I = s.begin(), E = s.end();  I != E; ++I)
    MutableVal K = *(++I);
}

void differentContainers() {
  T other;
  for (T::iterator I = t.begin(), E = other.end();  I != E; ++I)
    int K = *I;

  for (T::iterator I = other.begin(), E = t.end();  I != E; ++I)
    int K = *I;

  S otherS;
  for (S::iterator I = s.begin(), E = otherS.end();  I != E; ++I)
    MutableVal K = *I;

  for (S::iterator I = otherS.begin(), E = s.end();  I != E; ++I)
    MutableVal K = *I;
}

void wrongIterators() {
  T::iterator other;
  for (T::iterator I = t.begin(), E = t.end(); I != other; ++I)
    int K = *I;
}

struct EvilArrow : U {
  // Please, no one ever write code like this.
  U *operator->();
};

void differentMemberAccessTypes() {
  EvilArrow A;
  for (EvilArrow::iterator I = A.begin(), E = A->end();  I != E; ++I)
    Val K = *I;
  for (EvilArrow::iterator I = A->begin(), E = A.end();  I != E; ++I)
    Val K = *I;
}

void f(const T::iterator &it, int);
void f(const T &it, int);
void g(T &it, int);

void iteratorPassedToFunction() {
  for (T::iterator I = t.begin(), E = t.end();  I != E; ++I)
    f(I, *I);
}

// FIXME: These tests can be removed if this tool ever does enough analysis to
// decide that this is a safe transformation. Until then, we don't want it
// applied.
void iteratorDefinedOutside() {
  T::iterator TheEnd = t.end();
  for (T::iterator I = t.begin(); I != TheEnd; ++I)
    int K = *I;

  T::iterator TheBegin = t.begin();
  for (T::iterator E = t.end(); TheBegin != E; ++TheBegin)
    int K = *TheBegin;
}

} // namespace NegativeIterator

namespace NegativePseudoArray {

const int N = 6;
dependent<int> v;
dependent<int> *pv;

int Sum = 0;

// Checks for the Index start and end:
void IndexStartAndEnd() {
  for (int I = 0; I < v.size() + 1; ++I)
    Sum += v[I];

  for (int I = 0; I < v.size() - 1; ++I)
    Sum += v[I];

  for (int I = 1; I < v.size(); ++I)
    Sum += v[I];

  for (int I = 1; I < v.size(); ++I)
    Sum += v[I];

  for (int I = 0;; ++I)
    Sum += (*pv)[I];
}

// Checks for invalid increment steps:
void increment() {
  for (int I = 0; I < v.size(); --I)
    Sum += v[I];

  for (int I = 0; I < v.size(); I)
    Sum += v[I];

  for (int I = 0; I < v.size();)
    Sum += v[I];

  for (int I = 0; I < v.size(); I += 2)
    Sum++;
}

// Checks to make sure that the Index isn't used outside of the container:
void IndexUse() {
  for (int I = 0; I < v.size(); ++I)
    v[I] += 1 + I;
}

// Checks for incorrect loop variables.
void mixedVariables() {
  int badIndex;
  for (int I = 0; badIndex < v.size(); ++I)
    Sum += v[I];

  for (int I = 0; I < v.size(); ++badIndex)
    Sum += v[I];

  for (int I = 0; badIndex < v.size(); ++badIndex)
    Sum += v[I];

  for (int I = 0; badIndex < v.size(); ++badIndex)
    Sum += v[badIndex];
}

// Checks for an array Indexed in addition to the container.
void multipleArrays() {
  int badArr[N];

  for (int I = 0; I < v.size(); ++I)
    Sum += v[I] + badArr[I];

  for (int I = 0; I < v.size(); ++I)
    Sum += badArr[I];

  for (int I = 0; I < v.size(); ++I) {
    int K = badArr[I];
    Sum += K + 2;
  }

  for (int I = 0; I < v.size(); ++I) {
    int K = badArr[I];
    Sum += v[I] + K;
  }
}

// Checks for multiple containers being Indexed container.
void multipleContainers() {
  dependent<int> badArr;

  for (int I = 0; I < v.size(); ++I)
    Sum += v[I] + badArr[I];

  for (int I = 0; I < v.size(); ++I)
    Sum += badArr[I];

  for (int I = 0; I < v.size(); ++I) {
    int K = badArr[I];
    Sum += K + 2;
  }

  for (int I = 0; I < v.size(); ++I) {
    int K = badArr[I];
    Sum += v[I] + K;
  }
}

// Check to make sure that dereferenced pointers-to-containers behave nicely.
void derefContainer() {
  // Note the dependent<T>::operator*() returns another dependent<T>.
  // This test makes sure that we don't allow an arbitrary number of *'s.
  for (int I = 0; I < pv->size(); ++I)
    Sum += (**pv).at(I);

  for (int I = 0; I < pv->size(); ++I)
    Sum += (**pv)[I];
}

void wrongEnd() {
  int Bad;
  for (int I = 0, E = v.size(); I < Bad; ++I)
    Sum += v[I];
}

// Checks to see that non-const member functions are not called on the container
// object.
// These could be conceivably allowed with a lower required confidence level.
void memberFunctionCalled() {
  for (int I = 0; I < v.size(); ++I) {
    Sum += v[I];
    v.foo();
  }

  for (int I = 0; I < v.size(); ++I) {
    Sum += v[I];
    dependent<int>::iterator It = v.begin();
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
  for (S::iterator I = x.s.begin(), E = x.s.end();  I != E; ++I) {
    f(x);
    MutableVal K = *I;
  }

  for (T::iterator I = x.t.begin(), E = x.t.end();  I != E; ++I) {
    f(x);
    int K = *I;
  }

  for (S::iterator I = x.s.begin(), E = x.s.end();  I != E; ++I) {
    f(x.s);
    MutableVal K = *I;
  }

  for (T::iterator I = x.t.begin(), E = x.t.end();  I != E; ++I) {
    f(x.t);
    int K = *I;
  }

  for (S::iterator I = x.getS().begin(), E = x.getS().end();  I != E; ++I) {
    f(x.getS());
    MutableVal K = *I;
  }

  X Exes[5];
  int Index = 0;

  for (S::iterator I = Exes[Index].getS().begin(),
                   E = Exes[Index].getS().end();
        I != E; ++I) {
    Index++;
    MutableVal K = *I;
  }
}

} // namespace NegativeMultiEndCall

namespace NoUsages {

const int N = 6;
int arr[N] = {1, 2, 3, 4, 5, 6};
S s;
dependent<int> v;
int Count = 0;

void foo();

void f() {
  for (int I = 0; I < N; ++I) {}
  for (int I = 0; I < N; ++I)
    printf("Hello world\n");
  for (int I = 0; I < N; ++I)
    ++Count;
  for (int I = 0; I < N; ++I)
    foo();

  for (S::iterator I = s.begin(), E = s.end(); I != E; ++I) {}
  for (S::iterator I = s.begin(), E = s.end(); I != E; ++I)
    printf("Hello world\n");
  for (S::iterator I = s.begin(), E = s.end(); I != E; ++I)
    ++Count;
  for (S::iterator I = s.begin(), E = s.end(); I != E; ++I)
    foo();

  for (int I = 0; I < v.size(); ++I) {}
  for (int I = 0; I < v.size(); ++I)
    printf("Hello world\n");
  for (int I = 0; I < v.size(); ++I)
    ++Count;
  for (int I = 0; I < v.size(); ++I)
    foo();
}

} // namespace NoUsages
