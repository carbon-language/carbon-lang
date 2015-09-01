// RUN: %python %S/check_clang_tidy.py %s modernize-loop-convert %t -- -std=c++11 -I %S/Inputs/modernize-loop-convert

#include "structures.h"

namespace Dependency {

void f() {
  const int N = 6;
  const int M = 8;
  int arr[N][M];

  for (int i = 0; i < N; ++i) {
    int a = 0;
    int b = arr[i][a];
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & elem : arr) {
  // CHECK-FIXES-NEXT: int a = 0;
  // CHECK-FIXES-NEXT: int b = elem[a];
  // CHECK-FIXES-NEXT: }

  for (int j = 0; j < M; ++j) {
    int a = 0;
    int b = arr[a][j];
  }
}

} // namespace Dependency

namespace NamingAlias {

const int N = 10;

Val Arr[N];
dependent<Val> v;
dependent<Val> *pv;
Val &func(Val &);
void sideEffect(int);

void aliasing() {
  // If the loop container is only used for a declaration of a temporary
  // variable to hold each element, we can name the new variable for the
  // converted range-based loop as the temporary variable's name.

  // In the following case, "t" is used as a temporary variable to hold each
  // element, and thus we consider the name "t" aliased to the loop.
  // The extra blank braces are left as a placeholder for after the variable
  // declaration is deleted.
  for (int i = 0; i < N; ++i) {
    Val &t = Arr[i];
    {}
    int y = t.x;
  }
  // CHECK-MESSAGES: :[[@LINE-5]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & t : Arr)
  // CHECK-FIXES-NOT: Val &{{[a-z_]+}} =
  // CHECK-FIXES: {}
  // CHECK-FIXES-NEXT: int y = t.x;

  // The container was not only used to initialize a temporary loop variable for
  // the container's elements, so we do not alias the new loop variable.
  for (int i = 0; i < N; ++i) {
    Val &t = Arr[i];
    int y = t.x;
    int z = Arr[i].x + t.x;
  }
  // CHECK-MESSAGES: :[[@LINE-5]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & elem : Arr)
  // CHECK-FIXES-NEXT: Val &t = elem;
  // CHECK-FIXES-NEXT: int y = t.x;
  // CHECK-FIXES-NEXT: int z = elem.x + t.x;

  for (int i = 0; i < N; ++i) {
    Val t = Arr[i];
    int y = t.x;
    int z = Arr[i].x + t.x;
  }
  // CHECK-MESSAGES: :[[@LINE-5]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & elem : Arr)
  // CHECK-FIXES-NEXT: Val t = elem;
  // CHECK-FIXES-NEXT: int y = t.x;
  // CHECK-FIXES-NEXT: int z = elem.x + t.x;

  // The same for pseudo-arrays like std::vector<T> (or here dependent<Val>)
  // which provide a subscript operator[].
  for (int i = 0; i < v.size(); ++i) {
    Val &t = v[i];
    {}
    int y = t.x;
  }
  // CHECK-MESSAGES: :[[@LINE-5]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & t : v)
  // CHECK-FIXES: {}
  // CHECK-FIXES-NEXT: int y = t.x;

  // The same with a call to at()
  for (int i = 0; i < pv->size(); ++i) {
    Val &t = pv->at(i);
    {}
    int y = t.x;
  }
  // CHECK-MESSAGES: :[[@LINE-5]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & t : *pv)
  // CHECK-FIXES: {}
  // CHECK-FIXES-NEXT: int y = t.x;

  for (int i = 0; i < N; ++i) {
    Val &t = func(Arr[i]);
    int y = t.x;
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & elem : Arr)
  // CHECK-FIXES-NEXT: Val &t = func(elem);
  // CHECK-FIXES-NEXT: int y = t.x;

  int IntArr[N];
  for (unsigned i = 0; i < N; ++i) {
    if (int alias = IntArr[i]) {
      sideEffect(alias);
    }
  }
  // CHECK-MESSAGES: :[[@LINE-5]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto alias : IntArr)
  // CHECK-FIXES-NEXT: if (alias) {

  for (unsigned i = 0; i < N; ++i) {
    while (int alias = IntArr[i]) {
      sideEffect(alias);
    }
  }
  // CHECK-MESSAGES: :[[@LINE-5]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto alias : IntArr)
  // CHECK-FIXES-NEXT: while (alias) {

  for (unsigned i = 0; i < N; ++i) {
    switch (int alias = IntArr[i]) {
    default:
      sideEffect(alias);
    }
  }
  // CHECK-MESSAGES: :[[@LINE-6]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto alias : IntArr)
  // CHECK-FIXES-NEXT: switch (alias) {

  for (unsigned i = 0; i < N; ++i) {
    for (int alias = IntArr[i]; alias < N; ++alias) {
      sideEffect(alias);
    }
  }
  // CHECK-MESSAGES: :[[@LINE-5]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto alias : IntArr)
  // CHECK-FIXES-NEXT: for (; alias < N; ++alias) {

  for (unsigned i = 0; i < N; ++i) {
    for (unsigned j = 0; int alias = IntArr[i]; ++j) {
      sideEffect(alias);
    }
  }
  // CHECK-MESSAGES: :[[@LINE-5]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto alias : IntArr)
  // CHECK-FIXES-NEXT: for (unsigned j = 0; alias; ++j) {
}

void refs_and_vals() {
  // The following tests check that the transform correctly preserves the
  // reference or value qualifiers of the aliased variable. That is, if the
  // variable was declared as a value, the loop variable will be declared as a
  // value and vice versa for references.

  S s;
  const S s_const = s;

  for (S::const_iterator it = s_const.begin(); it != s_const.end(); ++it) {
    MutableVal alias = *it;
    {}
    alias.x = 0;
  }
  // CHECK-MESSAGES: :[[@LINE-5]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto alias : s_const)
  // CHECK-FIXES-NOT: MutableVal {{[a-z_]+}} =
  // CHECK-FIXES: {}
  // CHECK-FIXES-NEXT: alias.x = 0;

  for (S::iterator it = s.begin(), e = s.end(); it != e; ++it) {
    MutableVal alias = *it;
    {}
    alias.x = 0;
  }
  // CHECK-MESSAGES: :[[@LINE-5]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto alias : s)
  // CHECK-FIXES-NOT: MutableVal {{[a-z_]+}} =
  // CHECK-FIXES: {}
  // CHECK-FIXES-NEXT: alias.x = 0;

  for (S::iterator it = s.begin(), e = s.end(); it != e; ++it) {
    MutableVal &alias = *it;
    {}
    alias.x = 0;
  }
  // CHECK-MESSAGES: :[[@LINE-5]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & alias : s)
  // CHECK-FIXES-NOT: MutableVal &{{[a-z_]+}} =
  // CHECK-FIXES: {}
  // CHECK-FIXES-NEXT: alias.x = 0;

  dependent<int> dep, other;
  for (dependent<int>::iterator it = dep.begin(), e = dep.end(); it != e; ++it) {
    printf("%d\n", *it);
    const int& idx = other[0];
    unsigned othersize = other.size();
  }
  // CHECK-MESSAGES: :[[@LINE-5]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & elem : dep)
  // CHECK-FIXES-NEXT: printf("%d\n", elem);
  // CHECK-FIXES-NEXT: const int& idx = other[0];
  // CHECK-FIXES-NEXT: unsigned othersize = other.size();

  for (int i = 0, e = dep.size(); i != e; ++i) {
    int idx = other.at(i);
  }
}

} // namespace NamingAlias

namespace NamingConlict {

#define MAX(a, b) (a > b) ? a : b
#define DEF 5

const int N = 10;
int nums[N];
int sum = 0;

namespace ns {
struct st {
  int x;
};
}

void sameNames() {
  int num = 0;
  for (int i = 0; i < N; ++i) {
    printf("Fibonacci number is %d\n", nums[i]);
    sum += nums[i] + 2 + num;
    (void)nums[i];
  }
  // CHECK-MESSAGES: :[[@LINE-5]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & nums_i : nums)
  // CHECK-FIXES-NEXT: printf("Fibonacci number is %d\n", nums_i);
  // CHECK-FIXES-NEXT: sum += nums_i + 2 + num;
  // CHECK-FIXES-NOT: (void) num;
}

void macroConflict() {
  S MAXs;
  for (S::iterator it = MAXs.begin(), e = MAXs.end(); it != e; ++it) {
    printf("s has value %d\n", (*it).x);
    printf("Max of 3 and 5: %d\n", MAX(3, 5));
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & MAXs_it : MAXs)
  // CHECK-FIXES-NEXT: printf("s has value %d\n", (MAXs_it).x);
  // CHECK-FIXES-NEXT: printf("Max of 3 and 5: %d\n", MAX(3, 5));

  for (S::const_iterator it = MAXs.begin(), e = MAXs.end(); it != e; ++it) {
    printf("s has value %d\n", (*it).x);
    printf("Max of 3 and 5: %d\n", MAX(3, 5));
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (const auto & MAXs_it : MAXs)
  // CHECK-FIXES-NEXT: printf("s has value %d\n", (MAXs_it).x);
  // CHECK-FIXES-NEXT: printf("Max of 3 and 5: %d\n", MAX(3, 5));

  T DEFs;
  for (T::iterator it = DEFs.begin(), e = DEFs.end(); it != e; ++it) {
    if (*it == DEF) {
      printf("I found %d\n", *it);
    }
  }
  // CHECK-MESSAGES: :[[@LINE-5]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & DEFs_it : DEFs)
  // CHECK-FIXES-NEXT: if (DEFs_it == DEF) {
  // CHECK-FIXES-NEXT: printf("I found %d\n", DEFs_it);
}

void keywordConflict() {
  T ints;
  for (T::iterator it = ints.begin(), e = ints.end(); it != e; ++it) {
    *it = 5;
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & ints_it : ints)
  // CHECK-FIXES-NEXT: ints_it = 5;

  U __FUNCTION__s;
  for (U::iterator it = __FUNCTION__s.begin(), e = __FUNCTION__s.end();
       it != e; ++it) {
    int __FUNCTION__s_it = (*it).x + 2;
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & __FUNCTION__s_elem : __FUNCTION__s)
  // CHECK-FIXES-NEXT: int __FUNCTION__s_it = (__FUNCTION__s_elem).x + 2;
}

void typeConflict() {
  T Vals;
  // Using the name "Val", although it is the name of an existing struct, is
  // safe in this loop since it will only exist within this scope.
  for (T::iterator it = Vals.begin(), e = Vals.end(); it != e; ++it) {
  }
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & Val : Vals)

  // We cannot use the name "Val" in this loop since there is a reference to
  // it in the body of the loop.
  for (T::iterator it = Vals.begin(), e = Vals.end(); it != e; ++it) {
    *it = sizeof(Val);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & Vals_it : Vals)
  // CHECK-FIXES-NEXT: Vals_it = sizeof(Val);

  typedef struct Val TD;
  U TDs;
  // Naming the variable "TD" within this loop is safe because the typedef
  // was never used within the loop.
  for (U::iterator it = TDs.begin(), e = TDs.end(); it != e; ++it) {
  }
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & TD : TDs)

  // "TD" cannot be used in this loop since the typedef is being used.
  for (U::iterator it = TDs.begin(), e = TDs.end(); it != e; ++it) {
    TD V;
    V.x = 5;
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & TDs_it : TDs)
  // CHECK-FIXES-NEXT: TD V;
  // CHECK-FIXES-NEXT: V.x = 5;

  using ns::st;
  T sts;
  for (T::iterator it = sts.begin(), e = sts.end(); it != e; ++it) {
    *it = sizeof(st);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & sts_it : sts)
  // CHECK-FIXES-NEXT: sts_it = sizeof(st);
}

} // namespace NamingConflict

namespace FreeBeginEnd {

// FIXME: Loop Convert should detect free begin()/end() functions.

struct MyArray {
  unsigned size();
};

template <typename T>
struct MyContainer {
};

int *begin(const MyArray &Arr);
int *end(const MyArray &Arr);

template <typename T>
T *begin(const MyContainer<T> &C);
template <typename T>
T *end(const MyContainer<T> &C);

// The Loop Convert Transform doesn't detect free functions begin()/end() and
// so fails to transform these cases which it should.
void f() {
  MyArray Arr;
  for (unsigned i = 0, e = Arr.size(); i < e; ++i) {
  }

  MyContainer<int> C;
  for (int *I = begin(C), *E = end(C); I != E; ++I) {
  }
}

} // namespace FreeBeginEnd

namespace Nesting {

void g(S::iterator it);
void const_g(S::const_iterator it);
class Foo {
 public:
  void g(S::iterator it);
  void const_g(S::const_iterator it);
};

void f() {
  const int N = 10;
  const int M = 15;
  Val Arr[N];
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      int k = Arr[i].x + Arr[j].x;
      // The repeat is there to allow FileCheck to make sure the two variable
      // names aren't the same.
      int l = Arr[i].x + Arr[j].x;
    }
  }
  // CHECK-MESSAGES: :[[@LINE-8]]:3: warning: use range-based for loop instead
  // CHECK-MESSAGES: :[[@LINE-8]]:5: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & elem : Arr)
  // CHECK-FIXES-NEXT: for (auto & Arr_j : Arr)
  // CHECK-FIXES-NEXT: int k = elem.x + Arr_j.x;
  // CHECK-FIXES-NOT: int l = elem.x + elem.x;

  // The inner loop is also convertible, but doesn't need to be converted
  // immediately. FIXME: update this test when that changes.
  Val Nest[N][M];
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      printf("Got item %d", Nest[i][j].x);
    }
  }
  // CHECK-MESSAGES: :[[@LINE-5]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & elem : Nest)
  // CHECK-FIXES-NEXT: for (int j = 0; j < M; ++j)
  // CHECK-FIXES-NEXT: printf("Got item %d", elem[j].x);

  // Note that the order of M and N are switched for this test.
  for (int j = 0; j < M; ++j) {
    for (int i = 0; i < N; ++i) {
      printf("Got item %d", Nest[i][j].x);
    }
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:5: warning: use range-based for loop instead
  // CHECK-FIXES-NOT: for (auto & {{[a-zA-Z_]+}} : Nest[i])
  // CHECK-FIXES: for (int j = 0; j < M; ++j)
  // CHECK-FIXES-NEXT: for (auto & elem : Nest)
  // CHECK-FIXES-NEXT: printf("Got item %d", elem[j].x);

  // The inner loop is also convertible.
  Nested<T> NestT;
  for (Nested<T>::iterator I = NestT.begin(), E = NestT.end(); I != E; ++I) {
    for (T::iterator TI = (*I).begin(), TE = (*I).end(); TI != TE; ++TI) {
      printf("%d", *TI);
    }
  }
  // CHECK-MESSAGES: :[[@LINE-5]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & elem : NestT) {
  // CHECK-FIXES-NEXT: for (T::iterator TI = (elem).begin(), TE = (elem).end(); TI != TE; ++TI) {
  // CHECK-FIXES-NEXT: printf("%d", *TI);

  // The inner loop is also convertible.
  Nested<S> NestS;
  for (Nested<S>::const_iterator I = NestS.begin(), E = NestS.end(); I != E; ++I) {
    for (S::const_iterator SI = (*I).begin(), SE = (*I).end(); SI != SE; ++SI) {
      printf("%d", *SI);
    }
  }
  // CHECK-MESSAGES: :[[@LINE-5]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (const auto & elem : NestS) {
  // CHECK-FIXES-NEXT: for (S::const_iterator SI = (elem).begin(), SE = (elem).end(); SI != SE; ++SI) {
  // CHECK-FIXES-NEXT: printf("%d", *SI);

  for (Nested<S>::const_iterator I = NestS.begin(), E = NestS.end(); I != E; ++I) {
    const S &s = *I;
    for (S::const_iterator SI = s.begin(), SE = s.end(); SI != SE; ++SI) {
      printf("%d", *SI);
      const_g(SI);
    }
  }
  // CHECK-MESSAGES: :[[@LINE-7]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (const auto & s : NestS) {

  for (Nested<S>::iterator I = NestS.begin(), E = NestS.end(); I != E; ++I) {
    S &s = *I;
    for (S::iterator SI = s.begin(), SE = s.end(); SI != SE; ++SI) {
      printf("%d", *SI);
      g(SI);
    }
  }
  // CHECK-MESSAGES: :[[@LINE-7]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & s : NestS) {

  Foo foo;
  for (Nested<S>::const_iterator I = NestS.begin(), E = NestS.end(); I != E; ++I) {
    const S &s = *I;
    for (S::const_iterator SI = s.begin(), SE = s.end(); SI != SE; ++SI) {
      printf("%d", *SI);
      foo.const_g(SI);
    }
  }
  // CHECK-MESSAGES: :[[@LINE-7]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (const auto & s : NestS) {

  for (Nested<S>::iterator I = NestS.begin(), E = NestS.end(); I != E; ++I) {
    S &s = *I;
    for (S::iterator SI = s.begin(), SE = s.end(); SI != SE; ++SI) {
      printf("%d", *SI);
      foo.g(SI);
    }
  }
  // CHECK-MESSAGES: :[[@LINE-7]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & s : NestS) {

}

} // namespace Nesting

namespace SingleIterator {

void complexContainer() {
  X exes[5];
  int index = 0;

  for (S::iterator i = exes[index].getS().begin(), e = exes[index].getS().end(); i != e; ++i) {
    MutableVal k = *i;
    MutableVal j = *i;
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & elem : exes[index].getS())
  // CHECK-FIXES-NEXT: MutableVal k = elem;
  // CHECK-FIXES-NEXT: MutableVal j = elem;
}

void f() {
  /// begin()/end() - based for loops here:
  T t;
  for (T::iterator it = t.begin(); it != t.end(); ++it) {
    printf("I found %d\n", *it);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & elem : t)
  // CHECK-FIXES-NEXT: printf("I found %d\n", elem);

  T *pt;
  for (T::iterator it = pt->begin(); it != pt->end(); ++it) {
    printf("I found %d\n", *it);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & elem : *pt)
  // CHECK-FIXES-NEXT: printf("I found %d\n", elem);

  S s;
  for (S::iterator it = s.begin(); it != s.end(); ++it) {
    printf("s has value %d\n", (*it).x);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & elem : s)
  // CHECK-FIXES-NEXT: printf("s has value %d\n", (elem).x);

  S *ps;
  for (S::iterator it = ps->begin(); it != ps->end(); ++it) {
    printf("s has value %d\n", (*it).x);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & p : *ps)
  // CHECK-FIXES-NEXT: printf("s has value %d\n", (p).x);

  for (S::iterator it = s.begin(); it != s.end(); ++it) {
    printf("s has value %d\n", it->x);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & elem : s)
  // CHECK-FIXES-NEXT: printf("s has value %d\n", elem.x);

  for (S::iterator it = s.begin(); it != s.end(); ++it) {
    it->x = 3;
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & elem : s)
  // CHECK-FIXES-NEXT: elem.x = 3;

  for (S::iterator it = s.begin(); it != s.end(); ++it) {
    (*it).x = 3;
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & elem : s)
  // CHECK-FIXES-NEXT: (elem).x = 3;

  for (S::iterator it = s.begin(); it != s.end(); ++it) {
    it->nonConstFun(4, 5);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & elem : s)
  // CHECK-FIXES-NEXT: elem.nonConstFun(4, 5);

  U u;
  for (U::iterator it = u.begin(); it != u.end(); ++it) {
    printf("s has value %d\n", it->x);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & elem : u)
  // CHECK-FIXES-NEXT: printf("s has value %d\n", elem.x);

  for (U::iterator it = u.begin(); it != u.end(); ++it) {
    printf("s has value %d\n", (*it).x);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & elem : u)
  // CHECK-FIXES-NEXT: printf("s has value %d\n", (elem).x);

  U::iterator A;
  for (U::iterator i = u.begin(); i != u.end(); ++i)
    int k = A->x + i->x;
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & elem : u)
  // CHECK-FIXES-NEXT: int k = A->x + elem.x;

  dependent<int> v;
  for (dependent<int>::iterator it = v.begin();
       it != v.end(); ++it) {
    printf("Fibonacci number is %d\n", *it);
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & elem : v) {
  // CHECK-FIXES-NEXT: printf("Fibonacci number is %d\n", elem);

  for (dependent<int>::iterator it(v.begin());
       it != v.end(); ++it) {
    printf("Fibonacci number is %d\n", *it);
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & elem : v) {
  // CHECK-FIXES-NEXT: printf("Fibonacci number is %d\n", elem);

  doublyDependent<int, int> intmap;
  for (doublyDependent<int, int>::iterator it = intmap.begin();
       it != intmap.end(); ++it) {
    printf("intmap[%d] = %d", it->first, it->second);
  }
  // CHECK-MESSAGES: :[[@LINE-4]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & elem : intmap)
  // CHECK-FIXES-NEXT: printf("intmap[%d] = %d", elem.first, elem.second);
}

void different_type() {
  // Tests to verify the proper use of auto where the init variable type and the
  // initializer type differ or are mostly the same except for const qualifiers.

  // s.begin() returns a type 'iterator' which is just a non-const pointer and
  // differs from const_iterator only on the const qualification.
  S s;
  for (S::const_iterator it = s.begin(); it != s.end(); ++it) {
    printf("s has value %d\n", (*it).x);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (const auto & elem : s)
  // CHECK-FIXES-NEXT: printf("s has value %d\n", (elem).x);

  S *ps;
  for (S::const_iterator it = ps->begin(); it != ps->end(); ++it) {
    printf("s has value %d\n", (*it).x);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (const auto & p : *ps)
  // CHECK-FIXES-NEXT: printf("s has value %d\n", (p).x);

  // v.begin() returns a user-defined type 'iterator' which, since it's
  // different from const_iterator, disqualifies these loops from
  // transformation.
  dependent<int> v;
  for (dependent<int>::const_iterator it = v.begin(); it != v.end(); ++it) {
    printf("Fibonacci number is %d\n", *it);
  }

  for (dependent<int>::const_iterator it(v.begin()); it != v.end(); ++it) {
    printf("Fibonacci number is %d\n", *it);
  }
}

} // namespace SingleIterator


namespace Macros {

const int N = 10;
int arr[N];

void messing_with_macros() {
  for (int i = 0; i < N; ++i) {
    printf("Value: %d\n", arr[i]);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & elem : arr) {
  // CHECK-FIXES-NEXT:  printf("Value: %d\n", elem);

  for (int i = 0; i < N; ++i) {
    printf("Value: %d\n", CONT arr[i]);
  }
}

} // namespace Macros

namespace Templates {

template <class Container>
void set_union(Container &container) {
  for (typename Container::const_iterator SI = container.begin(),
       SE = container.end(); SI != SE; ++SI) {
  }
  S s;
  for (S::iterator SI = s.begin(), SE = s.end(); SI != SE; ++SI) {
  }
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (auto & elem : s) {
}

void template_instantiation() {
  S a;
  set_union(a);
}

} // namespace Templates
