// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: clang-modernize -loop-convert %t.cpp -- -I %S/Inputs
// RUN: FileCheck -input-file=%t.cpp %s

#include "structures.h"

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
  // CHECK: for (auto & elem : Arr)
  // CHECK-NEXT: for (auto & Arr_j : Arr)
  // CHECK-NEXT: int k = elem.x + Arr_j.x;
  // CHECK-NOT: int l = elem.x + elem.x;

  Val Nest[N][M];
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      printf("Got item %d", Nest[i][j].x);
    }
  }
  // The inner loop is also convertible, but doesn't need to be converted
  // immediately. Update this test when that changes!
  // CHECK: for (auto & elem : Nest)
  // CHECK-NEXT: for (int j = 0; j < M; ++j)
  // CHECK-NEXT: printf("Got item %d", elem[j].x);

  // Note that the order of M and N are switched for this test.
  for (int j = 0; j < M; ++j) {
    for (int i = 0; i < N; ++i) {
      printf("Got item %d", Nest[i][j].x);
    }
  }
  // CHECK-NOT: for (auto & {{[a-zA-Z_]+}} : Nest[i])
  // CHECK: for (int j = 0; j < M; ++j)
  // CHECK-NEXT: for (auto & elem : Nest)
  // CHECK-NEXT: printf("Got item %d", elem[j].x);
  Nested<T> NestT;
  for (Nested<T>::iterator I = NestT.begin(), E = NestT.end(); I != E; ++I) {
    for (T::iterator TI = (*I).begin(), TE = (*I).end(); TI != TE; ++TI) {
      printf("%d", *TI);
    }
  }
  // The inner loop is also convertible, but doesn't need to be converted
  // immediately. Update this test when that changes!
  // CHECK: for (auto & elem : NestT) {
  // CHECK-NEXT: for (T::iterator TI = (elem).begin(), TE = (elem).end(); TI != TE; ++TI) {
  // CHECK-NEXT: printf("%d", *TI);

  Nested<S> NestS;
  for (Nested<S>::const_iterator I = NestS.begin(), E = NestS.end(); I != E; ++I) {
    for (S::const_iterator SI = (*I).begin(), SE = (*I).end(); SI != SE; ++SI) {
      printf("%d", *SI);
    }
  }
  // The inner loop is also convertible, but doesn't need to be converted
  // immediately. Update this test when that changes!
  // CHECK: for (const auto & elem : NestS) {
  // CHECK-NEXT: for (S::const_iterator SI = (elem).begin(), SE = (elem).end(); SI != SE; ++SI) {
  // CHECK-NEXT: printf("%d", *SI);
}
