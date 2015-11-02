// RUN: %check_clang_tidy %s modernize-loop-convert %t -- \
// RUN:   -config="{CheckOptions: [{key: modernize-loop-convert.NamingStyle, value: 'UPPER_CASE'}]}" \
// RUN:   -- -std=c++11 -I %S/Inputs/modernize-loop-convert

#include "structures.h"

const int N = 10;
int ARR[N];
int NUMS[N];
int NUMS_[N];

void naming() {
  for (int I = 0; I < N; ++I) {
    printf("%d\n", ARR[I]);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead [modernize-loop-convert]
  // CHECK-FIXES: for (int ELEM : ARR)
  // CHECK-FIXES-NEXT: printf("%d\n", ELEM);

  for (int I = 0; I < N; ++I) {
    printf("%d\n", NUMS[I]);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int NUM : NUMS)
  // CHECK-FIXES-NEXT: printf("%d\n", NUM);

  for (int I = 0; I < N; ++I) {
    printf("%d\n", NUMS_[I]);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int NUM : NUMS_)
  // CHECK-FIXES-NEXT: printf("%d\n", NUM);

  int NUM = 0;
  for (int I = 0; I < N; ++I) {
    printf("%d\n", NUMS[I] + NUM);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int ELEM : NUMS)
  // CHECK-FIXES-NEXT: printf("%d\n", ELEM + NUM);

  int ELEM = 0;
  for (int I = 0; I < N; ++I) {
    printf("%d\n", NUMS[I] + NUM + ELEM);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int NUMS_I : NUMS)
  // CHECK-FIXES-NEXT: printf("%d\n", NUMS_I + NUM + ELEM);

  int NUMS_I = 0;
  for (int I = 0; I < N; ++I) {
    printf("%d\n", NUMS[I] + NUM + ELEM + NUMS_I);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int NUMS_ELEM : NUMS)
  // CHECK-FIXES-NEXT: printf("%d\n", NUMS_ELEM + NUM + ELEM + NUMS_I);

  int NUMS_ELEM = 0;
  for (int I = 0; I < N; ++I) {
    printf("%d\n", NUMS[I] + NUM + ELEM + NUMS_I + NUMS_ELEM);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int GIVE_ME_NAME_0 : NUMS)
  // CHECK-FIXES-NEXT: printf("%d\n", GIVE_ME_NAME_0 + NUM + ELEM + NUMS_I + NUMS_ELEM);

  int GIVE_ME_NAME_0 = 0;
  for (int I = 0; I < N; ++I) {
    printf("%d\n", NUMS[I] + NUM + ELEM + NUMS_I + NUMS_ELEM + GIVE_ME_NAME_0);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int GIVE_ME_NAME_1 : NUMS)
  // CHECK-FIXES-NEXT: printf("%d\n", GIVE_ME_NAME_1 + NUM + ELEM + NUMS_I + NUMS_ELEM + GIVE_ME_NAME_0);

  int NUMS_J = 0;
  for (int I = 0; I < N; ++I) {
    for (int J = 0; J < N; ++J) {
      printf("%d\n", NUMS[I] + NUMS[J] + NUM + ELEM + NUMS_I + NUMS_J + NUMS_ELEM);
    }
  }
  // CHECK-MESSAGES: :[[@LINE-5]]:3: warning: use range-based for loop instead
  // CHECK-MESSAGES: :[[@LINE-5]]:5: warning: use range-based for loop instead
  // CHECK-FIXES: for (int GIVE_ME_NAME_0 : NUMS)
  // CHECK-FIXES: for (int GIVE_ME_NAME_1 : NUMS)
  // CHECK-FIXES-NEXT: printf("%d\n", GIVE_ME_NAME_0 + GIVE_ME_NAME_1 + NUM + ELEM + NUMS_I + NUMS_J + NUMS_ELEM);
}
