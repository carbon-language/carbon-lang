// RUN: %check_clang_tidy %s modernize-loop-convert %t -- \
// RUN:   -config="{CheckOptions: [{key: modernize-loop-convert.NamingStyle, value: 'camelBack'}]}" \
// RUN:   -- -std=c++11 -I %S/Inputs/modernize-loop-convert

#include "structures.h"

const int n = 10;
int arr[n];
int nums[n];

void naming() {
  for (int i = 0; i < n; ++i) {
    printf("%d\n", arr[i]);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead [modernize-loop-convert]
  // CHECK-FIXES: for (int & elem : arr)
  // CHECK-FIXES-NEXT: printf("%d\n", elem);

  for (int i = 0; i < n; ++i) {
    printf("%d\n", nums[i]);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int & num : nums)
  // CHECK-FIXES-NEXT: printf("%d\n", num);

  int num = 0;
  for (int i = 0; i < n; ++i) {
    printf("%d\n", nums[i] + num);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int & elem : nums)
  // CHECK-FIXES-NEXT: printf("%d\n", elem + num);

  int elem = 0;
  for (int i = 0; i < n; ++i) {
    printf("%d\n", nums[i] + num + elem);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int & numsI : nums)
  // CHECK-FIXES-NEXT: printf("%d\n", numsI + num + elem);

  int numsI = 0;
  for (int i = 0; i < n; ++i) {
    printf("%d\n", nums[i] + num + elem + numsI);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int & numsElem : nums)
  // CHECK-FIXES-NEXT: printf("%d\n", numsElem + num + elem + numsI);

  int numsElem = 0;
  for (int i = 0; i < n; ++i) {
    printf("%d\n", nums[i] + num + elem + numsI + numsElem);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int & giveMeName0 : nums)
  // CHECK-FIXES-NEXT: printf("%d\n", giveMeName0 + num + elem + numsI + numsElem);

  int giveMeName0 = 0;
  for (int i = 0; i < n; ++i) {
    printf("%d\n", nums[i] + num + elem + numsI + numsElem + giveMeName0);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int & giveMeName1 : nums)
  // CHECK-FIXES-NEXT: printf("%d\n", giveMeName1 + num + elem + numsI + numsElem + giveMeName0);

  int numsJ = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      printf("%d\n", nums[i] + nums[j] + num + elem + numsI + numsJ + numsElem);
    }
  }
  // CHECK-MESSAGES: :[[@LINE-5]]:3: warning: use range-based for loop instead
  // CHECK-MESSAGES: :[[@LINE-5]]:5: warning: use range-based for loop instead
  // CHECK-FIXES: for (int & giveMeName0 : nums)
  // CHECK-FIXES: for (int & giveMeName1 : nums)
  // CHECK-FIXES-NEXT: printf("%d\n", giveMeName0 + giveMeName1 + num + elem + numsI + numsJ + numsElem);
}
