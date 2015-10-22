// RUN: %check_clang_tidy %s modernize-loop-convert %t -- \
// RUN:   -config="{CheckOptions: [{key: modernize-loop-convert.NamingStyle, value: 'lower_case'}]}" \
// RUN:   -- -std=c++11 -I %S/Inputs/modernize-loop-convert

#include "structures.h"

const int n = 10;
int arr[n];
int nums[n];
int nums_[n];

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

  for (int i = 0; i < n; ++i) {
    printf("%d\n", nums_[i]);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int & num : nums_)
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
  // CHECK-FIXES: for (int & nums_i : nums)
  // CHECK-FIXES-NEXT: printf("%d\n", nums_i + num + elem);

  int nums_i = 0;
  for (int i = 0; i < n; ++i) {
    printf("%d\n", nums[i] + num + elem + nums_i);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int & nums_elem : nums)
  // CHECK-FIXES-NEXT: printf("%d\n", nums_elem + num + elem + nums_i);

  int nums_elem = 0;
  for (int i = 0; i < n; ++i) {
    printf("%d\n", nums[i] + num + elem + nums_i + nums_elem);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int & give_me_name_0 : nums)
  // CHECK-FIXES-NEXT: printf("%d\n", give_me_name_0 + num + elem + nums_i + nums_elem);

  int give_me_name_0 = 0;
  for (int i = 0; i < n; ++i) {
    printf("%d\n", nums[i] + num + elem + nums_i + nums_elem + give_me_name_0);
  }
  // CHECK-MESSAGES: :[[@LINE-3]]:3: warning: use range-based for loop instead
  // CHECK-FIXES: for (int & give_me_name_1 : nums)
  // CHECK-FIXES-NEXT: printf("%d\n", give_me_name_1 + num + elem + nums_i + nums_elem + give_me_name_0);

  int nums_j = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      printf("%d\n", nums[i] + nums[j] + num + elem + nums_i + nums_j + nums_elem);
    }
  }
  // CHECK-MESSAGES: :[[@LINE-5]]:3: warning: use range-based for loop instead
  // CHECK-MESSAGES: :[[@LINE-5]]:5: warning: use range-based for loop instead
  // CHECK-FIXES: for (int & give_me_name_0 : nums)
  // CHECK-FIXES: for (int & give_me_name_1 : nums)
  // CHECK-FIXES-NEXT: printf("%d\n", give_me_name_0 + give_me_name_1 + num + elem + nums_i + nums_j + nums_elem);
}
