//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <algorithm>

// template <class PopulationIterator, class SampleIterator, class Distance,
//           class UniformRandomNumberGenerator>
// SampleIterator sample(PopulationIterator first, PopulationIterator last,
//                       SampleIterator out, Distance n,
//                       UniformRandomNumberGenerator &&g);

#include <algorithm>
#include <random>
#include <cassert>

#include "test_iterators.h"

template <class PopulationIterator, class SampleIterator> void test() {
  int ia[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const unsigned is = sizeof(ia) / sizeof(ia[0]);
  const unsigned os = 4;
  int oa[os];
  std::minstd_rand g;
  std::sample(PopulationIterator(ia), PopulationIterator(ia + is),
                            SampleIterator(oa), os, g);
}

int main(int, char**) {
  // expected-error-re@*:* {{static_assert failed{{( due to requirement '.*')?}} "SampleIterator must meet the requirements of RandomAccessIterator"}}
  // expected-error@*:* 2 {{does not provide a subscript operator}}
  // expected-error@*:* {{invalid operands}}
  test<cpp17_input_iterator<int *>, output_iterator<int *> >();

  return 0;
}
