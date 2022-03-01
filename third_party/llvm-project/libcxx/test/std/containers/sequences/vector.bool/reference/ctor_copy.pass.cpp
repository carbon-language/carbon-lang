//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <vector>

// reference(const reference&)

#include <cassert>
#include <vector>

bool test() {
  std::vector<bool> vec;
  typedef std::vector<bool>::reference Ref;
  vec.push_back(true);
  Ref ref = vec[0];
  Ref ref2 = ref;
  assert(ref == ref2 && ref2);
  ref.flip();
  assert(ref == ref2 && !ref2);

  return true;
}

int main(int, char**) {
  test();

  return 0;
}
