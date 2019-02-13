// Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Utility: before overwriting a variable, capture its value and
// ensure that it will be restored when the Restorer goes out of scope.
//
// int x{3};
// {
//   auto save{common::ScopedSet(x, 4)};
//   // x is now 4
// }
// // x is back to 3

#ifndef FORTRAN_COMMON_RESTORER_H_
#define FORTRAN_COMMON_RESTORER_H_
namespace Fortran::common {
template<typename A> class Restorer {
public:
  explicit Restorer(A &p) : p_{p}, original_{std::move(p)} {}
  ~Restorer() { p_ = std::move(original_); }

private:
  A &p_;
  A original_;
};

template<typename A, typename B> Restorer<A> ScopedSet(A &to, B &&from) {
  Restorer<A> result{to};
  to = std::move(from);
  return result;
}
}
#endif  // FORTRAN_COMMON_RESTORER_H_
