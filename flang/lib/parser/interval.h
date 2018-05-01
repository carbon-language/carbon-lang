// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

#ifndef FORTRAN_PARSER_INTERVAL_H_
#define FORTRAN_PARSER_INTERVAL_H_

// Defines a generalized template class Interval<A> to represent
// the half-open interval [x .. x+n).

#include "idioms.h"
#include <cstddef>
#include <utility>

namespace Fortran {
namespace parser {

template<typename A> class Interval {
public:
  using type = A;
  Interval() {}
  Interval(const A &s, std::size_t n = 1) : start_{s}, size_{n} {}
  Interval(A &&s, std::size_t n = 1) : start_{std::move(s)}, size_{n} {}
  Interval(const Interval &) = default;
  Interval(Interval &&) = default;
  Interval &operator=(const Interval &) = default;
  Interval &operator=(Interval &&) = default;

  bool operator==(const Interval &that) const {
    return start_ == that.start_ && size_ == that.size_;
  }
  bool operator!=(const Interval &that) const {
    return start_ != that.start_ || size_ != that.size_;
  }

  const A &start() const { return start_; }
  std::size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }

  bool Contains(const A &x) const { return start_ <= x && x < start_ + size_; }
  bool Contains(const Interval &that) const {
    return Contains(that.start_) && Contains(that.start_ + (that.size_ - 1));
  }
  bool ImmediatelyPrecedes(const Interval &that) const {
    return NextAfter() == that.start_;
  }
  void Annex(const Interval &that) {
    size_ = (that.start_ + that.size_) - start_;
  }
  bool AnnexIfPredecessor(const Interval &that) {
    if (ImmediatelyPrecedes(that)) {
      size_ += that.size_;
      return true;
    }
    return false;
  }

  std::size_t MemberOffset(const A &x) const {
    CHECK(Contains(x));
    return x - start_;
  }
  A OffsetMember(std::size_t n) const {
    CHECK(n < size_);
    return start_ + n;
  }

  A Last() const { return start_ + (size_ - 1); }
  A NextAfter() const { return start_ + size_; }
  Interval Prefix(std::size_t n) const { return {start_, std::min(size_, n)}; }
  Interval Suffix(std::size_t n) const {
    CHECK(n <= size_);
    return {start_ + n, size_ - n};
  }

private:
  A start_;
  std::size_t size_{0};
};
}  // namespace parser
}  // namespace Fortran
#endif  // FORTRAN_PARSER_INTERVAL_H_
