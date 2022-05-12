//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-no-concepts
// UNSUPPORTED: libcpp-has-no-incomplete-ranges

// template<class R>
//   explicit join_view(R&&) -> join_view<views::all_t<R>>;

#include <ranges>

struct Child {
  int *begin() const;
  int *end() const;
};

struct View : std::ranges::view_base {
  Child *begin() const;
  Child *end() const;
};

struct Range {
  Child *begin() const;
  Child *end() const;
};

struct BorrowedRange {
  Child *begin() const;
  Child *end() const;
};
template<>
inline constexpr bool std::ranges::enable_borrowed_range<BorrowedRange> = true;

void testCTAD() {
    View v;
    Range r;
    BorrowedRange br;

    static_assert(std::same_as<
        decltype(std::ranges::join_view(v)),
        std::ranges::join_view<View>
    >);
    static_assert(std::same_as<
        decltype(std::ranges::join_view(std::move(v))),
        std::ranges::join_view<View>
    >);
    static_assert(std::same_as<
        decltype(std::ranges::join_view(r)),
        std::ranges::join_view<std::ranges::ref_view<Range>>
    >);
    static_assert(std::same_as<
        decltype(std::ranges::join_view(std::move(r))),
        std::ranges::join_view<std::ranges::owning_view<Range>>
    >);
    static_assert(std::same_as<
        decltype(std::ranges::join_view(br)),
        std::ranges::join_view<std::ranges::ref_view<BorrowedRange>>
    >);
    static_assert(std::same_as<
        decltype(std::ranges::join_view(std::move(br))),
        std::ranges::join_view<std::ranges::owning_view<BorrowedRange>>
    >);
}
