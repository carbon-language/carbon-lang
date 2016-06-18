//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <experimental/filesystem>

// class path

// template <class Source>
//      path(const Source& source);
// template <class InputIterator>
//      path(InputIterator first, InputIterator last);


#include <experimental/filesystem>
#include <iterator>
#include <type_traits>
#include <cassert>

#include <iostream>

#include "test_macros.h"
#include "filesystem_test_helper.hpp"

namespace fs = std::experimental::filesystem;


template <class It>
std::reverse_iterator<It> mkRev(It it) {
  return std::reverse_iterator<It>(it);
}


template <class Iter1, class Iter2>
bool checkCollectionsEqualVerbose(
    Iter1 start1, Iter1 const end1
  , Iter2 start2, Iter2 const end2
  )
{
    while (start1 != end1 && start2 != end2) {
      std::cout << "Got start1 = " << *start1 << "\n"
                << "Got start2 = " << *start2 << "\n";
        if (*start1 != *start2) {
            return false;
        }
        ++start1; ++start2;
    }
  if (start1 != end1) {
    std::cout << "Got start1 = " << *start1 << " but expected end1\n";
  }
  if (start2 != end2) {
    std::cout << "Got start2 = " << *start2 << " but expected end2\n";
  }
    return (start1 == end1 && start2 == end2);
}

void checkIteratorConcepts() {
  using namespace fs;
  using It = path::iterator;
  using Traits = std::iterator_traits<It>;
  ASSERT_SAME_TYPE(Traits::iterator_category, std::bidirectional_iterator_tag);
  ASSERT_SAME_TYPE(Traits::value_type, path);
  ASSERT_SAME_TYPE(Traits::pointer,   path const*);
  ASSERT_SAME_TYPE(Traits::reference, path const&);
  {
    It it;
    ASSERT_SAME_TYPE(It&, decltype(++it));
    ASSERT_SAME_TYPE(It, decltype(it++));
    ASSERT_SAME_TYPE(It&, decltype(--it));
    ASSERT_SAME_TYPE(It, decltype(it--));
    ASSERT_SAME_TYPE(Traits::reference, decltype(*it));
    ASSERT_SAME_TYPE(Traits::pointer, decltype(it.operator->()));
    ASSERT_SAME_TYPE(std::string const&, decltype(it->native()));
    ASSERT_SAME_TYPE(bool, decltype(it == it));
    ASSERT_SAME_TYPE(bool, decltype(it != it));
  }
  {
    path const p;
    ASSERT_SAME_TYPE(It, decltype(p.begin()));
    ASSERT_SAME_TYPE(It, decltype(p.end()));
    assert(p.begin() == p.end());
  }
}

void checkBeginEndBasic() {
  using namespace fs;
  using It = path::iterator;
  {
    path const p;
    ASSERT_SAME_TYPE(It, decltype(p.begin()));
    ASSERT_SAME_TYPE(It, decltype(p.end()));
    assert(p.begin() == p.end());
  }
  {
    path const p("foo");
    It default_constructed;
    default_constructed = p.begin();
    assert(default_constructed == p.begin());
    assert(default_constructed != p.end());
    default_constructed = p.end();
    assert(default_constructed == p.end());
    assert(default_constructed != p.begin());
  }
  {
    path p("//root_name//first_dir////second_dir");
    const path expect[] = {"//root_name", "/", "first_dir", "second_dir"};
    assert(checkCollectionsEqual(p.begin(), p.end(), std::begin(expect), std::end(expect)));
    assert(checkCollectionsEqualVerbose(mkRev(p.end()), mkRev(p.begin()),
                                 mkRev(std::end(expect)),
                                 mkRev(std::begin(expect))));
  }
  {
    path p("////foo/bar/baz///");
    const path expect[] = {"/", "foo", "bar", "baz", "."};
    assert(checkCollectionsEqual(p.begin(), p.end(), std::begin(expect), std::end(expect)));
    assert(checkCollectionsEqual(mkRev(p.end()), mkRev(p.begin()),
                                 mkRev(std::end(expect)), mkRev(std::begin(expect))));
  }

}

int main() {
  using namespace fs;
  checkIteratorConcepts();
  checkBeginEndBasic(); // See path.decompose.pass.cpp for more tests.
}
