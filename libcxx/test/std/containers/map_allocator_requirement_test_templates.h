//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#ifndef MAP_ALLOCATOR_REQUIREMENT_TEST_TEMPLATES_H
#define MAP_ALLOCATOR_REQUIREMENT_TEST_TEMPLATES_H

// <map>
// <unordered_map>

// class map
// class unordered_map

// insert(...);
// emplace(...);
// emplace_hint(...);

// UNSUPPORTED: c++98, c++03

#include <cassert>

#include "test_macros.h"
#include "count_new.hpp"
#include "container_test_types.h"
#include "assert_checkpoint.h"


template <class Container>
void testMapInsert()
{
  typedef typename Container::value_type ValueTp;
  typedef Container C;
  typedef std::pair<typename C::iterator, bool> R;
  ConstructController* cc = getConstructController();
  cc->reset();
  {
    CHECKPOINT("Testing C::insert(const value_type&)");
    Container c;
    const ValueTp v(42, 1);
    cc->expect<const ValueTp&>();
    assert(c.insert(v).second);
    assert(!cc->unchecked());
    {
      DisableAllocationGuard g;
      const ValueTp v2(42, 1);
      assert(c.insert(v2).second == false);
    }
  }
  {
    CHECKPOINT("Testing C::insert(value_type&)");
    Container c;
    ValueTp v(42, 1);
    cc->expect<const ValueTp&>();
    assert(c.insert(v).second);
    assert(!cc->unchecked());
    {
      DisableAllocationGuard g;
      ValueTp v2(42, 1);
      assert(c.insert(v2).second == false);
    }
  }
  {
    CHECKPOINT("Testing C::insert(value_type&&)");
    Container c;
    ValueTp v(42, 1);
    cc->expect<ValueTp&&>();
    assert(c.insert(std::move(v)).second);
    assert(!cc->unchecked());
    {
      DisableAllocationGuard g;
      ValueTp v2(42, 1);
      assert(c.insert(std::move(v2)).second == false);
    }
  }
  {
    CHECKPOINT("Testing C::insert(const value_type&&)");
    Container c;
    const ValueTp v(42, 1);
    cc->expect<const ValueTp&>();
    assert(c.insert(std::move(v)).second);
    assert(!cc->unchecked());
    {
      DisableAllocationGuard g;
      const ValueTp v2(42, 1);
      assert(c.insert(std::move(v2)).second == false);
    }
  }
  {
    CHECKPOINT("Testing C::insert(std::initializer_list<ValueTp>)");
    Container c;
    std::initializer_list<ValueTp> il = { ValueTp(1, 1), ValueTp(2, 1) };
    cc->expect<ValueTp const&>(2);
    c.insert(il);
    assert(!cc->unchecked());
    {
      DisableAllocationGuard g;
      c.insert(il);
    }
  }
  {
    CHECKPOINT("Testing C::insert(Iter, Iter) for *Iter = value_type const&");
    Container c;
    const ValueTp ValueList[] = { ValueTp(1, 1), ValueTp(2, 1), ValueTp(3, 1) };
    cc->expect<ValueTp const&>(3);
    c.insert(std::begin(ValueList), std::end(ValueList));
    assert(!cc->unchecked());
    {
      DisableAllocationGuard g;
      c.insert(std::begin(ValueList), std::end(ValueList));
    }
  }
  {
    CHECKPOINT("Testing C::insert(Iter, Iter) for *Iter = value_type&&");
    Container c;
    ValueTp ValueList[] = { ValueTp(1, 1), ValueTp(2, 1) , ValueTp(3, 1) };
    cc->expect<ValueTp&&>(3);
    c.insert(std::move_iterator<ValueTp*>(std::begin(ValueList)),
             std::move_iterator<ValueTp*>(std::end(ValueList)));
    assert(!cc->unchecked());
    {
      DisableAllocationGuard g;
      ValueTp ValueList2[] = { ValueTp(1, 1), ValueTp(2, 1) , ValueTp(3, 1) };
      c.insert(std::move_iterator<ValueTp*>(std::begin(ValueList2)),
               std::move_iterator<ValueTp*>(std::end(ValueList2)));
    }
  }
  {
    CHECKPOINT("Testing C::insert(Iter, Iter) for *Iter = value_type&");
    Container c;
    ValueTp ValueList[] = { ValueTp(1, 1), ValueTp(2, 1) , ValueTp(3, 1) };
    cc->expect<ValueTp const&>(3);
    c.insert(std::begin(ValueList), std::end(ValueList));
    assert(!cc->unchecked());
    {
      DisableAllocationGuard g;
      c.insert(std::begin(ValueList), std::end(ValueList));
    }
  }
}

template <class Container>
void testMapEmplace()
{
  typedef typename Container::value_type ValueTp;
  typedef typename Container::key_type Key;
  typedef typename Container::mapped_type Mapped;
  typedef typename std::pair<Key, Mapped> NonConstKeyPair;
  typedef Container C;
  typedef std::pair<typename C::iterator, bool> R;
  ConstructController* cc = getConstructController();
  cc->reset();
  {
    CHECKPOINT("Testing C::emplace(const value_type&)");
    Container c;
    const ValueTp v(42, 1);
    cc->expect<const ValueTp&>();
    assert(c.emplace(v).second);
    assert(!cc->unchecked());
    {
      DisableAllocationGuard g;
      const ValueTp v2(42, 1);
      assert(c.emplace(v2).second == false);
    }
  }
  {
    CHECKPOINT("Testing C::emplace(value_type&)");
    Container c;
    ValueTp v(42, 1);
    cc->expect<ValueTp&>();
    assert(c.emplace(v).second);
    assert(!cc->unchecked());
    {
      DisableAllocationGuard g;
      ValueTp v2(42, 1);
      assert(c.emplace(v2).second == false);
    }
  }
  {
    CHECKPOINT("Testing C::emplace(value_type&&)");
    Container c;
    ValueTp v(42, 1);
    cc->expect<ValueTp&&>();
    assert(c.emplace(std::move(v)).second);
    assert(!cc->unchecked());
    {
      DisableAllocationGuard g;
      ValueTp v2(42, 1);
      assert(c.emplace(std::move(v2)).second == false);
    }
  }
  {
    CHECKPOINT("Testing C::emplace(const value_type&&)");
    Container c;
    const ValueTp v(42, 1);
    cc->expect<const ValueTp&&>();
    assert(c.emplace(std::move(v)).second);
    assert(!cc->unchecked());
    {
      DisableAllocationGuard g;
      const ValueTp v2(42, 1);
      assert(c.emplace(std::move(v2)).second == false);
    }
  }
  {
    CHECKPOINT("Testing C::emplace(pair<Key, Mapped> const&)");
    Container c;
    const NonConstKeyPair v(42, 1);
    cc->expect<const NonConstKeyPair&>();
    assert(c.emplace(v).second);
    assert(!cc->unchecked());
    {
      DisableAllocationGuard g;
      const NonConstKeyPair v2(42, 1);
      assert(c.emplace(v2).second == false);
    }
  }
  {
    CHECKPOINT("Testing C::emplace(pair<Key, Mapped> &&)");
    Container c;
    NonConstKeyPair v(42, 1);
    cc->expect<NonConstKeyPair&&>();
    assert(c.emplace(std::move(v)).second);
    assert(!cc->unchecked());
    {
      DisableAllocationGuard g;
      NonConstKeyPair v2(42, 1);
      assert(c.emplace(std::move(v2)).second == false);
    }
  }
}


template <class Container>
void testMapEmplaceHint()
{
  typedef typename Container::value_type ValueTp;
  typedef typename Container::key_type Key;
  typedef typename Container::mapped_type Mapped;
  typedef typename std::pair<Key, Mapped> NonConstKeyPair;
  typedef Container C;
  typedef typename C::iterator It;
  ConstructController* cc = getConstructController();
  cc->reset();
  {
    CHECKPOINT("Testing C::emplace_hint(p, const value_type&)");
    Container c;
    const ValueTp v(42, 1);
    cc->expect<const ValueTp&>();
    It ret = c.emplace_hint(c.end(), v);
    assert(ret != c.end());
    assert(c.size() == 1);
    assert(!cc->unchecked());
    {
      DisableAllocationGuard g;
      const ValueTp v2(42, 1);
      It ret2 = c.emplace_hint(c.begin(), v2);
      assert(&(*ret2) == &(*ret));
      assert(c.size() == 1);
    }
  }
  {
    CHECKPOINT("Testing C::emplace_hint(p, value_type&)");
    Container c;
    ValueTp v(42, 1);
    cc->expect<ValueTp&>();
    It ret = c.emplace_hint(c.end(), v);
    assert(ret != c.end());
    assert(c.size() == 1);
    assert(!cc->unchecked());
    {
      DisableAllocationGuard g;
      ValueTp v2(42, 1);
      It ret2 = c.emplace_hint(c.begin(), v2);
      assert(&(*ret2) == &(*ret));
      assert(c.size() == 1);
    }
  }
  {
    CHECKPOINT("Testing C::emplace_hint(p, value_type&&)");
    Container c;
    ValueTp v(42, 1);
    cc->expect<ValueTp&&>();
    It ret = c.emplace_hint(c.end(), std::move(v));
    assert(ret != c.end());
    assert(c.size() == 1);
    assert(!cc->unchecked());
    {
      DisableAllocationGuard g;
      ValueTp v2(42, 1);
      It ret2 = c.emplace_hint(c.begin(), std::move(v2));
      assert(&(*ret2) == &(*ret));
      assert(c.size() == 1);
    }
  }
  {
    CHECKPOINT("Testing C::emplace_hint(p, const value_type&&)");
    Container c;
    const ValueTp v(42, 1);
    cc->expect<const ValueTp&&>();
    It ret = c.emplace_hint(c.end(), std::move(v));
    assert(ret != c.end());
    assert(c.size() == 1);
    assert(!cc->unchecked());
    {
      DisableAllocationGuard g;
      const ValueTp v2(42, 1);
      It ret2 = c.emplace_hint(c.begin(), std::move(v2));
      assert(&(*ret2) == &(*ret));
      assert(c.size() == 1);
    }
  }
  {
    CHECKPOINT("Testing C::emplace_hint(p, pair<Key, Mapped> const&)");
    Container c;
    const NonConstKeyPair v(42, 1);
    cc->expect<const NonConstKeyPair&>();
    It ret = c.emplace_hint(c.end(), v);
    assert(ret != c.end());
    assert(c.size() == 1);
    assert(!cc->unchecked());
    {
      DisableAllocationGuard g;
      const NonConstKeyPair v2(42, 1);
      It ret2 = c.emplace_hint(c.begin(), v2);
      assert(&(*ret2) == &(*ret));
      assert(c.size() == 1);
    }
  }
  {
    CHECKPOINT("Testing C::emplace_hint(p, pair<Key, Mapped>&&)");
    Container c;
    NonConstKeyPair v(42, 1);
    cc->expect<NonConstKeyPair&&>();
    It ret = c.emplace_hint(c.end(), std::move(v));
    assert(ret != c.end());
    assert(c.size() == 1);
    assert(!cc->unchecked());
    {
      DisableAllocationGuard g;
      NonConstKeyPair v2(42, 1);
      It ret2 = c.emplace_hint(c.begin(), std::move(v2));
      assert(&(*ret2) == &(*ret));
      assert(c.size() == 1);
    }
  }
}


template <class Container>
void testMultimapInsert()
{
  typedef typename Container::value_type ValueTp;
  typedef Container C;
  ConstructController* cc = getConstructController();
  cc->reset();
  {
    CHECKPOINT("Testing C::insert(const value_type&)");
    Container c;
    const ValueTp v(42, 1);
    cc->expect<const ValueTp&>();
    c.insert(v);
    assert(!cc->unchecked());
  }
  {
    CHECKPOINT("Testing C::insert(value_type&)");
    Container c;
    ValueTp v(42, 1);
    cc->expect<ValueTp&>();
    c.insert(v);
    assert(!cc->unchecked());
  }
  {
    CHECKPOINT("Testing C::insert(value_type&&)");
    Container c;
    ValueTp v(42, 1);
    cc->expect<ValueTp&&>();
    c.insert(std::move(v));
    assert(!cc->unchecked());
  }
  {
    CHECKPOINT("Testing C::insert(std::initializer_list<ValueTp>)");
    Container c;
    std::initializer_list<ValueTp> il = { ValueTp(1, 1), ValueTp(2, 1) };
    cc->expect<ValueTp const&>(2);
    c.insert(il);
    assert(!cc->unchecked());
  }
  {
    CHECKPOINT("Testing C::insert(Iter, Iter) for *Iter = value_type const&");
    Container c;
    const ValueTp ValueList[] = { ValueTp(1, 1), ValueTp(2, 1), ValueTp(3, 1) };
    cc->expect<ValueTp const&>(3);
    c.insert(std::begin(ValueList), std::end(ValueList));
    assert(!cc->unchecked());
  }
  {
    CHECKPOINT("Testing C::insert(Iter, Iter) for *Iter = value_type&&");
    Container c;
    ValueTp ValueList[] = { ValueTp(1, 1), ValueTp(2, 1) , ValueTp(3, 1) };
    cc->expect<ValueTp&&>(3);
    c.insert(std::move_iterator<ValueTp*>(std::begin(ValueList)),
             std::move_iterator<ValueTp*>(std::end(ValueList)));
    assert(!cc->unchecked());
  }
  {
    CHECKPOINT("Testing C::insert(Iter, Iter) for *Iter = value_type&");
    Container c;
    ValueTp ValueList[] = { ValueTp(1, 1), ValueTp(2, 1) , ValueTp(3, 1) };
    cc->expect<ValueTp&>(3);
    c.insert(std::begin(ValueList), std::end(ValueList));
    assert(!cc->unchecked());
  }
}

#endif