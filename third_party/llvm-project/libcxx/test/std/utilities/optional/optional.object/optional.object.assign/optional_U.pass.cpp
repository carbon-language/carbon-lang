//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// <optional>

// From LWG2451:
// template <class U>
// optional<T>& operator=(optional<U>&& rhs);

#include <optional>

#include <array>
#include <cassert>
#include <memory>
#include <type_traits>

#include "test_macros.h"
#include "archetypes.h"

using std::optional;

struct X
{
    static bool throw_now;

    X() = default;
    X(int &&)
    {
        if (throw_now)
            TEST_THROW(6);
    }
};

bool X::throw_now = false;

struct Y1
{
    Y1() = default;
    Y1(const int&) {}
    Y1& operator=(const Y1&) = delete;
};

struct Y2
{
    Y2() = default;
    Y2(const int&) = delete;
    Y2& operator=(const int&) { return *this; }
};

struct B { virtual ~B() = default; };
class D : public B {};


template <class T>
struct AssignableFrom {
  static int type_constructed;
  static int type_assigned;
static int int_constructed;
  static int int_assigned;

  static void reset() {
      type_constructed = int_constructed = 0;
      type_assigned = int_assigned = 0;
  }

  AssignableFrom() = default;

  explicit AssignableFrom(T) { ++type_constructed; }
  AssignableFrom& operator=(T) { ++type_assigned; return *this; }

  AssignableFrom(int) { ++int_constructed; }
  AssignableFrom& operator=(int) { ++int_assigned; return *this; }
private:
  AssignableFrom(AssignableFrom const&) = delete;
  AssignableFrom& operator=(AssignableFrom const&) = delete;
};

template <class T> int AssignableFrom<T>::type_constructed = 0;
template <class T> int AssignableFrom<T>::type_assigned = 0;
template <class T> int AssignableFrom<T>::int_constructed = 0;
template <class T> int AssignableFrom<T>::int_assigned = 0;

void test_with_test_type() {
    using T = TestTypes::TestType;
    T::reset();
    { // non-empty to empty
        T::reset_constructors();
        optional<T> opt;
        optional<int> other(42);
        opt = std::move(other);
        assert(T::alive == 1);
        assert(T::constructed == 1);
        assert(T::value_constructed == 1);
        assert(T::assigned == 0);
        assert(T::destroyed == 0);
        assert(static_cast<bool>(other) == true);
        assert(*other == 42);
        assert(static_cast<bool>(opt) == true);
        assert(*opt == T(42));
    }
    assert(T::alive == 0);
    { // non-empty to non-empty
        optional<T> opt(101);
        optional<int> other(42);
        T::reset_constructors();
        opt = std::move(other);
        assert(T::alive == 1);
        assert(T::constructed == 0);
        assert(T::assigned == 1);
        assert(T::value_assigned == 1);
        assert(T::destroyed == 0);
        assert(static_cast<bool>(other) == true);
        assert(*other == 42);
        assert(static_cast<bool>(opt) == true);
        assert(*opt == T(42));
    }
    assert(T::alive == 0);
    { // empty to non-empty
        optional<T> opt(101);
        optional<int> other;
        T::reset_constructors();
        opt = std::move(other);
        assert(T::alive == 0);
        assert(T::constructed == 0);
        assert(T::assigned == 0);
        assert(T::destroyed == 1);
        assert(static_cast<bool>(other) == false);
        assert(static_cast<bool>(opt) == false);
    }
    assert(T::alive == 0);
    { // empty to empty
        optional<T> opt;
        optional<int> other;
        T::reset_constructors();
        opt = std::move(other);
        assert(T::alive == 0);
        assert(T::constructed == 0);
        assert(T::assigned == 0);
        assert(T::destroyed == 0);
        assert(static_cast<bool>(other) == false);
        assert(static_cast<bool>(opt) == false);
    }
    assert(T::alive == 0);
}


void test_ambiguous_assign() {
    using OptInt = std::optional<int>;
    {
        using T = AssignableFrom<OptInt&&>;
        T::reset();
        {
            OptInt a(42);
            std::optional<T> t;
            t = std::move(a);
            assert(T::type_constructed == 1);
            assert(T::type_assigned == 0);
            assert(T::int_constructed == 0);
            assert(T::int_assigned == 0);
        }
        {
            using Opt = std::optional<T>;
            static_assert(!std::is_assignable<Opt&, const OptInt&&>::value, "");
            static_assert(!std::is_assignable<Opt&, const OptInt&>::value, "");
            static_assert(!std::is_assignable<Opt&, OptInt&>::value, "");
        }
    }
    {
        using T = AssignableFrom<OptInt const&&>;
        T::reset();
        {
            const OptInt a(42);
            std::optional<T> t;
            t = std::move(a);
            assert(T::type_constructed == 1);
            assert(T::type_assigned == 0);
            assert(T::int_constructed == 0);
            assert(T::int_assigned == 0);
        }
        T::reset();
        {
            OptInt a(42);
            std::optional<T> t;
            t = std::move(a);
            assert(T::type_constructed == 1);
            assert(T::type_assigned == 0);
            assert(T::int_constructed == 0);
            assert(T::int_assigned == 0);
        }
        {
            using Opt = std::optional<T>;
            static_assert(std::is_assignable<Opt&, OptInt&&>::value, "");
            static_assert(!std::is_assignable<Opt&, const OptInt&>::value, "");
            static_assert(!std::is_assignable<Opt&, OptInt&>::value, "");
        }
    }
}


TEST_CONSTEXPR_CXX20 bool test()
{
    {
        optional<int> opt;
        optional<short> opt2;
        opt = std::move(opt2);
        assert(static_cast<bool>(opt2) == false);
        assert(static_cast<bool>(opt) == static_cast<bool>(opt2));
    }
    {
        optional<int> opt;
        optional<short> opt2(short{2});
        opt = std::move(opt2);
        assert(static_cast<bool>(opt2) == true);
        assert(*opt2 == 2);
        assert(static_cast<bool>(opt) == static_cast<bool>(opt2));
        assert(*opt == *opt2);
    }
    {
        optional<int> opt(3);
        optional<short> opt2;
        opt = std::move(opt2);
        assert(static_cast<bool>(opt2) == false);
        assert(static_cast<bool>(opt) == static_cast<bool>(opt2));
    }
    {
        optional<int> opt(3);
        optional<short> opt2(short{2});
        opt = std::move(opt2);
        assert(static_cast<bool>(opt2) == true);
        assert(*opt2 == 2);
        assert(static_cast<bool>(opt) == static_cast<bool>(opt2));
        assert(*opt == *opt2);
    }

    enum class state_t { inactive, constructed, copy_assigned, move_assigned };
    class StateTracker {
    public:
      constexpr StateTracker(state_t& s)
      : state_(&s)
      {
        *state_ = state_t::constructed;
      }

      StateTracker(StateTracker&&) = default;
      StateTracker(StateTracker const&) = default;

      constexpr StateTracker& operator=(StateTracker&& other) noexcept
      {
        *state_ = state_t::inactive;
        state_ = other.state_;
        *state_ = state_t::move_assigned;
        other.state_ = nullptr;
        return *this;
      }

      constexpr StateTracker& operator=(StateTracker const& other) noexcept
      {
        *state_ = state_t::inactive;
        state_ = other.state_;
        *state_ = state_t::copy_assigned;
        return *this;
      }
    private:
      state_t* state_;
    };
    {
      auto state = std::array{state_t::inactive, state_t::inactive};
      auto opt1 = std::optional<StateTracker>(state[0]);
      assert(state[0] == state_t::constructed);

      auto opt2 = std::optional<StateTracker>(state[1]);
      assert(state[1] == state_t::constructed);

      opt1 = std::move(opt2);
      assert(state[0] == state_t::inactive);
      assert(state[1] == state_t::move_assigned);
    }
    {
      auto state = std::array{state_t::inactive, state_t::inactive};
      auto opt1 = std::optional<StateTracker>(state[0]);
      assert(state[0] == state_t::constructed);

      auto opt2 = std::optional<StateTracker>(state[1]);
      assert(state[1] == state_t::constructed);

      opt1 = opt2;
      assert(state[0] == state_t::inactive);
      assert(state[1] == state_t::copy_assigned);
    }

    return true;
}


int main(int, char**)
{
#if TEST_STD_VER > 17
    static_assert(test());
#endif
    test_with_test_type();
    test_ambiguous_assign();
    test();
    {
        optional<std::unique_ptr<B>> opt;
        optional<std::unique_ptr<D>> other(new D());
        opt = std::move(other);
        assert(static_cast<bool>(opt) == true);
        assert(static_cast<bool>(other) == true);
        assert(opt->get() != nullptr);
        assert(other->get() == nullptr);
    }
#ifndef TEST_HAS_NO_EXCEPTIONS
    {
        optional<X> opt;
        optional<int> opt2(42);
        assert(static_cast<bool>(opt2) == true);
        try
        {
            X::throw_now = true;
            opt = std::move(opt2);
            assert(false);
        }
        catch (int i)
        {
            assert(i == 6);
            assert(static_cast<bool>(opt) == false);
        }
    }
#endif

  return 0;
}
