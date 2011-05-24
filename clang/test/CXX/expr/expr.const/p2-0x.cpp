// RUN: %clang_cc1 -fsyntax-only -std=c++0x -verify %s

// PR9999
template<bool v>
class bitWidthHolding {
public:
  static const
  unsigned int width = (v == 0 ? 0 : bitWidthHolding<(v >> 1)>::width + 1);
};

static const int width=bitWidthHolding<255>::width;

template<bool b>
struct always_false {
  static const bool value = false;
};

template<bool b>
struct and_or {
  static const bool and_value = b && and_or<always_false<b>::value>::and_value;
  static const bool or_value = !b || and_or<always_false<b>::value>::or_value;
};

static const bool and_value = and_or<true>::and_value;
static const bool or_value = and_or<true>::or_value;
