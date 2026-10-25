// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import Cpp inline '''
template<typename T> auto Scalar(T value) { return value; }
template<typename T> decltype(auto) Identity(T value) { return value; }
template<typename T> auto& AutoRef(T& value) { return value; }
template<typename T> decltype(auto) Ref(T& value) { return (value); }
template<typename T> decltype(auto) Discard(T& value) { value = 0; }

template<typename T> struct Holder {
  T value;
  template<typename... Args> decltype(auto) Set(Args&&... args) {
    value = T(static_cast<Args&&>(args)...);
    return (value);
  }
};
struct Empty {};
template<> struct Holder<Empty> {
  template<typename... Args> decltype(auto) Set(Args&&...) {}
};

template<typename T> struct Result {
  T first;
  T second;
  T third;
};
template<typename T> auto MakeResult(T value) {
  return Result<T>{value, value + 1, value + 2};
}

struct Value {
  int value;
  auto Get() const { return value; }
};
template<typename T> auto operator+(T left, T right) {
  return left.value + right.value;
}
''';

fn Run() -> i32 {
  var value: i32 = 1;
  var wide: i64 = 2;
  Cpp.Ref(ref value) = 42;
  if (value != 42) {
    return 1;
  }
  Cpp.Ref(i64, ref wide) = 43;
  if (wide != 43) {
    return 2;
  }
  if (Cpp.Scalar(value) != 42) {
    return 3;
  }
  if (Cpp.Scalar(i64, wide) != 43) {
    return 4;
  }
  Cpp.AutoRef(ref value) = 44;
  if (value != 44) {
    return 5;
  }
  if (Cpp.Identity(value) != 44) {
    return 6;
  }

  var holder: Cpp.Holder(i32) = Cpp.Holder(i32).Holder();
  holder.Set(ref value) = 47;
  if (holder.value != 47) {
    return 7;
  }
  holder.Set(i32, 45) = 46;
  if (holder.value != 46) {
    return 8;
  }

  Cpp.Discard(ref value);
  if (value != 0) {
    return 9;
  }
  var empty_holder: Cpp.Holder(Cpp.Empty) = Cpp.Holder(Cpp.Empty).Holder();
  empty_holder.Set();
  empty_holder.Set(ref value);

  // Three i64 fields exercise an indirect C++ return and its class dependency.
  if (Cpp.MakeResult(40 as i64).third != 42) {
    return 10;
  }

  var left: Cpp.Value = Cpp.Value.Value();
  var right: Cpp.Value = Cpp.Value.Value();
  left.value = 40;
  right.value = 2;
  if (left.Get() != 40) {
    return 11;
  }
  if (left + right != 42) {
    return 12;
  }
  return 0;
}
