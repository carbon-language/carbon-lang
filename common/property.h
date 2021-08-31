// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef COMMON_PROPERTY_H_
#define COMMON_PROPERTY_H_

#include "llvm/Support/raw_ostream.h"

namespace Carbon {

template <typename GetT>
class ReadOnlyProperty {
 public:
  ReadOnlyProperty(std::function<GetT()> getter) : getter(getter) {}
  auto operator()() const -> GetT { return getter(); }
  auto operator*() const -> GetT { return getter(); }
  operator decltype(auto)() const { return getter(); }

 private:
  std::function<GetT()> getter;
};

template <typename GetT, typename SetT>
class Property : public ReadOnlyProperty<GetT> {
 public:
  Property(std::function<GetT()> getter,
           std::function<void(const SetT&)> setter)
      : ReadOnlyProperty<GetT>(getter), setter(setter) {}

  void operator=(const SetT& v) { setter(v); }
  void operator+=(const SetT& v) { setter((*this)() + v); }

 private:
  std::function<void(const SetT&)> setter;
};

}  // namespace Carbon

#endif  // COMMON_PROPERTY_H_
