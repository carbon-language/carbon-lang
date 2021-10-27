// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXECUTABLE_SEMANTICS_COMMON_ANY_PRINTABLE_PTR_H_
#define EXECUTABLE_SEMANTICS_COMMON_ANY_PRINTABLE_PTR_H_

#include <memory>

#include "common/ostream.h"
#include "executable_semantics/common/nonnull.h"

namespace Carbon {

// Type-erased wrapper for a non-owning pointer to any type that can be
// streamed to a llvm::raw_ostream.
class AnyPrintablePtr {
 public:
  template <typename Printable>
  AnyPrintablePtr(Nonnull<const Printable*> ptr);

  AnyPrintablePtr(const AnyPrintablePtr&) = default;
  auto operator=(const AnyPrintablePtr&) -> AnyPrintablePtr& = default;

  void Print(llvm::raw_ostream& out) const { impl_->Print(out); }

 private:
  class PrintInterface {
   public:
    virtual void Print(llvm::raw_ostream& out) const = 0;
    virtual ~PrintInterface() = default;
  };

  template <typename Printable>
  class Impl;

  std::shared_ptr<const PrintInterface> impl_;
};

// Implementation details only below here

template <typename Printable>
AnyPrintablePtr::AnyPrintablePtr(Nonnull<const Printable*> ptr)
    : impl_(std::make_unique<Impl<Printable>>(ptr)) {}

template <typename Printable>
class AnyPrintablePtr::Impl : public AnyPrintablePtr::PrintInterface {
 public:
  explicit Impl(Nonnull<const Printable*> ptr) : ptr_(ptr) {}
  ~Impl() override = default;

  void Print(llvm::raw_ostream& out) const override { out << *ptr_; }

 private:
  Nonnull<const Printable*> ptr_;
};

}  // namespace Carbon

#endif  // EXECUTABLE_SEMANTICS_COMMON_ANY_PRINTABLE_PTR_H_
