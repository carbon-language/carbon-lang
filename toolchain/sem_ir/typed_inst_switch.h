// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_TYPED_INST_SWITCH_H_
#define CARBON_TOOLCHAIN_SEM_IR_TYPED_INST_SWITCH_H_

#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::SemIR {

// Provides switch-like behavior on an instruction, allowing `Case` handling
// with typed instructions.
//
// Usage:
//   TypedInstSwitch(inst)
//       .Case<MyInst>([&](auto inst) { ... })
//       .Cases<OtherInst1, OtherInst2>([&]() { ... })
//       .Default([&]() { ... });
//
// The default is optional.
//
// TODO: Should this check for unhandled or duplicate cases?
class TypedInstSwitch {
 public:
  //
  explicit TypedInstSwitch(const SemIR::Inst& inst) : inst_(inst) {}

  // Not copyable or movable.
  TypedInstSwitch(const TypedInstSwitch&) = delete;
  auto operator=(const TypedInstSwitch&) -> TypedInstSwitch& = delete;

  // If an instruction is the provided kind, calls the function with the typed
  // instruction.
  template <typename InstT>
  auto Case(llvm::function_ref<void(InstT)> fn) -> TypedInstSwitch& {
    if (!done_ && inst_.Is<InstT>()) {
      fn(inst_.As<InstT>());
      done_ = true;
    }
    return *this;
  }

  // If an instruction is any of the provided kinds, calls the function. This
  // doesn't provide a typed instruction because the instruction types aren't
  // guaranteed to overlap; use `Case<Any*>` for typed instructions if that's
  // needed.
  template <typename... InstT>
  auto Cases(llvm::function_ref<void()> fn) -> TypedInstSwitch& {
    if (!done_ && (inst_.Is<InstT>() || ...)) {
      fn();
      done_ = true;
    }
    return *this;
  }

  // Provides a default handler.
  auto Default(llvm::function_ref<void()> fn) -> void {
    if (!done_) {
      fn();
    }
  }

 private:
  // The instruction being matched.
  const SemIR::Inst& inst_;
  // Whether any prior case has matched.
  bool done_ = false;
};

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_TYPED_INST_SWITCH_H_
