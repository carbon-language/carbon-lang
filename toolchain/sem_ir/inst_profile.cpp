// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/inst_profile.h"

#include <type_traits>

#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::SemIR {

// Profiling for unused arguments.
namespace {
struct NoArg {
  int32_t value;
};
}  // namespace
static auto ProfileArg(llvm::FoldingSetNodeID& /*id*/, const File& /*sem_ir*/,
                       NoArg arg) -> void {
  CARBON_CHECK(arg.value == IdBase::InvalidIndex)
      << "Unexpected value for unused argument.";
}

// Profiling for ID arguments that should participate in the instruction's
// value but are already canonical.
static auto ProfileArg(llvm::FoldingSetNodeID& id, const File& /*sem_ir*/,
                       int32_t arg) -> void {
  id.AddInteger(arg);
}

// Profiling for block ID arguments for which the content of the block should be
// included.
static auto ProfileArg(llvm::FoldingSetNodeID& id, const File& sem_ir,
                       InstBlockId block_id) -> void {
  for (auto inst_id : sem_ir.inst_blocks().Get(block_id)) {
    id.AddInteger(inst_id.index);
  }
}

// Profiling for type block ID arguments for which the content of the block
// should be included.
static auto ProfileArg(llvm::FoldingSetNodeID& id, const File& sem_ir,
                       TypeBlockId block_id) -> void {
  for (auto type_id : sem_ir.type_blocks().Get(block_id)) {
    id.AddInteger(type_id.index);
  }
}

// Profiling for integer IDs.
static auto ProfileArg(llvm::FoldingSetNodeID& id, const File& sem_ir,
                       IntId int_id) -> void {
  sem_ir.ints().Get(int_id).Profile(id);
}

// Profiling for real number IDs.
static auto ProfileArg(llvm::FoldingSetNodeID& id, const File& sem_ir,
                       RealId real_id) -> void {
  const auto& real = sem_ir.reals().Get(real_id);
  // TODO: Profile the value rather than the syntactic form.
  real.mantissa.Profile(id);
  real.exponent.Profile(id);
  id.AddBoolean(real.is_decimal);
}

// Wrap is_invocable_v to avoid a compile error on an unused lambda operand.
template <typename... T, typename F>
constexpr auto IsInvocable(F /*f*/) -> bool {
  return std::is_invocable_v<F, T...>;
}

// Selects the argument type to use to profile argument N of instruction InstT.
// We map as many types as we can to `int32_t` so that we can reuse the
// profiling code for all instructions that are profiled in the same way. For
// example, all instructions that take two IDs that are profiled by value use
// the same profiling code.
template <typename InstT, int N>
static auto ComputeProfileArgType() -> auto {
  if constexpr (N >= InstLikeTypeInfo<InstT>::NumArgs) {
    // This argument is not used by this instruction; don't profile it.
    return NoArg{0};
  } else {
    using ArgT = typename InstLikeTypeInfo<InstT>::template ArgType<N>;
    if constexpr (IsInvocable<ArgT>(
                      [](auto&& arg)
                          -> decltype(ProfileArg(
                              std::declval<llvm::FoldingSetNodeID&>(),
                              std::declval<const File&>(), arg)) {})) {
      return std::declval<ArgT>();
    } else {
      return std::declval<int32_t>();
    }
  }
}

template <typename InstT, int N>
using ProfileArgType = decltype(ComputeProfileArgType<InstT, N>());

// Profiles the given instruction arguments using the specified functions.
template <typename Arg0Type, typename Arg1Type>
static auto ProfileArgs(llvm::FoldingSetNodeID& id, const File& sem_ir,
                        int32_t arg0, int32_t arg1) -> void {
  ProfileArg(id, sem_ir, Arg0Type{arg0});
  ProfileArg(id, sem_ir, Arg1Type{arg1});
}

auto ProfileConstant(llvm::FoldingSetNodeID& id, const File& sem_ir, Inst inst)
    -> void {
  using ProfileArgsFunction =
      auto(llvm::FoldingSetNodeID&, const File&, int32_t, int32_t)->void;
  static constexpr ProfileArgsFunction* ProfileFunctions[] = {
#define CARBON_SEM_IR_INST_KIND(KindName) \
  ProfileArgs<ProfileArgType<KindName, 0>, ProfileArgType<KindName, 1>>,
#include "toolchain/sem_ir/inst_kind.def"
  };

  inst.kind().Profile(id);
  id.AddInteger(inst.type_id().index);
  ProfileFunctions[inst.kind().AsInt()](id, sem_ir, inst.arg0(), inst.arg1());
}

}  // namespace Carbon::SemIR
