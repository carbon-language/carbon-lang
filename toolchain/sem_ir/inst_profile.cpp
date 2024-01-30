// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/inst_profile.h"

#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/inst.h"

namespace Carbon::SemIR {

// A function to profile an argument of an instruction.
using ProfileArgFunction = auto(llvm::FoldingSetNodeID&, const File& sem_ir,
                                int32_t arg) -> void;

// Profiling for unused arguments.
static auto NullProfileArgFunction(llvm::FoldingSetNodeID& /*id*/,
                                   const File& /*sem_ir*/, int32_t arg)
    -> void {
  CARBON_CHECK(arg == IdBase::InvalidIndex)
      << "Unexpected value for unused argument.";
}

// Profiling for ID arguments that should participate in the instruction's
// value.
static auto DefaultProfileArgFunction(llvm::FoldingSetNodeID& id,
                                      const File& /*sem_ir*/, int32_t arg)
    -> void {
  id.AddInteger(arg);
}

// Profiling for block ID arguments for which the content of the block should be
// included.
static auto InstBlockProfileArgFunction(llvm::FoldingSetNodeID& id,
                                        const File& sem_ir, int32_t arg)
    -> void {
  for (auto inst_id : sem_ir.inst_blocks().Get(InstBlockId(arg))) {
    id.AddInteger(inst_id.index);
  }
}

// Profiling for type block ID arguments for which the content of the block
// should be included.
static auto TypeBlockProfileArgFunction(llvm::FoldingSetNodeID& id,
                                        const File& sem_ir, int32_t arg)
    -> void {
  for (auto type_id : sem_ir.type_blocks().Get(TypeBlockId(arg))) {
    id.AddInteger(type_id.index);
  }
}

// Profiling for integer IDs.
static auto IntProfileArgFunction(llvm::FoldingSetNodeID& id,
                                  const File& sem_ir, int32_t arg) -> void {
  sem_ir.ints().Get(IntId(arg)).Profile(id);
}

// Profiling for real number IDs.
static auto RealProfileArgFunction(llvm::FoldingSetNodeID& id,
                                   const File& sem_ir, int32_t arg) -> void {
  const auto& real = sem_ir.reals().Get(RealId(arg));
  // TODO: Profile the value rather than the syntactic form.
  real.mantissa.Profile(id);
  real.exponent.Profile(id);
  id.AddBoolean(real.is_decimal);
}

// Selects the function to use to profile argument N of instruction InstT. We
// compute this in advance so that we can reuse the profiling code for all
// instructions that are profiled in the same way. For example, all instructions
// that take two IDs that are profiled by value use the same profiling code,
// namely `ProfileArgs<DefaultProfileArgFunction, DefaultProfileArgFunction>`.
template <typename InstT, int N>
static constexpr auto SelectProfileArgFunction() -> ProfileArgFunction* {
  if constexpr (N >= Internal::InstLikeTypeInfo<InstT>::NumArgs) {
    // This argument is not used by this instruction; don't profile it.
    return NullProfileArgFunction;
  } else {
    using ArgT = Internal::InstLikeTypeInfo<InstT>::template ArgType<N>;
    if constexpr (std::is_same_v<ArgT, InstBlockId>) {
      return InstBlockProfileArgFunction;
    } else if constexpr (std::is_same_v<ArgT, TypeBlockId>) {
      return TypeBlockProfileArgFunction;
    } else if constexpr (std::is_same_v<ArgT, IntId>) {
      return IntProfileArgFunction;
    } else if constexpr (std::is_same_v<ArgT, RealId>) {
      return RealProfileArgFunction;
    } else {
      return DefaultProfileArgFunction;
    }
  }
}

// Profiles the given instruction arguments using the specified functions.
template <ProfileArgFunction* ProfileArg0, ProfileArgFunction* ProfileArg1>
static auto ProfileArgs(llvm::FoldingSetNodeID& id, const File& sem_ir,
                        int32_t arg0, int32_t arg1) -> void {
  ProfileArg0(id, sem_ir, arg0);
  ProfileArg1(id, sem_ir, arg1);
}

auto ProfileConstant(llvm::FoldingSetNodeID& id, const File& sem_ir, Inst inst)
    -> void {
  using ProfileArgsFunction =
      auto(llvm::FoldingSetNodeID&, const File&, int32_t, int32_t)->void;
  static constexpr ProfileArgsFunction* ProfileFunctions[] = {
#define CARBON_SEM_IR_INST_KIND(KindName)              \
  ProfileArgs<SelectProfileArgFunction<KindName, 0>(), \
              SelectProfileArgFunction<KindName, 1>()>,
#include "toolchain/sem_ir/inst_kind.def"
  };

  inst.kind().Profile(id);
  id.AddInteger(inst.type_id().index);
  ProfileFunctions[inst.kind().AsInt()](id, sem_ir, inst.arg0(), inst.arg1());
}

}  // namespace Carbon::SemIR
