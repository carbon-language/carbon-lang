// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/inst_profile.h"

#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::SemIR {

// Profiling for ID arguments that should participate in the instruction's
// value.
template <typename ArgType>
static auto ProfileArg(llvm::FoldingSetNodeID& id, const File& /*sem_ir*/,
                       ArgType arg) -> void {
  id.AddInteger(arg.index);
}

// Profiling for unused arguments.
namespace {
struct NoArg {
  int32_t index;
};
}  // namespace
template <>
auto ProfileArg(llvm::FoldingSetNodeID& /*id*/, const File& /*sem_ir*/,
                NoArg arg) -> void {
  CARBON_CHECK(arg.index == IdBase::InvalidIndex)
      << "Unexpected value for unused argument.";
}

// Profiling for block ID arguments for which the content of the block should be
// included.
template <>
auto ProfileArg(llvm::FoldingSetNodeID& id, const File& sem_ir, InstBlockId arg)
    -> void {
  for (auto inst_id : sem_ir.inst_blocks().Get(arg)) {
    id.AddInteger(inst_id.index);
  }
}

// Profiling for type block ID arguments for which the content of the block
// should be included.
template <>
auto ProfileArg(llvm::FoldingSetNodeID& id, const File& sem_ir, TypeBlockId arg)
    -> void {
  for (auto type_id : sem_ir.type_blocks().Get(arg)) {
    id.AddInteger(type_id.index);
  }
}

// Profiling for integer IDs.
template <>
auto ProfileArg(llvm::FoldingSetNodeID& id, const File& sem_ir, IntId arg)
    -> void {
  sem_ir.ints().Get(arg).Profile(id);
}

// Profiling for real number IDs.
template <>
auto ProfileArg(llvm::FoldingSetNodeID& id, const File& sem_ir, RealId arg)
    -> void {
  const auto& real = sem_ir.reals().Get(arg);
  // TODO: Profile the value rather than the syntactic form.
  real.mantissa.Profile(id);
  real.exponent.Profile(id);
  id.AddBoolean(real.is_decimal);
}

using ProfileArgsFunction = auto(llvm::FoldingSetNodeID&, const File&, int32_t,
                                 int32_t) -> void;

// Profiles a pair of arguments. We separate this out from InstT so that any
// instruction with the same pair of arguments will share the same
// implementation.
template <typename ArgType0, typename ArgType1>
static auto ProfileArgs(llvm::FoldingSetNodeID& id, const File& sem_ir,
                        int32_t arg0, int32_t arg1) -> void {
  if constexpr (std::is_same_v<ArgType0, BuiltinKind> ||
                std::is_same_v<ArgType0, int32_t>) {
    id.AddInteger(arg0);
  } else {
    ProfileArg(id, sem_ir, ArgType0{arg0});
  }
  ProfileArg(id, sem_ir, ArgType1{arg1});
}

// Selects the function to use to profile arguments of instruction InstT.
template <typename InstT>
static constexpr auto ChooseProfileArgs() -> ProfileArgsFunction* {
  if constexpr (InstLikeTypeInfo<InstT>::NumArgs < 1) {
    return ProfileArgs<NoArg, NoArg>;
  } else {
    using ArgType0 = typename InstLikeTypeInfo<InstT>::template ArgType<0>;
    if constexpr (InstLikeTypeInfo<InstT>::NumArgs < 2) {
      return ProfileArgs<ArgType0, NoArg>;
    } else {
      using ArgType1 = typename InstLikeTypeInfo<InstT>::template ArgType<1>;
      return ProfileArgs<ArgType0, ArgType1>;
    }
  }
}

auto ProfileConstant(llvm::FoldingSetNodeID& id, const File& sem_ir, Inst inst)
    -> void {
  static constexpr ProfileArgsFunction* ProfileFunctions[] = {
#define CARBON_SEM_IR_INST_KIND(KindName) ChooseProfileArgs<KindName>(),
#include "toolchain/sem_ir/inst_kind.def"
  };

  inst.kind().Profile(id);
  id.AddInteger(inst.type_id().index);
  ProfileFunctions[inst.kind().AsInt()](id, sem_ir, inst.arg0(), inst.arg1());
}

}  // namespace Carbon::SemIR
