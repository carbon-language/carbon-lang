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

// Profiles the given instruction argument, which is of the specified kind.
static auto ProfileArg(llvm::FoldingSetNodeID& id, const File& sem_ir,
                       IdKind arg_kind, int32_t arg) -> void {
  static constexpr std::array<ProfileArgFunction*, IdKind::NumValues>
      ProfileFunctions = [] {
        std::array<ProfileArgFunction*, IdKind::NumValues> array;
        array.fill(DefaultProfileArgFunction);
        array[IdKind::None.ToIndex()] = NullProfileArgFunction;
        array[IdKind::For<InstBlockId>.ToIndex()] = InstBlockProfileArgFunction;
        array[IdKind::For<TypeBlockId>.ToIndex()] = TypeBlockProfileArgFunction;
        array[IdKind::For<IntId>.ToIndex()] = IntProfileArgFunction;
        array[IdKind::For<RealId>.ToIndex()] = RealProfileArgFunction;
        return array;
      }();
  ProfileFunctions[arg_kind.ToIndex()](id, sem_ir, arg);
}

auto ProfileConstant(llvm::FoldingSetNodeID& id, const File& sem_ir, Inst inst)
    -> void {
  inst.kind().Profile(id);
  id.AddInteger(inst.type_id().index);
  auto arg_kinds = inst.ArgKinds();
  ProfileArg(id, sem_ir, arg_kinds.first, inst.arg0());
  ProfileArg(id, sem_ir, arg_kinds.second, inst.arg1());
}

}  // namespace Carbon::SemIR
