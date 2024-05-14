// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/inst_profile.h"

#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/ids.h"
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
  auto inst_block_id = InstBlockId(arg);
  if (!inst_block_id.is_valid()) {
    id.AddInteger(-1);
    return;
  }

  auto inst_block = sem_ir.inst_blocks().Get(inst_block_id);
  id.AddInteger(inst_block.size());
  for (auto inst_id : inst_block) {
    id.AddInteger(inst_id.index);
  }
}

// Profiling for type block ID arguments for which the content of the block
// should be included.
static auto TypeBlockProfileArgFunction(llvm::FoldingSetNodeID& id,
                                        const File& sem_ir, int32_t arg)
    -> void {
  auto type_block_id = TypeBlockId(arg);
  if (!type_block_id.is_valid()) {
    id.AddInteger(-1);
    return;
  }

  auto type_block = sem_ir.type_blocks().Get(type_block_id);
  id.AddInteger(type_block.size());
  for (auto type_id : type_block) {
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

// Profiling for BindNameInfo.
static auto BindNameIdProfileArgFunction(llvm::FoldingSetNodeID& id,
                                         const File& sem_ir, int32_t arg)
    -> void {
  const auto& [name_id, enclosing_scope_id, bind_index] =
      sem_ir.bind_names().Get(BindNameId(arg));
  id.AddInteger(name_id.index);
  id.AddInteger(enclosing_scope_id.index);
  id.AddInteger(bind_index.index);
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
        array[IdKind::For<BindNameId>.ToIndex()] = BindNameIdProfileArgFunction;
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
