// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/inst_profile.h"

#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/inst.h"
#include "toolchain/sem_ir/typed_insts.h"

namespace Carbon::SemIR {

// A function to profile an argument of an instruction.
using ProfileArgFunction = auto(llvm::FoldingSetNodeID&, File* context,
                                int32_t arg) -> void;

// Profiling for unused arguments.
static auto NullProfileArgFunction(llvm::FoldingSetNodeID& /*id*/,
                                   File* /*context*/, int32_t arg) -> void {
  CARBON_CHECK(arg == IdBase::InvalidIndex)
      << "Unexpected value for unused argument.";
}

// Profiling for ID arguments that should participate in the instruction's value.
static auto DefaultProfileArgFunction(llvm::FoldingSetNodeID& id,
                                      File* /*context*/, int32_t arg) -> void {
  id.AddInteger(arg);
}

// Profiling for block ID arguments for which the content of the block should be
// included.
static auto BlockProfileArgFunction(llvm::FoldingSetNodeID& id, File* context,
                                    int32_t arg) -> void {
  for (auto inst_id : context->inst_blocks().Get(InstBlockId(arg))) {
    id.AddInteger(inst_id.index);
  }
}

// Profiling for integer IDs.
static auto IntProfileArgFunction(llvm::FoldingSetNodeID& id, File* context,
                                  int32_t arg) -> void {
  context->ints().Get(IntId(arg)).Profile(id);
}

// Profiling for real number IDs.
static auto RealProfileArgFunction(llvm::FoldingSetNodeID& id, File* context,
                                   int32_t arg) -> void {
  const auto& real = context->reals().Get(RealId(arg));
  // TODO: Profile the value rather than the syntactic form.
  real.mantissa.Profile(id);
  real.exponent.Profile(id);
  id.AddBoolean(real.is_decimal);
}

// Selects the function to use to profile argument N of instruction InstT. This
// default implementation is used when there is no such argument, and adds
// nothing to the profile.
template <typename InstT, int N,
          bool InRange = (N < InstLikeTypeInfo<InstT>::NumArgs)>
static constexpr ProfileArgFunction* SelectProfileArgFunction =
    NullProfileArgFunction;

// Selects the function to use to profile an argument of type ArgT. This default
// implementation adds the ID itself to the profile.
template <typename ArgT>
static constexpr ProfileArgFunction* SelectProfileArgTypeFunction =
    DefaultProfileArgFunction;

// If InstT has an argument N, then profile it based on its type.
template <typename InstT, int N>
constexpr ProfileArgFunction* SelectProfileArgFunction<InstT, N, true> =
    SelectProfileArgTypeFunction<
        typename InstLikeTypeInfo<InstT>::template ArgType<N>>;

// If ArgT is InstBlockId, then profile the contents of the block.
// TODO: Consider deduplicating the block operands of constants themselves.
template <>
constexpr ProfileArgFunction* SelectProfileArgTypeFunction<InstBlockId> =
    BlockProfileArgFunction;

// If ArgT is IntId, then profile the `APInt`, which profiles the bit-width and
// value.
// TODO: Should IntIds themselves be deduplicated?
template <>
constexpr ProfileArgFunction* SelectProfileArgTypeFunction<IntId> =
    IntProfileArgFunction;

// If ArgT is RealId, then profile the value.
// TODO: Should RealIds themselves be deduplicated?
template <>
constexpr ProfileArgFunction* SelectProfileArgTypeFunction<RealId> =
    RealProfileArgFunction;

// Profiles the given instruction arguments using the specified functions.
template <ProfileArgFunction* ProfileArg0, ProfileArgFunction* ProfileArg1>
static auto ProfileArgs(llvm::FoldingSetNodeID& id, File* context, int32_t arg0,
                        int32_t arg1) -> void {
  ProfileArg0(id, context, arg0);
  ProfileArg1(id, context, arg1);
}

static constexpr auto KindIndex(InstKind::RawEnumType kind) -> int {
  return static_cast<InstKind::UnderlyingType>(kind);
}

auto ProfileConstant(llvm::FoldingSetNodeID& id, File* sem_ir, Inst inst) -> void {
  using ProfileArgsFunction =
      auto(llvm::FoldingSetNodeID&, File*, int32_t, int32_t)->void;
  static constexpr ProfileArgsFunction* ProfileFunctions[] = {
#define CARBON_SEM_IR_INST_KIND(KindName)            \
  ProfileArgs<SelectProfileArgFunction<KindName, 0>, \
              SelectProfileArgFunction<KindName, 1>>,
#include "toolchain/sem_ir/inst_kind.def"
  };

  inst.kind().Profile(id);
  ProfileFunctions[KindIndex(inst.kind())](id, sem_ir, inst.arg0(), inst.arg1());
}

}  // namespace Carbon::SemIR
