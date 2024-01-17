// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/value_stores.h"
#include <ucontext.h>
#include <cstdint>

#include "llvm/ADT/StringSwitch.h"
#include "toolchain/base/index_base.h"
#include "toolchain/sem_ir/file.h"

namespace Carbon::SemIR {

auto ConstantStore::GetOrAdd(Inst inst) -> std::pair<InstId, bool> {
  // Compute the instruction's profile.
  ConstantNode node = {.inst = inst, .inst_id = InstId::Invalid};
  llvm::FoldingSetNodeID id;
  node.Profile(id, constants_.getContext());

  // Check if we have already created this constant.
  void* insert_pos;
  if (ConstantNode *found = constants_.FindNodeOrInsertPos(id, insert_pos)) {
    return {found->inst_id, false};
  }

  // Create the new inst and insert the new node.
  node.inst_id = constants_.getContext()->insts().AddInNoBlock(
      ParseNodeAndInst::Untyped(Parse::NodeId::Invalid, inst));
  constants_.InsertNode(new (*allocator_) ConstantNode(node), insert_pos);
  return {node.inst_id, true};
}

auto ConstantStore::vector() const -> std::vector<InstId> {
  std::vector<InstId> result;
  result.reserve(constants_.size());
  for (const ConstantNode& node : constants_) {
    result.push_back(node.inst_id);
  }
  // For stability, put the results into index order. This happens to also be
  // insertion order.
  std::sort(result.begin(), result.end(),
            [](InstId a, InstId b) { return a.index < b.index; });
  return result;
}

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

// Profiling for the block ID argument of a struct type.
//
// For struct types, we don't promote the `StructTypeField`s into the constant
// block, and instead leave them as unattached instructions. This means we need
// to recurse into them here.
//
// TODO: Should we add the `StructTypeField`s to the constants block?
static auto StructTypeProfileArgFunction(llvm::FoldingSetNodeID& id,
                                         File* context, int32_t arg) -> void {
  for (auto inst_id : context->inst_blocks().Get(InstBlockId(arg))) {
    auto field = context->insts().GetAs<StructTypeField>(inst_id);
    id.AddInteger(field.name_id.index);
    id.AddInteger(field.field_type_id.index);
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
  auto& real = context->reals().Get(RealId(arg));
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

static constexpr auto KindIndex(InstKind::RawEnumType kind) -> int {
  return static_cast<InstKind::UnderlyingType>(kind);
}

auto ConstantStore::ConstantNode::Profile(llvm::FoldingSetNodeID& id,
                                          File* sem_ir) -> void {
  // TODO: This counter, the table below, and `KindIndex` should probably all
  // get moved into `inst_kind.h`.
  static constexpr int NumInstKinds =
#define CARBON_SEM_IR_INST_KIND(KindName) +1
#include "toolchain/sem_ir/inst_kind.def"
      ;
  using InstKindProfileFunctions = std::pair<ProfileArgFunction*, ProfileArgFunction*>;
  using Table = std::array<InstKindProfileFunctions, NumInstKinds>;
  static constexpr Table ProfileArgFunctions = [] {
    Table result = {
#define CARBON_SEM_IR_INST_KIND(KindName)                         \
  InstKindProfileFunctions{SelectProfileArgFunction<KindName, 0>, \
                           SelectProfileArgFunction<KindName, 1>},
#include "toolchain/sem_ir/inst_kind.def"
    };
    result[KindIndex(StructType::Kind)].first = StructTypeProfileArgFunction;
    return result;
  }();

  inst.kind().Profile(id);
  auto kind_index = KindIndex(inst.kind());
  ProfileArgFunctions[kind_index].first(id, sem_ir, inst.arg0());
  ProfileArgFunctions[kind_index].second(id, sem_ir, inst.arg1());
}

// Get the spelling to use for a special name.
static auto GetSpecialName(NameId name_id, bool for_ir) -> llvm::StringRef {
  switch (name_id.index) {
    case NameId::Invalid.index:
      return for_ir ? "" : "<invalid>";
    case NameId::SelfValue.index:
      return "self";
    case NameId::SelfType.index:
      return "Self";
    case NameId::ReturnSlot.index:
      return for_ir ? "return" : "<return slot>";
    case NameId::PackageNamespace.index:
      return "package";
    case NameId::Base.index:
      return "base";
    default:
      CARBON_FATAL() << "Unknown special name";
  }
}

auto NameStoreWrapper::GetFormatted(NameId name_id) const -> llvm::StringRef {
  // If the name is an identifier name with a keyword spelling, format it with
  // an `r#` prefix. Format any other identifier name as just the identifier.
  if (auto string_name = GetAsStringIfIdentifier(name_id)) {
    return llvm::StringSwitch<llvm::StringRef>(*string_name)
#define CARBON_KEYWORD_TOKEN(Name, Spelling) .Case(Spelling, "r#" Spelling)
#include "toolchain/lex/token_kind.def"
        .Default(*string_name);
  }
  return GetSpecialName(name_id, /*for_ir=*/false);
}

auto NameStoreWrapper::GetIRBaseName(NameId name_id) const -> llvm::StringRef {
  if (auto string_name = GetAsStringIfIdentifier(name_id)) {
    return *string_name;
  }
  return GetSpecialName(name_id, /*for_ir=*/true);
}

auto NameScope::Print(llvm::raw_ostream& out) const -> void {
  out << "{inst: " << inst_id << ", enclosing_scope: " << enclosing_scope_id
      << ", has_error: " << (has_error ? "true" : "false");

  out << ", extended_scopes: [";
  llvm::ListSeparator scope_sep;
  for (auto id : extended_scopes) {
    out << scope_sep << id;
  }
  out << "]";

  out << ", names: {";
  // Sort name keys to get stable output.
  llvm::SmallVector<NameId> keys;
  for (auto [key, _] : names) {
    keys.push_back(key);
  }
  llvm::sort(keys,
             [](NameId lhs, NameId rhs) { return lhs.index < rhs.index; });
  llvm::ListSeparator key_sep;
  for (auto key : keys) {
    out << key_sep << key << ": " << names.find(key)->second;
  }
  out << "}";

  out << "}";
}

}  // namespace Carbon::SemIR
