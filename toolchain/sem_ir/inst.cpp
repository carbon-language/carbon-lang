// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/sem_ir/inst.h"

#include <utility>

#include "toolchain/base/block_value_store_impl.h"
#include "toolchain/base/value_store_impl.h"
#include "toolchain/sem_ir/file.h"

namespace Carbon::SemIR {

auto Inst::Print(llvm::raw_ostream& out) const -> void {
  out << "{kind: " << kind();

  auto print_args = [&](auto info) {
    using Info = decltype(info);
    if constexpr (Info::NumArgs > 0) {
      out << ", arg0: " << FromRaw<typename Info::template ArgType<0>>(arg0_);
    }
    if constexpr (Info::NumArgs > 1) {
      out << ", arg1: " << FromRaw<typename Info::template ArgType<1>>(arg1_);
    }
  };

  switch (kind()) {
#define CARBON_SEM_IR_INST_KIND(Name)               \
  case Name::Kind:                                  \
    print_args(Internal::InstLikeTypeInfo<Name>()); \
    break;
#include "toolchain/sem_ir/inst_kind.def"
  }
  if (type_id_.has_value()) {
    out << ", type: " << type_id_;
  }
  out << "}";
}

template <typename T>
static auto CacheIfBundleId(T /*arg*/, const BundleStore& /*bundle_store*/)
    -> void {}

template <typename BundleT>
static auto CacheIfBundleId(BundleId<BundleT> arg,
                            const BundleStore& bundle_store) -> void {
  bundle_store.CacheDebugKind(arg);
}

auto Inst::CacheBundleDebugKinds(const BundleStore& bundles) const -> void {
  auto cache_args = [&](auto info) {
    using Info = decltype(info);
    if constexpr (Info::NumArgs > 0) {
      CacheIfBundleId(FromRaw<typename Info::template ArgType<0>>(arg0_),
                      bundles);
    }
    if constexpr (Info::NumArgs > 1) {
      CacheIfBundleId(FromRaw<typename Info::template ArgType<1>>(arg1_),
                      bundles);
    }
  };

  switch (kind()) {
#define CARBON_SEM_IR_INST_KIND(Name)               \
  case Name::Kind:                                  \
    cache_args(Internal::InstLikeTypeInfo<Name>()); \
    break;
#include "toolchain/sem_ir/inst_kind.def"
  }
}

// Returns the IdKind of an instruction's argument, or None if there is no
// argument with that index.
template <typename InstKind, int ArgIndex>
static constexpr auto IdKindFor() -> IdKind {
  using Info = Internal::InstLikeTypeInfo<InstKind>;
  if constexpr (ArgIndex < Info::NumArgs) {
    return IdKind::For<typename Info::template ArgType<ArgIndex>>;
  } else {
    return IdKind::None;
  }
}

const std::pair<IdKind, IdKind> Inst::ArgKindTable[] = {
#define CARBON_SEM_IR_INST_KIND(Name) \
  {IdKindFor<Name, 0>(), IdKindFor<Name, 1>()},
#include "toolchain/sem_ir/inst_kind.def"
};

InstStore::InstStore(File* file, int32_t reserved_inst_ids)
    : file_(file), values_(file->check_ir_id(), reserved_inst_ids) {}

auto InstStore::GetUnattachedType(TypeId type_id) const -> TypeId {
  return file_->types().GetUnattachedType(type_id);
}

// Returns whether the imported and local instruction kinds are compatible.
// Instruction kinds are compatible when the kinds are the same, or when the
// imported kind is intentionally transformed into the local kind. For example,
// imports form namespace-like entries in their original IR, and are imported
// as namespaces locally.
static auto HasCompatibleImportedInstKind(InstKind imported_kind,
                                          InstKind local_kind) -> bool {
  if (imported_kind == local_kind) {
    return true;
  }
  if (imported_kind == ImportDecl::Kind && local_kind == Namespace::Kind) {
    // Namespace node kinds should be a superset of ImportDecl node kinds.
    static_assert(
        std::is_convertible_v<decltype(ImportDecl::Kind)::TypedNodeId,
                              decltype(Namespace::Kind)::TypedNodeId>);
    return true;
  }
  return false;
}

auto LocIdAndInst::RuntimeVerified(const File& file, LocId loc_id, Inst inst)
    -> LocIdAndInst {
  switch (loc_id.kind()) {
    case LocId::Kind::ImportIRInstId: {
      CARBON_CHECK(!IsSingletonInstKind(inst.kind()),
                   "Should never import builtins/singletons: {0}", inst);
      if (inst.IsOneOf<ImportRefLoaded, ImportRefUnloaded>()) {
        // These don't represent the in-use `InstKind`, so should not be
        // validated.
        break;
      }

      const auto& import_ir_inst =
          file.import_ir_insts().Get(loc_id.import_ir_inst_id());
      // We don't require a matching node kind if the location is in C++,
      // because there isn't a node.
      if (import_ir_inst.ir_id() == ImportIRId::Cpp) {
        break;
      }
      const auto* import_ir =
          file.import_irs().Get(import_ir_inst.ir_id()).sem_ir;
      auto imported_kind =
          import_ir->insts().Get(import_ir_inst.inst_id()).kind();
      CARBON_CHECK(HasCompatibleImportedInstKind(imported_kind, inst.kind()),
                   "Unexpected imported `InstKind` {0} for {1}", imported_kind,
                   inst);
      break;
    }

    case LocId::Kind::InstId:
      // TODO: Figure out right verification.
      break;

    case LocId::Kind::NodeId: {
      auto node_kind = file.parse_tree().node_kind(loc_id.node_id());
      CARBON_CHECK(inst.kind().IsAllowedNodeKind(node_kind),
                   "Unexpected `NodeKind` {0} for {1}", node_kind, inst);
      break;
    }

    case LocId::Kind::None:
      break;
  }
  return LocIdAndInst(loc_id, inst, /*is_unchecked=*/true);
}

}  // namespace Carbon::SemIR

namespace Carbon {
template class ValueStore<SemIR::InstBlockId,
                          llvm::MutableArrayRef<SemIR::InstId>,
                          Tag<SemIR::CheckIRId>>;
template class BlockValueStore<SemIR::InstBlockId, SemIR::InstId,
                               Tag<SemIR::CheckIRId>>;
}  // namespace Carbon
