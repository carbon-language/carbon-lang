// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/impl_validation.h"

#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "toolchain/base/kind_switch.h"
#include "toolchain/check/diagnostic_helpers.h"
#include "toolchain/check/impl_lookup.h"
#include "toolchain/check/import_ref.h"
#include "toolchain/check/type_structure.h"
#include "toolchain/sem_ir/entity_with_params_base.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/type_iterator.h"

namespace Carbon::Check {

namespace {

// All information about a `SemIR::Impl` needed for validation.
struct ImplInfo {
  SemIR::ImplId impl_id;
  bool is_final;
  SemIR::InstId witness_id;
  SemIR::TypeInstId self_id;
  SemIR::InstId latest_decl_id;
  SemIR::SpecificInterface interface;
  // Whether the `impl` decl was imported or from the local file.
  bool is_local;
  // If imported, the IR from which the `impl` decl was imported.
  SemIR::ImportIRId ir_id;
  std::optional<TypeStructure> type_structure;
};

}  // namespace

static auto GetIRId(Context& context, SemIR::InstId owning_inst_id)
    -> SemIR::ImportIRId {
  if (!owning_inst_id.has_value()) {
    return SemIR::ImportIRId::None;
  }
  return GetCanonicalImportIRInst(context, owning_inst_id).ir_id();
}

static auto GetImplInfo(Context& context, SemIR::ImplId impl_id) -> ImplInfo {
  const auto& impl = context.impls().Get(impl_id);
  auto ir_id = GetIRId(context, impl.first_owning_decl_id);
  return {.impl_id = impl_id,
          .is_final = impl.is_final,
          .witness_id = impl.witness_id,
          .self_id = impl.self_id,
          .latest_decl_id = impl.latest_decl_id(),
          .interface = impl.interface,
          .is_local = !ir_id.has_value(),
          .ir_id = ir_id,
          .type_structure =
              BuildTypeStructure(context, impl.self_id, impl.interface)};
}

// A final impl must be in the same file as either its root self type or the
// interface in its constraint.
//
// Returns true if an error was diagnosed.
static auto DiagnoseFinalImplNotInSameFileAsRootSelfTypeOrInterface(
    Context& context, const ImplInfo& impl, SemIR::ImportIRId interface_ir_id)
    -> bool {
  bool self_type_same_file = false;

  auto type_iter = SemIR::TypeIterator(&context.sem_ir());
  type_iter.Add(impl.self_id);
  auto step = type_iter.Next();

  using Step = SemIR::TypeIterator::Step;
  CARBON_KIND_SWITCH(step.any) {
    case CARBON_KIND(Step::ClassStart start): {
      auto inst_id = context.classes().Get(start.class_id).first_owning_decl_id;
      if (!GetIRId(context, inst_id).has_value()) {
        self_type_same_file = true;
      }
      break;
    }
    case CARBON_KIND(Step::ClassStartOnly start): {
      auto inst_id = context.classes().Get(start.class_id).first_owning_decl_id;
      if (!GetIRId(context, inst_id).has_value()) {
        self_type_same_file = true;
      }
      break;
    }

    case CARBON_KIND(Step::Done _): {
      CARBON_FATAL("self type is empty?");
    }

    default:
      break;
  }

  bool interface_same_file = !interface_ir_id.has_value();

  if (!self_type_same_file && !interface_same_file) {
    CARBON_DIAGNOSTIC(FinalImplInvalidFile, Error,
                      "`final impl` found in file that does not contain "
                      "the root self type nor the interface definition");
    context.emitter().Emit(impl.latest_decl_id, FinalImplInvalidFile);
    return true;
  }

  return false;
}

// The type structure each non-final `impl` must differ from all other non-final
// `impl` for the same interface visible from the file.
//
// Returns true if an error was diagnosed.
static auto DiagnoseNonFinalImplsWithSameTypeStructure(Context& context,
                                                       const ImplInfo& impl_a,
                                                       const ImplInfo& impl_b)
    -> bool {
  if (impl_a.type_structure == impl_b.type_structure) {
    CARBON_DIAGNOSTIC(ImplNonFinalSameTypeStructure, Error,
                      "found non-final `impl` with the same type "
                      "structure as another non-final `impl`");
    auto builder = context.emitter().Build(impl_b.latest_decl_id,
                                           ImplNonFinalSameTypeStructure);
    CARBON_DIAGNOSTIC(ImplNonFinalSameTypeStructureNote, Note,
                      "other `impl` here");
    builder.Note(impl_a.latest_decl_id, ImplNonFinalSameTypeStructureNote);
    builder.Emit();
    return true;
  }

  return false;
}

// An impl's self type and constraint can not match (as a lookup query)
// against any final impl, or it would never be used instead of that
// final impl.
//
// Returns true if an error was diagnosed.
static auto DiagnoseUnmatchableNonFinalImplWithFinalImpl(Context& context,
                                                         const ImplInfo& impl_a,
                                                         const ImplInfo& impl_b)
    -> bool {
  auto diagnose_unmatchable_impl = [&](const ImplInfo& query_impl,
                                       const ImplInfo& final_impl) -> bool {
    if (LookupMatchesImpl(context, SemIR::LocId(query_impl.latest_decl_id),
                          context.constant_values().Get(query_impl.self_id),
                          query_impl.interface, final_impl.impl_id)) {
      CARBON_DIAGNOSTIC(ImplFinalOverlapsNonFinal, Error,
                        "`impl` will never be used");
      auto builder = context.emitter().Build(query_impl.latest_decl_id,
                                             ImplFinalOverlapsNonFinal);
      CARBON_DIAGNOSTIC(
          ImplFinalOverlapsNonFinalNote, Note,
          "`final impl` declared here would always be used instead");
      builder.Note(final_impl.latest_decl_id, ImplFinalOverlapsNonFinalNote);
      builder.Emit();
      return true;
    }
    return false;
  };

  CARBON_CHECK(impl_a.is_final || impl_b.is_final);

  if (impl_b.is_final) {
    return diagnose_unmatchable_impl(impl_a, impl_b);
  } else {
    return diagnose_unmatchable_impl(impl_b, impl_a);
  }
}

// Final impls that overlap in their type structure must be in the
// same file.
//
// Returns true if an error was diagnosed.
static auto DiagnoseFinalImplsOverlapInDifferentFiles(Context& context,
                                                      const ImplInfo& impl_a,
                                                      const ImplInfo& impl_b)
    -> bool {
  if (impl_a.ir_id != impl_b.ir_id) {
    CARBON_DIAGNOSTIC(
        FinalImplOverlapsDifferentFile, Error,
        "`final impl` overlaps with `final impl` from another file");
    CARBON_DIAGNOSTIC(FinalImplOverlapsDifferentFileNote, Note,
                      "imported `final impl` here");
    if (impl_a.is_local) {
      auto builder = context.emitter().Build(impl_a.latest_decl_id,
                                             FinalImplOverlapsDifferentFile);
      builder.Note(impl_b.latest_decl_id, FinalImplOverlapsDifferentFileNote);
      builder.Emit();
    } else {
      auto builder = context.emitter().Build(impl_b.latest_decl_id,
                                             FinalImplOverlapsDifferentFile);
      builder.Note(impl_a.latest_decl_id, FinalImplOverlapsDifferentFileNote);
      builder.Emit();
    }
    return true;
  }

  return false;
}

// Two final impls in the same file can not overlap in their type
// structure if they are not in the same match_first block.
//
// TODO: Support for match_first needed here when they exist in the
// toolchain.
//
// Returns true if an error was diagnosed.
static auto DiagnoseFinalImplsOverlapOutsideMatchFirst(Context& context,
                                                       const ImplInfo& impl_a,
                                                       const ImplInfo& impl_b)
    -> bool {
  if (impl_a.is_local && impl_b.is_local) {
    CARBON_DIAGNOSTIC(FinalImplOverlapsSameFile, Error,
                      "`final impl` overlaps with `final impl` from same file "
                      "outside a `match_first` block");
    auto builder = context.emitter().Build(impl_b.latest_decl_id,
                                           FinalImplOverlapsSameFile);
    CARBON_DIAGNOSTIC(FinalImplOverlapsSameFileNote, Note,
                      "other `final impl` here");
    builder.Note(impl_a.latest_decl_id, FinalImplOverlapsSameFileNote);
    builder.Emit();
    return true;
  }

  return false;
}

static auto ValidateImplsForInterface(Context& context,
                                      llvm::ArrayRef<ImplInfo> impls) -> void {
  // All `impl`s we look at here have the same `InterfaceId` (though different
  // `SpecificId`s in their `SpecificInterface`s). So we can grab the
  // `ImportIRId` for the interface a single time up front.
  auto interface_decl_id = context.interfaces()
                               .Get(impls[0].interface.interface_id)
                               .first_owning_decl_id;
  auto interface_ir_id = GetIRId(context, interface_decl_id);

  for (const auto& impl : impls) {
    if (impl.is_final && impl.is_local) {
      // =======================================================================
      /// Rules for an individual final impl.
      // =======================================================================
      DiagnoseFinalImplNotInSameFileAsRootSelfTypeOrInterface(context, impl,
                                                              interface_ir_id);
    }
  }

  // TODO: We should revisit this and look for a way to do these checks in less
  // than quadratic time. From @zygoloid: Possibly by converting the set of
  // impls into a decision tree.
  //
  // For each impl, we compare it pair-wise which each impl found before it, so
  // that diagnostics are attached to the later impl, as the earlier impl on its
  // own does not generate a diagnostic.
  size_t num_impls = impls.size();
  for (auto [split_point, impl_b] : llvm::drop_begin(llvm::enumerate(impls))) {
    // Prevent diagnosing the same error multiple times for the same `impl_b`
    // against different impls before it. But still ensure we do give one of
    // each diagnostic when they are different errors.
    bool did_diagnose_non_final_impls_with_same_type_structure = false;
    bool did_diagnose_unmatchable_non_final_impl_with_final_impl = false;
    bool did_diagnose_final_impls_overlap_in_different_files = false;
    bool did_diagnose_final_impls_overlap_outside_match_first = false;

    auto impls_before = llvm::drop_end(impls, num_impls - split_point);
    for (const auto& impl_a : impls_before) {
      // Don't diagnose structures that contain errors.
      if (!impl_a.type_structure || !impl_b.type_structure) {
        continue;
      }

      // Only enforce rules when at least one of the impls was written in this
      // file.
      if (!impl_a.is_local && !impl_b.is_local) {
        continue;
      }

      if (!impl_a.is_final && !impl_b.is_final) {
        // =====================================================================
        // Rules between two non-final impls.
        // =====================================================================
        if (!did_diagnose_non_final_impls_with_same_type_structure) {
          // Two impls in separate files will need to have some different
          // concrete element in their type structure, as enforced by the orphan
          // rule. So we don't need to check against non-local impls.
          if (impl_a.is_local && impl_b.is_local) {
            if (DiagnoseNonFinalImplsWithSameTypeStructure(context, impl_a,
                                                           impl_b)) {
              // The same final `impl_a` may overlap with multiple `impl_b`s,
              // and we want to diagnose each `impl_b`.
              did_diagnose_non_final_impls_with_same_type_structure = true;
            }
          }
        }
      } else if (!impl_a.is_final || !impl_b.is_final) {
        // =====================================================================
        // Rules between final impl and non-final impl.
        // =====================================================================
        if (!did_diagnose_unmatchable_non_final_impl_with_final_impl) {
          if (DiagnoseUnmatchableNonFinalImplWithFinalImpl(context, impl_a,
                                                           impl_b)) {
            did_diagnose_unmatchable_non_final_impl_with_final_impl = true;
          }
        }
      } else if (impl_a.type_structure->CompareStructure(
                     TypeStructure::CompareTest::HasOverlap,
                     *impl_b.type_structure)) {
        // =====================================================================
        // Rules between two overlapping final impls.
        // =====================================================================
        CARBON_CHECK(impl_a.is_final && impl_b.is_final);
        if (!did_diagnose_final_impls_overlap_in_different_files) {
          if (DiagnoseFinalImplsOverlapInDifferentFiles(context, impl_a,
                                                        impl_b)) {
            did_diagnose_final_impls_overlap_in_different_files = true;
          }
        }
        if (!did_diagnose_final_impls_overlap_outside_match_first) {
          if (DiagnoseFinalImplsOverlapOutsideMatchFirst(context, impl_a,
                                                         impl_b)) {
            did_diagnose_final_impls_overlap_outside_match_first = true;
          }
        }
      }
    }
  }
}

// For each `impl` seen in this file, ensure that we import every available
// `final impl` for the same interface, so that we can to check for
// diagnostics about the relationship between them and the `impl`s in this
// file.
static auto ImportFinalImplsWithImplInFile(Context& context) -> void {
  struct InterfaceToImport {
    SemIR::ImportIRId ir_id;
    SemIR::InterfaceId interface_id;

    constexpr auto operator==(const InterfaceToImport& rhs) const
        -> bool = default;
    constexpr auto operator<=>(const InterfaceToImport& rhs) const -> auto {
      if (ir_id != rhs.ir_id) {
        return ir_id.index <=> rhs.ir_id.index;
      }
      return interface_id.index <=> rhs.interface_id.index;
    }
  };

  llvm::SmallVector<InterfaceToImport> interfaces_to_import;
  for (const auto& impl : context.impls().values()) {
    if (impl.witness_id == SemIR::ErrorInst::InstId) {
      continue;
    }
    auto impl_import_ir_id = GetIRId(context, impl.first_owning_decl_id);
    if (impl_import_ir_id.has_value()) {
      // Only import `impl`s of interfaces for which there is a local `impl` of
      // that that interface.
      continue;
    }

    auto interface_id = impl.interface.interface_id;
    const auto& interface = context.interfaces().Get(interface_id);
    auto import_ir_id = GetIRId(context, interface.first_owning_decl_id);
    if (!import_ir_id.has_value()) {
      continue;
    }
    interfaces_to_import.push_back(
        {.ir_id = import_ir_id, .interface_id = interface_id});
  }

  llvm::sort(interfaces_to_import);
  llvm::unique(interfaces_to_import);

  struct ImplToImport {
    SemIR::ImportIRId ir_id;
    SemIR::ImplId import_impl_id;

    constexpr auto operator==(const ImplToImport& rhs) const -> bool = default;
    constexpr auto operator<=>(const ImplToImport& rhs) const -> auto {
      if (ir_id != rhs.ir_id) {
        return ir_id.index <=> rhs.ir_id.index;
      }
      return import_impl_id.index <=> rhs.import_impl_id.index;
    }
  };

  llvm::SmallVector<ImplToImport> impls_to_import;
  for (auto [ir_id, interface_id] : interfaces_to_import) {
    const SemIR::File& sem_ir = *context.import_irs().Get(ir_id).sem_ir;
    for (auto [impl_id, impl] : sem_ir.impls().enumerate()) {
      if (impl.is_final && impl.interface.interface_id == interface_id) {
        impls_to_import.push_back({.ir_id = ir_id, .import_impl_id = impl_id});
      }
    }
  }

  llvm::sort(impls_to_import);
  llvm::unique(impls_to_import);

  for (auto [ir_id, import_impl_id] : impls_to_import) {
    ImportImpl(context, ir_id, import_impl_id);
  }
}

auto ValidateImplsInFile(Context& context) -> void {
  ImportFinalImplsWithImplInFile(context);

  // Collect all of the impls sorted into contiguous segments by their
  // interface. We only need to compare impls within each such segment. We don't
  // keep impls with an Error in them, as they may be missing other values
  // needed to check the diagnostics and they already have a diagnostic printed
  // about them anyhow. We also verify the impl has an `InterfaceId` since it
  // can be missing, in which case a diagnostic would have been generated
  // already as well.
  llvm::SmallVector<ImplInfo> impl_ids_by_interface(llvm::map_range(
      llvm::make_filter_range(
          context.impls().enumerate(),
          [](std::pair<SemIR::ImplId, const SemIR::Impl&> pair) {
            return pair.second.witness_id != SemIR::ErrorInst::InstId &&
                   pair.second.interface.interface_id.has_value();
          }),
      [&](std::pair<SemIR::ImplId, const SemIR::Impl&> pair) {
        return GetImplInfo(context, pair.first);
      }));
  llvm::stable_sort(impl_ids_by_interface, [](const ImplInfo& lhs,
                                              const ImplInfo& rhs) {
    return lhs.interface.interface_id.index < rhs.interface.interface_id.index;
  });

  const auto* it = impl_ids_by_interface.begin();
  while (it != impl_ids_by_interface.end()) {
    const auto* segment_begin = it;
    auto begin_interface_id = segment_begin->interface.interface_id;
    do {
      ++it;
    } while (it != impl_ids_by_interface.end() &&
             it->interface.interface_id == begin_interface_id);
    const auto* segment_end = it;

    ValidateImplsForInterface(context, {segment_begin, segment_end});
  }
}

}  // namespace Carbon::Check
