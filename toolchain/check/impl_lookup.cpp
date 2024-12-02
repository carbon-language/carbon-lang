// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/impl_lookup.h"

#include "toolchain/check/deduce.h"
#include "toolchain/check/generic.h"
#include "toolchain/check/import_ref.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

static auto FindAssociatedImportIRs(Context& context,
                                    SemIR::ConstantId type_const_id,
                                    SemIR::ConstantId interface_const_id)
    -> llvm::SmallVector<SemIR::ImportIRId> {
  llvm::SmallVector<SemIR::ImportIRId> result;

  // Add an entity to our result.
  auto add_entity = [&](const SemIR::EntityWithParamsBase& entity) {
    // We will look for impls in the import IR associated with the first owning
    // declaration.
    auto decl_id = entity.first_owning_decl_id;
    if (!decl_id.is_valid()) {
      return;
    }
    auto loc_id = context.insts().GetLocId(decl_id);
    if (loc_id.is_import_ir_inst_id()) {
      result.push_back(
          context.import_ir_insts().Get(loc_id.import_ir_inst_id()).ir_id);
    }
  };

  llvm::SmallVector<SemIR::InstId> worklist;
  worklist.push_back(context.constant_values().GetInstId(type_const_id));
  worklist.push_back(context.constant_values().GetInstId(interface_const_id));

  // Push the contents of an instruction block onto our worklist.
  auto push_block = [&](SemIR::InstBlockId block_id) {
    if (block_id.is_valid()) {
      auto block = context.inst_blocks().Get(block_id);
      worklist.append(block.begin(), block.end());
    }
  };

  // Add the arguments of a specific to the worklist.
  auto push_args = [&](SemIR::SpecificId specific_id) {
    if (specific_id.is_valid()) {
      push_block(context.specifics().Get(specific_id).args_id);
    }
  };

  while (!worklist.empty()) {
    auto inst_id = worklist.pop_back_val();

    // Visit the operands of the constant.
    auto inst = context.insts().Get(inst_id);
    auto [arg0_kind, arg1_kind] = inst.ArgKinds();
    for (auto [arg, kind] :
         {std::pair{inst.arg0(), arg0_kind}, {inst.arg1(), arg1_kind}}) {
      switch (kind) {
        case SemIR::IdKind::For<SemIR::InstId>: {
          if (auto id = SemIR::InstId(arg); id.is_valid()) {
            worklist.push_back(id);
          }
          break;
        }
        case SemIR::IdKind::For<SemIR::InstBlockId>: {
          push_block(SemIR::InstBlockId(arg));
          break;
        }
        case SemIR::IdKind::For<SemIR::ClassId>: {
          add_entity(context.classes().Get(SemIR::ClassId(arg)));
          break;
        }
        case SemIR::IdKind::For<SemIR::InterfaceId>: {
          add_entity(context.interfaces().Get(SemIR::InterfaceId(arg)));
          break;
        }
        case SemIR::IdKind::For<SemIR::FacetTypeId>: {
          const auto& facet_type_info =
              context.facet_types().Get(SemIR::FacetTypeId(arg));
          for (const auto& impl : facet_type_info.impls_constraints) {
            add_entity(context.interfaces().Get(impl.interface_id));
            push_args(impl.specific_id);
          }
          break;
        }
        case SemIR::IdKind::For<SemIR::FunctionId>: {
          add_entity(context.functions().Get(SemIR::FunctionId(arg)));
          break;
        }
        case SemIR::IdKind::For<SemIR::SpecificId>: {
          push_args(SemIR::SpecificId(arg));
          break;
        }
        default: {
          break;
        }
      }
    }
  }

  // Deduplicate.
  std::sort(result.begin(), result.end(),
            [](SemIR::ImportIRId a, SemIR::ImportIRId b) {
              return a.index < b.index;
            });
  result.erase(std::unique(result.begin(), result.end()), result.end());

  return result;
}

auto LookupInterfaceWitness(Context& context, SemIR::LocId loc_id,
                            SemIR::ConstantId type_const_id,
                            SemIR::ConstantId interface_const_id)
    -> SemIR::InstId {
  auto import_irs =
      FindAssociatedImportIRs(context, type_const_id, interface_const_id);
  for (auto import_ir : import_irs) {
    // TODO: Instead of importing all impls, only import ones that are in some
    // way connected to this query.
    for (auto impl_index : llvm::seq(
             context.import_irs().Get(import_ir).sem_ir->impls().size())) {
      // TODO: Track the relevant impls and only consider those ones and any
      // local impls, rather than looping over all impls below.
      ImportImpl(context, import_ir, SemIR::ImplId(impl_index));
    }
  }

  // TODO: Add a better impl lookup system. At the very least, we should only be
  // considering impls that are for the same interface we're querying. We can
  // also skip impls that mention any types that aren't part of our impl query.
  for (const auto& impl : context.impls().array_ref()) {
    auto specific_id = SemIR::SpecificId::Invalid;
    if (impl.generic_id.is_valid()) {
      specific_id = DeduceImplArguments(context, loc_id, impl, type_const_id,
                                        interface_const_id);
      if (!specific_id.is_valid()) {
        continue;
      }
    }
    if (!context.constant_values().AreEqualAcrossDeclarations(
            SemIR::GetConstantValueInSpecific(context.sem_ir(), specific_id,
                                              impl.self_id),
            type_const_id)) {
      continue;
    }
    if (!context.constant_values().AreEqualAcrossDeclarations(
            SemIR::GetConstantValueInSpecific(context.sem_ir(), specific_id,
                                              impl.constraint_id),
            interface_const_id)) {
      // TODO: An impl of a constraint type should be treated as implementing
      // the constraint's interfaces.
      continue;
    }
    if (!impl.witness_id.is_valid()) {
      // TODO: Diagnose if the impl isn't defined yet?
      return SemIR::InstId::Invalid;
    }
    LoadImportRef(context, impl.witness_id);
    if (specific_id.is_valid()) {
      // We need a definition of the specific `impl` so we can access its
      // witness.
      ResolveSpecificDefinition(context, specific_id);
    }
    return context.constant_values().GetInstId(
        SemIR::GetConstantValueInSpecific(context.sem_ir(), specific_id,
                                          impl.witness_id));
  }
  return SemIR::InstId::Invalid;
}

}  // namespace Carbon::Check
