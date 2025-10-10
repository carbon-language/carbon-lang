// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lower/specific_coalescer.h"

#include "common/check.h"
#include "common/vlog.h"

namespace Carbon::Lower {

SpecificCoalescer::SpecificCoalescer(llvm::raw_ostream* vlog_stream,
                                     const SemIR::SpecificStore& specifics)
    : vlog_stream_(vlog_stream),
      lowered_specifics_type_fingerprint_(specifics, {}),
      lowered_specific_fingerprint_(specifics, {}),
      equivalent_specifics_(specifics, SemIR::SpecificId::None) {}

auto SpecificCoalescer::CoalesceEquivalentSpecifics(
    LoweredSpecificsStore& lowered_specifics,
    LoweredLlvmFunctionStore& lowered_llvm_functions) -> void {
  for (auto& specifics : lowered_specifics.values()) {
    // Collect specifics to delete for each generic. Replace and remove each
    // after processing all specifics for a generic. Note, we could also
    // replace and remove all specifics after processing all generics.
    llvm::SmallVector<SemIR::SpecificId> specifics_to_delete;
    // i cannot be unsigned due to the comparison with a negative number when
    // the specifics vector is empty.
    for (int i = 0; i < static_cast<int>(specifics.size()) - 1; ++i) {
      // This specific was already replaced, skip it.
      if (equivalent_specifics_.Get(specifics[i]).has_value() &&
          equivalent_specifics_.Get(specifics[i]) != specifics[i]) {
        specifics_to_delete.push_back(specifics[i]);
        specifics[i] = specifics[specifics.size() - 1];
        specifics.pop_back();
        --i;
        continue;
      }
      // TODO: Improve quadratic behavior by using a single hash based on
      // `lowered_specifics_type_fingerprint_` and `common_fingerprint`.
      for (int j = i + 1; j < static_cast<int>(specifics.size()); ++j) {
        // When the specific was already replaced, skip it.
        if (equivalent_specifics_.Get(specifics[j]).has_value() &&
            equivalent_specifics_.Get(specifics[j]) != specifics[j]) {
          specifics_to_delete.push_back(specifics[j]);
          specifics[j] = specifics[specifics.size() - 1];
          specifics.pop_back();
          --j;
          continue;
        }

        // When the two specifics are not equivalent due to the function type
        // info stored in lowered_specifics_types, mark non-equivalance. This
        // can be reused to short-cut another path and continue the search for
        // other equivalences.
        if (!AreFunctionTypesEquivalent(specifics[i], specifics[j])) {
          InsertPair(specifics[i], specifics[j], non_equivalent_specifics_);
          continue;
        }

        Set<std::pair<SemIR::SpecificId, SemIR::SpecificId>>
            visited_equivalent_specifics;
        InsertPair(specifics[i], specifics[j], visited_equivalent_specifics);
        // Function type information matches; check usages inside the function
        // body that are dependent on the specific. This information has been
        // stored in lowered_states while lowering each function body.
        if (AreFunctionBodiesEquivalent(specifics[i], specifics[j],
                                        visited_equivalent_specifics)) {
          // When processing equivalences, we may change the canonical specific
          // multiple times, so we don't delete replaced specifics until the
          // end.
          visited_equivalent_specifics.ForEach(
              [&](std::pair<SemIR::SpecificId, SemIR::SpecificId>
                      equivalent_entry) {
                CARBON_VLOG("Found equivalent specifics: {0}, {1}",
                            equivalent_entry.first, equivalent_entry.second);
                ProcessSpecificEquivalence(equivalent_entry);
              });

          // Removed the replaced specific from the list of emitted specifics.
          // Only the top level, since the others are somewhere else in the
          // vector, they will be found and removed during processing.
          if (equivalent_specifics_.Get(specifics[j]).has_value() &&
              equivalent_specifics_.Get(specifics[j]) != specifics[j]) {
            specifics_to_delete.push_back(specifics[j]);
            specifics[j] = specifics[specifics.size() - 1];
            specifics.pop_back();
            --j;
          } else {
            // j was the canonical one, remove specifics[i], exit j loop.
            specifics_to_delete.push_back(specifics[i]);
            specifics[i] = specifics[specifics.size() - 1];
            specifics.pop_back();
            --i;
            break;
          }
        } else {
          // Only mark non-equivalence based on state for starting specifics.
          InsertPair(specifics[i], specifics[j], non_equivalent_specifics_);
        }
      }
    }

    // Once all equivalences are found for a generic, update and delete up
    // equivalent specifics.
    for (auto specific_id : specifics_to_delete) {
      UpdateAndDeleteLLVMFunction(lowered_llvm_functions, specific_id);
    }
  }
}

auto SpecificCoalescer::ProcessSpecificEquivalence(
    std::pair<SemIR::SpecificId, SemIR::SpecificId> pair) -> void {
  auto [specific_id1, specific_id2] = pair;
  CARBON_CHECK(specific_id1.has_value() && specific_id2.has_value(),
               "Expected values in equivalence check");

  auto get_canon = [&](SemIR::SpecificId specific_id) {
    auto equiv_id = equivalent_specifics_.Get(specific_id);
    return equiv_id.has_value() ? equiv_id : specific_id;
  };
  auto canon_id1 = get_canon(specific_id1);
  auto canon_id2 = get_canon(specific_id2);

  if (canon_id1 == canon_id2) {
    // Already equivalent, there was a previous replacement.
    return;
  }

  if (canon_id1.index >= canon_id2.index) {
    // Prefer the earlier index for canonical values.
    std::swap(canon_id1, canon_id2);
  }

  // Update equivalent_specifics_ for all. This is used as an indicator that
  // this specific_id may be the canonical one when reducing the equivalence
  // chains in `IsKnownEquivalence`.
  equivalent_specifics_.Set(specific_id1, canon_id1);
  equivalent_specifics_.Set(specific_id2, canon_id1);
  equivalent_specifics_.Set(canon_id1, canon_id1);
  equivalent_specifics_.Set(canon_id2, canon_id1);
}

auto SpecificCoalescer::UpdateEquivalentSpecific(SemIR::SpecificId specific_id)
    -> void {
  if (!equivalent_specifics_.Get(specific_id).has_value()) {
    return;
  }

  llvm::SmallVector<SemIR::SpecificId> stack;
  SemIR::SpecificId specific_to_update = specific_id;
  SemIR::SpecificId equivalent = equivalent_specifics_.Get(specific_to_update);
  SemIR::SpecificId equivalent_next = equivalent_specifics_.Get(equivalent);
  while (equivalent != equivalent_next) {
    stack.push_back(specific_to_update);
    specific_to_update = equivalent;
    equivalent = equivalent_next;
    equivalent_next = equivalent_specifics_.Get(equivalent_next);
  }

  for (auto specific : stack) {
    equivalent_specifics_.Set(specific, equivalent);
  }
}

auto SpecificCoalescer::UpdateAndDeleteLLVMFunction(
    LoweredLlvmFunctionStore& lowered_llvm_functions,
    SemIR::SpecificId specific_id) -> void {
  UpdateEquivalentSpecific(specific_id);
  auto* old_function = lowered_llvm_functions.Get(specific_id);
  auto* new_function =
      lowered_llvm_functions.Get(equivalent_specifics_.Get(specific_id));
  old_function->replaceAllUsesWith(new_function);
  old_function->eraseFromParent();
  lowered_llvm_functions.Set(specific_id, new_function);
}

auto SpecificCoalescer::IsKnownEquivalence(SemIR::SpecificId specific_id1,
                                           SemIR::SpecificId specific_id2)
    -> bool {
  if (!equivalent_specifics_.Get(specific_id1).has_value() ||
      !equivalent_specifics_.Get(specific_id2).has_value()) {
    return false;
  }

  UpdateEquivalentSpecific(specific_id1);
  UpdateEquivalentSpecific(specific_id2);

  return equivalent_specifics_.Get(specific_id1) ==
         equivalent_specifics_.Get(specific_id2);
}

auto SpecificCoalescer::AreFunctionTypesEquivalent(
    SemIR::SpecificId specific_id1, SemIR::SpecificId specific_id2) -> bool {
  CARBON_CHECK(specific_id1.has_value() && specific_id2.has_value());
  return lowered_specifics_type_fingerprint_.Get(specific_id1) ==
         lowered_specifics_type_fingerprint_.Get(specific_id2);
}

auto SpecificCoalescer::AreFunctionBodiesEquivalent(
    SemIR::SpecificId specific_id1, SemIR::SpecificId specific_id2,
    Set<std::pair<SemIR::SpecificId, SemIR::SpecificId>>&
        visited_equivalent_specifics) -> bool {
  llvm::SmallVector<std::pair<SemIR::SpecificId, SemIR::SpecificId>> worklist;
  worklist.push_back({specific_id1, specific_id2});

  while (!worklist.empty()) {
    auto outer_pair = worklist.pop_back_val();
    auto [specific_id1, specific_id2] = outer_pair;

    auto state1 = lowered_specific_fingerprint_.Get(specific_id1);
    auto state2 = lowered_specific_fingerprint_.Get(specific_id2);
    if (state1.common_fingerprint != state2.common_fingerprint) {
      InsertPair(specific_id1, specific_id2, non_equivalent_specifics_);
      return false;
    }
    if (state1.specific_fingerprint == state2.specific_fingerprint) {
      continue;
    }

    // A size difference should have been detected by the common fingerprint.
    CARBON_CHECK(state1.calls.size() == state2.calls.size(),
                 "Number of specific calls expected to be the same.");

    for (auto [state1_call, state2_call] :
         llvm::zip(state1.calls, state2.calls)) {
      if (state1_call != state2_call) {
        if (ContainsPair(state1_call, state2_call, non_equivalent_specifics_)) {
          return false;
        }
        if (IsKnownEquivalence(state1_call, state2_call)) {
          continue;
        }
        if (!InsertPair(state1_call, state2_call,
                        visited_equivalent_specifics)) {
          continue;
        }
        // Leave the added equivalence pair in place and continue.
        worklist.push_back({state1_call, state2_call});
      }
    }
  }
  return true;
}

auto SpecificCoalescer::InsertPair(
    SemIR::SpecificId specific_id1, SemIR::SpecificId specific_id2,
    Set<std::pair<SemIR::SpecificId, SemIR::SpecificId>>& set_of_pairs)
    -> bool {
  if (specific_id1.index > specific_id2.index) {
    std::swap(specific_id1.index, specific_id2.index);
  }
  auto insert_result =
      set_of_pairs.Insert(std::make_pair(specific_id1, specific_id2));
  return insert_result.is_inserted();
}

auto SpecificCoalescer::ContainsPair(
    SemIR::SpecificId specific_id1, SemIR::SpecificId specific_id2,
    const Set<std::pair<SemIR::SpecificId, SemIR::SpecificId>>& set_of_pairs)
    -> bool {
  if (specific_id1.index > specific_id2.index) {
    std::swap(specific_id1.index, specific_id2.index);
  }
  return set_of_pairs.Contains(std::make_pair(specific_id1, specific_id2));
}

}  // namespace Carbon::Lower
