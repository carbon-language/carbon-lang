// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LOWER_SPECIFIC_COALESCER_H_
#define CARBON_TOOLCHAIN_LOWER_SPECIFIC_COALESCER_H_

#include "llvm/Support/BLAKE3.h"
#include "toolchain/lower/context.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Lower {

// Coalescing functionality for lowering fewer specifics of the same generic.
class SpecificCoalescer {
 public:
  using LoweredSpecificsStore =
      FixedSizeValueStore<SemIR::GenericId,
                          llvm::SmallVector<SemIR::SpecificId>>;
  using LoweredLlvmFunctionStore =
      FixedSizeValueStore<SemIR::SpecificId, llvm::Function*>;

  // Describes a specific function's body fingerprint.
  struct SpecificFunctionFingerprint {
    // Fingerprint with all specific-dependent instructions, except specific
    // calls. This is built by the `FunctionContext` while lowering each
    // instruction in the definition of a specific function.
    // TODO: This can be merged with the function type fingerprint, for a
    // single upfront non-equivalence check, and hash bucketing for deeper
    // equivalence evaluation.
    llvm::BLAKE3Result<32> common_fingerprint;
    // Fingerprint for all calls to specific functions (hashes all calls to
    // other specifics). This is built by the `FunctionContext` while lowering.
    llvm::BLAKE3Result<32> specific_fingerprint;
    // All non-hashed specific_ids of functions called.
    llvm::SmallVector<SemIR::SpecificId> calls;
  };

  // Takes a `SpecificStore` to help initialize related `FixedSizeValueStore`s.
  explicit SpecificCoalescer(llvm::raw_ostream* vlog_stream,
                             const SemIR::SpecificStore& specifics);

  // Entry point for coalescing equivalent specifics. Two function definitions,
  // from the same generic, with different specific_ids are considered
  // equivalent if, at the LLVM level, one can be replaced with the other, with
  // no change in behavior. All LLVM types and instructions must be equivalent.
  auto CoalesceEquivalentSpecifics(
      LoweredSpecificsStore& lowered_specifics,
      LoweredLlvmFunctionStore& lowered_llvm_functions) -> void;

  // Initializes and returns a SpecificFunctionFingerprint* instance for a
  // specific. The internal of the fingerprint are populated during and after
  // lowering the function body of that specific.
  auto InitializeFingerprintForSpecific(SemIR::SpecificId specific_id)
      -> SpecificFunctionFingerprint* {
    if (!specific_id.has_value()) {
      return nullptr;
    }
    return &lowered_specific_fingerprint_.Get(specific_id);
  }

  auto CreateTypeFingerprint(SemIR::SpecificId specific_id,
                             llvm::Type* llvm_type) -> void {
    llvm::BLAKE3 function_type_fingerprint;
    RawStringOstream os;
    llvm_type->print(os);
    function_type_fingerprint.update(os.TakeStr());
    function_type_fingerprint.final(
        lowered_specifics_type_fingerprint_.Get(specific_id));
  }

 private:
  // While coalescing specifics, returns whether the function types for two
  // specifics are equivalent. This uses a fingerprint generated for each
  // function type.
  auto AreFunctionTypesEquivalent(SemIR::SpecificId specific_id1,
                                  SemIR::SpecificId specific_id2) -> bool;

  // While coalescing specifics, compare the function bodies for two specifics.
  // This uses fingerprints generated during lowering of the function body.
  // The `visited_equivalent_specifics` parameter is used to track cycles in
  // the function callgraph, and will also return equivalent pairs of specifics
  // found, if the two specifics given as arguments are found to be equivalent.
  auto AreFunctionBodiesEquivalent(
      SemIR::SpecificId specific_id1, SemIR::SpecificId specific_id2,
      Set<std::pair<SemIR::SpecificId, SemIR::SpecificId>>&
          visited_equivalent_specifics) -> bool;

  // Given an equivalent pair of specifics, updates the canonical specific to
  // use for each of the two Specifics found to be equivalent.
  auto ProcessSpecificEquivalence(
      std::pair<SemIR::SpecificId, SemIR::SpecificId> pair) -> void;

  // Checks if two specific_ids are equivalent and also reduces the equivalence
  // chains/paths. This update ensures the canonical specific is always "one
  // hop away".
  auto IsKnownEquivalence(SemIR::SpecificId specific_id1,
                          SemIR::SpecificId specific_id2) -> bool;

  // Update the tracked equivalent specific for the `SpecificId`. This may
  // occur a replacement was performed and a chain of such replacements needs
  // to be followed to discover the canonical specific for the given argument.
  auto UpdateEquivalentSpecific(SemIR::SpecificId specific_id) -> void;

  // Update the LLVM function to use for a `SpecificId` that has been found to
  // have another equivalent LLVM function. Replace all uses of the original
  // LLVM function with the equivalent one found, and delete the previous LLVM
  // function body.
  auto UpdateAndDeleteLLVMFunction(
      LoweredLlvmFunctionStore& lowered_llvm_functions,
      SemIR::SpecificId specific_id) -> void;

  // Inserts a pair into a set of pairs in canonical form. Also implicitly
  // checks entry already existed if it cannot be inserted.
  auto InsertPair(
      SemIR::SpecificId specific_id1, SemIR::SpecificId specific_id2,
      Set<std::pair<SemIR::SpecificId, SemIR::SpecificId>>& set_of_pairs)
      -> bool;

  // Checks if a pair is contained into a set of pairs, in canonical form.
  auto ContainsPair(
      SemIR::SpecificId specific_id1, SemIR::SpecificId specific_id2,
      const Set<std::pair<SemIR::SpecificId, SemIR::SpecificId>>& set_of_pairs)
      -> bool;

  // The optional vlog stream.
  llvm::raw_ostream* vlog_stream_;

  // For specifics that exist in lowered_specifics, a hash of their function
  // type information.
  FixedSizeValueStore<SemIR::SpecificId, llvm::BLAKE3Result<32>>
      lowered_specifics_type_fingerprint_;

  // This is initialized and populated while lowering a specific.
  FixedSizeValueStore<SemIR::SpecificId, SpecificFunctionFingerprint>
      lowered_specific_fingerprint_;

  // Equivalent specifics that have been found. For each specific, this points
  // to the canonical equivalent specific, which may also be self. We currently
  // define the canonical specific as the one with the lowest
  // `SpecificId.index`.
  //
  // Entries are initialized to `SpecificId::None`, which defines that there is
  // no other equivalent specific to this `SpecificId`.
  FixedSizeValueStore<SemIR::SpecificId, SemIR::SpecificId>
      equivalent_specifics_;

  // Non-equivalent specifics found.
  // TODO: Revisit this due to its quadratic space growth.
  Set<std::pair<SemIR::SpecificId, SemIR::SpecificId>>
      non_equivalent_specifics_;
};

}  // namespace Carbon::Lower

#endif  // CARBON_TOOLCHAIN_LOWER_SPECIFIC_COALESCER_H_
