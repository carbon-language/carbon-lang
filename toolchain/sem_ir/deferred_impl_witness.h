// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_DEFERRED_IMPL_WITNESS_H_
#define CARBON_TOOLCHAIN_SEM_IR_DEFERRED_IMPL_WITNESS_H_

#include <cstring>
#include <type_traits>

#include "toolchain/base/canonical_value_store.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/inst.h"

namespace Carbon::SemIR {

struct DeferredImplWitness : public Printable<DeferredImplWitness> {
  // The specific interface of the impl. Can be used for diagnostics to find the
  // name of an element being accessed through this witness.
  SpecificInterface specific_interface;
  // The `ImplWitness`, once it becomes known. This can be `None` for deferred
  // witnesses used while the impl is still being declared.
  InstId impl_witness;

  auto Print(llvm::raw_ostream& out) const -> void {
    out << "{" << "specific_interface: " << specific_interface
        << ", impl_witness: " << impl_witness << "}";
  }
};

inline auto CarbonHashtableEq(const DeferredImplWitness& lhs,
                              const DeferredImplWitness& rhs) -> bool {
  // This requires that there are no padding bits in the type.
  static_assert(std::has_unique_object_representations_v<DeferredImplWitness>);
  return std::memcmp(&lhs, &rhs, sizeof(DeferredImplWitness)) == 0;
}

inline auto CarbonHashValue(const DeferredImplWitness& value, uint64_t seed)
    -> HashCode {
  Hasher hasher(seed);
  hasher.HashRaw(value);
  return static_cast<HashCode>(hasher);
}

using DeferredImplWitnessStore =
    CanonicalValueStore<DeferredImplWitnessId, DeferredImplWitness,
                        Tag<CheckIRId>>;

}  // namespace Carbon::SemIR

namespace Carbon {
extern template class CanonicalValueStore<SemIR::DeferredImplWitnessId,
                                          SemIR::DeferredImplWitness,
                                          Tag<SemIR::CheckIRId>>;
}  // namespace Carbon

#endif  // CARBON_TOOLCHAIN_SEM_IR_DEFERRED_IMPL_WITNESS_H_
