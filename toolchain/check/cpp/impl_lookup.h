// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_CPP_IMPL_LOOKUP_H_
#define CARBON_TOOLCHAIN_CHECK_CPP_IMPL_LOOKUP_H_

#include "toolchain/check/context.h"
#include "toolchain/check/custom_witness.h"
#include "toolchain/check/impl_lookup.h"
#include "toolchain/check/interface.h"
#include "toolchain/check/type_structure.h"
#include "toolchain/sem_ir/ids.h"
#include "toolchain/sem_ir/specific_interface.h"

namespace Carbon::Check {

// Performs lookup for an impl witness for a query involving C++ types.
// Shouldn't be called with `CoreInterface::Unknown`, because only core
// interfaces can have lookup results. Returns a witness value, or `None` if a
// synthesized C++ witness should not be used.
//
// Given a known `core_interface`, we can synthesize a witness based on C++
// operator overloads or special member functions. Performs the suitable C++
// lookup to determine if this interface should be considered implemented for
// the specified type, and if so, synthesizes and returns a suitable witness.
//
// `best_impl_type_structure` provides the type structure of the best-matching
// impl declaration. If this is better than every viable C++ candidate, a "none"
// result will be returned. If this is worse than the best viable C++ candidate
// according to C++ rules, a witness for the C++ candidate will be returned.
// Otherwise, it is at least as good as the best viable C++ candidate, but there
// is some C++ candidate that has a better type structure, in which case the
// result is ambiguous and we diagnose an error. This parameter can be null if
// there is no usable impl for this query.
//
// `best_impl_loc_id` gives the location of the impl corresponding to the best
// type structure, and can be `None` if `best_impl_type_structure` is null. This
// parameter is used only for ambiguity diagnostics.
auto LookupCppImpl(Context& context, SemIR::LocId loc_id,
                   CoreInterface core_interface,
                   SemIR::ConstantId query_self_const_id,
                   SemIR::SpecificInterfaceId query_specific_interface_id,
                   const TypeStructure* best_impl_type_structure,
                   SemIR::LocId best_impl_loc_id) -> SemIR::InstId;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_CPP_IMPL_LOOKUP_H_
