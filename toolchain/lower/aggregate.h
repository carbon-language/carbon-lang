// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LOWER_AGGREGATE_H_
#define CARBON_TOOLCHAIN_LOWER_AGGREGATE_H_

#include "llvm/ADT/Twine.h"
#include "llvm/IR/Value.h"
#include "toolchain/lower/function_context.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Lower {

// Extracts an element of an aggregate, such as a struct, tuple, or class, by
// index. Depending on the expression category and value representation of the
// aggregate input, this will either produce a value or a reference.
auto GetAggregateElement(FunctionContext& context, SemIR::InstId aggr_inst_id,
                         SemIR::ElementIndex idx, SemIR::InstId result_inst_id,
                         llvm::Twine name) -> llvm::Value*;

// Emits the value representation for a struct or tuple whose elements are the
// contents of `refs_id`.
auto EmitAggregateValueRepr(FunctionContext& context,
                            SemIR::InstId value_inst_id,
                            SemIR::InstBlockId refs_id) -> llvm::Value*;

// Emits the initialization for a struct or tuple.
auto EmitAggregateInitializer(FunctionContext& context,
                              SemIR::InstId init_inst_id,
                              SemIR::InstBlockId refs_id, llvm::Twine name)
    -> llvm::Value*;

// Given that `element_id` is the `index` element of an aggregate of type
// `aggr_type`, returns a reference to that aggregate.
auto GetEnclosingAggregate(FunctionContext& context,
                           FunctionContext::TypeInFile aggr_type,
                           SemIR::InstId element_id, SemIR::ElementIndex index)
    -> llvm::Value*;

}  // namespace Carbon::Lower

#endif  // CARBON_TOOLCHAIN_LOWER_AGGREGATE_H_
