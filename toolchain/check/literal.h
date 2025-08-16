// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_LITERAL_H_
#define CARBON_TOOLCHAIN_CHECK_LITERAL_H_

#include "toolchain/base/value_ids.h"
#include "toolchain/check/context.h"
#include "toolchain/check/convert.h"
#include "toolchain/lex/token_info.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// Forms an IntValue instruction with type `IntLiteral` for a given literal
// integer value, which is assumed to be unsigned.
auto MakeIntLiteral(Context& context, Parse::NodeId node_id, IntId int_id)
    -> SemIR::InstId;

// Forms an integer type expression for either an `iN` or `uN` literal.
auto MakeIntTypeLiteral(Context& context, Parse::NodeId node_id,
                        SemIR::IntKind int_kind, IntId size_id)
    -> SemIR::InstId;

// Forms an integer type of the specified kind and bit-width.
auto MakeIntType(Context& context, Parse::NodeId node_id,
                 SemIR::IntKind int_kind, IntId size_id) -> SemIR::TypeId;

// Forms a floating point type expression for `fN` literal.
auto MakeFloatTypeLiteral(Context& context, Parse::NodeId node_id,
                          IntId size_id) -> SemIR::InstId;

// Forms a string literal value instruction for a given string literal.
auto MakeStringLiteral(Context& context, Parse::StringLiteralId node_id,
                       StringLiteralValueId value_id) -> SemIR::InstId;

// Forms a string literal type expression for a `str` literal.
auto MakeStringTypeLiteral(Context& context, SemIR::LocId loc_id)
    -> SemIR::InstId;

// Forms a string type.
auto MakeStringType(Context& context, SemIR::LocId) -> TypeExpr;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_LITERAL_H_
