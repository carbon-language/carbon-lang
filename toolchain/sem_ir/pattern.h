// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_SEM_IR_PATTERN_H_
#define CARBON_TOOLCHAIN_SEM_IR_PATTERN_H_

#include "toolchain/sem_ir/file.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::SemIR {

// Returns true if `pattern_id` is a `self` parameter pattern, such as
// `self: Foo` or `ref self: Self`.
auto IsSelfPattern(const File& sem_ir, InstId pattern_id) -> bool;

// Returns the given parameter patterns with a leading `self` parameter removed,
// or the patterns unchanged if the first one is not `self`.
auto ParamPatternsWithoutSelf(const File& sem_ir,
                              llvm::ArrayRef<InstId> param_patterns)
    -> llvm::ArrayRef<InstId>;
// As above, given the parameter patterns as an `InstBlockId`.
auto ParamPatternsWithoutSelf(const File& sem_ir, InstBlockId param_patterns_id)
    -> llvm::ArrayRef<InstId>;

// Returns the parameter patterns matched against a call's explicit arguments.
//
// `self` is a method's first parameter. A method call `obj.M(args)` provides
// `obj` as the receiver, binding it to `self`; `self_provided_as_receiver` is
// then true and the leading `self` parameter is excluded here. A method may
// also be called as an ordinary function, `Type.M(obj, args)`, with no
// receiver; `self` is then passed as the first explicit argument and is
// included.
auto CallArgParamPatterns(const File& sem_ir, InstBlockId param_patterns_id,
                          bool self_provided_as_receiver)
    -> llvm::ArrayRef<InstId>;

// If `pattern_id` introduces any name bindings, this returns the `EntityNameId`
// of the lexically-first such binding. Otherwise, returns `None`.
auto GetFirstBindingNameFromPatternId(const File& sem_ir, InstId pattern_id)
    -> EntityNameId;

// If `pattern_id` is a declaration of a single name, this returns that name,
// and otherwise returns `None`. This tries to "see through" wrappers like
// `AddrPattern` and `*ParamPattern`, so this may return the same name for
// different insts if one is an ancestor of the other (or if they represent
// separate declarations of the same name).
//
// This should only be used for decorative purposes such as SemIR
// pretty-printing or LLVM parameter naming.
auto GetPrettyNameFromPatternId(const File& sem_ir, InstId pattern_id)
    -> NameId;

}  // namespace Carbon::SemIR

#endif  // CARBON_TOOLCHAIN_SEM_IR_PATTERN_H_
