
// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_SCOPE_INDEX_H_
#define CARBON_TOOLCHAIN_CHECK_SCOPE_INDEX_H_

#include "toolchain/base/index_base.h"

namespace Carbon::Check {

// An index for a pushed scope. This may correspond to a permanent scope with a
// corresponding `NameScope`, in which case a different index will be assigned
// each time the scope is entered. Alternatively, it may be a temporary scope
// such as is created for a block, and will only be entered once.
//
// `ScopeIndex` values are comparable. Lower `ScopeIndex` values correspond to
// scopes entered earlier in the file.
struct ScopeIndex : public IndexBase, public Printable<ScopeIndex> {
  static const ScopeIndex Package;

  using IndexBase::IndexBase;
};

constexpr ScopeIndex ScopeIndex::Package = ScopeIndex(0);

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_SCOPE_INDEX_H_
