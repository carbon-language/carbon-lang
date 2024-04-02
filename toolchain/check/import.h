// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_IMPORT_H_
#define CARBON_TOOLCHAIN_CHECK_IMPORT_H_

#include "toolchain/check/context.h"
#include "toolchain/parse/node_ids.h"
#include "toolchain/sem_ir/file.h"

namespace Carbon::Check {

// Add imports from a single library in the current package. This pulls in all
// names; conflicts for things such as `package.a.b.c` will be flagged even
// though they are several layers deep.
auto ImportLibraryFromCurrentPackage(Context& context,
                                     SemIR::TypeId namespace_type_id,
                                     Parse::ImportDirectiveId node_id,
                                     bool is_api_for_impl,
                                     const SemIR::File& import_sem_ir) -> void;

// Adds another package's imports to name lookup, with all libraries together.
// This only adds the package name to lookup, so that `package.ImportedPackage`
// will resolve, and will provide a name scope that can be used for further
// qualified name lookups.
//
// import_irs may be empty. has_load_error is used to indicate if any library in
// the package failed to import correctly.
auto ImportLibrariesFromOtherPackage(Context& context,
                                     SemIR::TypeId namespace_type_id,
                                     Parse::ImportDirectiveId node_id,
                                     IdentifierId package_id,
                                     llvm::ArrayRef<SemIR::ImportIR> import_irs,
                                     bool has_load_error) -> void;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_IMPORT_H_
