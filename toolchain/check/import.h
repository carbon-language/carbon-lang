// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_IMPORT_H_
#define CARBON_TOOLCHAIN_CHECK_IMPORT_H_

#include "toolchain/check/context.h"
#include "toolchain/parse/node_ids.h"
#include "toolchain/sem_ir/file.h"

namespace Carbon::Check {

// Imports the API file's name lookup information into a corresponding
// implementation file. Only information for the current package will be copied;
// information for other packages should be handled through
// ImportLibrariesFromOtherPackage.
auto ImportApiFile(Context& context, SemIR::TypeId namespace_type_id,
                   Parse::ImportDeclId node_id, const SemIR::File& api_sem_ir)
    -> void;

// Add the current package's imports to name lookup. This pulls in all names;
// conflicts for things such as `package.a.b.c` will be flagged even though they
// are several layers deep.
auto ImportLibrariesFromCurrentPackage(
    Context& context, SemIR::TypeId namespace_type_id,
    llvm::ArrayRef<SemIR::ImportIR> import_irs) -> void;

// Adds another package's imports to name lookup. This only adds the package
// name to lookup, so that `package.ImportedPackage` will resolve, and will
// provide a name scope that can be used for further qualified name lookups.
//
// import_irs may be empty. has_load_error is used to indicate if any library in
// the package failed to import correctly.
auto ImportLibrariesFromOtherPackage(Context& context,
                                     SemIR::TypeId namespace_type_id,
                                     Parse::ImportDeclId node_id,
                                     IdentifierId package_id,
                                     llvm::ArrayRef<SemIR::ImportIR> import_irs,
                                     bool has_load_error) -> void;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_IMPORT_H_
