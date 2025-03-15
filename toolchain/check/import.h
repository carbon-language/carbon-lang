// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_IMPORT_H_
#define CARBON_TOOLCHAIN_CHECK_IMPORT_H_

#include "toolchain/check/context.h"
#include "toolchain/parse/node_ids.h"
#include "toolchain/sem_ir/file.h"

namespace Carbon::Check {

struct AddImportNamespaceResult {
  // The namespace scope id.
  SemIR::NameScopeId name_scope_id;

  // The namespace instruction id.
  SemIR::InstId inst_id;
};

// Adds a namespace to the IR. Associates the namespace with the import referred
// in `import_id`, if any. Do not try to add the namespace `name_id` to the
// scope and don't check if it already exists. Used when the name is added
// afterwards.
auto AddImportNamespace(Context& context, SemIR::TypeId namespace_type_id,
                        SemIR::NameId name_id,
                        SemIR::NameScopeId parent_scope_id,
                        SemIR::InstId import_id) -> AddImportNamespaceResult;

struct AddImportNamespaceToScopeResult {
  AddImportNamespaceResult add_result;

  // When trying to add the namespace name, whether it already exists and refers
  // to a namespace.
  bool is_duplicate_of_namespace_in_current_package = false;
};

// Adds a namespace to the IR. Associates the namespace with the import returned
// from `make_import_id`, which must return a value. Tries to add the namespace
// `name_id` to the scope. If the name already exists, diagnose only if
// `diagnose_duplicate_namespace` is true or if it doesn't refer to a namespace.
// `diagnose_duplicate_namespace` is used when handling a cross-package import,
// where an existing namespace is in the current package and the new namespace
// is a different package.
auto AddImportNamespaceToScope(
    Context& context, SemIR::TypeId namespace_type_id, SemIR::NameId name_id,
    SemIR::NameScopeId parent_scope_id, bool diagnose_duplicate_namespace,
    llvm::function_ref<SemIR::InstId()> make_import_id)
    -> AddImportNamespaceToScopeResult;

// Imports the API file's name lookup information into a corresponding
// implementation file. Only information for the current package will be copied;
// information for other packages should be handled through
// ImportLibrariesFromOtherPackage.
auto ImportApiFile(Context& context, SemIR::TypeId namespace_type_id,
                   const SemIR::File& api_sem_ir) -> void;

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
                                     SemIR::InstId import_decl_id,
                                     PackageNameId package_id,
                                     llvm::ArrayRef<SemIR::ImportIR> import_irs,
                                     bool has_load_error) -> void;

// Given a name scope that corresponds to another package (having one or more
// `import_irs`), looks for the name in imports. Name resolution results are
// added to the scope, and the `InstId` (possibly `None`) is returned.
//
// In general, this will add an `ImportRef` and load it; it's never left
// unloaded because the result is expected to be immediately used. Namespaces
// will be directly produced, similar to how they function for imports from the
// current package. Conflicts will be resolved and diagnosed.
//
// Arguments are all in the context of the current IR. Scope lookup is expected
// to be resolved first.
auto ImportNameFromOtherPackage(
    Context& context, SemIRLoc loc, SemIR::NameScopeId scope_id,
    llvm::ArrayRef<std::pair<SemIR::ImportIRId, SemIR::NameScopeId>>
        import_ir_scopes,
    SemIR::NameId name_id) -> SemIR::InstId;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_IMPORT_H_
