// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_IMPORT_CPP_H_
#define CARBON_TOOLCHAIN_CHECK_IMPORT_CPP_H_

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "toolchain/check/context.h"
#include "toolchain/check/diagnostic_helpers.h"
#include "toolchain/diagnostics/diagnostic_emitter.h"

namespace Carbon::Check {

// Generates a C++ header that includes the imported cpp files, parses it,
// generates the AST from it and links `SemIR::File` to it. Report C++ errors
// and warnings. If successful, adds a `Cpp` namespace and returns the AST.
auto ImportCppFiles(Context& context,
                    llvm::ArrayRef<Parse::Tree::PackagingNames> imports,
                    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> fs,
                    std::shared_ptr<clang::CompilerInvocation> invocation)
    -> std::unique_ptr<clang::ASTUnit>;

// Looks up the given name in the Clang AST generated when importing C++ code
// and returns a lookup result. If using the injected class name (`X.X()`),
// imports the class constructor as a function named as the class.
auto ImportNameFromCpp(Context& context, SemIR::LocId loc_id,
                       SemIR::NameScopeId scope_id, SemIR::NameId name_id)
    -> SemIR::ScopeLookupResult;

// Given a Carbon class declaration that was imported from some kind of C++
// declaration, such as a class or enum, attempt to import a corresponding class
// definition. Returns true if nothing went wrong (whether or not a definition
// could be imported), false if a diagnostic was produced.
auto ImportClassDefinitionForClangDecl(Context& context, SemIR::LocId loc_id,
                                       SemIR::ClassId class_id,
                                       SemIR::ClangDeclId clang_decl_id)
    -> bool;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_IMPORT_CPP_H_
