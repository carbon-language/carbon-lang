// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_CPP_EXPORT_H_
#define CARBON_TOOLCHAIN_CHECK_CPP_EXPORT_H_

#include "clang/AST/Decl.h"
#include "toolchain/check/context.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

// Exports a Carbon name scope into C++ as a namespace or class, or returns the
// C++ namespace or class declaration that it was imported from.
//
// If the name scope has already been exported, returns the existing context.
// Otherwise, creates a new C++ declaration context and returns it. Returns
// nullptr if the name scope could not be exported and an error was diagnosed.
auto ExportNameScopeToCpp(Context& context, SemIR::LocId loc_id,
                          SemIR::NameScopeId name_scope_id)
    -> clang::DeclContext*;

// Exports a Carbon class into C++ as a class, or returns the C++ tag type that
// the class was imported from.
//
// If the class has already been exported, returns the existing C++ class.
// Otherwise, creates a new C++ class and returns it. Returns nullptr if the
// class could not be exported and an error was diagnosed.
auto ExportClassToCpp(Context& context, SemIR::LocId loc_id,
                      SemIR::ClassType class_type) -> clang::TagDecl*;

// Export all `SemIR::FieldDecl`s in the class body as `clang::FieldDecl`s.
auto ExportAllFieldsToCpp(Context& context, SemIR::Class& class_info) -> void;

// Exports a Carbon class field into C++.
//
// If the field has already been exported, returns the existing C++
// field.
//
// If the field has not already been exported, *all* fields of the class
// are exported, and then the requested C++ field is returned.
//
// Returns nullptr if the class could not be exported and an error was
// diagnosed.
auto ExportFieldToCpp(Context& context, SemIR::InstId field_inst_id,
                      SemIR::FieldDecl field_decl) -> clang::FieldDecl*;

// Get the field offset for each field in a class.
//
// Returns true on success, false if any error occurs.
auto CalculateCppFieldOffsets(
    Context& context, SemIR::ClassId class_id,
    llvm::DenseMap<const clang::FieldDecl*, uint64_t>& field_offsets) -> bool;

// Get a `clang::FunctionDecl` that can be used to call a Carbon function.
auto ExportFunctionToCpp(Context& context, SemIR::LocId loc_id,
                         SemIR::FunctionId function_id) -> clang::FunctionDecl*;

// Export a Carbon destructor into C++.
//
// The destructor calls the `Destroy` operator.
auto ExportDestructorToCpp(Context& context, const SemIR::Class& class_info,
                           clang::CXXRecordDecl* record_decl)
    -> clang::CXXDestructorDecl*;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_CPP_EXPORT_H_
