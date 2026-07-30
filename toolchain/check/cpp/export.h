// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_CPP_EXPORT_H_
#define CARBON_TOOLCHAIN_CHECK_CPP_EXPORT_H_

#include "clang/AST/Decl.h"
#include "toolchain/check/context.h"
#include "toolchain/sem_ir/ids.h"

namespace clang {
class CXXDestructorDecl;
class CXXMethodDecl;
class CXXRecordDecl;
}  // namespace clang

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
auto ExportClassToCpp(Context& context, SemIR::ClassType class_type)
    -> clang::TagDecl*;

// Exports a generic Carbon class into C++ as a templated class.
//
// If the generic class has already been exported, returns the existing
// C++ class template.  Otherwise, creates a new C++ class template and
// returns it. Returns nullptr if the class could not be exported and an
// error was diagnosed.
auto ExportGenericClassToCpp(Context& context, SemIR::InstId inst_id,
                             SemIR::GenericClassType generic_class_type)
    -> clang::ClassTemplateDecl*;

// Creates a C++ class template specialization for a generic Carbon
// class.
//
// Returns true if a specialization was added, false otherwise.
auto ExportClassSpecializationToCpp(
    Context& context, clang::ClassTemplateDecl* class_template_decl,
    llvm::ArrayRef<clang::TemplateArgument> template_args) -> bool;

// Export all `SemIR::FieldDecl`s in the class body as `clang::FieldDecl`s.
auto ExportAllFieldsToCpp(Context& context,
                          SemIR::TypeInstId class_type_inst_id) -> void;

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
                      SemIR::FieldDecl field_decl,
                      SemIR::SpecificId specific_id) -> clang::FieldDecl*;

// Get a `clang::FunctionDecl` that can be used to call a Carbon function.
// If the function is generic, a `clang::FunctionTemplateDecl` will be
// created instead.
auto ExportFunctionToCpp(Context& context, SemIR::LocId loc_id,
                         SemIR::FunctionId function_id) -> clang::NamedDecl*;

// Exports a Carbon virtual function as a C++ `clang::FunctionDecl` declaration.
// Does not emit a definition.
auto ExportVirtualFunctionDeclToCpp(Context& context, SemIR::LocId loc_id,
                                    clang::CXXRecordDecl* parent,
                                    SemIR::FunctionId callee_function_id)
    -> clang::CXXMethodDecl*;

// Defines an virtual function that was previously exported to C++ with
// ExportVirtualFunctionDeclToCpp.
auto DefineExportedVirtualFunction(Context& context, SemIR::LocId loc_id,
                                   SemIR::FunctionId callee_function_id,
                                   clang::CXXMethodDecl* method_decl) -> void;

// Creates a C++ function template specialization for a generic Carbon
// function.
//
// Returns true if a specialization was added, false otherwise.
auto ExportFunctionSpecializationToCpp(
    Context& context, clang::FunctionTemplateDecl* function_template_decl,
    llvm::ArrayRef<clang::TemplateArgument> template_args) -> bool;

// Export a Carbon destructor into C++.
//
// The destructor calls the `Destroy` operator.
auto ExportDestructorToCpp(Context& context, const SemIR::Class& class_info,
                           clang::CXXRecordDecl* record_decl)
    -> clang::CXXDestructorDecl*;

// Export a Carbon variable into C++.
//
// Returns nullptr if the variable could not be exported an an error was
// diagnosed.
auto ExportVarToCpp(Context& context, SemIR::InstId inst_id,
                    SemIR::VarStorage var_storage) -> clang::VarDecl*;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_CPP_EXPORT_H_
