//==--- DiagnosticCommonKinds.td - common diagnostics ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Common Helpers
//===----------------------------------------------------------------------===//

let Component = "Common" in {

// Basic.

def fatal_too_many_errors
  : Error<"too many errors emitted, stopping now">, DefaultFatal; 

def note_previous_definition : Note<"previous definition is here">;
def note_previous_declaration : Note<"previous declaration is here">;
def note_previous_implicit_declaration : Note<
  "previous implicit declaration is here">;
def note_previous_use : Note<"previous use is here">;
def note_duplicate_case_prev : Note<"previous case defined here">;
def note_forward_declaration : Note<"forward declaration of %0">;
def note_type_being_defined : Note<
  "definition of %0 is not complete until the closing '}'">;
/// note_matching - this is used as a continuation of a previous diagnostic,
/// e.g. to specify the '(' when we expected a ')'.
def note_matching : Note<"to match this '%0'">;

def note_using : Note<"using">;
def note_possibility : Note<"one possibility">;
def note_also_found : Note<"also found">;

// Parse && Lex
def err_expected_colon : Error<"expected ':'">;
def err_expected_colon_after_setter_name : Error<
  "method name referenced in property setter attribute "
  "must end with ':'">;

// Parse && Sema
def ext_no_declarators : ExtWarn<"declaration does not declare anything">,
  InGroup<MissingDeclarations>;
def err_param_redefinition : Error<"redefinition of parameter %0">;
def warn_method_param_redefinition : Warning<"redefinition of method parameter %0">;
def warn_method_param_declaration : Warning<"redeclaration of method parameter %0">,
  InGroup<DuplicateArgDecl>, DefaultIgnore;
def err_invalid_storage_class_in_func_decl : Error<
  "invalid storage class specifier in function declarator">;
def err_expected_namespace_name : Error<"expected namespace name">;
def ext_variadic_templates : ExtWarn<
  "variadic templates are a C++0x extension">, InGroup<CXX0x>;
def err_default_special_members : Error<
  "only special member functions may be defaulted">;
def err_friends_define_only_namespace_scope : Error<
  "cannot define a function with non-namespace scope in a friend declaration">;
def err_deleted_non_function : Error<
  "only functions can have deleted definitions">;
def err_module_not_found : Error<"module '%0' not found">, DefaultFatal;
  
// Sema && Lex
def ext_longlong : Extension<
  "'long long' is an extension when C99 mode is not enabled">,
  InGroup<LongLong>;
def warn_integer_too_large : Warning<
  "integer constant is too large for its type">;
def warn_integer_too_large_for_signed : Warning<
  "integer constant is so large that it is unsigned">;

// Sema && AST
def note_invalid_subexpr_in_ice : Note<
  "subexpression not valid in an integer constant expression">;

// Targets

def err_target_unknown_triple : Error<
  "unknown target triple '%0', please use -triple or -arch">;
def err_target_unknown_cpu : Error<"unknown target CPU '%0'">;
def err_target_unknown_abi : Error<"unknown target ABI '%0'">;
def err_target_unknown_cxxabi : Error<"unknown C++ ABI '%0'">;
def err_target_invalid_feature : Error<"invalid target feature '%0'">;

// Source manager
def err_cannot_open_file : Error<"cannot open file '%0': %1">, DefaultFatal;
def err_file_modified : Error<
  "file '%0' modified since it was first processed">, DefaultFatal;
def err_unsupported_bom : Error<"%0 byte order mark detected in '%1', but "
  "encoding is not supported">, DefaultFatal;
}
