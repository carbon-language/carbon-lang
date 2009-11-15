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

def note_using_decl : Note<"using">;
def note_also_found_decl : Note<"also found">;

// Parse && Lex
def err_expected_colon : Error<"expected ':'">;

// Parse && Sema
def err_no_declarators : Error<"declaration does not declare anything">;
def err_param_redefinition : Error<"redefinition of parameter %0">;
def err_invalid_storage_class_in_func_decl : Error<
  "invalid storage class specifier in function declarator">;
def err_expected_namespace_name : Error<"expected namespace name">;

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
def err_target_unknown_abi : Error<"unknown target ABI '%0'">;
def err_target_invalid_feature : Error<"invalid target feature '%0'">;

}
