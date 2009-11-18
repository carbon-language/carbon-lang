//==--- DiagnosticParseKinds.td - libparse diagnostics --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Parser Diagnostics
//===----------------------------------------------------------------------===//

let Component = "Parse" in {

def w_asm_qualifier_ignored : Warning<"ignored %0 qualifier on asm">;

def ext_empty_source_file : Extension<"ISO C forbids an empty source file">;
def ext_top_level_semi : Extension<
  "extra ';' outside of a function">;
def ext_extra_struct_semi : Extension<
  "extra ';' inside a struct or union">;

def ext_duplicate_declspec : Extension<"duplicate '%0' declaration specifier">;
def ext_plain_complex : ExtWarn<
  "plain '_Complex' requires a type specifier; assuming '_Complex double'">;
def ext_integer_complex : Extension<
  "complex integer types are an extension">;
def ext_thread_before : Extension<"'__thread' before 'static'">;

def ext_empty_struct_union_enum : Extension<"use of empty %0 extension">;

def err_invalid_sign_spec : Error<"'%0' cannot be signed or unsigned">;
def err_invalid_short_spec : Error<"'short %0' is invalid">;
def err_invalid_long_spec : Error<"'long %0' is invalid">;
def err_invalid_longlong_spec : Error<"'long long %0' is invalid">;
def err_invalid_complex_spec : Error<"'_Complex %0' is invalid">;
def err_friend_storage_spec : Error<"'%0' is invalid in friend declarations">;

def ext_ident_list_in_param : Extension<
  "type-less parameter names in function declaration">;
def ext_c99_variable_decl_in_for_loop : Extension<
  "variable declaration in for loop is a C99-specific feature">;
def ext_c99_compound_literal : Extension<
  "compound literals are a C99-specific feature">;
def ext_enumerator_list_comma : Extension<
  "commas at the end of enumerator lists are a %select{C99|C++0x}0-specific feature">;

def ext_gnu_indirect_goto : Extension<
  "use of GNU indirect-goto extension">;
def ext_gnu_address_of_label : Extension<
  "use of GNU address-of-label extension">;
def ext_gnu_statement_expr : Extension<
  "use of GNU statement expression extension">;
def ext_gnu_conditional_expr : Extension<
  "use of GNU ?: expression extension, eliding middle term">;
def ext_gnu_empty_initializer : Extension<
  "use of GNU empty initializer extension">;
def ext_gnu_array_range : Extension<"use of GNU array range extension">;
def ext_gnu_missing_equal_designator : ExtWarn<
  "use of GNU 'missing =' extension in designator">;
def err_expected_equal_designator : Error<"expected '=' or another designator">;
def ext_gnu_old_style_field_designator : ExtWarn<
  "use of GNU old-style field designator extension">;
def ext_gnu_case_range : Extension<"use of GNU case range extension">;

// Generic errors.
def err_parse_error : Error<"parse error">;
def err_expected_expression : Error<"expected expression">;
def err_expected_type : Error<"expected a type">;
def err_expected_external_declaration : Error<"expected external declaration">;
def err_expected_ident : Error<"expected identifier">;
def err_expected_ident_lparen : Error<"expected identifier or '('">;
def err_expected_ident_lbrace : Error<"expected identifier or '{'">;
def err_expected_lbrace : Error<"expected '{'">;
def err_expected_lparen : Error<"expected '('">;
def err_expected_rparen : Error<"expected ')'">;
def err_expected_rsquare : Error<"expected ']'">;
def err_expected_rbrace : Error<"expected '}'">;
def err_expected_greater : Error<"expected '>'">;
def err_expected_semi_declaration : Error<
  "expected ';' at end of declaration">;
def err_expected_semi_decl_list : Error<
  "expected ';' at end of declaration list">;
def ext_expected_semi_decl_list : Extension<
  "expected ';' at end of declaration list">;
def err_expected_member_name_or_semi : Error<
  "expected member name or ';' after declaration specifiers">;
def err_function_declared_typedef : Error<
  "function definition declared 'typedef'">;
def err_expected_fn_body : Error<
  "expected function body after function declarator">;
def err_expected_method_body : Error<"expected method body">;
def err_invalid_token_after_toplevel_declarator : Error<
  "invalid token after top level declarator">;
def err_expected_statement : Error<"expected statement">;
def err_expected_lparen_after : Error<"expected '(' after '%0'">;
def err_expected_lparen_after_id : Error<"expected '(' after %0">;
def err_expected_less_after : Error<"expected '<' after '%0'">;
def err_expected_comma : Error<"expected ','">;
def err_expected_lbrace_in_compound_literal : Error<
  "expected '{' in compound literal">;
def err_expected_while : Error<"expected 'while' in do/while loop">;

def err_expected_semi_after : Error<"expected ';' after %0">;
def err_expected_semi_after_stmt : Error<"expected ';' after %0 statement">;
def err_expected_semi_after_expr : Error<"expected ';' after expression">;
def err_expected_semi_after_method_proto : Error<
  "expected ';' after method prototype">;
def err_expected_semi_after_namespace_name : Error<
  "expected ';' after namespace name">;
def err_unexpected_namespace_attributes_alias : Error<
  "attributes can not be specified on namespace alias">;
def err_expected_semi_after_attribute_list : Error<
  "expected ';' after attribute list">;
def err_expected_semi_after_static_assert : Error<
  "expected ';' after static_assert">;
def err_expected_semi_for : Error<"expected ';' in 'for' statement specifier">;
def err_expected_colon_after : Error<"expected ':' after %0">;
def err_label_end_of_compound_statement : Error<
  "label at end of compound statement: expected statement">;
def err_expected_string_literal : Error<"expected string literal">;
def err_expected_asm_operand : Error<
  "expected string literal or '[' for asm operand">;
def err_expected_selector_for_method : Error<
  "expected selector for Objective-C method">;
def err_expected_property_name : Error<"expected property name">;

def err_unexpected_at : Error<"unexpected '@' in program">;

def err_invalid_reference_qualifier_application : Error<
  "'%0' qualifier may not be applied to a reference">;
def err_illegal_decl_reference_to_reference : Error<
  "%0 declared as a reference to a reference">;
def err_rvalue_reference : Error<
  "rvalue references are only allowed in C++0x">;
def err_argument_required_after_attribute : Error<
  "argument required after attribute">;
def err_missing_param : Error<"expected parameter declarator">;
def err_missing_comma_before_ellipsis : Error<
  "C requires a comma prior to the ellipsis in a variadic function type">;
def err_unexpected_typedef_ident : Error<
  "unexpected type name %0: expected identifier">;
def err_expected_class_name : Error<"expected class name">;
def err_destructor_class_name : Error<
  "expected the class name after '~' to name a destructor">;
def err_unspecified_vla_size_with_static : Error<
  "'static' may not be used with an unspecified variable length array size">;

// Declarations.
def err_typename_requires_specqual : Error<
  "type name requires a specifier or qualifier">;
def err_typename_invalid_storageclass : Error<
  "type name does not allow storage class to be specified">;
def err_typename_invalid_functionspec : Error<
  "type name does not allow function specifier to be specified">;
def err_invalid_decl_spec_combination : Error<
  "cannot combine with previous '%0' declaration specifier">;
def err_friend_invalid_in_context : Error<
  "'friend' used outside of class">;
def err_unknown_typename : Error<
  "unknown type name %0">;
def err_use_of_tag_name_without_tag : Error<
  "use of tagged type %0 without '%1' tag">;
def err_expected_ident_in_using : Error<
  "expected an identifier in using directive">;

/// Objective-C parser diagnostics
def err_objc_no_attributes_on_category : Error<
  "attributes may not be specified on a category">;
def err_objc_missing_end : Error<"missing @end">;
def warn_objc_protocol_qualifier_missing_id : Warning<
  "protocol qualifiers without 'id' is archaic">;

def err_objc_illegal_visibility_spec : Error<
  "illegal visibility specification">;
def err_objc_illegal_interface_qual : Error<"illegal interface qualifier">;
def err_objc_expected_equal : Error<
  "setter/getter expects '=' followed by name">;
def err_objc_property_requires_field_name : Error<
  "property requires fields to be named">;
def err_objc_property_bitfield : Error<"property name cannot be a bitfield">;
def err_objc_expected_property_attr : Error<"unknown property attribute %0">;
def err_objc_propertoes_require_objc2 : Error<
  "properties are an Objective-C 2 feature">;
def err_objc_unexpected_attr : Error<
  "prefix attribute must be followed by an interface or protocol">;
def err_objc_directive_only_in_protocol : Error<
  "directive may only be specified in protocols only">;
def err_missing_catch_finally : Error<
  "@try statement without a @catch and @finally clause">;
def err_objc_concat_string : Error<"unexpected token after Objective-C string">;
def err_missing_sel_definition : Error<"cannot find definition of 'SEL'">;
def err_missing_id_definition : Error<"cannot find definition of 'id'">;
def err_missing_proto_definition : Error<
  "cannot find definition of 'Protocol'">;
def err_missing_class_definition : Error<"cannot find definition of 'Class'">;
def warn_expected_implementation : Warning<
  "@end must appear in an @implementation context">;
def error_property_ivar_decl : Error<
  "property synthesize requires specification of an ivar">;
def err_synthesized_property_name : Error<
  "expected a property name in @synthesize">;
def warn_semicolon_before_method_body : Warning<
  "semicolon before method body is ignored">,
  InGroup<DiagGroup<"semicolon-before-method-body">>, DefaultIgnore;

def err_expected_field_designator : Error<
  "expected a field designator, such as '.field = 4'">;

def err_declaration_does_not_declare_param : Error<
  "declaration does not declare a parameter">;
def err_no_matching_param : Error<"parameter named %0 is missing">;

/// C++ parser diagnostics
def err_expected_unqualified_id : Error<
  "expected %select{identifier|unqualified-id}0">;
def err_func_def_no_params : Error<
  "function definition does not declare parameters">;
def err_expected_lparen_after_type : Error<
  "expected '(' for function-style cast or type construction">;
def err_expected_equal_after_declarator : Error<
  "expected '=' after declarator">;
def warn_parens_disambiguated_as_function_decl : Warning<
  "parentheses were disambiguated as a function declarator">;
def err_expected_member_or_base_name : Error<
  "expected class member or base class name">;
def ext_ellipsis_exception_spec : Extension<
  "exception specification of '...' is a Microsoft extension">;
def err_expected_catch : Error<"expected catch">;
def err_expected_lbrace_or_comma : Error<"expected '{' or ','">;
def err_using_namespace_in_class : Error<
  "'using namespace' is not allowed in classes">;
def err_ident_in_pseudo_dtor_not_a_type : Error<
  "identifier %0 in pseudo-destructor expression does not name a type">;

// C++ derived classes
def err_dup_virtual : Error<"duplicate 'virtual' in base specifier">;

// C++ operator overloading
def err_operator_missing_type_specifier : Error<
  "missing type specifier after 'operator'">;

// Classes.
def err_anon_type_definition : Error<
  "declaration of anonymous %0 must be a definition">;


/// C++ Templates
def err_expected_template : Error<"expected template">;
def err_expected_comma_greater : Error<
  "expected ',' or '>' in template-parameter-list">;
def err_expected_type_id_after : Error<"expected type-id after '%0'">;
def err_expected_class_before : Error<"expected 'class' before '%0'">;
def err_template_spec_syntax_non_template : Error<
  "identifier followed by '<' indicates a class template specialization but "
  "%0 %select{does not refer to a template|refers to a function "
  "template|<unused>|refers to a template template parameter}1">;
def err_id_after_template_in_nested_name_spec : Error<
  "expected template name after 'template' keyword in nested name specifier">;
def err_id_after_template_in_typename_spec : Error<
  "expected template name after 'template' keyword in typename specifier">;
def err_less_after_template_name_in_nested_name_spec : Error<
  "expected '<' after 'template %0' in nested name specifier">;
def err_two_right_angle_brackets_need_space : Error<
  "a space is required between consecutive right angle brackets (use '> >')">;
def warn_cxx0x_right_shift_in_template_arg : Warning<
  "use of right-shift operator ('>>') in template argument will require "
  "parentheses in C++0x">;
def err_multiple_template_declarators : Error<
    "%select{|a template declaration|an explicit template specialization|"
    "an explicit template instantiation}0 can "
    "only %select{|declare|declare|instantiate}0 a single entity">;
def err_explicit_instantiation_with_definition : Error<
    "explicit template instantiation cannot have a definition; if this "
    "definition is meant to be an explicit specialization, add '<>' after the "
    "'template' keyword">;

def err_expected_qualified_after_typename : Error<
  "expected a qualified name after 'typename'">;
def err_typename_refers_to_non_type_template : Error<
  "typename specifier refers to a non-template">;
def err_expected_type_name_after_typename : Error<
  "expected an identifier or template-id after '::'">;
def err_explicit_spec_non_template : Error<
  "explicit %select{specialization|instantiation}0 of non-template "
  "%select{class|struct|union}1 %2">;

def err_variadic_templates : Error<
  "variadic templates are only allowed in C++0x">;
  
def err_default_template_template_parameter_not_template : Error<
  "default template argument for a template template parameter must be a class "
  "template">;
  
// C++ declarations
def err_friend_decl_defines_class : Error<
  "cannot define a type in a friend declaration">;
  
// Language specific pragmas
// - Generic warnings
def warn_pragma_expected_lparen : Warning<
  "missing '(' after '#pragma %0' - ignoring">;
def warn_pragma_expected_rparen : Warning<
  "missing ')' after '#pragma %0' - ignoring">;
def warn_pragma_expected_identifier : Warning<
  "expected identifier in '#pragma %0' - ignored">;  
def warn_pragma_extra_tokens_at_eol : Warning<
  "extra tokens at end of '#pragma %0' - ignored">; 
// - #pragma pack
def warn_pragma_pack_invalid_action : Warning<
  "unknown action for '#pragma pack' - ignored">;
def warn_pragma_pack_invalid_constant : Warning<
  "invalid constant for '#pragma pack', expected %0 - ignored">;
def warn_pragma_pack_malformed : Warning<
  "expected integer or identifier in '#pragma pack' - ignored">;
// - #pragma unused
def warn_pragma_unused_expected_var : Warning<
  "expected '#pragma unused' argument to be a variable name">;
def warn_pragma_unused_expected_punc : Warning<
  "expected ')' or ',' in '#pragma unused'">;

} // end of Parser diagnostics
