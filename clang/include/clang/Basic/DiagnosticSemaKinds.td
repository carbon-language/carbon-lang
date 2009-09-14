//==--- DiagnosticSemaKinds.td - libsema diagnostics ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Semantic Analysis
//===----------------------------------------------------------------------===//

let Component = "Sema" in {

// Constant expressions
def err_expr_not_ice : Error<
  "expression is not an integer constant expression">;
def ext_expr_not_ice : Extension<
  "expression is not integer constant expression "
  "(but is allowed as an extension)">;

def ext_null_pointer_expr_not_ice : Extension<
  "null pointer expression is not an integer constant expression "
  "(but is allowed as an extension)">;



// Semantic analysis of string and character constant literals.
def ext_predef_outside_function : Warning<
  "predefined identifier is only valid inside function">;

// C99 Designated Initializers
def err_array_designator_negative : Error<
  "array designator value '%0' is negative">;
def err_array_designator_empty_range : Error<
  "array designator range [%0, %1] is empty">;
def err_array_designator_non_array : Error<
  "array designator cannot initialize non-array type %0">;
def err_array_designator_too_large : Error<
  "array designator index (%0) exceeds array bounds (%1)">;
def err_field_designator_non_aggr : Error<
  "field designator cannot initialize a "
  "%select{non-struct, non-union|non-class}0 type %1">;
def err_field_designator_unknown : Error<
  "field designator %0 does not refer to any field in type %1">;
def err_field_designator_nonfield : Error<
  "field designator %0 does not refer to a non-static data member">;
def note_field_designator_found : Note<"field designator refers here">;
def err_designator_for_scalar_init : Error<
  "designator in initializer for scalar type %0">;
def warn_subobject_initializer_overrides : Warning<
  "subobject initialization overrides initialization of other fields "
  "within its enclosing subobject">;
def warn_initializer_overrides : Warning<
  "initializer overrides prior initialization of this subobject">;
def note_previous_initializer : Note<
  "previous initialization %select{|with side effects }0is here"
  "%select{| (side effects may not occur at run time)}0">;
def err_designator_into_flexible_array_member : Error<
  "designator into flexible array member subobject">;
def note_flexible_array_member : Note<
  "initialized flexible array member %0 is here">;
def ext_flexible_array_init : Extension<
  "flexible array initialization is a GNU extension">;

// Declarations.
def ext_vla : Extension<
  "variable length arrays are a C99 feature, accepted as an extension">;
def err_vla_cxx : Error<
  "variable length arrays are not permitted in C++">;
  
def ext_anon_param_requires_type_specifier : Extension<
  "type specifier required for unnamed parameter, defaults to int">;
def err_bad_variable_name : Error<
  "'%0' cannot be the name of a variable or data member">;
def err_parameter_name_omitted : Error<"parameter name omitted">;
def warn_unused_parameter : Warning<"unused parameter %0">,
  InGroup<UnusedParameter>, DefaultIgnore;
def warn_decl_in_param_list : Warning<
  "declaration of %0 will not be visible outside of this function">;

def warn_implicit_function_decl : Warning<
  "implicit declaration of function %0">,
  InGroup<ImplicitFunctionDeclare>, DefaultIgnore;
def ext_implicit_function_decl : Extension<
  "implicit declaration of function %0 is invalid in C99">,
  InGroup<ImplicitFunctionDeclare>;

def err_ellipsis_first_arg : Error<
  "ISO C requires a named argument before '...'">;
def err_declarator_need_ident : Error<"declarator requires an identifier">;
def err_bad_language : Error<"unknown linkage language">;
def warn_use_out_of_scope_declaration : Warning<
  "use of out-of-scope declaration of %0">;
def err_inline_non_function : Error<
  "'inline' can only appear on functions">;
  
// C++ using declarations
def err_using_requires_qualname : Error<
  "using declaration requires a qualified name">;
def err_using_typename_non_type : Error<
  "'typename' keyword used on a non-type">;
def err_using_decl_nested_name_specifier_is_not_a_base_class : Error<
  "using declaration refers into %0, which is not a base class of %1">;
def err_using_decl_can_not_refer_to_class_member : Error<
  "using declaration can not refer to class member">;
 def err_using_decl_can_not_refer_to_namespace : Error<
  "using declaration can not refer to namespace">;

def err_invalid_thread : Error<
  "'__thread' is only allowed on variable declarations">;
def err_thread_non_global : Error<
  "'__thread' variables must have global storage">;
def err_thread_unsupported : Error<
  "thread-local storage is unsupported for the current target">;

def warn_maybe_falloff_nonvoid_function : Warning<
  "control may reach end of non-void function">,
  InGroup<ReturnType>;
def warn_falloff_nonvoid_function : Warning<
  "control reaches end of non-void function">,
  InGroup<ReturnType>;
def err_maybe_falloff_nonvoid_block : Error<
  "control may reach end of non-void block">;
def err_falloff_nonvoid_block : Error<
  "control reaches end of non-void block">;
def warn_suggest_noreturn_function : Warning<
  "function could be attribute 'noreturn'">,
  InGroup<DiagGroup<"missing-noreturn">>, DefaultIgnore;
def warn_suggest_noreturn_block : Warning<
  "block could be attribute 'noreturn'">,
  InGroup<DiagGroup<"missing-noreturn">>, DefaultIgnore;

/// Built-in functions.
def ext_implicit_lib_function_decl : ExtWarn<
  "implicitly declaring C library function '%0' with type %1">;
def note_please_include_header : Note<
  "please include the header <%0> or explicitly provide a "
  "declaration for '%1'">;
def note_previous_builtin_declaration : Note<"%0 is a builtin with type %1">;
def err_implicit_decl_requires_stdio : Error<
  "implicit declaration of '%0' requires inclusion of the header <stdio.h>">;
def err_implicit_decl_requires_setjmp : Error<
  "implicit declaration of '%0' requires inclusion of the header <setjmp.h>">;
def warn_redecl_library_builtin : Warning<
  "incompatible redeclaration of library function %0">;
def err_builtin_definition : Error<"definition of builtin function %0">;
def err_types_compatible_p_in_cplusplus : Error<
  "__builtin_types_compatible_p is not valid in C++">;

/// main()
// static/inline main() are not errors in C, just in C++.
def warn_unusual_main_decl : Warning<"'main' should not be declared "
    "%select{static|inline|static or inline}0">;
def err_unusual_main_decl : Error<"'main' is not allowed to be declared "
    "%select{static|inline|static or inline}0">;
def err_main_returns_nonint : Error<"'main' must return 'int'">;
def err_main_surplus_args : Error<"%0 is too many arguments for 'main': "
    "must be 0, 2, or 3">;
def warn_main_one_arg : Warning<"one-argument 'main' is usually a mistake">;
def err_main_arg_wrong : Error<"%select{first|second|third}0 argument of "
    "'main' should be of type %1">;

/// parser diagnostics
def ext_typedef_without_a_name : ExtWarn<"typedef requires a name">;
def err_statically_allocated_object : Error<
  "interface type cannot be statically allocated">;
def err_object_cannot_be_passed_returned_by_value : Error<
  "interface type %1 cannot be %select{returned|passed}0 by value">;
def warn_enum_value_overflow : Warning<"overflow in enumeration value">;
def warn_pragma_pack_invalid_alignment : Warning<
  "expected #pragma pack parameter to be '1', '2', '4', '8', or '16'">;
// Follow the MSVC implementation.
def warn_pragma_pack_show : Warning<"value of #pragma pack(show) == %0">;
// FIXME: Dehardcode.
def warn_pragma_pack_pop_identifer_and_alignment : Warning<
  "specifying both a name and alignment to 'pop' is undefined">;
def warn_pragma_pack_pop_failed : Warning<"#pragma pack(pop, ...) failed: %0">;

def warn_pragma_unused_undeclared_var : Warning<
  "undeclared variable %0 used as an argument for '#pragma unused'">;
def warn_pragma_unused_expected_localvar : Warning<
  "only local variables can be arguments to '#pragma unused'">;
def err_unsupported_pragma_weak : Error<
  "using '#pragma weak' to refer to an undeclared identifier is not yet supported">;

/// Objective-C parser diagnostics
def err_duplicate_class_def : Error<
  "duplicate interface definition for class %0">;
def err_undef_superclass : Error<
  "cannot find interface declaration for %0, superclass of %1">;
def err_recursive_superclass : Error<
  "trying to recursively use %0 as superclass of %1">;
def warn_previous_alias_decl : Warning<"previously declared alias is ignored">;
def err_conflicting_aliasing_type : Error<"conflicting types for alias %0">;
def warn_undef_interface : Warning<"cannot find interface declaration for %0">;
def warn_duplicate_protocol_def : Warning<"duplicate protocol definition of %0 is ignored">;
def err_protocol_has_circular_dependency : Error<
  "protocol has circular dependency">;
def err_undeclared_protocol : Error<"cannot find protocol declaration for %0">;
def warn_undef_protocolref : Warning<"cannot find protocol definition for %0">;
def warn_readonly_property : Warning<
  "attribute 'readonly' of property %0 restricts attribute "
  "'readwrite' of property inherited from %1">;

def warn_property_attribute : Warning<
  "property %0 '%1' attribute does not match the property inherited from %2">;
def warn_property_types_are_incompatible : Warning<
  "property type %0 is incompatible with type %1 inherited from %2">;
def err_undef_interface : Error<"cannot find interface declaration for %0">;
def warn_dup_category_def : Warning<
  "duplicate definition of category %1 on interface %0">;
def err_conflicting_super_class : Error<"conflicting super class name %0">;
def err_dup_implementation_class : Error<"reimplementation of class %0">;
def err_dup_implementation_category : Error<
  "reimplementation of category %1 for class %0">;
def err_conflicting_ivar_type : Error<
  "instance variable %0 has conflicting type: %1 vs %2">;
def err_conflicting_ivar_bitwidth : Error<
  "instance variable %0 has conflicting bitfield width">;
def err_conflicting_ivar_name : Error<
  "conflicting instance variable names: %0 vs %1">;
def err_inconsistant_ivar_count : Error<
  "inconsistent number of instance variables specified">;
def warn_incomplete_impl : Warning<"incomplete implementation">;
def warn_undef_method_impl : Warning<"method definition for %0 not found">;

def warn_conflicting_ret_types : Warning<
  "conflicting return type in implementation of %0: %1 vs %2">;

def warn_conflicting_param_types : Warning<
  "conflicting parameter types in implementation of %0: %1 vs %2">;

def warn_implements_nscopying : Warning<
"default assign attribute on property %0 which implements "
"NSCopying protocol is not appropriate with -fobjc-gc[-only]">;

def warn_multiple_method_decl : Warning<"multiple methods named %0 found">;
def warn_accessor_property_type_mismatch : Warning<
  "type of property %0 does not match type of accessor %1">;
def note_declared_at : Note<"declared at">;
def err_setter_type_void : Error<"type of setter must be void">;
def err_duplicate_method_decl : Error<"duplicate declaration of method %0">;
def err_objc_var_decl_inclass : 
    Error<"cannot declare variable inside @interface or @protocol">;
def error_missing_method_context : Error<
  "missing context for method declaration">;
def err_objc_property_attr_mutually_exclusive : Error<
  "property attributes '%0' and '%1' are mutually exclusive">;
def err_objc_property_requires_object : Error<
  "property with '%0' attribute must be of object type">;
def warn_objc_property_no_assignment_attribute : Warning<
  "no 'assign', 'retain', or 'copy' attribute is specified - "
  "'assign' is assumed">;
def warn_objc_property_default_assign_on_object : Warning<
  "default property attribute 'assign' not appropriate for non-gc object">;
def warn_property_attr_mismatch : Warning<
  "property attribute in continuation class does not match the primary class">;
def warn_objc_property_copy_missing_on_block : Warning<
    "'copy' attribute must be specified for the block property "
    "when -fobjc-gc-only is specified">;
def err_use_continuation_class : Error<
  "attribute of property in continuation class of %0 can only  be 'readwrite'">;
def err_continuation_class : Error<"continuation class has no primary class">;
def err_property_type : Error<"property cannot have array or function type %0">;
def error_missing_property_context : Error<
  "missing context for property implementation declaration">;
def error_bad_property_decl : Error<
  "property implementation must have its declaration in interface %0">;
def error_synthesize_category_decl : Error<
  "@synthesize not allowed in a category's implementation">;
def error_missing_property_interface : Error<
  "property implementation in a category with no category declaration">;
def error_bad_category_property_decl : Error<
  "property implementation must have its declaration in the category %0">;
def error_bad_property_context : Error<
  "property implementation must be in a class or category implementation">;
def error_missing_property_ivar_decl : Error<
  "synthesized property %0 must either be named the same as a compatible"
  " ivar or must explicitly name an ivar">;

def error_synthesized_ivar_yet_not_supported : Error<
  "instance variable synthesis not yet supported"
  " (need to declare %0 explicitly)">;

def error_property_ivar_type : Error<
  "type of property %0 does not match type of ivar %1">;
def error_ivar_in_superclass_use : Error<
  "property %0 attempting to use ivar %1 declared in super class %2">;
def error_weak_property : Error<
  "existing ivar %1 for __weak property %0 must be __weak">;
def error_strong_property : Error<
  "existing ivar %1 for a __strong property %0 must be garbage collectable">;
def error_dynamic_property_ivar_decl : Error<
  "dynamic property can not have ivar specification">;
def error_duplicate_ivar_use : Error<
  "synthesized properties %0 and %1 both claim ivar %2">;
def error_property_implemented : Error<"property %0 is already implemented">;
def warn_objc_property_attr_mutually_exclusive : Warning<
  "property attributes '%0' and '%1' are mutually exclusive">,
  InGroup<ReadOnlySetterAttrs>, DefaultIgnore;
def warn_undeclared_selector : Warning<
  "undeclared selector %0">, InGroup<UndeclaredSelector>, DefaultIgnore;

// C++ declarations
def err_static_assert_expression_is_not_constant : Error<
  "static_assert expression is not an integral constant expression">;
def err_static_assert_failed : Error<"static_assert failed \"%0\"">;

def err_friend_decl_defines_class : Error<
  "cannot define a type in a friend declaration">;
def err_unexpected_friend : Error<
  "friends can only be classes or functions">;
def err_friend_is_member : Error<
  "friends cannot be members of the declaring class">;
def ext_friend_inner_class : Extension<
  "C++ 98 does not allow inner classes as friends">;
def err_unelaborated_friend_type : Error<
  "must specify '%select{class|union}0' in a friend "
  "%select{class|union}0 declaration">;
def err_qualified_friend_not_found : Error<
  "no function named %0 with type %1 was found in the specified scope">;
def err_introducing_special_friend : Error<
  "must use a qualified name when declaring a %select{constructor|"
  "destructor|conversion operator}0 as a friend">;

def err_abstract_type_in_decl : Error<
  "%select{return|parameter|variable|field}0 type %1 is an abstract class">;
def err_allocation_of_abstract_type : Error<
  "allocation of an object of abstract type %0">;
  
def err_type_defined_in_type_specifier : Error<
  "%0 can not be defined in a type specifier">;
def err_type_defined_in_result_type : Error<
  "%0 can not be defined in the result type of a function">;
def err_type_defined_in_param_type : Error<
  "%0 can not be defined in a parameter type">;

def note_pure_virtual_function : Note<
  "pure virtual function %0">;

def err_deleted_non_function : Error<
  "only functions can have deleted definitions">;
def err_deleted_decl_not_first : Error<
  "deleted definition must be first declaration">;

// C++ exception specifications
def err_exception_spec_in_typedef : Error<
  "exception specifications are not allowed in typedefs">;
def err_distant_exception_spec : Error<
  "exception specifications are not allowed beyond a single level "
  "of indirection">;
def err_incomplete_in_exception_spec : Error<
  "%select{|pointer to |reference to }1incomplete type %0 is not allowed "
  "in exception specification">;
def err_mismatched_exception_spec : Error<
  "exception specification in declaration does not match previous declaration">;
def err_override_exception_spec : Error<
  "exception specification of overriding function is more lax than "
  "base version">;

// C++ access checking
def err_class_redeclared_with_different_access : Error<
  "%0 redeclared with '%1' access">;
def note_previous_access_declaration : Note<
  "previously declared '%1' here">;
  
// C++ name lookup
def err_incomplete_nested_name_spec : Error<
  "incomplete type %0 named in nested name specifier">;
def err_nested_name_member_ref_lookup_ambiguous : Error<
  "lookup of %0 in member access expression is ambiguous">;
def note_ambig_member_ref_object_type : Note<
  "lookup in the object type %0 refers here">;
def note_ambig_member_ref_scope : Note<
  "lookup from the current scope refers here">;

// C++ class members
def err_storageclass_invalid_for_member : Error<
  "storage class specified for a member declaration">;
def err_mutable_function : Error<"'mutable' cannot be applied to functions">;
def err_mutable_reference : Error<"'mutable' cannot be applied to references">;
def err_mutable_const : Error<"'mutable' and 'const' cannot be mixed">;
def err_mutable_nonmember : Error<
  "'mutable' can only be applied to member variables">;
def err_virtual_non_function : Error<
  "'virtual' can only appear on non-static member functions">;
def err_explicit_non_function : Error<
  "'explicit' can only appear on non-static member functions">;
def err_virtual_out_of_class : Error<
  "'virtual' can only be specified inside the class definition">;
def err_static_not_bitfield : Error<"static member %0 cannot be a bit-field">;
def err_static_out_of_line : Error<
  "'static' can only be specified inside the class definition">;
def err_typedef_not_bitfield : Error<"typedef member %0 cannot be a bit-field">;
def err_not_integral_type_bitfield : Error<
  "bit-field %0 has non-integral type %1">;
def err_not_integral_type_anon_bitfield : Error<
  "anonymous bit-field has non-integral type %0">;
def err_member_initialization : Error<
  "%0 can only be initialized if it is a static const integral data member">;
def err_member_function_initialization : Error<
  "initializer on function does not look like a pure-specifier">;
def err_non_virtual_pure : Error<
  "%0 is not virtual and cannot be declared pure">;
def err_implicit_object_parameter_init : Error<
  "cannot initialize object parameter of type %0 with an expression "
  "of type %1">;

def err_missing_default_constructor : Error<
  "default constructor for %1 is missing in initialization of "
  "%select{base class|member}0">;
def err_illegal_union_member : Error<
  "union member %0 has a non-trivial %select{constructor|"
  "copy constructor|copy assignment operator|destructor}1">;
def note_nontrivial_has_virtual : Note<
  "because type %0 has a virtual %select{member function|base class}1">;
def note_nontrivial_has_nontrivial : Note<
  "because type %0 has a %select{member|base class}1 with a non-trivial "
  "%select{constructor|copy constructor|copy assignment operator|destructor}2">;
def note_nontrivial_user_defined : Note<
  "because type %0 has a user-declared %select{constructor|copy constructor|"
  "copy assignment operator|destructor}1">;

def err_different_return_type_for_overriding_virtual_function : Error<
  "virtual function %0 has a different return type (%1) than the "
  "function it overrides (which has return type %2)">;
def note_overridden_virtual_function : Note<
  "overridden virtual function is here">;

def err_covariant_return_inaccessible_base : Error<
  "return type of virtual function %2 is not covariant with the return type "
  "of the function it overrides "
  "(conversion from %0 to inaccessible base class %1)">, NoSFINAE;
def err_covariant_return_ambiguous_derived_to_base_conv : Error<
  "return type of virtual function %3 is not covariant with the return type of "
  "the function it overrides (ambiguous conversion from derived class "
  "%0 to base class %1:%2)">;
def err_covariant_return_not_derived : Error<
  "return type of virtual function %0 is not covariant with the return type of "
  "the function it overrides (%1 is not derived from %2)">;
def err_covariant_return_type_different_qualifications : Error<
  "return type of virtual function %0 is not covariant with the return type of "
  "the function it overrides (%1 has different qualifiers than %2)">;
def err_covariant_return_type_class_type_more_qualified : Error<
  "return type of virtual function %0 is not covariant with the return type of "
  "the function it overrides (class type %1 is more qualified than class "
  "type %2">;
// C++ constructors
def err_constructor_cannot_be : Error<"constructor cannot be declared '%0'">;
def err_invalid_qualified_constructor : Error<
  "'%0' qualifier is not allowed on a constructor">;
def err_constructor_return_type : Error<
  "constructor cannot have a return type">;
def err_constructor_redeclared : Error<"constructor cannot be redeclared">;
def err_constructor_byvalue_arg : Error<
  "copy constructor must pass its first argument by reference">;

// C++ destructors
def err_destructor_not_member : Error<
  "destructor must be a non-static member function">;
def err_destructor_cannot_be : Error<"destructor cannot be declared '%0'">;
def err_invalid_qualified_destructor : Error<
  "'%0' qualifier is not allowed on a destructor">;
def err_destructor_return_type : Error<"destructor cannot have a return type">;
def err_destructor_redeclared : Error<"destructor cannot be redeclared">;
def err_destructor_with_params : Error<"destructor cannot have any parameters">;
def err_destructor_variadic : Error<"destructor cannot be variadic">;
def err_destructor_typedef_name : Error<
  "destructor cannot be declared using a typedef %0 of the class name">;
def err_destructor_name : Error<
  "expected the class name after '~' to name the enclosing class">;

// C++ initialization
def err_lvalue_to_rvalue_ref : Error<"rvalue reference cannot bind to lvalue">;
// FIXME: passing in an English string as %1!
def err_not_reference_to_const_init : Error<
  "non-const lvalue reference to type %0 cannot be initialized "
  "with a %1 of type %2">;
// FIXME: passing in an English string as %1!
def err_reference_init_drops_quals : Error<
  "initialization of reference to type %0 with a %1 of type %2 drops "
  "qualifiers">;
def err_reference_var_requires_init : Error<
  "declaration of reference variable %0 requires an initializer">;
def err_const_var_requires_init : Error<
  "declaration of const variable '%0' requires an initializer">;
def err_init_non_aggr_init_list : Error<
  "initialization of non-aggregate type %0 with an initializer list">;
def err_init_reference_member_uninitialized : Error<
  "reference member of type %0 uninitialized">;
def note_uninit_reference_member : Note<
  "uninitialized reference member is here">;

// C++0x decltype
def err_cannot_determine_declared_type_of_overloaded_function : Error<
    "can't determine the declared type of an overloaded function">;
    
// C++0x auto
def err_auto_variable_cannot_appear_in_own_initializer : Error<
  "variable %0 declared with 'auto' type cannot appear in its own initializer">;
def err_illegal_decl_array_of_auto : Error<
  "'%0' declared as array of 'auto'">;
def err_auto_not_allowed : Error<
  "'auto' not allowed in %select{function prototype|struct member|union member"
  "|class member|exception declaration|template parameter|block literal}0">;
def err_auto_var_requires_init : Error<
  "declaration of variable %0 with type %1 requires an initializer">;
  
// Objective-C++
def err_objc_decls_may_only_appear_in_global_scope : Error<
  "Objective-C declarations may only appear in global scope">;
def err_nsobject_attribute : Error<
  "__attribute ((NSObject)) is for pointer types only">;

// Attributes
def err_attribute_can_be_applied_only_to_symbol_declaration : Error<
  "%0 attribute can be applied only to symbol declaration">;
def err_attributes_are_not_compatible : Error<
  "%0 and %1 attributes are not compatible">;
def err_attribute_wrong_number_arguments : Error<
  "attribute requires %0 argument(s)">;
def err_attribute_missing_parameter_name : Error<
  "attribute requires unquoted parameter">;
def err_attribute_invalid_vector_type : Error<"invalid vector type %0">;
def err_attribute_argument_not_int : Error<
  "'%0' attribute requires integer constant">;
def err_attribute_argument_n_not_int : Error<
  "'%0' attribute requires parameter %1 to be an integer constant">;
def err_attribute_argument_n_not_string : Error<
  "'%0' attribute requires parameter %1 to be a string">;
def err_attribute_argument_out_of_bounds : Error<
  "'%0' attribute parameter %1 is out of bounds">;
def err_attribute_requires_objc_interface : Error<
  "attribute may only be applied to an Objective-C interface">;
def err_nonnull_pointers_only : Error<
  "nonnull attribute only applies to pointer arguments">;
def err_format_strftime_third_parameter : Error<
  "strftime format attribute requires 3rd parameter to be 0">;
def err_format_attribute_requires_variadic : Error<
  "format attribute requires variadic function">;
def err_format_attribute_not : Error<"format argument not %0">;
def err_format_attribute_result_not : Error<"function does not return %0">;
def err_attribute_invalid_size : Error<
  "vector size not an integral multiple of component size">;
def err_attribute_zero_size : Error<"zero vector size">;
def err_typecheck_vector_not_convertable : Error<
  "can't convert between vector values of different size (%0 and %1)">;
def err_typecheck_ext_vector_not_typedef : Error<
  "ext_vector_type only applies to types, not variables">;
def err_unsupported_vector_size : Error<
  "unsupported type %0 for vector_size attribute, please use on typedef">;
def err_ext_vector_component_exceeds_length : Error<
  "vector component access exceeds type %0">;
def err_ext_vector_component_requires_even : Error<
  "vector component access invalid for odd-sized type %0">;
def err_ext_vector_component_name_illegal : Error<
  "illegal vector component name '%0'">;
def err_attribute_address_space_not_int : Error<
  "address space attribute requires an integer constant">;
def err_attribute_address_space_negative : Error<
  "address space is negative">;
def err_attribute_address_space_too_high : Error<
  "address space is larger than the maximum supported (%0)">;
def err_attribute_address_multiple_qualifiers : Error<
  "multiple address spaces specified for type">;
def err_implicit_pointer_address_space_cast : Error<
  "illegal implicit cast between two pointers with different address spaces">;
def err_as_qualified_auto_decl : Error<
  "automatic variable qualified with an address space">;
def err_attribute_not_string : Error<
  "argument to %0 attribute was not a string literal">;
def err_attribute_section_invalid_for_target : Error<
  "argument to 'section' attribute is not valid for this target: %0">;
def err_attribute_aligned_not_power_of_two : Error<
  "requested alignment is not a power of 2">;
def warn_redeclaration_without_attribute_prev_attribute_ignored : Warning<
  "'%0' redeclared without %1 attribute: previous %1 ignored">;
def warn_attribute_ignored : Warning<"%0 attribute ignored">;
def warn_attribute_precede_definition : Warning<
  "attribute declaration must precede definition">;
def warn_attribute_weak_on_field : Warning<
  "__weak attribute cannot be specified on a field declaration">;
def warn_attribute_weak_on_local : Warning<
  "__weak attribute cannot be specified on an automatic variable">;
def warn_weak_identifier_undeclared : Warning<
  "weak identifier %0 never declared">;
def err_attribute_weak_static : Error<
  "weak declaration of '%0' must be public">;
def warn_attribute_weak_import_invalid_on_definition : Warning<
  "'weak_import' attribute cannot be specified on a definition">;
def warn_attribute_wrong_decl_type : Warning<
  "%0 attribute only applies to %select{function|union|"
  "variable and function|function or method|parameter|parameter or Objective-C method |"
  "function, method or block}1 types">;
def warn_gnu_inline_attribute_requires_inline : Warning<
  "'gnu_inline' attribute requires function to be marked 'inline',"
  " attribute ignored">;

def warn_attribute_ignored_for_field_of_type : Warning<
  "%0 attribute ignored for field of type %1">;
def warn_transparent_union_attribute_field_size_align : Warning<
  "%select{alignment|size}0 of field %1 (%2 bits) does not match the "
  "%select{alignment|size}0 of the first field in transparent union; "
  "transparent_union attribute ignored">;
def note_transparent_union_first_field_size_align : Note<
  "%select{alignment|size}0 of first field is %1 bits">;
def warn_transparent_union_attribute_not_definition : Warning<
  "transparent_union attribute can only be applied to a union definition; "
  "attribute ignored">;
def warn_transparent_union_attribute_floating : Warning<
  "first field of a transparent union cannot have floating point or vector "
  "type; transparent_union attribute ignored">;
def warn_transparent_union_attribute_zero_fields : Warning<
  "transparent union definition must contain at least one field; "
  "transparent_union attribute ignored">;
def warn_attribute_type_not_supported : Warning<
  "'%0' attribute argument not supported: %1">;
def warn_attribute_unknown_visibility : Warning<"unknown visibility '%1'">;
def err_unknown_machine_mode : Error<"unknown machine mode %0">;
def err_unsupported_machine_mode : Error<"unsupported machine mode %0">;
def err_mode_not_primitive : Error<
  "mode attribute only supported for integer and floating-point types">;
def err_mode_wrong_type : Error<
  "type of machine mode does not match type of base type">;
def err_attr_wrong_decl : Error<
  "'%0' attribute invalid on this declaration, requires typedef or value">;
def warn_attribute_nonnull_no_pointers : Warning<
  "'nonnull' attribute applied to function with no pointer arguments">;
def warn_attribute_malloc_pointer_only : Warning<
  "'malloc' attribute only applies to functions returning a pointer type">;
def warn_transparent_union_nonpointer : Warning<
  "'transparent_union' attribute support incomplete; only supported for "
  "pointer unions">;

def warn_attribute_sentinel_named_arguments : Warning<
  "'sentinel' attribute requires named arguments">;
def warn_attribute_sentinel_not_variadic : Warning<
  "'sentinel' attribute only supported for variadic %select{functions|blocks}0">;
def err_attribute_sentinel_less_than_zero : Error<
  "'sentinel' parameter 1 less than zero">;
def err_attribute_sentinel_not_zero_or_one : Error<
  "'sentinel' parameter 2 not 0 or 1">;
def err_attribute_cleanup_arg_not_found : Error<
  "'cleanup' argument %0 not found">;
def err_attribute_cleanup_arg_not_function : Error<
  "'cleanup' argument %0 is not a function">;
def err_attribute_cleanup_func_must_take_one_arg : Error<
  "'cleanup' function %0 must take 1 parameter">;
def err_attribute_cleanup_func_arg_incompatible_type : Error<
  "'cleanup' function %0 parameter has type %1 which is incompatible with "
  "type %2">;
def err_attribute_regparm_wrong_platform : Error<
  "'regparm' is not valid on this platform">;
def err_attribute_regparm_invalid_number : Error<
  "'regparm' parameter must be between 0 and %0 inclusive">;


// Clang-Specific Attributes
def err_attribute_iboutlet : Error<
  "'iboutlet' attribute can only be applied to instance variables or "
  "properties">;
def err_attribute_overloadable_not_function : Error<
  "'overloadable' attribute can only be applied to a function">;
def err_attribute_overloadable_missing : Error<
  "%select{overloaded function|redeclaration of}0 %1 must have the "
  "'overloadable' attribute">;
def note_attribute_overloadable_prev_overload : Note<
  "previous overload of function is here">;
def err_attribute_overloadable_no_prototype : Error<
  "'overloadable' function %0 must have a prototype">;
def warn_ns_attribute_wrong_return_type : Warning<
    "%0 attribute only applies to functions or methods that "
    "return a pointer or Objective-C object">;

// Function Parameter Semantic Analysis.
def err_param_with_void_type : Error<"argument may not have 'void' type">;
def err_void_only_param : Error<
  "'void' must be the first and only parameter if specified">;
def err_void_param_qualified : Error<
  "'void' as parameter must not have type qualifiers">;
def err_ident_list_in_fn_declaration : Error<
  "a parameter list without types is only allowed in a function definition">;
def ext_param_not_declared : Extension<
  "parameter %0 was not declared, defaulting to type 'int'">;
def err_param_typedef_of_void : Error<
  "empty parameter list defined with a typedef of 'void' not allowed in C++">;
def err_param_default_argument : Error<
  "C does not support default arguments">;
def err_param_default_argument_redefinition : Error<
  "redefinition of default argument">;
def err_param_default_argument_missing : Error<
  "missing default argument on parameter">;
def err_param_default_argument_missing_name : Error<
  "missing default argument on parameter %0">;
def err_param_default_argument_references_param : Error<
  "default argument references parameter %0">;
def err_param_default_argument_references_local : Error<
  "default argument references local variable %0 of enclosing function">;
def err_param_default_argument_references_this : Error<
  "default argument references 'this'">;
def err_param_default_argument_nonfunc : Error<
  "default arguments can only be specified for parameters in a function "
  "declaration">;
def err_param_default_argument_template_redecl : Error<
  "default arguments cannot be added to a function template that has already "
  "been declared">;
def err_param_default_argument_member_template_redecl : Error<
  "default arguments cannot be added to an out-of-line definition of a member "
  "of a %select{class template|class template partial specialization|nested "
  "class in a template}0">;
def err_defining_default_ctor : Error<
  "cannot define the implicit default constructor for %0, because %select{base class|member}1 "
  "%2 does not have any default constructor">;
def note_previous_class_decl : Note<
  "%0 declared here">;
def err_uninitialized_member_for_assign : Error<
  "cannot define the implicit default assignment operator for %0, because "
  "non-static %select{reference|const}1 member %2 can't use default "
  "assignment operator">;
def note_first_required_here : Note<
  "synthesized method is first required here">;
def err_unintialized_member : Error<
  "cannot define the implicit default constructor for %0, because "
  "%select{reference|const}1 member %2 cannot be default-initialized">;
def err_null_intialized_reference_member : Error<
  "cannot initialize the member to null in default constructor because "
  "reference member %0 cannot be null-initialized">;
def err_unintialized_member_in_ctor : Error<
  "constructor for %0 must explicitly initialize the "
  "%select{reference|const}1 member %2 ">;

def err_use_of_default_argument_to_function_declared_later : Error<
  "use of default argument to function %0 that is declared later in class %1">;
def note_default_argument_declared_here : Note<
  "default argument declared here">;

def ext_param_promoted_not_compatible_with_prototype : ExtWarn<
  "promoted type %0 of K&R function parameter is not compatible with the "
  "parameter type %1 declared in a previous prototype">;


// C++ Overloading Semantic Analysis.
def err_ovl_diff_return_type : Error<
  "functions that differ only in their return type cannot be overloaded">;
def err_ovl_static_nonstatic_member : Error<
  "static and non-static member functions with the same parameter types "
  "cannot be overloaded">;

def err_ovl_no_viable_function_in_call : Error<
  "no matching function for call to %0">;
def err_ovl_no_viable_member_function_in_call : Error<
  "no matching member function for call to %0">;
def err_ovl_ambiguous_call : Error<
  "call to %0 is ambiguous">;
def err_ovl_deleted_call : Error<
  "call to %select{unavailable|deleted}0 function %1">;
def err_ovl_ambiguous_member_call : Error<
  "call to member function %0 is ambiguous">;
def err_ovl_deleted_member_call : Error<
  "call to %select{unavailable|deleted}0 member function %1">;
def err_ovl_candidate : Note<"candidate function">;
def err_ovl_candidate_deleted : Note<
  "candidate function has been explicitly %select{made unavailable|deleted}0">;
def err_ovl_builtin_candidate : Note<"built-in candidate function %0">;
def err_ovl_no_viable_function_in_init : Error<
  "no matching constructor for initialization of %0">;
def err_ovl_ambiguous_init : Error<"call to constructor of %0 is ambiguous">;
def err_ovl_deleted_init : Error<
  "call to %select{unavailable|deleted}0 constructor of %1">;
def err_ovl_ambiguous_oper : Error<
  "use of overloaded operator '%0' is ambiguous">;
def err_ovl_no_viable_oper : Error<"no viable overloaded '%0'">;
def err_ovl_deleted_oper : Error<
  "overload resolution selected %select{unavailable|deleted}0 operator '%1'">;

def err_ovl_no_viable_object_call : Error<
  "no matching function for call to object of type %0">;
def err_ovl_ambiguous_object_call : Error<
  "call to object of type %0 is ambiguous">;
def err_ovl_deleted_object_call : Error<
  "call to %select{unavailable|deleted}0 function call operator in type %1">;
def err_ovl_surrogate_cand : Note<"conversion candidate of type %0">;
def err_member_call_without_object : Error<
  "call to non-static member function without an object argument">;

// C++ Address of Overloaded Function
def err_addr_ovl_ambiguous : Error<
  "address of overloaded function %0 is ambiguous">;

// C++ Template Declarations
def err_template_param_shadow : Error<
  "declaration of %0 shadows template parameter">;
def note_template_param_here : Note<"template parameter is declared here">;
def note_template_export_unsupported : Note<
  "exported templates are unsupported">;
def err_template_outside_namespace_or_class_scope : Error<
  "templates can only be declared in namespace or class scope">;
def err_template_linkage : Error<"templates must have C++ linkage">;
def err_template_typedef : Error<"a typedef cannot be a template">;
def err_template_unnamed_class : Error<
  "cannot declare a class template with no name">;
def err_template_param_list_different_arity : Error<
  "%select{too few|too many}0 template parameters in template "
  "%select{|template parameter }1redeclaration">;
def note_template_param_list_different_arity : Note<
  "%select{too few|too many}0 template parameters in template template "
  "argument">;
def note_template_prev_declaration : Note<
  "previous template %select{declaration|template parameter}0 is here">;
def err_template_param_different_kind : Error<
  "template parameter has a different kind in template "
  "%select{|template parameter }0redeclaration">;
def note_template_param_different_kind : Note<
  "template parameter has a different kind in template argument">;
def err_template_nontype_parm_different_type : Error<
  "template non-type parameter has a different type %0 in template "
  "%select{|template parameter }1redeclaration">;

def note_template_nontype_parm_different_type : Note<
  "template non-type parameter has a different type %0 in template argument">;
def note_template_nontype_parm_prev_declaration : Note<
  "previous non-type template parameter with type %0 is here">;
def err_template_nontype_parm_bad_type : Error<
  "a non-type template parameter cannot have type %0">;
def err_template_param_default_arg_redefinition : Error<
  "template parameter redefines default argument">;
def note_template_param_prev_default_arg : Note<
  "previous default template argument defined here">;
def err_template_param_default_arg_missing : Error<
  "template parameter missing a default argument">;
  
def err_template_variable : Error<"variable %0 declared as a template">;
def err_template_variable_noparams : Error<
  "extraneous 'template<>' in declaration of variable %0">;

// C++ Template Argument Lists
def err_template_arg_list_different_arity : Error<
  "%select{too few|too many}0 template arguments for "
  "%select{class template|function template|template template parameter"
  "|template}1 %2">;
def note_template_decl_here : Note<"template is declared here">;
def note_member_of_template_here : Note<"member is declared here">;
def err_template_arg_must_be_type : Error<
  "template argument for template type parameter must be a type">;
def err_template_arg_must_be_expr : Error<
  "template argument for non-type template parameter must be an expression">;
def err_template_arg_nontype_ambig : Error<
  "template argument for non-type template parameter is treated as type %0">;
def err_template_arg_must_be_template : Error<
  "template argument for template template parameter must be a class template">;
def err_template_arg_local_type : Error<"template argument uses local type %0">;
def err_template_arg_unnamed_type : Error<
  "template argument uses unnamed type">;
def note_template_unnamed_type_here : Note<
  "unnamed type used in template argument was declared here">;
def err_template_arg_not_class_template : Error<
  "template argument does not refer to a class template or template "
  "template parameter">;
def note_template_arg_refers_here_func : Note<
  "template argument refers to function template %0, here">;
def err_template_arg_template_params_mismatch : Error<
  "template template argument has different template parameters than its "
  "corresponding template template parameter">;
def err_template_arg_not_integral_or_enumeral : Error<
  "non-type template argument of type %0 must have an integral or enumeration"
  " type">;
def err_template_arg_not_ice : Error<
  "non-type template argument of type %0 is not an integral constant "
  "expression">;
def err_template_arg_not_convertible : Error<
  "non-type template argument of type %0 cannot be converted to a value "
  "of type %1">;
def err_template_arg_negative : Error<
  "non-type template argument provides negative value '%0' for unsigned "
  "template parameter of type %1">;
def err_template_arg_too_large : Error<
  "non-type template argument value '%0' is too large for template "
  "parameter of type %1">;
def err_template_arg_no_ref_bind : Error<
  "non-type template parameter of reference type %0 cannot bind to template "
  "argument of type %1">;
def err_template_arg_ref_bind_ignores_quals : Error<
  "reference binding of non-type template parameter of type %0 to template "
  "argument of type %1 ignores qualifiers">;
def err_template_arg_not_object_or_func_form : Error<
  "non-type template argument does not directly refer to an object or "
  "function">;
def err_template_arg_field : Error<
  "non-type template argument refers to non-static data member %0">;
def err_template_arg_method : Error<
  "non-type template argument refers to non-static member function %0">;
def err_template_arg_function_not_extern : Error<
  "non-template argument refers to function %0 with internal linkage">;
def err_template_arg_object_not_extern : Error<
  "non-template argument refers to object %0 that does not have external "
  "linkage">;
def note_template_arg_internal_object : Note<
  "non-template argument refers to %select{function|object}0 here">;
def note_template_arg_refers_here : Note<"non-template argument refers here">;
def err_template_arg_not_object_or_func : Error<
  "non-type template argument does not refer to an object or function">;
def err_template_arg_not_pointer_to_member_form : Error<
  "non-type template argument is not a pointer to member constant">;
def err_template_arg_extra_parens : Error<
  "non-type template argument cannot be surrounded by parentheses">;

// C++ class template specializations and out-of-line definitions
def err_template_spec_needs_header : Error<
  "template specialization requires 'template<>'">;
def err_template_spec_needs_template_parameters : Error<
  "template specialization or definition requires a template parameter list"
  "corresponding to the nested type %0">;
def err_template_param_list_matches_nontemplate : Error<
  "template parameter list matching the non-templated nested type %0 should "
  "be empty ('template<>')">;
def err_template_spec_extra_headers : Error<
  "extraneous template parameter list in template specialization or "
  "out-of-line template definition">;
def err_template_qualified_declarator_no_match : Error<
  "nested name specifier %0 for declaration does not refer into a class, "
  "class template or class template partial specialization">;
def err_template_spec_decl_out_of_scope_global : Error<
  "class template %select{|partial }0specialization of %1 must occur in the "
  "global scope">;
def err_template_spec_decl_out_of_scope : Error<
  "class template %select{|partial }0specialization of %1 not in namespace %2">;
def err_template_spec_decl_function_scope : Error<
  "%select{class template specialization|class template partial specialization|"
  "explicit instantiation}0 of %1 in function scope">;
def err_template_spec_redecl_out_of_scope : Error<
  "%select{class template specialization|class template partial specialization|"
  "explicit instantiation}0 of %1 not in a namespace enclosing %2">;
def err_template_spec_redecl_global_scope : Error<
  "%select{class template specialization|class template partial specialization|"
  "explicit instantiation}0 of %1 must occur at global scope">;

// C++ Class Template Partial Specialization
def err_default_arg_in_partial_spec : Error<
    "default template argument in a class template partial specialization">;
def err_dependent_non_type_arg_in_partial_spec : Error<
    "non-type template argument depends on a template parameter of the "
    "partial specialization">;
def err_dependent_typed_non_type_arg_in_partial_spec : Error<
    "non-type template argument specializes a template parameter with "
    "dependent type %0">;
def err_partial_spec_args_match_primary_template : Error<
    "class template partial specialization does not specialize any template "
    "argument; to %select{declare|define}0 the primary template, remove the "
    "template argument list">; 
def warn_partial_specs_not_deducible : Warning<
    "class template partial specialization contains "
    "%select{a template parameter|template parameters}0 that can not be "
    "deduced; this partial specialization will never be used">;
def note_partial_spec_unused_parameter : Note<
    "non-deducible template parameter %0">;
def unsup_template_partial_spec_ordering : Error<
  "partial ordering of class template partial specializations is not yet "
  "supported">;

// C++ Template Instantiation
def err_template_recursion_depth_exceeded : Error<
  "recursive template instantiation exceeded maximum depth of %0">,
  DefaultFatal, NoSFINAE;
def note_template_recursion_depth : Note<
  "use -ftemplate-depth-N to increase recursive template instantiation depth">;

def err_template_instantiate_undefined : Error<
  "%select{implicit|explicit}0 instantiation of undefined template %1">;
def err_implicit_instantiate_member_undefined : Error<
  "implicit instantiation of undefined member %0">;
def note_template_class_instantiation_here : Note<
  "in instantiation of template class %0 requested here">;
def note_template_member_class_here : Note<
  "in instantiation of member class %0 requested here">;
def note_template_member_function_here : Note<
  "in instantiation of member function %q0 requested here">;
def note_function_template_spec_here : Note<
  "in instantiation of function template specialization %q0 requested here">;
def note_template_static_data_member_def_here : Note<
  "in instantiation of static data member %q0 requested here">;
  
def note_default_arg_instantiation_here : Note<
  "in instantiation of default argument for '%0' required here">;
def note_default_function_arg_instantiation_here : Note<
  "in instantiation of default function argument expression "
  "for '%0' required here">;
def note_explicit_template_arg_substitution_here : Note<
  "while substituting explicitly-specified template arguments into function "
  "template %f, here">;
def note_function_template_deduction_instantiation_here : Note<
  "while substituting deduced template arguments into function template %0, "
  "here">;
def note_partial_spec_deduct_instantiation_here : Note<
  "during template argument deduction for class template partial "
  "specialization %0, here">;
def err_field_instantiates_to_function : Error<
  "data member instantiated with function type %0">;
def err_nested_name_spec_non_tag : Error<
  "type %0 cannot be used prior to '::' because it has no members">;

// C++ Explicit Instantiation
def err_explicit_instantiation_duplicate : Error<
    "duplicate explicit instantiation of %0">;
def note_previous_explicit_instantiation : Note<
    "previous explicit instantiation is here">;
def ext_explicit_instantiation_after_specialization : Extension<
    "explicit instantiation of %0 that occurs after an explicit "
    "specialization will be ignored (C++0x extension)">;
def note_previous_template_specialization : Note<
    "previous template specialization is here">;
def err_explicit_instantiation_enum : Error<
    "explicit instantiation of enumeration type %0">;
def err_explicit_instantiation_nontemplate_type : Error<
    "explicit instantiation of non-templated type %0">;
def note_nontemplate_decl_here : Note<
    "non-templated declaration is here">;
def err_explicit_instantiation_out_of_scope : Error<
  "explicit instantiation of %0 not in a namespace enclosing %1">;

// C++ typename-specifiers
def err_typename_nested_not_found : Error<"no type named %0 in %1">;
def err_typename_nested_not_found_global : Error<
    "no type named %0 in the global namespace">;
def err_typename_nested_not_type : Error<
    "typename specifier refers to non-type member %0">;
def note_typename_refers_here : Note<
    "referenced member %0 is declared here">;

def err_template_kw_refers_to_non_template : Error<
    "%0 following the 'template' keyword does not refer to a template">;
def err_template_kw_refers_to_function_template : Error<
    "%0 following the 'template' keyword refers to a function template">;

// C++0x Variadic Templates
def err_template_param_pack_default_arg : Error<
  "template parameter pack cannot have a default argument">;
def err_template_param_pack_must_be_last_template_parameter : Error<
  "template parameter pack must be the last template parameter">;

def err_unexpected_typedef : Error<
  "unexpected type name %0: expected expression">;
def err_unexpected_namespace : Error<
  "unexpected namespace name %0: expected expression">;
def err_undeclared_var_use : Error<"use of undeclared identifier %0">;
def err_undeclared_use : Error<"use of undeclared '%0'">;
def warn_deprecated : Warning<"%0 is deprecated">,
    InGroup<DiagGroup<"deprecated-declarations">>;
def warn_unavailable : Warning<"%0 is unavailable">,
    InGroup<DiagGroup<"unavailable-declarations">>;
def note_unavailable_here : Note<
  "function has been explicitly marked %select{unavailable|deleted}0 here">;
def warn_not_enough_argument : Warning<
  "not enough variable arguments in %0 declaration to fit a sentinel">;
def warn_missing_sentinel : Warning <
  "missing sentinel in %select{function call|method dispatch|block call}0">;
def note_sentinel_here : Note<
  "%select{function|method|block}0 has been explicitly marked sentinel here">;
def warn_missing_prototype : Warning<
  "no previous prototype for function %0">,
  InGroup<DiagGroup<"missing-prototypes">>, DefaultIgnore;
def err_redefinition : Error<"redefinition of %0">;

def warn_redefinition_of_typedef : Warning<
  "redefinition of typedef %0 is invalid in C">,
  InGroup<DiagGroup<"typedef-redefinition"> >, DefaultError;

def err_static_non_static : Error<
  "static declaration of %0 follows non-static declaration">;
def err_non_static_static : Error<
  "non-static declaration of %0 follows static declaration">;
def err_non_thread_thread : Error<
  "non-thread-local declaration of %0 follows thread-local declaration">;
def err_thread_non_thread : Error<
  "thread-local declaration of %0 follows non-thread-local declaration">;
def err_redefinition_different_type : Error<
  "redefinition of %0 with a different type">;
def err_redefinition_different_kind : Error<
  "redefinition of %0 as different kind of symbol">;
def err_redefinition_different_typedef : Error<
  "typedef redefinition with different types (%0 vs %1)">;
def err_tag_definition_of_typedef : Error<
  "definition of type %0 conflicts with typedef of the same name">;
def err_conflicting_types : Error<"conflicting types for %0">;
def err_nested_redefinition : Error<"nested redefinition of %0">;
def err_use_with_wrong_tag : Error<
  "use of %0 with tag type that does not match previous declaration">;
def warn_struct_class_tag_mismatch : Warning<
    "%select{struct|class}0 %select{|template}1 %2 was previously declared "
    "as a %select{class|struct}0 %select{|template}1">,
    InGroup<MismatchedTags>, DefaultIgnore;
def ext_forward_ref_enum : Extension<
  "ISO C forbids forward references to 'enum' types">;
def err_forward_ref_enum : Error<
  "ISO C++ forbids forward references to 'enum' types">;
def err_redefinition_of_enumerator : Error<"redefinition of enumerator %0">;
def err_duplicate_member : Error<"duplicate member %0">;
def ext_enum_value_not_int : Extension<
  "ISO C restricts enumerator values to range of 'int' (%0 is too large)">;
def warn_enum_too_large : Warning<
  "enumeration values exceed range of largest integer">;
def warn_illegal_constant_array_size : Extension<
  "size of static array must be an integer constant expression">;
def err_vla_decl_in_file_scope : Error<
  "variable length array declaration not allowed at file scope">;
def err_vla_decl_has_static_storage : Error<
  "variable length array declaration can not have 'static' storage duration">;
def err_vla_decl_has_extern_linkage : Error<
  "variable length array declaration can not have 'extern' linkage">;
def err_vm_decl_in_file_scope : Error<
  "variably modified type declaration not allowed at file scope">;
def err_vm_decl_has_extern_linkage : Error<
  "variably modified type declaration can not have 'extern' linkage">;
def err_typecheck_field_variable_size : Error<
  "fields must have a constant size: 'variable length array in structure' "
  "extension will never be supported">;
def err_vm_func_decl : Error<
  "function declaration cannot have variably modified type">;

def err_typecheck_negative_array_size : Error<"array size is negative">;
def warn_typecheck_function_qualifiers : Warning<
  "qualifier on function type %0 has unspecified behavior">;
def err_typecheck_invalid_restrict_not_pointer : Error<
  "restrict requires a pointer or reference (%0 is invalid)">;
def err_typecheck_invalid_restrict_invalid_pointee : Error<
  "pointer to function type %0 may not be 'restrict' qualified">;
def ext_typecheck_zero_array_size : Extension<
  "zero size arrays are an extension">;
def err_at_least_one_initializer_needed_to_size_array : Error<
  "at least one initializer value required to size array">;
def err_array_size_non_int : Error<"size of array has non-integer type %0">;
def err_init_element_not_constant : Error<
  "initializer element is not a compile-time constant">;
def err_block_extern_cant_init : Error<
  "'extern' variable cannot have an initializer">;
def warn_extern_init : Warning<"'extern' variable has an initializer">;
def err_variable_object_no_init : Error<
  "variable-sized object may not be initialized">;
def err_array_init_list_required : Error<
  "initialization with '{...}' expected for array">;
def err_excess_initializers : Error<
  "excess elements in %select{array|vector|scalar|union|struct}0 initializer">;
def warn_excess_initializers : ExtWarn<
  "excess elements in %select{array|vector|scalar|union|struct}0 initializer">;
def err_excess_initializers_in_char_array_initializer : Error<
  "excess elements in char array initializer">;
def warn_excess_initializers_in_char_array_initializer : ExtWarn<
  "excess elements in char array initializer">;
def warn_initializer_string_for_char_array_too_long : ExtWarn<
  "initializer-string for char array is too long">;
def warn_braces_around_scalar_init : Warning<
  "braces around scalar initializer">;
def err_many_braces_around_scalar_init : Error<
  "too many braces around scalar initializer">;
def err_empty_scalar_initializer : Error<"scalar initializer cannot be empty">;
def err_illegal_initializer : Error<
  "illegal initializer (only variables can be initialized)">;
def err_illegal_initializer_type : Error<"illegal initializer type %0">;
def err_implicit_empty_initializer : Error<
  "initializer for aggregate with no elements requires explicit braces">;
def err_bitfield_has_negative_width : Error<
  "bit-field %0 has negative width (%1)">;
def err_anon_bitfield_has_negative_width : Error<
  "anonymous bit-field has negative width (%0)">;
def err_bitfield_has_zero_width : Error<"named bit-field %0 has zero width">;
def err_bitfield_width_exceeds_type_size : Error<
  "size of bit-field %0 exceeds size of its type (%1 bits)">;
def err_anon_bitfield_width_exceeds_type_size : Error<
  "size of anonymous bitfield exceeds size of its type (%0 bits)">;

def err_redefinition_of_label : Error<"redefinition of label '%0'">;
def err_undeclared_label_use : Error<"use of undeclared label '%0'">;

def err_goto_into_protected_scope : Error<"illegal goto into protected scope">;
def err_switch_into_protected_scope : Error<
  "illegal switch case into protected scope">;
def err_indirect_goto_in_protected_scope : Error<
  "illegal indirect goto in protected scope, unknown effect on scopes">;
def err_addr_of_label_in_protected_scope : Error<
  "address taken of label in protected scope, jump to it would have "
  "unknown effect on scope">;
def note_protected_by_vla_typedef : Note<
  "jump bypasses initialization of VLA typedef">;
def note_protected_by_vla : Note<
  "jump bypasses initialization of variable length array">;
def note_protected_by_cleanup : Note<
  "jump bypasses initialization of declaration with __attribute__((cleanup))">;
def note_protected_by_objc_try : Note<
  "jump bypasses initialization of @try block">;
def note_protected_by_objc_catch : Note<
  "jump bypasses initialization of @catch block">;
def note_protected_by_objc_finally : Note<
  "jump bypasses initialization of @finally block">;
def note_protected_by_objc_synchronized : Note<
  "jump bypasses initialization of @synchronized block">;
def note_protected_by_cxx_try : Note<
  "jump bypasses initialization of try block">;
def note_protected_by_cxx_catch : Note<
  "jump bypasses initialization of catch block">;
def note_protected_by___block : Note<
  "jump bypasses setup of __block variable">;

def err_func_returning_array_function : Error<
  "function cannot return array or function type %0">;
def err_field_declared_as_function : Error<"field %0 declared as a function">;
def err_field_incomplete : Error<"field has incomplete type %0">;
def ext_variable_sized_type_in_struct : ExtWarn<
  "field %0 with variable sized type %1 not at the end of a struct or class is"
  " a GNU extension">;

def err_flexible_array_empty_struct : Error<
  "flexible array %0 not allowed in otherwise empty struct">;
def ext_flexible_array_in_struct : Extension<
  "%0 may not be nested in a struct due to flexible array member">;
def ext_flexible_array_in_array : Extension<
  "%0 may not be used as an array element due to flexible array member">;
def err_flexible_array_init_nonempty : Error<
  "non-empty initialization of flexible array member inside subobject">;
def err_flexible_array_init_needs_braces : Error<
  "flexible array requires brace-enclosed initializer">;
def err_illegal_decl_array_of_functions : Error<
  "'%0' declared as array of functions">;
def err_illegal_decl_array_incomplete_type : Error<
  "array has incomplete element type %0">;
def err_illegal_decl_array_of_references : Error<
  "'%0' declared as array of references">;
def err_array_star_outside_prototype : Error<
  "star modifier used outside of function prototype">;
def err_illegal_decl_pointer_to_reference : Error<
  "'%0' declared as a pointer to a reference">;
def err_illegal_decl_mempointer_to_reference : Error<
  "'%0' declared as a member pointer to a reference">;
def err_illegal_decl_mempointer_to_void : Error<
  "'%0' declared as a member pointer to void">;
def err_illegal_decl_mempointer_in_nonclass : Error<
  "'%0' does not point into a class">;
def err_mempointer_in_nonclass_type : Error<
  "member pointer refers into non-class type %0">;
def err_reference_to_void : Error<"cannot form a reference to 'void'">;
def err_qualified_block_pointer_type : Error<
  "qualifier specification on block pointer type not allowed">;
def err_nonfunction_block_type : Error<
  "block pointer to non-function type is invalid">;
def err_return_block_has_expr : Error<"void block should not return a value">;
def err_block_return_missing_expr : Error<
  "non-void block should return a value">;
def err_block_with_return_type_requires_args : Error<
  "block with explicit return type requires argument list">;
def err_func_def_incomplete_result : Error<
  "incomplete result type %0 in function definition">;

// Expressions.
def ext_sizeof_function_type : Extension<
  "invalid application of 'sizeof' to a function type">, InGroup<PointerArith>;
def ext_sizeof_void_type : Extension<
  "invalid application of '%0' to a void type">, InGroup<PointerArith>;
// FIXME: merge with %select
def err_sizeof_incomplete_type : Error<
  "invalid application of 'sizeof' to an incomplete type %0">;
def err_alignof_incomplete_type : Error<
  "invalid application of '__alignof' to an incomplete type %0">;
def err_sizeof_alignof_bitfield : Error<
  "invalid application of '%select{sizeof|__alignof}0' to bitfield">;
def err_offsetof_record_type : Error<
  "offsetof requires struct, union, or class type, %0 invalid">;
def err_offsetof_array_type : Error<"offsetof requires array type, %0 invalid">;
def ext_offsetof_extended_field_designator : Extension<
  "using extended field designator is an extension">;
def warn_offsetof_non_pod_type : ExtWarn<"offset of on non-POD type %0">,
  InGroup<InvalidOffsetof>;

def warn_floatingpoint_eq : Warning<
  "comparing floating point with == or != is unsafe">,
  InGroup<DiagGroup<"float-equal">>, DefaultIgnore;

def warn_shift_negative : Warning<
  "shift count is negative">;
def warn_shift_gt_typewidth : Warning<
  "shift count >= width of type">;

def err_sizeof_nonfragile_interface : Error<
  "invalid application of '%select{alignof|sizeof}1' to interface %0 in "
  "non-fragile ABI">;
def err_atdef_nonfragile_interface : Error<
  "invalid application of @defs in non-fragile ABI">;
def err_subscript_nonfragile_interface : Error<
  "subscript requires size of interface %0, which is not constant in "
  "non-fragile ABI">;

def err_arithmetic_nonfragile_interface : Error<
  "arithmetic on pointer to interface %0, which is not a constant size in "
  "non-fragile ABI">;


def ext_subscript_non_lvalue : Extension<
  "ISO C90 does not allow subscripting non-lvalue array">;
def err_typecheck_subscript_value : Error<
  "subscripted value is not an array, pointer, or vector">;
def err_typecheck_subscript_not_integer : Error<
  "array subscript is not an integer">;
def err_subscript_function_type : Error<
  "subscript of pointer to function type %0">;
def err_subscript_incomplete_type : Error<
  "subscript of pointer to incomplete type %0">;
def err_typecheck_member_reference_struct_union : Error<
  "member reference base type %0 is not a structure or union">;
def err_typecheck_member_reference_ivar : Error<
  "%0 does not have a member named %1">;
def err_typecheck_member_reference_arrow : Error<
  "member reference type %0 is not a pointer">;
def err_typecheck_member_reference_type : Error<
  "cannot refer to type member %0 with '%select{.|->}1'">;
def err_typecheck_member_reference_unknown : Error<
  "cannot refer to member %0 with '%select{.|->}1'">;
def note_member_reference_needs_call : Note<
  "perhaps you meant to call this function with '()'?">;
def warn_subscript_is_char : Warning<"array subscript is of type 'char'">,
  InGroup<CharSubscript>, DefaultIgnore;

def err_typecheck_incomplete_tag : Error<"incomplete definition of type %0">;
def err_typecheck_no_member_deprecated : Error<"no member named %0">;
def err_typecheck_record_no_member : Error<
  "no member named %0 in %select{struct|union|class|enum}1 %q2">;
def err_typecheck_namespace_no_member : Error<
  "no member named %0 in namespace %q1">;
def err_typecheck_global_namespace_no_member : Error<
  "no member named %0 in the global namespace">;

def err_member_redeclared : Error<"class member cannot be redeclared">;
def err_member_def_does_not_match : Error<
  "out-of-line definition does not match any declaration in %0">;
def err_nonstatic_member_out_of_line : Error<
  "non-static data member defined out-of-line">;
def err_qualified_typedef_declarator : Error<
  "typedef declarator cannot be qualified">;
def err_qualified_param_declarator : Error<
  "parameter declarator cannot be qualified">;
def err_out_of_line_declaration : Error<
  "out-of-line declaration of a member must be a definition">;
def note_member_def_close_match : Note<"member declaration nearly matches">;
def err_typecheck_ivar_variable_size : Error<
  "instance variables must have a constant size">;
// FIXME: Improve with %select
def err_typecheck_illegal_increment_decrement : Error<
  "cannot modify value of type %0">;
def err_typecheck_arithmetic_incomplete_type : Error<
  "arithmetic on pointer to incomplete type %0">;
def err_typecheck_pointer_arith_function_type : Error<
  "arithmetic on pointer to function type %0">;
def err_typecheck_pointer_arith_void_type : Error<
  "arithmetic on pointer to void type">;
def err_typecheck_decl_incomplete_type : Error<
  "variable has incomplete type %0">;
def ext_typecheck_decl_incomplete_type : ExtWarn<
  "tentative definition of variable with internal linkage has incomplete non-array type %0">;
def err_tentative_def_incomplete_type : Error<
  "tentative definition has type %0 that is never completed">;
def err_tentative_def_incomplete_type_arr : Error<
  "tentative definition has array of type %0 that is never completed">;
def warn_tentative_incomplete_array : Warning<
  "tentative array definition assumed to have one element">;

def err_realimag_invalid_type : Error<"invalid type %0 to %1 operator">;
def err_typecheck_sclass_fscope : Error<
  "illegal storage class on file-scoped variable">;
def err_unsupported_global_register : Error<
  "global register variables are not supported">;
def err_typecheck_sclass_func : Error<"illegal storage class on function">;
def err_static_block_func : Error<
  "function declared in block scope cannot have 'static' storage class">;
def err_typecheck_address_of : Error<"address of %0 requested">;
def ext_typecheck_addrof_void : Extension<
  "ISO C forbids taking the address of an expression of type 'void'">;
def err_typecheck_invalid_lvalue_addrof : Error<
  "address expression must be an lvalue or a function designator">;
def err_typecheck_unary_expr : Error<
  "invalid argument type %0 to unary expression">;
def err_typecheck_indirection_requires_pointer : Error<
  "indirection requires pointer operand (%0 invalid)">;
def err_indirection_requires_nonfragile_object : Error<
  "indirection cannot be to an interface in non-fragile ABI (%0 invalid)">;
def err_direct_interface_unsupported : Error<
  "indirection to an interface is not supported (%0 invalid)">;
def err_typecheck_invalid_operands : Error<
  "invalid operands to binary expression (%0 and %1)">;
def err_typecheck_sub_ptr_object : Error<
  "subtraction of pointer %0 requires pointee to be a complete object type">;
def err_typecheck_sub_ptr_compatible : Error<
  "%0 and %1 are not pointers to compatible types">;
def ext_typecheck_ordered_comparison_of_pointer_integer : ExtWarn<
  "ordered comparison between pointer and integer (%0 and %1)">;
def ext_typecheck_ordered_comparison_of_pointer_and_zero : Extension<
  "ordered comparison between pointer and zero (%0 and %1) is an extension">;
def ext_typecheck_ordered_comparison_of_function_pointers : ExtWarn<
  "ordered comparison of function pointers (%0 and %1)">;
def ext_typecheck_comparison_of_fptr_to_void : Extension<
  "equality comparison between function pointer and void pointer (%0 and %1)">;
def ext_typecheck_comparison_of_pointer_integer : ExtWarn<
  "comparison between pointer and integer (%0 and %1)">;
def ext_typecheck_comparison_of_distinct_pointers : ExtWarn<
  "comparison of distinct pointer types (%0 and %1)">;
def ext_typecheck_cond_incompatible_operands : ExtWarn<
  "incompatible operand types (%0 and %1)">;
def err_typecheck_comparison_of_distinct_pointers : Error<
  "comparison of distinct pointer types (%0 and %1)">;
def err_typecheck_vector_comparison : Error<
  "comparison of vector types (%0 and %1) not supported yet">;
def err_typecheck_assign_const : Error<"read-only variable is not assignable">;
def err_stmtexpr_file_scope : Error<
  "statement expression not allowed at file scope">;

def err_invalid_this_use : Error<
  "invalid use of 'this' outside of a nonstatic member function">;
def err_invalid_member_use_in_static_method : Error<
  "invalid use of member %0 in static member function">;
def err_invalid_qualified_function_type : Error<
  "type qualifier is not allowed on this function">;
def err_invalid_qualified_typedef_function_type_use : Error<
  "a qualified function type cannot be used to declare a nonmember function "
  "or a static member function">;

def err_invalid_non_static_member_use : Error<
  "invalid use of nonstatic data member %0">;
def err_invalid_incomplete_type_use : Error<
  "invalid use of incomplete type %0">;
def err_builtin_func_cast_more_than_one_arg : Error<
  "function-style cast to a builtin type can only take one argument">;
def err_builtin_direct_init_more_than_one_arg : Error<
  "initializer of a builtin type can only take one argument">;
def err_value_init_for_array_type : Error<
  "array types cannot be value-initialized">;
def warn_printf_nonliteral_noargs : Warning<
  "format string is not a string literal (potentially insecure)">,
  InGroup<FormatSecurity>;
def warn_printf_nonliteral : Warning<
  "format string is not a string literal">,
  InGroup<FormatNonLiteral>, DefaultIgnore;

def err_unexpected_interface : Error<
  "unexpected interface name %0: expected expression">;
def err_property_not_found : Error<
  "property %0 not found on object of type %1">;
def ext_gnu_void_ptr : Extension<
  "use of GNU void* extension">, InGroup<PointerArith>;
def ext_gnu_ptr_func_arith : Extension<
  "arithmetic on pointer to function type %0 is a GNU extension">,
  InGroup<PointerArith>;
def error_readonly_property_assignment : Error<
  "assigning to property with 'readonly' attribute not allowed">;
def ext_integer_increment_complex : Extension<
  "ISO C does not support '++'/'--' on complex integer type %0">;
def ext_integer_complement_complex : Extension<
  "ISO C does not support '~' for complex conjugation of %0">;
def error_nosetter_property_assignment : Error<
  "setter method is needed to assign to object using property" " assignment syntax">;

def ext_freestanding_complex : Extension<
  "complex numbers are an extension in a freestanding C99 implementation">;


// Obj-c expressions
def warn_root_inst_method_not_found : Warning<
  "instance method %0 is being used on 'Class' which is not in the root class">;
def warn_class_method_not_found : Warning<
  "method %objcclass0 not found (return type defaults to 'id')">;
def warn_inst_method_not_found : Warning<
  "method %objcinstance0 not found (return type defaults to 'id')">;
def error_no_super_class_message : Error<
  "no @interface declaration found in class messaging of %0">;
def error_no_super_class : Error<
  "no super class declared in @interface for %0">;
def err_invalid_receiver_to_message : Error<
  "invalid receiver to message expression">;
def warn_bad_receiver_type : Warning<
  "receiver type %0 is not 'id' or interface pointer, consider "
  "casting it to 'id'">;
def err_bad_receiver_type : Error<"bad receiver type %0">;
def error_objc_throw_expects_object : Error<
  "@throw requires an Objective-C object type (%0 invalid)">;
def error_objc_synchronized_expects_object : Error<
  "@synchronized requires an Objective-C object type (%0 invalid)">;
def error_rethrow_used_outside_catch : Error<
  "@throw (rethrow) used outside of a @catch block">;
def err_attribute_multiple_objc_gc : Error<
  "multiple garbage collection attributes specified for type">;
def err_catch_param_not_objc_type : Error<
  "@catch parameter is not a pointer to an interface type">;
def err_illegal_qualifiers_on_catch_parm : Error<
  "illegal qualifiers on @catch parameter">;
def err_illegal_super_cast : Error<
  "cannot cast 'super' (it isn't an expression)">;
def warn_setter_getter_impl_required : Warning<
  "property %0 requires method %1 to be defined - "
  "use @synthesize, @dynamic or provide a method implementation">;
def note_property_impl_required : Note<
  "implementation is here">;


// C++ casts
// These messages adhere to the TryCast pattern: %0 is an int specifying the
// cast type, %1 is the source type, %2 is the destination type.
def err_bad_cxx_cast_generic : Error<
  "%select{const_cast|static_cast|reinterpret_cast|dynamic_cast|C-style cast|"
  "functional-style cast}0 from %1 to %2 is not allowed">;
def err_bad_cxx_cast_rvalue : Error<
  "%select{const_cast|static_cast|reinterpret_cast|dynamic_cast|C-style cast|"
  "functional-style cast}0 from rvalue to reference type %2">;
def err_bad_cxx_cast_const_away : Error<
  "%select{const_cast|static_cast|reinterpret_cast|dynamic_cast|C-style cast|"
  "functional-style cast}0 from %1 to %2 casts away constness">;
def err_bad_const_cast_dest : Error<
  "%select{const_cast||||C-style cast|functional-style cast}0 to %2, "
  "which is not a reference, pointer-to-object, or pointer-to-data-member">;
def ext_cast_fn_obj : Extension<
  "cast between pointer-to-function and pointer-to-object is an extension">;
def err_bad_reinterpret_cast_small_int : Error<
  "cast from pointer to smaller type %2 loses information">;
def err_bad_lvalue_to_rvalue_cast : Error<
  "cannot cast from lvalue of type %1 to rvalue reference type %2; types are "
  "not compatible">;
def err_bad_static_cast_pointer_nonpointer : Error<
  "cannot cast from type %1 to pointer type %2">;
def err_bad_static_cast_member_pointer_nonmp : Error<
  "cannot cast from type %1 to member pointer type %2">;
def err_bad_static_cast_incomplete : Error<"%0 is an incomplete type">;

// These messages don't adhere to the pattern.
// FIXME: Display the path somehow better.
def err_ambiguous_base_to_derived_cast : Error<
  "ambiguous cast from base %0 to derived %1:%2">;
def err_static_downcast_via_virtual : Error<
  "cannot cast %0 to %1 via virtual base %2">;
def err_downcast_from_inaccessible_base : Error<
  "cannot cast %1 to %0 due to inaccessible conversion path">;
def err_bad_dynamic_cast_not_ref_or_ptr : Error<
  "%0 is not a reference or pointer">;
def err_bad_dynamic_cast_not_class : Error<"%0 is not a class">;
def err_bad_dynamic_cast_incomplete : Error<"%0 is an incomplete type">;
def err_bad_dynamic_cast_not_ptr : Error<"%0 is not a pointer">;
def err_bad_dynamic_cast_not_polymorphic : Error<"%0 is not polymorphic">;

// Other C++ expressions
def err_need_header_before_typeid : Error<
  "you need to include <typeinfo> before using the 'typeid' operator">;
def err_static_illegal_in_new : Error<
  "the 'static' modifier for the array size is not legal in new expressions">;
def err_array_new_needs_size : Error<
  "array size must be specified in new expressions">;
def err_bad_new_type : Error<
  "cannot allocate %select{function|reference}1 type %0 with new">;
def err_new_incomplete_type : Error<
  "allocation of incomplete type %0">;
def err_new_array_nonconst : Error<
  "only the first dimension of an allocated array may be non-const">;
def err_array_size_not_integral : Error<
  "array size expression must have integral or enumerated type, not %0">;
def err_new_uninitialized_const : Error<
  "must provide an initializer if the allocated object is 'const'">;
def err_delete_operand : Error<"cannot delete expression of type %0">;
def warn_delete_incomplete : Warning<
  "deleting pointer to incomplete type %0 may cause undefined behaviour">;
def err_decrement_bool : Error<"cannot decrement expression of type bool">;
def warn_increment_bool : Warning<
  "incrementing expression of type bool is deprecated">;
def err_catch_incomplete_ptr : Error<
  "cannot catch pointer to incomplete type %0">;
def err_catch_incomplete_ref : Error<
  "cannot catch reference to incomplete type %0">;
def err_catch_incomplete : Error<"cannot catch incomplete type %0">;
def err_catch_rvalue_ref : Error<"cannot catch exceptions by rvalue reference">;
def err_qualified_catch_declarator : Error<
  "exception declarator cannot be qualified">;
def err_early_catch_all : Error<"catch-all handler must come last">;
def err_bad_memptr_rhs : Error<
  "right hand operand to %0 has non pointer-to-member type %1">;
def err_bad_memptr_lhs : Error<
  "left hand operand to %0 must be a %select{|pointer to }1class "
  "compatible with the right hand operand, but is %2">;
def warn_exception_caught_by_earlier_handler : Warning<
  "exception of type %0 will be caught by earlier handler">;
def note_previous_exception_handler : Note<"for type %0">;

def err_conditional_void_nonvoid : Error<
  "%select{left|right}1 operand to ? is void, but %select{right|left}1 operand "
  "is of type %0">;
def err_conditional_ambiguous : Error<
  "conditional expression is ambiguous; %0 can be converted to %1 "
  "and vice versa">;
def err_conditional_ambiguous_ovl : Error<
  "conditional expression is ambiguous; %0 and %1 can be converted to several "
  "common types">;

def err_throw_incomplete : Error<
  "cannot throw object of incomplete type %0">;
def err_throw_incomplete_ptr : Error<
  "cannot throw pointer to object of incomplete type %0">;
def err_return_in_constructor_handler : Error<
  "return in the catch of a function try block of a constructor is illegal">;

def err_ident_in_pseudo_dtor_not_a_type : Error<
  "identifier %0 in pseudo-destructor expression does not name a type">;
def err_pseudo_dtor_base_not_scalar : Error<
  "object expression of non-scalar type %0 cannot be used in a "
  "pseudo-destructor expression">;
def err_pseudo_dtor_type_mismatch : Error<
  "the type of object expression (%0) does not match the type being destroyed "
  "(%1) in pseudo-destructor expression">;
def err_pseudo_dtor_call_with_args : Error<
  "call to pseudo-destructor cannot have any arguments">;
def err_dtor_expr_without_call : Error<
  "%select{destructor reference|pseudo-destructor expression}0 must be "
  "called immediately with '()'">;

def err_invalid_use_of_function_type : Error<
  "a function type is not allowed here">;
def err_invalid_use_of_array_type : Error<"an array type is not allowed here">;
def err_type_defined_in_condition : Error<
  "types may not be defined in conditions">;
def err_typecheck_bool_condition : Error<
  "value of type %0 is not contextually convertible to 'bool'">;
def err_expected_class_or_namespace : Error<"expected a class or namespace">;
def err_invalid_declarator_scope : Error<
  "definition or redeclaration of %0 not in a namespace enclosing %1">;
def err_invalid_declarator_global_scope : Error<
  "definition or redeclaration of %0 cannot name the global scope">;
def err_invalid_declarator_in_function : Error<
  "definition or redeclaration of %0 not allowed inside a function">;
def err_not_tag_in_scope : Error<
  "%0 does not name a tag member in the specified scope">;

def err_cannot_form_pointer_to_member_of_reference_type : Error<
  "cannot form a pointer-to-member to member %0 of reference type %1">;
  
def warn_value_always_zero : Warning<"%0 is always zero in this context">;
def warn_value_always_false : Warning<"%0 is always false in this context">;

// assignment related diagnostics (also for argument passing, returning, etc).
// FIXME: %2 is an english string here.
def err_typecheck_convert_incompatible : Error<
  "incompatible type %2 %1, expected %0">;
def err_cannot_initialize_decl_noname : Error<
  "cannot initialize a value of type %0 with an %select{rvalue|lvalue}1 "
  "of type %2">;
def err_cannot_initialize_decl : Error<
  "cannot initialize %0 with an %select{rvalue|lvalue}1 of type %2">;
def warn_incompatible_qualified_id : Warning<
  "incompatible type %2 %1, expected %0">;
def ext_typecheck_convert_pointer_int : ExtWarn<
  "incompatible pointer to integer conversion %2 %1, expected %0">;
def ext_typecheck_convert_int_pointer : ExtWarn<
  "incompatible integer to pointer conversion %2 %1, expected %0">;
def ext_typecheck_convert_pointer_void_func : Extension<
  "%2 %1 converts between void* and function pointer, expected %0">;
def ext_typecheck_convert_incompatible_pointer_sign : ExtWarn<
 "pointer types point to integer types with different sign %2 %1, expected %0">,
  InGroup<DiagGroup<"pointer-sign">>;
def ext_typecheck_convert_incompatible_pointer : ExtWarn<
  "incompatible pointer types %2 %1, expected %0">;
def ext_typecheck_convert_discards_qualifiers : ExtWarn<
  "%2 %1 discards qualifiers, expected %0">;
def warn_incompatible_vectors : Warning<
  "incompatible vector types %2 %1, expected %0">,
  InGroup<VectorConversions>, DefaultIgnore;
def err_int_to_block_pointer : Error<
  "invalid conversion %2 integer %1, expected block pointer %0">;
def err_typecheck_comparison_of_distinct_blocks : Error<
  "comparison of distinct block types (%0 and %1)">;
def err_typecheck_convert_incompatible_block_pointer : Error<
  "incompatible block pointer types %2 %1, expected %0">;

def err_typecheck_array_not_modifiable_lvalue : Error<
  "array type %0 is not assignable">;
def err_typecheck_non_object_not_modifiable_lvalue : Error<
  "non-object type %0 is not assignable">;
def err_typecheck_expression_not_modifiable_lvalue : Error<
  "expression is not assignable">;
def err_typecheck_incomplete_type_not_modifiable_lvalue : Error<
  "incomplete type %0 is not assignable">;
def err_typecheck_lvalue_casts_not_supported : Error<
  "assignment to cast is illegal, lvalue casts are not supported">;

def err_typecheck_duplicate_vector_components_not_mlvalue : Error<
  "vector is not assignable (contains duplicate components)">;
def err_block_decl_ref_not_modifiable_lvalue : Error<
  "variable is not assignable (missing __block type specifier)">;
def err_typecheck_call_not_function : Error<
  "called object type %0 is not a function or function pointer">;
def err_call_incomplete_return : Error<
  "return type of called function (%0) is incomplete">;
def err_call_incomplete_argument : Error<
  "argument type %0 is incomplete">;
def err_typecheck_call_too_few_args : Error<
  "too few arguments to %select{function|block|method}0 call">;
def err_typecheck_call_too_many_args : Error<
  "too many arguments to %select{function|block|method}0 call">;
def warn_call_wrong_number_of_arguments : Warning<
  "too %select{few|many}0 arguments in call to %1">;
def err_atomic_builtin_must_be_pointer : Error<
  "first argument to atomic builtin must be a pointer (%0 invalid)">;
def err_atomic_builtin_must_be_pointer_intptr : Error<
  "first argument to atomic builtin must be a pointer to integer or pointer"
  " (%0 invalid)">;
def err_atomic_builtin_pointer_size : Error<
  "first argument to atomic builtin must be a pointer to 1,2,4,8 or 16 byte "
  "type (%0 invalid)">;

  
def err_deleted_function_use : Error<"attempt to use a deleted function">;

def err_cannot_pass_objc_interface_to_vararg : Error<
  "cannot pass object with interface type %0 by-value through variadic "
  "%select{function|block|method}1">;

def warn_cannot_pass_non_pod_arg_to_vararg : Warning<
  "cannot pass object of non-POD type %0 through variadic "
  "%select{function|block|method|constructor}1; call will abort at runtime">;

def err_typecheck_call_invalid_ordered_compare : Error<
  "ordered compare requires two args of floating point type (%0 and %1)">;
def err_typecheck_call_invalid_unary_fp : Error<
  "floating point classification requires argument of floating point type "
  "(passed in %0)">;
def err_typecheck_cond_expect_scalar : Error<
  "used type %0 where arithmetic or pointer type is required">;
def ext_typecheck_cond_one_void : Extension<
  "C99 forbids conditional expressions with only one void side">;
def ext_typecheck_cast_nonscalar : Extension<
  "C99 forbids casting nonscalar type %0 to the same type">;
def ext_typecheck_cast_to_union : Extension<"C99 forbids casts to union type">;
def err_typecheck_cast_to_union_no_type : Error<
  "cast to union type from type %0 not present in union">;
def err_cast_pointer_from_non_pointer_int : Error<
  "operand of type %0 cannot be cast to a pointer type">;
def err_cast_pointer_to_non_pointer_int : Error<
  "pointer cannot be cast to type %0">;
def err_typecheck_expect_scalar_operand : Error<
  "operand of type %0 where arithmetic or pointer type is required">;
def err_typecheck_cond_incompatible_operands : Error<
  "incompatible operand types (%0 and %1)">;
def err_cast_selector_expr : Error<
  "cannot type cast @selector expression">;
def warn_typecheck_cond_incompatible_pointers : ExtWarn<
  "pointer type mismatch (%0 and %1)">;
def warn_typecheck_cond_pointer_integer_mismatch : ExtWarn<
  "pointer/integer type mismatch in conditional expression (%0 and %1)">;
def err_typecheck_choose_expr_requires_constant : Error<
  "'__builtin_choose_expr' requires a constant expression">;
def ext_typecheck_expression_not_constant_but_accepted : Extension<
  "expression is not a constant, but is accepted as one by GNU extensions">;
def warn_unused_expr : Warning<"expression result unused">,
  InGroup<UnusedValue>;
def warn_unused_property_expr : Warning<
  "property access result unused - getters should not have side effects">,
  InGroup<UnusedValue>;

def err_incomplete_type_used_in_type_trait_expr : Error<
  "incomplete type %0 used in type trait expression">;
  
// inline asm.
def err_asm_wide_character : Error<"wide string is invalid in 'asm'">;
def err_asm_invalid_lvalue_in_output : Error<"invalid lvalue in asm output">;
def err_asm_invalid_output_constraint : Error<
  "invalid output constraint '%0' in asm">;
def err_asm_invalid_lvalue_in_input : Error<
  "invalid lvalue in asm input for constraint '%0'">;
def err_asm_invalid_input_constraint : Error<
  "invalid input constraint '%0' in asm">;
def err_asm_invalid_type_in_input : Error<
  "invalid type %0 in asm input for constraint '%1'">;
def err_asm_tying_incompatible_types : Error<
  "unsupported inline asm: input with type %0 matching output with type %1">;
def err_asm_unknown_register_name : Error<"unknown register name '%0' in asm">;
def err_invalid_asm_cast_lvalue : Error<
  "invalid use of a cast in a inline asm context requiring an l-value: "
  "remove the cast or build with -fheinous-gnu-extensions">;

def warn_invalid_asm_cast_lvalue : Warning<
  "invalid use of a cast in a inline asm context requiring an l-value: "
  "accepted due to -fheinous-gnu-extensions, but clang may remove support "
  "for this in the future">;



def err_invalid_conversion_between_vectors : Error<
  "invalid conversion between vector type %0 and %1 of different size">;
def err_invalid_conversion_between_vector_and_integer : Error<
  "invalid conversion between vector type %0 and integer type %1 "
  "of different size">;

def err_invalid_conversion_between_vector_and_scalar : Error<
  "invalid conversion between vector type %0 and scalar type %1">;
def err_overload_expr_requires_non_zero_constant : Error<
  "overload requires a non-zero constant expression as first argument">;
def err_overload_incorrect_fntype : Error<
  "argument is not a function, or has wrong number of parameters">;

// FIXME: PASSING TYPES AS STRING.
def err_overload_no_match : Error<
  "no matching overload found for arguments of type '%0'">;
def err_overload_multiple_match : Error<
  "more than one matching function found in __builtin_overload">;

// C++ member initializers.
def err_only_constructors_take_base_inits : Error<
  "only constructors take base initializers">;

def error_multiple_mem_initialization : Error <
  "multiple initializations given for non-static member '%0'">;

def error_multiple_base_initialization : Error <
  "multiple initializations given for base %0">;

def err_mem_init_not_member_or_class : Error<
  "member initializer %0 does not name a non-static data member or base "
  "class">;
def err_mem_initializer_mismatch : Error<
  "Too many arguments for member initializer %0">;

def warn_field_initialized : Warning<
  "member '%0' will be initialized after">,
  InGroup<Reorder>, DefaultIgnore;
def warn_base_initialized : Warning<
  "base class %0 will be initialized after">,
  InGroup<Reorder>, DefaultIgnore;
def note_fieldorbase_initialized_here : Note<
  "%select{field|base}0 %1">;

def err_base_init_does_not_name_class : Error<
  "constructor initializer %0 does not name a class">;
def err_base_init_direct_and_virtual : Error<
  "base class initializer %0 names both a direct base class and an "
  "inherited virtual base class">;
def err_not_direct_base_or_virtual : Error<
  "type %0 is not a direct or virtual base of '%1'">;

def err_in_class_initializer_non_integral_type : Error<
  "in-class initializer has non-integral, non-enumeration type %0">;
def err_in_class_initializer_non_constant : Error<
  "in-class initializer is not an integral constant expression">;

// C++ anonymous unions and GNU anonymous structs/unions
def ext_anonymous_union : Extension<
  "anonymous unions are a GNU extension in C">;
def ext_anonymous_struct : Extension<
  "anonymous structs are a GNU extension">;
def err_anonymous_union_not_static : Error<
  "anonymous unions at namespace or global scope must be declared 'static'">;
def err_anonymous_union_with_storage_spec : Error<
  "anonymous union at class scope must not have a storage specifier">;
def err_anonymous_struct_not_member : Error<
  "anonymous %select{structs|structs and classes}0 must be "
  "%select{struct or union|class}0 members">;
def err_anonymous_union_member_redecl : Error<
  "member of anonymous union redeclares %0">;
def err_anonymous_struct_member_redecl : Error<
  "member of anonymous struct redeclares %0">;
def err_anonymous_record_with_type : Error<
  "types cannot be declared in an anonymous %select{struct|union}0">;
def err_anonymous_record_with_function : Error<
  "functions cannot be declared in an anonymous %select{struct|union}0">;
def err_anonymous_record_with_static : Error<
  "static members cannot be declared in an anonymous %select{struct|union}0">;
def err_anonymous_record_bad_member : Error<
  "anonymous %select{struct|union}0 can only contain non-static data members">;
def err_anonymous_record_nonpublic_member : Error<
  "anonymous %select{struct|union}0 cannot contain a "
  "%select{private|protected}1 data member">;

// C++ local classes
def err_reference_to_local_var_in_enclosing_function : Error<
  "reference to local variable %0 declared in enclosed function %1">;
def note_local_variable_declared_here : Note<
  "%0 declared here">;
def err_static_data_member_not_allowed_in_local_class : Error<
  "static data member %0 not allowed in local class %1">; 
  
// C++ derived classes
def err_base_clause_on_union : Error<"unions cannot have base classes">;
def err_base_must_be_class : Error<"base specifier must name a class">;
def err_union_as_base_class : Error<"unions cannot be base classes">;
def err_incomplete_base_class : Error<"base class has incomplete type">;
def err_duplicate_base_class : Error<
  "base class %0 specified more than once as a direct base class">;
// FIXME: better way to display derivation?  Pass entire thing into diagclient?
def err_ambiguous_derived_to_base_conv : Error<
  "ambiguous conversion from derived class %0 to base class %1:%2">;
def err_ambiguous_memptr_conv : Error<
  "ambiguous conversion from pointer to member of %select{base|derived}0 "
  "class %1 to pointer to member of %select{derived|base}0 class %2:%3">;

def err_memptr_conv_via_virtual : Error<
  "conversion from pointer to member of class %0 to pointer to member "
  "of class %1 via virtual base %2 is not allowed">;

// C++ access control
def err_conv_to_inaccessible_base : Error<
  "conversion from %0 to inaccessible base class %1">, NoSFINAE;
def note_inheritance_specifier_here : Note<
  "'%0' inheritance specifier here">;
def note_inheritance_implicitly_private_here : Note<
  "inheritance is implicitly 'private'">;

// C++ member name lookup
def err_ambiguous_member_multiple_subobjects : Error<
  "non-static member %0 found in multiple base-class subobjects of type %1:%2">;
def err_ambiguous_member_multiple_subobject_types : Error<
  "member %0 found in multiple base classes of different types">;
def note_ambiguous_member_found : Note<"member found by ambiguous name lookup">;
def err_ambiguous_reference : Error<"reference to %0 is ambiguous">;
def note_ambiguous_candidate : Note<"candidate found by name lookup is %q0">;

// C++ operator overloading
def err_operator_overload_needs_class_or_enum : Error<
  "overloaded %0 must have at least one parameter of class "
  "or enumeration type">;

def err_operator_overload_variadic : Error<"overloaded %0 cannot be variadic">;
def err_operator_overload_static : Error<
  "overloaded %0 cannot be a static member function">;
def err_operator_overload_default_arg : Error<
  "parameter of overloaded %0 cannot have a default argument">;
def err_operator_overload_must_be : Error<
  "overloaded %0 must be a %select{unary|binary|unary or binary}2 operator "
  "(has %1 parameter%s1)">;

def err_operator_overload_must_be_member : Error<
  "overloaded %0 must be a non-static member function">;
def err_operator_overload_post_incdec_must_be_int : Error<
  "parameter of overloaded post-%select{increment|decrement}1 operator must "
  "have type 'int' (not %0)">;


// C++ conversion functions
def err_conv_function_not_member : Error<
  "conversion function must be a non-static member function">;
def err_conv_function_return_type : Error<
  "conversion function cannot have a return type">;
def err_conv_function_with_params : Error<
  "conversion function cannot have any parameters">;
def err_conv_function_variadic : Error<
  "conversion function cannot be variadic">;
def err_conv_function_to_array : Error<
  "conversion function cannot convert to an array type">;
def err_conv_function_to_function : Error<
  "conversion function cannot convert to a function type">;
def err_conv_function_redeclared : Error<
  "conversion function cannot be redeclared">;
def warn_conv_to_self_not_used : Warning<
  "conversion function converting %0 to itself will never be used">;
def warn_conv_to_base_not_used : Warning<
  "conversion function converting %0 to its base class %1 will never be used">;
def warn_conv_to_void_not_used : Warning<
  "conversion function converting %0 to %1 will never be used">;

def warn_not_compound_assign : Warning<
  "use of unary operator that may be intended as compound assignment (%0=)">;

// C++0x explicit conversion operators
def warn_explicit_conversion_functions : Warning<
  "explicit conversion functions are a C++0x extension">;

def warn_printf_write_back : Warning<
  "use of '%%n' in format string discouraged (potentially insecure)">,
  InGroup<FormatSecurity>;
def warn_printf_insufficient_data_args : Warning<
  "more '%%' conversions than data arguments">, InGroup<Format>;
def warn_printf_too_many_data_args : Warning<
  "more data arguments than '%%' conversions">, InGroup<FormatExtraArgs>;
def warn_printf_invalid_conversion : Warning<
  "invalid conversion '%0'">, InGroup<Format>;
def warn_printf_missing_format_string : Warning<
  "format string missing">, InGroup<Format>;
def warn_null_arg : Warning<
  "null passed to a callee which requires a non-null argument">,
  InGroup<NonNull>;
def warn_printf_empty_format_string : Warning<
  "format string is empty">, InGroup<FormatZeroLength>;
def warn_printf_format_string_is_wide_literal : Warning<
  "format string should not be a wide string">, InGroup<Format>;
def warn_printf_format_string_contains_null_char : Warning<
  "format string contains '\\0' within the string body">, InGroup<Format>;
def warn_printf_asterisk_width_missing_arg : Warning<
  "'*' specified field width is missing a matching 'int' argument">;
def warn_printf_asterisk_precision_missing_arg : Warning<
  "'.*' specified field precision is missing a matching 'int' argument">;
def warn_printf_asterisk_width_wrong_type : Warning<
  "field width should have type 'int', but argument has type %0">,
  InGroup<Format>;
def warn_printf_asterisk_precision_wrong_type : Warning<
  "field precision should have type 'int', but argument has type %0">,
  InGroup<Format>;

// CHECK: returning address/reference of stack memory
def warn_ret_stack_addr : Warning<
  "address of stack memory associated with local variable %0 returned">;
def warn_ret_stack_ref : Warning<
  "reference to stack memory associated with local variable %0 returned">;


// For non-floating point, expressions of the form x == x or x != x
// should result in a warning, since these always evaluate to a constant.
def warn_selfcomparison : Warning<
  "self-comparison always results in a constant value">;
def warn_stringcompare : Warning<
  "result of comparison against %select{a string literal|@encode}0 is "
  "unspecified (use strcmp instead)">;



// Blocks
def err_blocks_disable : Error<"blocks support disabled - compile with -fblocks"
  " or pick a deployment target that supports them">;
def err_expected_block_lbrace : Error<"expected '{' in block literal">;
def err_goto_in_block : Error<
  "goto not allowed in block literal">;
def err_return_in_block_expression : Error<
  "return not allowed in block expression literal">;
def err_block_returns_array : Error<
  "block declared as returning an array">;

def err_ret_local_block : Error<
  "returning block that lives on the local stack">;

// CFString checking
def err_cfstring_literal_not_string_constant : Error<
  "CFString literal is not a string constant">;
def warn_cfstring_literal_contains_nul_character : Warning<
  "CFString literal contains NUL character">;

// Statements.
def err_continue_not_in_loop : Error<
  "'continue' statement not in loop statement">;
def err_break_not_in_loop_or_switch : Error<
  "'break' statement not in loop or switch statement">;
def err_default_not_in_switch : Error<
  "'default' statement not in switch statement">;
def err_case_not_in_switch : Error<"'case' statement not in switch statement">;
def warn_case_value_overflow : Warning<
  "overflow converting case value to switch condition type (%0 to %1)">;
def err_duplicate_case : Error<"duplicate case value '%0'">;
def warn_case_empty_range : Warning<"empty case range specified">;
def err_typecheck_statement_requires_scalar : Error<
  "statement requires expression of scalar type (%0 invalid)">;
def err_typecheck_statement_requires_integer : Error<
  "statement requires expression of integer type (%0 invalid)">;
def err_multiple_default_labels_defined : Error<
  "multiple default labels in one switch">;
def warn_empty_if_body : Warning<
  "if statement has empty body">, InGroup<EmptyBody>;
def err_va_start_used_in_non_variadic_function : Error<
  "'va_start' used in function with fixed args">;
def warn_second_parameter_of_va_start_not_last_named_argument : Warning<
  "second parameter of 'va_start' not last named argument">;
def err_first_argument_to_va_arg_not_of_type_va_list : Error<
  "first argument to 'va_arg' is of type %0 and not 'va_list'">;

def warn_return_missing_expr : Warning<
  "non-void %select{function|method}1 %0 should return a value">,
  InGroup<ReturnType>;
def ext_return_missing_expr : ExtWarn<
  "non-void %select{function|method}1 %0 should return a value">,
  InGroup<ReturnType>;
def ext_return_has_expr : ExtWarn<
  "void %select{function|method}1 %0 should not return a value">,
  InGroup<ReturnType>;
def ext_return_has_void_expr : Extension<
  "void %select{function|method}1 %0 should not return void expression">;
def warn_noreturn_function_has_return_expr : Warning<
  "function %0 declared 'noreturn' should not return">, DefaultError,
  InGroup<DiagGroup<"invalid-noreturn">>;
def warn_falloff_noreturn_function : Warning<
  "function declared 'noreturn' should not return">, DefaultError,
  InGroup<DiagGroup<"invalid-noreturn">>;
def err_noreturn_block_has_return_expr : Error<
  "block declared 'noreturn' should not return">;
def err_block_on_nonlocal : Error<
  "__block attribute not allowed, only allowed on local variables">;
def err_block_on_vm : Error<
  "__block attribute not allowed on declaration with a variably modified type">;

def err_shufflevector_non_vector : Error<
  "first two arguments to __builtin_shufflevector must be vectors">;
def err_shufflevector_incompatible_vector : Error<
  "first two arguments to __builtin_shufflevector must have the same type">;
def err_shufflevector_nonconstant_argument : Error<
  "index for __builtin_shufflevector must be a constant integer">;
def err_shufflevector_argument_too_large : Error<
  "index for __builtin_shufflevector must be less than the total number "
  "of vector elements">;

def err_vector_incorrect_num_initializers : Error<
  "%select{too many|too few}0 elements in vector initialization (expected %1 elements, have %2)">;
def err_altivec_empty_initializer : Error<"expected initializer">;

def err_stack_const_level : Error<
  "level argument for a stack address builtin must be constant">;

def err_prefetch_invalid_argument : Error<
  "argument to __builtin_prefetch must be a constant integer">;
def err_argument_invalid_range : Error<
  "argument should be a value from %0 to %1">;

def err_object_size_invalid_argument : Error<
  "argument to __builtin_object_size must be a constant integer">;

def err_builtin_longjmp_invalid_val : Error<
  "argument to __builtin_longjmp must be a constant 1">;

def ext_mixed_decls_code : Extension<
  "ISO C90 forbids mixing declarations and code">;
def err_non_variable_decl_in_for : Error<
  "declaration of non-local variable in 'for' loop">;
def err_toomany_element_decls : Error<
  "only one element declaration is allowed">;
def err_selector_element_not_lvalue : Error<
  "selector element is not a valid lvalue">;
def err_selector_element_type : Error<
  "selector element type %0 is not a valid object">;
def err_collection_expr_type : Error<
  "collection expression type %0 is not a valid object">;

def err_invalid_conversion_between_ext_vectors : Error<
  "invalid conversion between ext-vector type %0 and %1">;

// Type
def ext_invalid_sign_spec : Extension<"'%0' cannot be signed or unsigned">;
def warn_receiver_forward_class : Warning<
    "receiver %0 is a forward class and corresponding @interface may not exist">;
def note_method_sent_forward_class : Note<"method %0 is used for the forward class">;
def ext_missing_declspec : ExtWarn<
  "declaration specifier missing, defaulting to 'int'">;
def ext_missing_type_specifier : ExtWarn<
  "type specifier missing, defaults to 'int'">,
  InGroup<ImplicitInt>;
def err_decimal_unsupported : Error<
  "GNU decimal type extension not supported">;
def err_missing_type_specifier : Error<
  "C++ requires a type specifier for all declarations">;
def err_missing_param_declspec : Error<
  "parameter requires a declaration specifier">;
def err_objc_array_of_interfaces : Error<
  "array of interface %0 is invalid (probably should be an array of pointers)">;
def ext_c99_array_usage : Extension<
  "use of C99-specific array features, accepted as an extension">;
def err_c99_array_usage_cxx : Error<
  "C99-specific array features are not permitted in C++">;
  
def err_invalid_protocol_qualifiers : Error<
  "invalid protocol qualifiers on non-ObjC type">;
def warn_ivar_use_hidden : Warning<
  "local declaration of %0 hides instance variable">;
def error_ivar_use_in_class_method : Error<
  "instance variable %0 accessed in class method">;
def error_private_ivar_access : Error<"instance variable %0 is private">,
    NoSFINAE;
def error_protected_ivar_access : Error<"instance variable %0 is protected">,
    NoSFINAE;
def warn_maynot_respond : Warning<"%0  may not respond to %1">;
def warn_attribute_method_def : Warning<
  "method attribute can only be specified on method declarations">;
def ext_typecheck_base_super : Warning<
  "method parameter type %0 does not match "
  "super class method parameter type %1">, InGroup<SuperSubClassMismatch>, DefaultIgnore;
}
