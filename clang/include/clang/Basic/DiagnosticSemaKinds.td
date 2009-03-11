//==--- DiagnosticSemaKinds.td - libsema diagnostics ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Semantic Analysis
//
//===----------------------------------------------------------------------===//

let Component = "Sema" in {

def err_expr_not_ice : Error<
  "expression is not an integer constant expression">;
def ext_expr_not_ice : Extension<
  "expression is not integer constant expression (but is allowed as an extension)">;
def ext_null_pointer_expr_not_ice : Extension<
  "null pointer expression is not an integer constant expression (but is allowed as an extension)">;

def ext_predef_outside_function : Warning<
  "predefined identifier is only valid inside function">;

def err_array_designator_nonconstant : Error<
  "array designator value must be a constant expression">;
def err_array_designator_negative : Error<
  "array designator value '%0' is negative">;
def err_array_designator_empty_range : Error<
  "array designator range [%0, %1] is empty">;
def err_array_designator_non_array : Error<
  "array designator cannot initialize non-array type %0">;
def err_array_designator_too_large : Error<
  "array designator index (%0) exceeds array bounds (%1)">;
def err_field_designator_non_aggr : Error<
  "field designator cannot initialize a %select{non-struct, non-union|non-class}0 type %1">;
def err_field_designator_unknown : Error<
  "field designator %0 does not refer to any field in type %1">;
def err_field_designator_nonfield : Error<
  "field designator %0 does not refer to a non-static data member">;
def note_field_designator_found : Note<"field designator refers here">;
def err_field_designator_anon_class : Error<
  "field designator %0 refers to a member of an anonymous %select{struct|class|union}1">;
def err_designator_for_scalar_init : Error<
  "designator in initializer for scalar type %0">;
def warn_subobject_initializer_overrides : Warning<
  "subobject initialization overrides initialization of other fields within its enclosing subobject">;
def warn_initializer_overrides : Warning<
  "initializer overrides prior initialization of this subobject">;
def note_previous_initializer : Note<
  "previous initialization %select{|with side effects }0is here%select{| (side effects may not occur at run time)}0">;
def err_designator_into_flexible_array_member : Error<
  "designator into flexible array member subobject">;
def note_flexible_array_member : Note<
  "initialized flexible array member %0 is here">;

def ext_vla : Extension<
  "variable length arrays are a C99 feature, accepted as an extension">;
def ext_anon_param_requires_type_specifier : Extension<
  "type specifier required for unnamed parameter, defaults to int">;
def err_bad_variable_name : Error<
  "'%0' cannot be the name of a variable or data member">;
def err_parameter_name_omitted : Error<"parameter name omitted">;
def warn_decl_in_param_list : Warning<
  "declaration of %0 will not be visible outside of this function">;
def warn_implicit_function_decl : Warning<
  "implicit declaration of function %0">;
def err_ellipsis_first_arg : Error<
  "ISO C requires a named argument before '...'">;
def err_declarator_need_ident : Error<"declarator requires an identifier">;
def err_bad_language : Error<"unknown linkage language">;
def warn_use_out_of_scope_declaration : Warning<
  "use of out-of-scope declaration of %0">;

def ext_implicit_lib_function_decl : ExtWarn<
  "implicitly declaring C library function '%0' with type %1">;
def note_please_include_header : Note<
  "please include the header <%0> or explicitly provide a declaration for '%1'">;
def note_previous_builtin_declaration : Note<"%0 is a builtin with type %1">;
def err_implicit_decl_requires_stdio : Error<
  "implicit declaration of '%0' requires inclusion of the header <stdio.h>">;
def warn_redecl_library_builtin : Warning<
  "incompatible redeclaration of library function %0 will be ignored">;
def err_builtin_definition : Error<"definition of builtin function %0">;

def ext_typedef_without_a_name : ExtWarn<"typedef requires a name">;
def err_statically_allocated_object : Error<
  "Objective-C type cannot be statically allocated">;
def err_object_cannot_be_by_value : Error<
  "Objective-C type cannot be %0 by value">;
def warn_enum_value_overflow : Warning<"overflow in enumeration value">;

def warn_pragma_pack_invalid_alignment : Warning<
  "expected #pragma pack parameter to be '1', '2', '4', '8', or '16'">;
def warn_pragma_pack_show : Warning<"value of #pragma pack(show) == %0">;
def warn_pragma_pack_pop_identifer_and_alignment : Warning<
  "specifying both a name and alignment to 'pop' is undefined">;
def warn_pragma_pack_pop_failed : Warning<"#pragma pack(pop, ...) failed: %0">;

def err_duplicate_class_def : Error<
  "duplicate interface definition for class %0">;
def err_undef_superclass : Error<
  "cannot find interface declaration for %0, superclass of %1">;
def warn_previous_alias_decl : Warning<"previously declared alias is ignored">;
def err_conflicting_aliasing_type : Error<"conflicting types for alias %0">;
def warn_undef_interface : Warning<"cannot find interface declaration for %0">;
def err_duplicate_protocol_def : Error<"duplicate protocol definition of %0">;
def err_undeclared_protocol : Error<"cannot find protocol declaration for %0">;
def warn_undef_protocolref : Warning<"cannot find protocol definition for %0">;
def warn_readonly_property : Warning<
  "attribute 'readonly' of property %0 restricts attribute 'readwrite' of property inherited from %1">;

def warn_property_attribute : Warning<
  "property %0 '%1' attribute does not match the property inherited from %2">;
def warn_property_types_are_incompatible : Warning<
  "property type %0 is incompatible with type %1 inherited from %2">;
def err_undef_interface : Error<"cannot find interface declaration for %0">;
def warn_dup_category_def : Warning<
  "duplicate definition of category %1 on interface %0">;
def err_conflicting_super_class : Error<"conflicting super class name %0">;
def err_dup_implementation_class : Error<"reimplementation of class %0">;
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
def warn_conflicting_types : Warning<"conflicting types for %0">;
def warn_multiple_method_decl : Warning<"multiple methods named %0 found">;
def err_accessor_property_type_mismatch : Error<
  "type of property %0 does not match type of accessor %1">;
def note_declared_at : Note<"declared at">;
def err_setter_type_void : Error<"type of setter must be void">;
def err_duplicate_method_decl : Error<"duplicate declaration of method %0">;
def error_missing_method_context : Error<
  "missing context for method declaration">;
def err_objc_property_attr_mutually_exclusive : Error<
  "property attributes '%0' and '%1' are mutually exclusive">;
def err_objc_property_requires_object : Error<
  "property with '%0' attribute must be of object type">;
def warn_objc_property_no_assignment_attribute : Warning<
  "no 'assign', 'retain', or 'copy' attribute is specified - 'assign' is assumed">;
def warn_objc_property_default_assign_on_object : Warning<
  "default property attribute 'assign' not appropriate for non-gc object">;
def warn_property_attr_mismatch : Warning<
  "property attribute in continuation class does not match the primary class">;
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
  "synthesized property %0 must either be named the same as a compatible ivar or must explicitly name an ivar">;
def error_property_ivar_type : Error<
  "type of property %0 does not match type of ivar %1">;
def error_weak_property : Error<
  "existing ivar %1 for __weak property %0 must be __weak">;
def error_strong_property : Error<
  "existing ivar %1 for a __strong property %0 must be garbage collectable">;
def error_dynamic_property_ivar_decl : Error<
  "dynamic property can not have ivar specification">;
def error_duplicate_ivar_use : Error<
  "synthesized properties %0 and %1 both claim ivar %2">;
def error_property_implemented : Error<
  "property %0 is already implemented">;
def warn_objc_property_attr_mutually_exclusive : Warning<
  "property attributes '%0' and '%1' are mutually exclusive">;

def err_storageclass_invalid_for_member : Error<
  "storage class specified for a member declaration">;
def err_mutable_function : Error<"'mutable' cannot be applied to functions">;
def err_mutable_reference : Error<"'mutable' cannot be applied to references">;
def err_mutable_const : Error<"'mutable' and 'const' cannot be mixed">;
def err_mutable_nonmember : Error<
  "'mutable' can only be applied to member variables">;
def err_virtual_non_function : Error<
  "'virtual' can only appear on non-static member functions">;
def err_not_bitfield_type : Error<"cannot declare %0 to be a bit-field type">;
def err_static_not_bitfield : Error<"static member %0 cannot be a bit-field">;
def err_not_integral_type_bitfield : Error<
  "bit-field %0 with non-integral type">;
def err_member_initialization : Error<
  "%0 can only be initialized if it is a static const integral data member">;
def err_member_function_initialization : Error<
  "initializer on function does not look like a pure-specifier">;
def err_non_virtual_pure : Error<
  "%0 is not virtual and cannot be declared pure">;
def err_implicit_object_parameter_init : Error<
  "cannot initialize object parameter of type %0 with an expression of type %1">;

def err_constructor_cannot_be : Error<"constructor cannot be declared '%0'">;
def err_invalid_qualified_constructor : Error<
  "'%0' qualifier is not allowed on a constructor">;
def err_constructor_return_type : Error<
  "constructor cannot have a return type">;
def err_constructor_redeclared : Error<"constructor cannot be redeclared">;
def err_constructor_byvalue_arg : Error<
  "copy constructor must pass its first argument by reference">;

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

def err_not_reference_to_const_init : Error<
  "non-const reference to type %0 cannot be initialized with a %1 of type %2">;
def err_reference_init_drops_quals : Error<
  "initialization of reference to type %0 with a %1 of type %2 drops qualifiers">;
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

def err_objc_decls_may_only_appear_in_global_scope : Error<
  "Objective-C declarations may only appear in global scope">;
def err_nsobject_attribute : Error<
  "__attribute ((NSObject)) is for pointer types only">;

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
def err_attribute_invalid_size : Error<
  "vector size not an integral multiple of component size">;
def err_attribute_zero_size : Error<"zero vector size">;
def err_typecheck_vector_not_convertable : Error<
  "can't convert between vector values of different size (%0 and %1)">;
def err_typecheck_ext_vector_not_typedef : Error<
  "ext_vector_type only applies to types, not variables">;
def err_ext_vector_component_exceeds_length : Error<
  "vector component access exceeds type %0">;
def err_ext_vector_component_requires_even : Error<
  "vector component access invalid for odd-sized type %0">;
def err_ext_vector_component_name_illegal : Error<
  "illegal vector component name '%0'">;
def err_attribute_address_space_not_int : Error<
  "address space attribute requires an integer constant">;
def err_attribute_address_multiple_qualifiers : Error<
  "multiple address spaces specified for type">;
def err_implicit_pointer_address_space_cast : Error<
  "illegal implicit cast between two pointers with different address spaces">;
def err_as_qualified_auto_decl : Error<
  "automatic variable qualified with an address space">;
def err_attribute_annotate_no_string : Error<
  "argument to annotate attribute was not a string literal">;
def err_attribute_aligned_not_power_of_two : Error<
  "requested alignment is not a power of 2">;
def warn_redeclaration_without_attribute_prev_attribute_ignored : Warning<
  "'%0' redeclared without %1 attribute: previous %1 ignored">;
def warn_attribute_ignored : Warning<"%0 attribute ignored">;
def warn_attribute_weak_on_field : Warning<
  "__weak attribute cannot be specified on a field declaration">;
def warn_attribute_weak_on_local : Warning<
  "__weak attribute cannot be specified on an automatic variable">;
def warn_attribute_wrong_decl_type : Warning<
  "'%0' attribute only applies to %select{function|union|variable and function|function or method}1 types">;
def warn_attribute_ignored_for_field_of_type : Warning<
  "%0 attribute ignored for field of type %1">;
def warn_transparent_union_attribute_field_size : Warning<
  "transparent_union attribute ignored, size of type %0 must match type of first field">;
def warn_transparent_union_attribute_not_difinition : Warning<
  "transparent_union attribute ignored, union type must be defined">;
def warn_transparent_union_attribute_floating : Warning<
  "transparent_union attribute ignored, first field cannot be a floating-point or vector type">;
def warn_transparent_union_attribute_zero_fields : Warning<
  "transparent_union attribute ignored, the union does not contain any fields">;
def warn_transparent_union_attribute_not_c : Warning<
  "transparent_union attribute ignored, attribute is c only">;
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
def warn_transparent_union_nonpointer : Warning<
  "'transparent_union' attribute support incomplete; only supported for pointer unions">;
def warn_attribute_sentinel_not_variadic : Warning<
  "'sentinel' attribute only supported for variadic functions">;
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
  "'cleanup' function %0 parameter has type %1 which is incompatible with type %2">;

def err_attribute_iboutlet : Error<
  "'iboutlet' attribute can only be applied to instance variables or properties">;
def err_attribute_overloadable_not_function : Error<
  "'overloadable' attribute can only be applied to a function">;
def err_attribute_overloadable_missing : Error<
  "%select{overloaded function|redeclaration of}0 %1 must have the 'overloadable' attribute">;
def note_attribute_overloadable_prev_overload : Note<
  "previous overload of function is here">;
def err_attribute_overloadable_no_prototype : Error<
  "'overloadable' function %0 must have a prototype">;

def err_param_with_void_type : Error<"argument may not have 'void' type">;
def err_void_only_param : Error<
  "'void' must be the first and only parameter if specified">;
def err_void_param_qualified : Error<
  "'void' as parameter must not have type qualifiers">;
def err_ident_list_in_fn_declaration : Error<
  "a parameter list without types is only allowed in a function definition">;
def ext_param_not_declared : Extension<
  "parameter %0 was not declared, defaulting to type 'int'">;
def ext_param_typedef_of_void : Extension<
  "empty parameter list defined with a typedef of 'void' not allowed in C++">;
def err_param_default_argument : Error<"C does not support default arguments">;
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
  "default arguments can only be specified for parameters in a function declaration">;

def err_ovl_diff_return_type : Error<
  "functions that differ only in their return type cannot be overloaded">;
def err_ovl_static_nonstatic_member : Error<
  "static and non-static member functions with the same parameter types cannot be overloaded">;
def err_ovl_no_viable_function_in_call : Error<
  "no matching function for call to %0">;
def err_ovl_no_viable_member_function_in_call : Error<
  "no matching member function for call to %0">;
def err_ovl_ambiguous_call : Error<"call to %0 is ambiguous">;
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

}
