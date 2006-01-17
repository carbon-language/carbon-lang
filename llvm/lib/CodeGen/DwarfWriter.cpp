//===-- llvm/CodeGen/DwarfWriter.cpp - Dwarf Framework ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by James M. Laskey and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing dwarf debug info into asm files.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/DwarfWriter.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineDebugInfo.h"
#include "llvm/Support/CommandLine.h"

#include <iostream>

using namespace llvm;

static cl::opt<bool>
DwarfVerbose("dwarf-verbose", cl::Hidden,
                                cl::desc("Add comments to Dwarf directives."));

//===----------------------------------------------------------------------===//
// Dwarf abbreviations used by this emitter.
//

static const unsigned char AbbrevTAG_compile_unit[] = {
  DW_TAG_compile_unit,     DW_CHILDREN_yes,
  DW_AT_stmt_list,         DW_FORM_data4,
  DW_AT_high_pc,           DW_FORM_addr,
  DW_AT_low_pc,            DW_FORM_addr,
  DW_AT_producer,          DW_FORM_string,
  DW_AT_language,          DW_FORM_data1,
  DW_AT_name,              DW_FORM_string,
  DW_AT_comp_dir,          DW_FORM_string,
  0,                       0
};
  
static const unsigned char AbbrevTAG_subprogram[] = {
  DW_TAG_subprogram,       DW_CHILDREN_yes,
  DW_AT_sibling,           DW_FORM_ref4,
  DW_AT_external,          DW_FORM_flag,
  DW_AT_name,              DW_FORM_string,
  DW_AT_decl_file,         DW_FORM_data1,
  DW_AT_decl_line,         DW_FORM_data1,
  DW_AT_prototyped,        DW_FORM_flag,
  DW_AT_type,              DW_FORM_ref4,
  DW_AT_low_pc,            DW_FORM_addr,
  DW_AT_high_pc,           DW_FORM_addr,
  DW_AT_frame_base,        DW_FORM_block1,
  0,                       0
};
  
static const unsigned char AbbrevTAG_formal_parameter[] = {
  DW_TAG_formal_parameter, DW_CHILDREN_no,
  DW_AT_name,              DW_FORM_string,
  DW_AT_decl_file,         DW_FORM_data1,
  DW_AT_decl_line,         DW_FORM_data1,
  DW_AT_type,              DW_FORM_ref4,
  DW_AT_location,          DW_FORM_block1,
  0,                       0
};
  
static const unsigned char AbbrevTAG_base_type[] = {
  DW_TAG_base_type,        DW_CHILDREN_no,
  DW_AT_name,              DW_FORM_string,
  DW_AT_byte_size,         DW_FORM_data1,
  DW_AT_encoding,          DW_FORM_data1,
  0,                       0
};
  
static const unsigned char AbbrevTAG_pointer_type[] = {
  DW_TAG_pointer_type,     DW_CHILDREN_no,
  DW_AT_byte_size,         DW_FORM_data1,
  DW_AT_type,              DW_FORM_ref4,
  0,                       0
};
  
static const unsigned char AbbrevTAG_array_type[] = {
  DW_TAG_array_type,       DW_CHILDREN_yes,
  DW_AT_sibling,           DW_FORM_ref4,
  DW_AT_type,              DW_FORM_ref4,
  0,                       0
};
  
static const unsigned char AbbrevTAG_subrange_type[] = {
  DW_TAG_subrange_type,    DW_CHILDREN_no,
  0,                       0
};
  
static const unsigned char AbbrevTAG_variable[] = {
  DW_TAG_variable,         DW_CHILDREN_no,
  DW_AT_name,              DW_FORM_string,
  DW_AT_type,              DW_FORM_ref4,
  DW_AT_external,          DW_FORM_flag,
  DW_AT_artificial,        DW_FORM_flag,
  DW_AT_declaration,       DW_FORM_flag,
  0,                       0
};

//===----------------------------------------------------------------------===//

/// TagString - Return the string for the specified tag.
///
static const char *TagString(unsigned Tag) {
  switch(Tag) {
    case DW_TAG_array_type:                return "TAG_array_type";
    case DW_TAG_class_type:                return "TAG_class_type";
    case DW_TAG_entry_point:               return "TAG_entry_point";
    case DW_TAG_enumeration_type:          return "TAG_enumeration_type";
    case DW_TAG_formal_parameter:          return "TAG_formal_parameter";
    case DW_TAG_imported_declaration:      return "TAG_imported_declaration";
    case DW_TAG_label:                     return "TAG_label";
    case DW_TAG_lexical_block:             return "TAG_lexical_block";
    case DW_TAG_member:                    return "TAG_member";
    case DW_TAG_pointer_type:              return "TAG_pointer_type";
    case DW_TAG_reference_type:            return "TAG_reference_type";
    case DW_TAG_compile_unit:              return "TAG_compile_unit";
    case DW_TAG_string_type:               return "TAG_string_type";
    case DW_TAG_structure_type:            return "TAG_structure_type";
    case DW_TAG_subroutine_type:           return "TAG_subroutine_type";
    case DW_TAG_typedef:                   return "TAG_typedef";
    case DW_TAG_union_type:                return "TAG_union_type";
    case DW_TAG_unspecified_parameters:    return "TAG_unspecified_parameters";
    case DW_TAG_variant:                   return "TAG_variant";
    case DW_TAG_common_block:              return "TAG_common_block";
    case DW_TAG_common_inclusion:          return "TAG_common_inclusion";
    case DW_TAG_inheritance:               return "TAG_inheritance";
    case DW_TAG_inlined_subroutine:        return "TAG_inlined_subroutine";
    case DW_TAG_module:                    return "TAG_module";
    case DW_TAG_ptr_to_member_type:        return "TAG_ptr_to_member_type";
    case DW_TAG_set_type:                  return "TAG_set_type";
    case DW_TAG_subrange_type:             return "TAG_subrange_type";
    case DW_TAG_with_stmt:                 return "TAG_with_stmt";
    case DW_TAG_access_declaration:        return "TAG_access_declaration";
    case DW_TAG_base_type:                 return "TAG_base_type";
    case DW_TAG_catch_block:               return "TAG_catch_block";
    case DW_TAG_const_type:                return "TAG_const_type";
    case DW_TAG_constant:                  return "TAG_constant";
    case DW_TAG_enumerator:                return "TAG_enumerator";
    case DW_TAG_file_type:                 return "TAG_file_type";
    case DW_TAG_friend:                    return "TAG_friend";
    case DW_TAG_namelist:                  return "TAG_namelist";
    case DW_TAG_namelist_item:             return "TAG_namelist_item";
    case DW_TAG_packed_type:               return "TAG_packed_type";
    case DW_TAG_subprogram:                return "TAG_subprogram";
    case DW_TAG_template_type_parameter:   return "TAG_template_type_parameter";
    case DW_TAG_template_value_parameter: return "TAG_template_value_parameter";
    case DW_TAG_thrown_type:               return "TAG_thrown_type";
    case DW_TAG_try_block:                 return "TAG_try_block";
    case DW_TAG_variant_part:              return "TAG_variant_part";
    case DW_TAG_variable:                  return "TAG_variable";
    case DW_TAG_volatile_type:             return "TAG_volatile_type";
    case DW_TAG_dwarf_procedure:           return "TAG_dwarf_procedure";
    case DW_TAG_restrict_type:             return "TAG_restrict_type";
    case DW_TAG_interface_type:            return "TAG_interface_type";
    case DW_TAG_namespace:                 return "TAG_namespace";
    case DW_TAG_imported_module:           return "TAG_imported_module";
    case DW_TAG_unspecified_type:          return "TAG_unspecified_type";
    case DW_TAG_partial_unit:              return "TAG_partial_unit";
    case DW_TAG_imported_unit:             return "TAG_imported_unit";
    case DW_TAG_condition:                 return "TAG_condition";
    case DW_TAG_shared_type:               return "TAG_shared_type";
    case DW_TAG_lo_user:                   return "TAG_lo_user";
    case DW_TAG_hi_user:                   return "TAG_hi_user";
  }
  assert(0 && "Unknown Dwarf Tag");
  return "";
}

/// ChildrenString - Return the string for the specified children flag.
///
static const char *ChildrenString(unsigned Children) {
  switch(Children) {
    case DW_CHILDREN_no:                   return "CHILDREN_no";
    case DW_CHILDREN_yes:                  return "CHILDREN_yes";
  }
  assert(0 && "Unknown Dwarf ChildrenFlag");
  return "";
}

/// AttributeString - Return the string for the specified attribute.
///
static const char *AttributeString(unsigned Attribute) {
  switch(Attribute) {
    case DW_AT_sibling:                    return "AT_sibling";
    case DW_AT_location:                   return "AT_location";
    case DW_AT_name:                       return "AT_name";
    case DW_AT_ordering:                   return "AT_ordering";
    case DW_AT_byte_size:                  return "AT_byte_size";
    case DW_AT_bit_offset:                 return "AT_bit_offset";
    case DW_AT_bit_size:                   return "AT_bit_size";
    case DW_AT_stmt_list:                  return "AT_stmt_list";
    case DW_AT_low_pc:                     return "AT_low_pc";
    case DW_AT_high_pc:                    return "AT_high_pc";
    case DW_AT_language:                   return "AT_language";
    case DW_AT_discr:                      return "AT_discr";
    case DW_AT_discr_value:                return "AT_discr_value";
    case DW_AT_visibility:                 return "AT_visibility";
    case DW_AT_import:                     return "AT_import";
    case DW_AT_string_length:              return "AT_string_length";
    case DW_AT_common_reference:           return "AT_common_reference";
    case DW_AT_comp_dir:                   return "AT_comp_dir";
    case DW_AT_const_value:                return "AT_const_value";
    case DW_AT_containing_type:            return "AT_containing_type";
    case DW_AT_default_value:              return "AT_default_value";
    case DW_AT_inline:                     return "AT_inline";
    case DW_AT_is_optional:                return "AT_is_optional";
    case DW_AT_lower_bound:                return "AT_lower_bound";
    case DW_AT_producer:                   return "AT_producer";
    case DW_AT_prototyped:                 return "AT_prototyped";
    case DW_AT_return_addr:                return "AT_return_addr";
    case DW_AT_start_scope:                return "AT_start_scope";
    case DW_AT_bit_stride:                 return "AT_bit_stride";
    case DW_AT_upper_bound:                return "AT_upper_bound";
    case DW_AT_abstract_origin:            return "AT_abstract_origin";
    case DW_AT_accessibility:              return "AT_accessibility";
    case DW_AT_address_class:              return "AT_address_class";
    case DW_AT_artificial:                 return "AT_artificial";
    case DW_AT_base_types:                 return "AT_base_types";
    case DW_AT_calling_convention:         return "AT_calling_convention";
    case DW_AT_count:                      return "AT_count";
    case DW_AT_data_member_location:       return "AT_data_member_location";
    case DW_AT_decl_column:                return "AT_decl_column";
    case DW_AT_decl_file:                  return "AT_decl_file";
    case DW_AT_decl_line:                  return "AT_decl_line";
    case DW_AT_declaration:                return "AT_declaration";
    case DW_AT_discr_list:                 return "AT_discr_list";
    case DW_AT_encoding:                   return "AT_encoding";
    case DW_AT_external:                   return "AT_external";
    case DW_AT_frame_base:                 return "AT_frame_base";
    case DW_AT_friend:                     return "AT_friend";
    case DW_AT_identifier_case:            return "AT_identifier_case";
    case DW_AT_macro_info:                 return "AT_macro_info";
    case DW_AT_namelist_item:              return "AT_namelist_item";
    case DW_AT_priority:                   return "AT_priority";
    case DW_AT_segment:                    return "AT_segment";
    case DW_AT_specification:              return "AT_specification";
    case DW_AT_static_link:                return "AT_static_link";
    case DW_AT_type:                       return "AT_type";
    case DW_AT_use_location:               return "AT_use_location";
    case DW_AT_variable_parameter:         return "AT_variable_parameter";
    case DW_AT_virtuality:                 return "AT_virtuality";
    case DW_AT_vtable_elem_location:       return "AT_vtable_elem_location";
    case DW_AT_allocated:                  return "AT_allocated";
    case DW_AT_associated:                 return "AT_associated";
    case DW_AT_data_location:              return "AT_data_location";
    case DW_AT_byte_stride:                return "AT_byte_stride";
    case DW_AT_entry_pc:                   return "AT_entry_pc";
    case DW_AT_use_UTF8:                   return "AT_use_UTF8";
    case DW_AT_extension:                  return "AT_extension";
    case DW_AT_ranges:                     return "AT_ranges";
    case DW_AT_trampoline:                 return "AT_trampoline";
    case DW_AT_call_column:                return "AT_call_column";
    case DW_AT_call_file:                  return "AT_call_file";
    case DW_AT_call_line:                  return "AT_call_line";
    case DW_AT_description:                return "AT_description";
    case DW_AT_binary_scale:               return "AT_binary_scale";
    case DW_AT_decimal_scale:              return "AT_decimal_scale";
    case DW_AT_small:                      return "AT_small";
    case DW_AT_decimal_sign:               return "AT_decimal_sign";
    case DW_AT_digit_count:                return "AT_digit_count";
    case DW_AT_picture_string:             return "AT_picture_string";
    case DW_AT_mutable:                    return "AT_mutable";
    case DW_AT_threads_scaled:             return "AT_threads_scaled";
    case DW_AT_explicit:                   return "AT_explicit";
    case DW_AT_object_pointer:             return "AT_object_pointer";
    case DW_AT_endianity:                  return "AT_endianity";
    case DW_AT_elemental:                  return "AT_elemental";
    case DW_AT_pure:                       return "AT_pure";
    case DW_AT_recursive:                  return "AT_recursive";
    case DW_AT_lo_user:                    return "AT_lo_user";
    case DW_AT_hi_user:                    return "AT_hi_user";
  }
  assert(0 && "Unknown Dwarf Attribute");
  return "";
}

/// FormEncodingString - Return the string for the specified form encoding.
///
static const char *FormEncodingString(unsigned Encoding) {
  switch(Encoding) {
    case DW_FORM_addr:                     return "FORM_addr";
    case DW_FORM_block2:                   return "FORM_block2";
    case DW_FORM_block4:                   return "FORM_block4";
    case DW_FORM_data2:                    return "FORM_data2";
    case DW_FORM_data4:                    return "FORM_data4";
    case DW_FORM_data8:                    return "FORM_data8";
    case DW_FORM_string:                   return "FORM_string";
    case DW_FORM_block:                    return "FORM_block";
    case DW_FORM_block1:                   return "FORM_block1";
    case DW_FORM_data1:                    return "FORM_data1";
    case DW_FORM_flag:                     return "FORM_flag";
    case DW_FORM_sdata:                    return "FORM_sdata";
    case DW_FORM_strp:                     return "FORM_strp";
    case DW_FORM_udata:                    return "FORM_udata";
    case DW_FORM_ref_addr:                 return "FORM_ref_addr";
    case DW_FORM_ref1:                     return "FORM_ref1";
    case DW_FORM_ref2:                     return "FORM_ref2";
    case DW_FORM_ref4:                     return "FORM_ref4";
    case DW_FORM_ref8:                     return "FORM_ref8";
    case DW_FORM_ref_udata:                return "FORM_ref_udata";
    case DW_FORM_indirect:                 return "FORM_indirect";
  }
  assert(0 && "Unknown Dwarf Form Encoding");
  return "";
}

/// OperationEncodingString - Return the string for the specified operation
/// encoding.
static const char *OperationEncodingString(unsigned Encoding) {
  switch(Encoding) {
    case DW_OP_addr:                       return "OP_addr";
    case DW_OP_deref:                      return "OP_deref";
    case DW_OP_const1u:                    return "OP_const1u";
    case DW_OP_const1s:                    return "OP_const1s";
    case DW_OP_const2u:                    return "OP_const2u";
    case DW_OP_const2s:                    return "OP_const2s";
    case DW_OP_const4u:                    return "OP_const4u";
    case DW_OP_const4s:                    return "OP_const4s";
    case DW_OP_const8u:                    return "OP_const8u";
    case DW_OP_const8s:                    return "OP_const8s";
    case DW_OP_constu:                     return "OP_constu";
    case DW_OP_consts:                     return "OP_consts";
    case DW_OP_dup:                        return "OP_dup";
    case DW_OP_drop:                       return "OP_drop";
    case DW_OP_over:                       return "OP_over";
    case DW_OP_pick:                       return "OP_pick";
    case DW_OP_swap:                       return "OP_swap";
    case DW_OP_rot:                        return "OP_rot";
    case DW_OP_xderef:                     return "OP_xderef";
    case DW_OP_abs:                        return "OP_abs";
    case DW_OP_and:                        return "OP_and";
    case DW_OP_div:                        return "OP_div";
    case DW_OP_minus:                      return "OP_minus";
    case DW_OP_mod:                        return "OP_mod";
    case DW_OP_mul:                        return "OP_mul";
    case DW_OP_neg:                        return "OP_neg";
    case DW_OP_not:                        return "OP_not";
    case DW_OP_or:                         return "OP_or";
    case DW_OP_plus:                       return "OP_plus";
    case DW_OP_plus_uconst:                return "OP_plus_uconst";
    case DW_OP_shl:                        return "OP_shl";
    case DW_OP_shr:                        return "OP_shr";
    case DW_OP_shra:                       return "OP_shra";
    case DW_OP_xor:                        return "OP_xor";
    case DW_OP_skip:                       return "OP_skip";
    case DW_OP_bra:                        return "OP_bra";
    case DW_OP_eq:                         return "OP_eq";
    case DW_OP_ge:                         return "OP_ge";
    case DW_OP_gt:                         return "OP_gt";
    case DW_OP_le:                         return "OP_le";
    case DW_OP_lt:                         return "OP_lt";
    case DW_OP_ne:                         return "OP_ne";
    case DW_OP_lit0:                       return "OP_lit0";
    case DW_OP_lit1:                       return "OP_lit1";
    case DW_OP_lit31:                      return "OP_lit31";
    case DW_OP_reg0:                       return "OP_reg0";
    case DW_OP_reg1:                       return "OP_reg1";
    case DW_OP_reg31:                      return "OP_reg31";
    case DW_OP_breg0:                      return "OP_breg0";
    case DW_OP_breg1:                      return "OP_breg1";
    case DW_OP_breg31:                     return "OP_breg31";
    case DW_OP_regx:                       return "OP_regx";
    case DW_OP_fbreg:                      return "OP_fbreg";
    case DW_OP_bregx:                      return "OP_bregx";
    case DW_OP_piece:                      return "OP_piece";
    case DW_OP_deref_size:                 return "OP_deref_size";
    case DW_OP_xderef_size:                return "OP_xderef_size";
    case DW_OP_nop:                        return "OP_nop";
    case DW_OP_push_object_address:        return "OP_push_object_address";
    case DW_OP_call2:                      return "OP_call2";
    case DW_OP_call4:                      return "OP_call4";
    case DW_OP_call_ref:                   return "OP_call_ref";
    case DW_OP_form_tls_address:           return "OP_form_tls_address";
    case DW_OP_call_frame_cfa:             return "OP_call_frame_cfa";
    case DW_OP_lo_user:                    return "OP_lo_user";
    case DW_OP_hi_user:                    return "OP_hi_user";
  }
  assert(0 && "Unknown Dwarf Operation Encoding");
  return "";
}

/// AttributeEncodingString - Return the string for the specified attribute
/// encoding.
static const char *AttributeEncodingString(unsigned Encoding) {
  switch(Encoding) {
    case DW_ATE_address:                   return "ATE_address";
    case DW_ATE_boolean:                   return "ATE_boolean";
    case DW_ATE_complex_float:             return "ATE_complex_float";
    case DW_ATE_float:                     return "ATE_float";
    case DW_ATE_signed:                    return "ATE_signed";
    case DW_ATE_signed_char:               return "ATE_signed_char";
    case DW_ATE_unsigned:                  return "ATE_unsigned";
    case DW_ATE_unsigned_char:             return "ATE_unsigned_char";
    case DW_ATE_imaginary_float:           return "ATE_imaginary_float";
    case DW_ATE_packed_decimal:            return "ATE_packed_decimal";
    case DW_ATE_numeric_string:            return "ATE_numeric_string";
    case DW_ATE_edited:                    return "ATE_edited";
    case DW_ATE_signed_fixed:              return "ATE_signed_fixed";
    case DW_ATE_unsigned_fixed:            return "ATE_unsigned_fixed";
    case DW_ATE_decimal_float:             return "ATE_decimal_float";
    case DW_ATE_lo_user:                   return "ATE_lo_user";
    case DW_ATE_hi_user:                   return "ATE_hi_user";
  }
  assert(0 && "Unknown Dwarf Attribute Encoding");
  return "";
}

/// DecimalSignString - Return the string for the specified decimal sign
/// attribute.
static const char *DecimalSignString(unsigned Sign) {
  switch(Sign) {
    case DW_DS_unsigned:                   return "DS_unsigned";
    case DW_DS_leading_overpunch:          return "DS_leading_overpunch";
    case DW_DS_trailing_overpunch:         return "DS_trailing_overpunch";
    case DW_DS_leading_separate:           return "DS_leading_separate";
    case DW_DS_trailing_separate:          return "DS_trailing_separate";
  }
  assert(0 && "Unknown Dwarf Decimal Sign Attribute");
  return "";
}

/// EndianityString - Return the string for the specified endianity.
///
static const char *EndianityString(unsigned Endian) {
  switch(Endian) {
    case DW_END_default:                   return "END_default";
    case DW_END_big:                       return "END_big";
    case DW_END_little:                    return "END_little";
    case DW_END_lo_user:                   return "END_lo_user";
    case DW_END_hi_user:                   return "END_hi_user";
  }
  assert(0 && "Unknown Dwarf Endianity");
  return "";
}

/// AccessibilityString - Return the string for the specified accessibility.
///
static const char *AccessibilityString(unsigned Access) {
  switch(Access) {
    // Accessibility codes
    case DW_ACCESS_public:                 return "ACCESS_public";
    case DW_ACCESS_protected:              return "ACCESS_protected";
    case DW_ACCESS_private:                return "ACCESS_private";
  }
  assert(0 && "Unknown Dwarf Accessibility");
  return "";
}

/// VisibilityString - Return the string for the specified visibility.
///
static const char *VisibilityString(unsigned Visibility) {
  switch(Visibility) {
    case DW_VIS_local:                     return "VIS_local";
    case DW_VIS_exported:                  return "VIS_exported";
    case DW_VIS_qualified:                 return "VIS_qualified";
  }
  assert(0 && "Unknown Dwarf Visibility");
  return "";
}

/// VirtualityString - Return the string for the specified virtuality.
///
static const char *VirtualityString(unsigned Virtuality) {
  switch(Virtuality) {
    case DW_VIRTUALITY_none:               return "VIRTUALITY_none";
    case DW_VIRTUALITY_virtual:            return "VIRTUALITY_virtual";
    case DW_VIRTUALITY_pure_virtual:       return "VIRTUALITY_pure_virtual";
  }
  assert(0 && "Unknown Dwarf Virtuality");
  return "";
}

/// LanguageString - Return the string for the specified language.
///
static const char *LanguageString(unsigned Language) {
  switch(Language) {
    case DW_LANG_C89:                      return "LANG_C89";
    case DW_LANG_C:                        return "LANG_C";
    case DW_LANG_Ada83:                    return "LANG_Ada83";
    case DW_LANG_C_plus_plus:              return "LANG_C_plus_plus";
    case DW_LANG_Cobol74:                  return "LANG_Cobol74";
    case DW_LANG_Cobol85:                  return "LANG_Cobol85";
    case DW_LANG_Fortran77:                return "LANG_Fortran77";
    case DW_LANG_Fortran90:                return "LANG_Fortran90";
    case DW_LANG_Pascal83:                 return "LANG_Pascal83";
    case DW_LANG_Modula2:                  return "LANG_Modula2";
    case DW_LANG_Java:                     return "LANG_Java";
    case DW_LANG_C99:                      return "LANG_C99";
    case DW_LANG_Ada95:                    return "LANG_Ada95";
    case DW_LANG_Fortran95:                return "LANG_Fortran95";
    case DW_LANG_PLI:                      return "LANG_PLI";
    case DW_LANG_ObjC:                     return "LANG_ObjC";
    case DW_LANG_ObjC_plus_plus:           return "LANG_ObjC_plus_plus";
    case DW_LANG_UPC:                      return "LANG_UPC";
    case DW_LANG_D:                        return "LANG_D";
    case DW_LANG_lo_user:                  return "LANG_lo_user";
    case DW_LANG_hi_user:                  return "LANG_hi_user";
  }
  assert(0 && "Unknown Dwarf Language");
  return "";
}

/// CaseString - Return the string for the specified identifier case.
///
static const char *CaseString(unsigned Case) {
   switch(Case) {
    case DW_ID_case_sensitive:             return "ID_case_sensitive";
    case DW_ID_up_case:                    return "ID_up_case";
    case DW_ID_down_case:                  return "ID_down_case";
    case DW_ID_case_insensitive:           return "ID_case_insensitive";
  }
  assert(0 && "Unknown Dwarf Identifier Case");
  return "";
}

/// ConventionString - Return the string for the specified calling convention.
///
static const char *ConventionString(unsigned Convention) {
   switch(Convention) {
    case DW_CC_normal:                     return "CC_normal";
    case DW_CC_program:                    return "CC_program";
    case DW_CC_nocall:                     return "CC_nocall";
    case DW_CC_lo_user:                    return "CC_lo_user";
    case DW_CC_hi_user:                    return "CC_hi_user";
  }
  assert(0 && "Unknown Dwarf Calling Convention");
  return "";
}

/// InlineCodeString - Return the string for the specified inline code.
///
static const char *InlineCodeString(unsigned Code) {
   switch(Code) {
    case DW_INL_not_inlined:               return "INL_not_inlined";
    case DW_INL_inlined:                   return "INL_inlined";
    case DW_INL_declared_not_inlined:      return "INL_declared_not_inlined";
    case DW_INL_declared_inlined:          return "INL_declared_inlined";
  }
  assert(0 && "Unknown Dwarf Inline Code");
  return "";
}

/// ArrayOrderString - Return the string for the specified array order.
///
static const char *ArrayOrderString(unsigned Order) {
   switch(Order) {
    case DW_ORD_row_major:                 return "ORD_row_major";
    case DW_ORD_col_major:                 return "ORD_col_major";
  }
  assert(0 && "Unknown Dwarf Array Order");
  return "";
}

/// DiscriminantString - Return the string for the specified discriminant
/// descriptor.
static const char *DiscriminantString(unsigned Discriminant) {
   switch(Discriminant) {
    case DW_DSC_label:                     return "DSC_label";
    case DW_DSC_range:                     return "DSC_range";
  }
  assert(0 && "Unknown Dwarf Discriminant Descriptor");
  return "";
}

/// LNStandardString - Return the string for the specified line number standard.
///
static const char *LNStandardString(unsigned Standard) {
   switch(Standard) {
    case DW_LNS_copy:                      return "LNS_copy";
    case DW_LNS_advance_pc:                return "LNS_advance_pc";
    case DW_LNS_advance_line:              return "LNS_advance_line";
    case DW_LNS_set_file:                  return "LNS_set_file";
    case DW_LNS_set_column:                return "LNS_set_column";
    case DW_LNS_negate_stmt:               return "LNS_negate_stmt";
    case DW_LNS_set_basic_block:           return "LNS_set_basic_block";
    case DW_LNS_const_add_pc:              return "LNS_const_add_pc";
    case DW_LNS_fixed_advance_pc:          return "LNS_fixed_advance_pc";
    case DW_LNS_set_prologue_end:          return "LNS_set_prologue_end";
    case DW_LNS_set_epilogue_begin:        return "LNS_set_epilogue_begin";
    case DW_LNS_set_isa:                   return "LNS_set_isa";
  }
  assert(0 && "Unknown Dwarf Line Number Standard");
  return "";
}

/// LNExtendedString - Return the string for the specified line number extended
/// opcode encodings.
static const char *LNExtendedString(unsigned Encoding) {
   switch(Encoding) {
    // Line Number Extended Opcode Encodings
    case DW_LNE_end_sequence:              return "LNE_end_sequence";
    case DW_LNE_set_address:               return "LNE_set_address";
    case DW_LNE_define_file:               return "LNE_define_file";
    case DW_LNE_lo_user:                   return "LNE_lo_user";
    case DW_LNE_hi_user:                   return "LNE_hi_user";
  }
  assert(0 && "Unknown Dwarf Line Number Extended Opcode Encoding");
  return "";
}

/// MacinfoString - Return the string for the specified macinfo type encodings.
///
static const char *MacinfoString(unsigned Encoding) {
   switch(Encoding) {
    // Macinfo Type Encodings
    case DW_MACINFO_define:                return "MACINFO_define";
    case DW_MACINFO_undef:                 return "MACINFO_undef";
    case DW_MACINFO_start_file:            return "MACINFO_start_file";
    case DW_MACINFO_end_file:              return "MACINFO_end_file";
    case DW_MACINFO_vendor_ext:            return "MACINFO_vendor_ext";
  }
  assert(0 && "Unknown Dwarf Macinfo Type Encodings");
  return "";
}

/// CallFrameString - Return the string for the specified call frame instruction
/// encodings.
static const char *CallFrameString(unsigned Encoding) {
   switch(Encoding) {
    case DW_CFA_advance_loc:               return "CFA_advance_loc";
    case DW_CFA_offset:                    return "CFA_offset";
    case DW_CFA_restore:                   return "CFA_restore";
    case DW_CFA_set_loc:                   return "CFA_set_loc";
    case DW_CFA_advance_loc1:              return "CFA_advance_loc1";
    case DW_CFA_advance_loc2:              return "CFA_advance_loc2";
    case DW_CFA_advance_loc4:              return "CFA_advance_loc4";
    case DW_CFA_offset_extended:           return "CFA_offset_extended";
    case DW_CFA_restore_extended:          return "CFA_restore_extended";
    case DW_CFA_undefined:                 return "CFA_undefined";
    case DW_CFA_same_value:                return "CFA_same_value";
    case DW_CFA_register:                  return "CFA_register";
    case DW_CFA_remember_state:            return "CFA_remember_state";
    case DW_CFA_restore_state:             return "CFA_restore_state";
    case DW_CFA_def_cfa:                   return "CFA_def_cfa";
    case DW_CFA_def_cfa_register:          return "CFA_def_cfa_register";
    case DW_CFA_def_cfa_offset:            return "CFA_def_cfa_offset";
    case DW_CFA_def_cfa_expression:        return "CFA_def_cfa_expression";
    case DW_CFA_expression:                return "CFA_expression";
    case DW_CFA_offset_extended_sf:        return "CFA_offset_extended_sf";
    case DW_CFA_def_cfa_sf:                return "CFA_def_cfa_sf";
    case DW_CFA_def_cfa_offset_sf:         return "CFA_def_cfa_offset_sf";
    case DW_CFA_val_offset:                return "CFA_val_offset";
    case DW_CFA_val_offset_sf:             return "CFA_val_offset_sf";
    case DW_CFA_val_expression:            return "CFA_val_expression";
    case DW_CFA_lo_user:                   return "CFA_lo_user";
    case DW_CFA_hi_user:                   return "CFA_hi_user";
  }
  assert(0 && "Unknown Dwarf Call Frame Instruction Encodings");
  return "";
}

//===----------------------------------------------------------------------===//

/// EmitValue - Emit integer of appropriate size.
///
void DIEInteger::EmitValue(const DwarfWriter &DW, unsigned Form) const {
  switch (Form) {
  case DW_FORM_data1: DW.EmitByte(Value);  break;
  case DW_FORM_data2: DW.EmitShort(Value); break;
  case DW_FORM_data4: DW.EmitLong(Value);  break;
  default: assert(0 && "DIE Value form not supported yet"); break;
  }
}

/// SizeOf - Determine size of integer value in bytes.
///
unsigned DIEInteger::SizeOf(const DwarfWriter &DW, unsigned Form) const {
  switch (Form) {
  case DW_FORM_data1: return sizeof(int8_t);
  case DW_FORM_data2: return sizeof(int16_t);
  case DW_FORM_data4: return sizeof(int32_t);
  default: assert(0 && "DIE Value form not supported yet"); break;
  }
  return 0;
}

//===----------------------------------------------------------------------===//

/// EmitValue - Emit string value.
///
void DIEString::EmitValue(const DwarfWriter &DW, unsigned Form) const {
  DW.EmitString(Value);
}

/// SizeOf - Determine size of string value in bytes.
///
unsigned DIEString::SizeOf(const DwarfWriter &DW, unsigned Form) const {
  return Value.size() + sizeof(int8_t);
}

//===----------------------------------------------------------------------===//

/// EmitValue - Emit label value.
///
void DIELabel::EmitValue(const DwarfWriter &DW, unsigned Form) const {
  DW.EmitLabelReference(Value);
}

/// SizeOf - Determine size of label value in bytes.
///
unsigned DIELabel::SizeOf(const DwarfWriter &DW, unsigned Form) const {
  return DW.getAddressSize();
}
    
//===----------------------------------------------------------------------===//

/// EmitValue - Emit delta value.
///
void DIEDelta::EmitValue(const DwarfWriter &DW, unsigned Form) const {
  DW.EmitDifference(Value1, Value2);
}

/// SizeOf - Determine size of delta value in bytes.
///
unsigned DIEDelta::SizeOf(const DwarfWriter &DW, unsigned Form) const {
  return DW.getAddressSize();
}

//===----------------------------------------------------------------------===//

/// PrintHex - Print a value as a hexidecimal value.
///
void DwarfWriter::PrintHex(int Value) const { 
  O << "0x" << std::hex << Value << std::dec;
}

/// EOL - Print a newline character to asm stream.  If a comment is present
/// then it will be printed first.  Comments should not contain '\n'.
void DwarfWriter::EOL(const std::string &Comment) const {
  if (DwarfVerbose) {
    O << "\t"
      << Asm->CommentString
      << " "
      << Comment;
  }
  O << "\n";
}

/// EmitULEB128Bytes - Emit an assembler byte data directive to compose an
/// unsigned leb128 value.
void DwarfWriter::EmitULEB128Bytes(unsigned Value) const {
  if (hasLEB128) {
    O << "\t.uleb128\t"
      << Value;
  } else {
    O << Asm->Data8bitsDirective;
    PrintULEB128(Value);
  }
}

/// EmitSLEB128Bytes - Emit an assembler byte data directive to compose a
/// signed leb128 value.
void DwarfWriter::EmitSLEB128Bytes(int Value) const {
  if (hasLEB128) {
    O << "\t.sleb128\t"
      << Value;
  } else {
    O << Asm->Data8bitsDirective;
    PrintSLEB128(Value);
  }
}

/// PrintULEB128 - Print a series of hexidecimal values (separated by commas)
/// representing an unsigned leb128 value.
void DwarfWriter::PrintULEB128(unsigned Value) const {
  do {
    unsigned Byte = Value & 0x7f;
    Value >>= 7;
    if (Value) Byte |= 0x80;
    PrintHex(Byte);
    if (Value) O << ", ";
  } while (Value);
}

/// SizeULEB128 - Compute the number of bytes required for an unsigned leb128
/// value.
unsigned DwarfWriter::SizeULEB128(unsigned Value) {
  unsigned Size = 0;
  do {
    Value >>= 7;
    Size += sizeof(int8_t);
  } while (Value);
  return Size;
}

/// PrintSLEB128 - Print a series of hexidecimal values (separated by commas)
/// representing a signed leb128 value.
void DwarfWriter::PrintSLEB128(int Value) const {
  int Sign = Value >> (8 * sizeof(Value) - 1);
  bool IsMore;
  
  do {
    unsigned Byte = Value & 0x7f;
    Value >>= 7;
    IsMore = Value != Sign || ((Byte ^ Sign) & 0x40) != 0;
    if (IsMore) Byte |= 0x80;
    PrintHex(Byte);
    if (IsMore) O << ", ";
  } while (IsMore);
}

/// SizeSLEB128 - Compute the number of bytes required for a signed leb128
/// value.
unsigned DwarfWriter::SizeSLEB128(int Value) {
  unsigned Size = 0;
  int Sign = Value >> (8 * sizeof(Value) - 1);
  bool IsMore;
  
  do {
    unsigned Byte = Value & 0x7f;
    Value >>= 7;
    IsMore = Value != Sign || ((Byte ^ Sign) & 0x40) != 0;
    Size += sizeof(int8_t);
  } while (IsMore);
  return Size;
}

/// EmitByte - Emit a byte directive and value.
///
void DwarfWriter::EmitByte(int Value) const {
  O << Asm->Data8bitsDirective;
  PrintHex(Value);
}

/// EmitShort - Emit a short directive and value.
///
void DwarfWriter::EmitShort(int Value) const {
  O << Asm->Data16bitsDirective;
  PrintHex(Value);
}

/// EmitLong - Emit a long directive and value.
///
void DwarfWriter::EmitLong(int Value) const {
  O << Asm->Data32bitsDirective;
  PrintHex(Value);
}

/// EmitString - Emit a string with quotes and a null terminator.
/// Special characters are emitted properly. (Eg. '\t')
void DwarfWriter::EmitString(const std::string &String) const {
  O << Asm->AsciiDirective
    << "\"";
  for (unsigned i = 0, N = String.size(); i < N; i++) {
    unsigned char C = String[i];
    
    if (!isascii(C) || iscntrl(C)) {
      switch(C) {
      case '\b': O << "\\b"; break;
      case '\f': O << "\\f"; break;
      case '\n': O << "\\n"; break;
      case '\r': O << "\\r"; break;
      case '\t': O << "\\t"; break;
      default:
        O << '\\';
        O << char('0' + (C >> 6));
        O << char('0' + (C >> 3));
        O << char('0' + (C >> 0));
        break;
      }
    } else if (C == '\"') {
      O << "\\\"";
    } else if (C == '\'') {
      O << "\\\'";
    } else {
     O << C;
    }
  }
  O << "\\0\"";
}

/// PrintLabelName - Print label name in form used by Dwarf writer.
///
void DwarfWriter::PrintLabelName(const char *Tag, unsigned Number) const {
  O << Asm->PrivateGlobalPrefix
    << "debug_"
    << Tag
    << Number;
}

/// EmitLabel - Emit location label for internal use by Dwarf.
///
void DwarfWriter::EmitLabel(const char *Tag, unsigned Number) const {
  PrintLabelName(Tag, Number);
  O << ":\n";
}

/// EmitLabelReference - Emit a reference to a label.
///
void DwarfWriter::EmitLabelReference(const char *Tag, unsigned Number) const {
  if (AddressSize == 4)
    O << Asm->Data32bitsDirective;
  else
    O << Asm->Data64bitsDirective;
    
  PrintLabelName(Tag, Number);
}

/// EmitDifference - Emit an label difference as sizeof(pointer) value.  Some
/// assemblers do not accept absolute expressions with data directives, so there 
/// is an option (needsSet) to use an intermediary 'set' expression.
void DwarfWriter::EmitDifference(const char *Tag1, unsigned Number1,
                                 const char *Tag2, unsigned Number2) const {
  if (needsSet) {
    static unsigned SetCounter = 0;
    O << "\t.set\t";
    PrintLabelName("set", SetCounter);
    O << ",";
    PrintLabelName(Tag1, Number1);
    O << "-";
    PrintLabelName(Tag2, Number2);
    O << "\n";
    
    if (AddressSize == sizeof(int32_t))
      O << Asm->Data32bitsDirective;
    else
      O << Asm->Data64bitsDirective;
      
    PrintLabelName("set", SetCounter);
    
    SetCounter++;
  } else {
    if (AddressSize == sizeof(int32_t))
      O << Asm->Data32bitsDirective;
    else
      O << Asm->Data64bitsDirective;
      
    PrintLabelName(Tag1, Number1);
    O << "-";
    PrintLabelName(Tag2, Number2);
  }
}

/// NewDIE - Construct a new structured debug information entry.
///  
DIE *DwarfWriter::NewDIE(const unsigned char *AbbrevData) {
  // Get the abbreviation ID.
  unsigned AbbrevID = Abbreviations.insert(DIEAbbrev(AbbrevData));
  // Allocate new new structured DIE.
  DIE *Die = new DIE(AbbrevID);
  // Return  structured DIE.
  return Die;
}

/// NewCompileUnit - Create new compile unit information.
///
DIE *DwarfWriter::NewCompileUnit(const std::string &Directory,
                                 const std::string &SourceName) {
  DIE *Die = NewDIE(AbbrevTAG_compile_unit);
  // FIXME - use the correct line set.
  Die->AddValue(DWLabel("line", 0));
  Die->AddValue(DWLabel("text_end", 0));
  Die->AddValue(DWLabel("text_begin", 0));
  // FIXME - The producer needs to be in this form, but should come from
  // an appropriate source.
  Die->AddValue("llvm 3.4.x (LLVM Research Group)");
  Die->AddValue(DW_LANG_C89);
  Die->AddValue(SourceName);
  Die->AddValue(Directory);
  
  return Die;
}

/// EmitInitial - Emit initial Dwarf declarations.  This is necessary for cc
/// tools to recognize the object file contains Dwarf information.
///
void DwarfWriter::EmitInitial() const {
  // Dwarf sections base addresses.
  Asm->SwitchSection(DwarfAbbrevSection, 0);
  EmitLabel("abbrev", 0);
  Asm->SwitchSection(DwarfInfoSection, 0);
  EmitLabel("info", 0);
  Asm->SwitchSection(DwarfLineSection, 0);
  EmitLabel("line", 0);
  
  // Standard sections base addresses.
  Asm->SwitchSection(TextSection, 0);
  EmitLabel("text_begin", 0);
  Asm->SwitchSection(DataSection, 0);
  EmitLabel("data_begin", 0);
}

/// EmitDIE - Recusively Emits a debug information entry.
///
void DwarfWriter::EmitDIE(DIE *Die) const {
  // Get the abbreviation for this DIE.
  unsigned AbbrevID = Die->getAbbrevID();
  const DIEAbbrev &Abbrev = Abbreviations[AbbrevID];

  // Emit the code (index) for the abbreviation.
  EmitULEB128Bytes(AbbrevID);
  EOL(std::string("Abbrev [" +
      utostr(AbbrevID) +
      TagString(Abbrev.getTag())) +
      " ");
  
  const std::vector<DIEValue *> &Values = Die->getValues();
  
  // Emit the DIE attribute values.
  for (unsigned i = 0, N = Values.size(); i < N; i++) {
    unsigned Attr = Abbrev.getAttribute(i);
    unsigned Form = Abbrev.getForm(i);
    assert(Form && "Too many attributes for DIE (check abbreviation)");
    
    switch (Attr) {
    case DW_AT_sibling: {
      EmitLong(Die->SiblingOffset());
      break;
    }
    default: {
      // Emit an attribute using the defined form.
      Values[i]->EmitValue(*this, Form);
      break;
    }
    }
    
    EOL(AttributeString(Attr));
  }
  
  // Emit the DIE children if any.
  if (Abbrev.getChildrenFlag() == DW_CHILDREN_yes) {
    const std::vector<DIE *> &Children = Die->getChildren();
    
    for (unsigned j = 0, M = Children.size(); j < M; j++) {
      // FIXME - handle sibling offsets.
      // FIXME - handle all DIE types.
      EmitDIE(Children[j]);
    }
    
    EmitByte(0); EOL("End Of Children Mark");
  }
}

/// SizeAndOffsetDie - Compute the size and offset of a DIE.
///
unsigned DwarfWriter::SizeAndOffsetDie(DIE *Die, unsigned Offset) const {
  // Get the abbreviation for this DIE.
  unsigned AbbrevID = Die->getAbbrevID();
  const DIEAbbrev &Abbrev = Abbreviations[AbbrevID];

  // Set DIE offset
  Die->setOffset(Offset);
  
  // Start the size with the size of abbreviation code.
  Offset += SizeULEB128(AbbrevID);
  
  const std::vector<DIEValue *> &Values = Die->getValues();

  // Emit the DIE attribute values.
  for (unsigned i = 0, N = Values.size(); i < N; i++) {
    // Size attribute value.
    Offset += Values[i]->SizeOf(*this, Abbrev.getForm(i));
  }
  
  // Emit the DIE children if any.
  if (Abbrev.getChildrenFlag() == DW_CHILDREN_yes) {
    const std::vector<DIE *> &Children = Die->getChildren();
    
    for (unsigned j = 0, M = Children.size(); j < M; j++) {
      // FIXME - handle sibling offsets.
      // FIXME - handle all DIE types.
      Offset = SizeAndOffsetDie(Children[j], Offset);
    }
    
    // End of children marker.
    Offset += sizeof(int8_t);
  }

  Die->setSize(Offset - Die->getOffset());
  return Offset;
}

/// SizeAndOffsets - Compute the size and offset of all the DIEs.
///
void DwarfWriter::SizeAndOffsets() {
  // Compute size of debug unit header
  unsigned Offset = sizeof(int32_t) + // Length of Compilation Unit Info
                    sizeof(int16_t) + // DWARF version number
                    sizeof(int32_t) + // Offset Into Abbrev. Section
                    sizeof(int8_t);  // Pointer Size (in bytes)
  
  // Process each compile unit.
  for (unsigned i = 0, N = CompileUnits.size(); i < N; i++) {
    Offset = SizeAndOffsetDie(CompileUnits[i], Offset);
  }
}

/// EmitDebugInfo - Emit the debug info section.
///
void DwarfWriter::EmitDebugInfo() const {
  // Start debug info section.
  Asm->SwitchSection(DwarfInfoSection, 0);
  
  // Get the number of compile units.
  unsigned N = CompileUnits.size();
  
  // If there are any compile units.
  if (N) {
    // Emit the compile units header.

    // Emit size of content not including length itself
    unsigned ContentSize = CompileUnits[N - 1]->SiblingOffset();
    EmitLong(ContentSize - sizeof(int32_t));
    EOL("Length of Compilation Unit Info");
    
    EmitShort(DWARF_VERSION); EOL("DWARF version number");

    EmitLabelReference("abbrev", 0); EOL("Offset Into Abbrev. Section");

    EmitByte(AddressSize); EOL("Address Size (in bytes)");
    
    // Process each compile unit.
    for (unsigned i = 0; i < N; i++) {
      EmitDIE(CompileUnits[i]);
    }
  }
}

/// EmitAbbreviations - Emit the abbreviation section.
///
void DwarfWriter::EmitAbbreviations() const {
  // Start the debug abbrev section.
  Asm->SwitchSection(DwarfAbbrevSection, 0);
  
  // For each abbrevation.
  for (unsigned AbbrevID = 1, NAID = Abbreviations.size();
                AbbrevID <= NAID; AbbrevID++) {
    // Get abbreviation data
    const DIEAbbrev &Abbrev = Abbreviations[AbbrevID];
    
    // Emit the abbrevations code (base 1 index.)
    EmitULEB128Bytes(AbbrevID); EOL("Abbreviation Code");
    
    // Emit its Dwarf tag type.
    EmitULEB128Bytes(Abbrev.getTag());
    EOL(TagString(Abbrev.getTag()));
    
    // Emit whether it has children DIEs.
    EmitULEB128Bytes(Abbrev.getChildrenFlag());
    EOL(ChildrenString(Abbrev.getChildrenFlag()));
    
    // For each attribute description.
    for (unsigned i = 0; ; i++) {
      unsigned Attr = Abbrev.getAttribute(i);
      unsigned Form = Abbrev.getForm(i);
      
      // Attributes are null terminated.
      if (!Attr) break;
      
      // Emit attribute type.
      EmitULEB128Bytes(Attr);
      EOL(AttributeString(Attr));
      
      // Emit form type.
      EmitULEB128Bytes(Form);
      EOL(FormEncodingString(Form));
    }

    // Mark end of abbreviation.
    EmitULEB128Bytes(0); EOL("EOM(1)");
    EmitULEB128Bytes(0); EOL("EOM(2)");
  }
}

/// EmitDebugLines - Emit source line information.
///
void DwarfWriter::EmitDebugLines() const {
  // Minimum line delta, thus ranging from -10..(255-10).
  const int MinLineDelta = -(DW_LNS_fixed_advance_pc + 1);
  // Maximum line delta, thus ranging from -10..(255-10).
  const int MaxLineDelta = 255 + MinLineDelta;

  // Start the dwarf line section.
  Asm->SwitchSection(DwarfLineSection, 0);
  
  // Construct the section header.
  
  EmitDifference("line_end", 0, "line_begin", 0);
  EOL("Length of Source Line Info");
  EmitLabel("line_begin", 0);
  
  EmitShort(DWARF_VERSION); EOL("DWARF version number");
  
  EmitDifference("line_prolog_end", 0, "line_prolog_begin", 0);
  EOL("Prolog Length");
  EmitLabel("line_prolog_begin", 0);
  
  EmitByte(1); EOL("Minimum Instruction Length");

  EmitByte(1); EOL("Default is_stmt_start flag");

  EmitByte(MinLineDelta);  EOL("Line Base Value (Special Opcodes)");
  
  EmitByte(MaxLineDelta); EOL("Line Range Value (Special Opcodes)");

  EmitByte(-MinLineDelta); EOL("Special Opcode Base");
  
  // Line number standard opcode encodings argument count
  EmitByte(0); EOL("DW_LNS_copy arg count");
  EmitByte(1); EOL("DW_LNS_advance_pc arg count");
  EmitByte(1); EOL("DW_LNS_advance_line arg count");
  EmitByte(1); EOL("DW_LNS_set_file arg count");
  EmitByte(1); EOL("DW_LNS_set_column arg count");
  EmitByte(0); EOL("DW_LNS_negate_stmt arg count");
  EmitByte(0); EOL("DW_LNS_set_basic_block arg count");
  EmitByte(0); EOL("DW_LNS_const_add_pc arg count");
  EmitByte(1); EOL("DW_LNS_fixed_advance_pc arg count");

  const UniqueVector<std::string> &Directories = DebugInfo->getDirectories();
  const UniqueVector<SourceFileInfo> &SourceFiles = DebugInfo->getSourceFiles();

  // Emit directories.
  for (unsigned DirectoryID = 1, NDID = Directories.size();
                DirectoryID <= NDID; DirectoryID++) {
    EmitString(Directories[DirectoryID]); EOL("Directory");
  }
  EmitByte(0); EOL("End of directories");
  
  // Emit files.
  for (unsigned SourceID = 1, NSID = SourceFiles.size();
               SourceID <= NSID; SourceID++) {
    const SourceFileInfo &SourceFile = SourceFiles[SourceID];
    EmitString(SourceFile.getName()); EOL("Source");
    EmitULEB128Bytes(SourceFile.getDirectoryID());  EOL("Directory #");
    EmitULEB128Bytes(0);  EOL("Mod date");
    EmitULEB128Bytes(0);  EOL("File size");
  }
  EmitByte(0); EOL("End of files");
  
  EmitLabel("line_prolog_end", 0);
  
  // Emit line information
  const std::vector<SourceLineInfo *> &LineInfos = DebugInfo->getSourceLines();
  
  // Dwarf assumes we start with first line of first source file.
  unsigned Source = 1;
  unsigned Line = 1;
  
  // Construct rows of the address, source, line, column matrix.
  for (unsigned i = 0, N = LineInfos.size(); i < N; i++) {
    SourceLineInfo *LineInfo = LineInfos[i];

    // Define the line address.
    EmitByte(0); EOL("Extended Op");
    EmitByte(4 + 1); EOL("Op size");
    EmitByte(DW_LNE_set_address); EOL("DW_LNE_set_address");
    EmitLabelReference("loc", i + 1); EOL("Location label");
    
    // If change of source, then switch to the new source.
    if (Source != LineInfo->getSourceID()) {
      Source = LineInfo->getSourceID();
      EmitByte(DW_LNS_set_file); EOL("DW_LNS_set_file");
      EmitULEB128Bytes(0); EOL("New Source");
    }
    
    // If change of line.
    if (Line != LineInfo->getLine()) {
      // Determine offset.
      int Offset = LineInfo->getLine() - Line;
      int Delta = Offset - MinLineDelta;
      
      // Update line.
      Line = LineInfo->getLine();
      
      // If delta is small enough and in range...
      if (Delta >= 0 && Delta < (MaxLineDelta - 1)) {
        // ... then use fast opcode.
        EmitByte(Delta - MinLineDelta); EOL("Line Delta");
      } else {
        // ... otherwise use long hand.
        EmitByte(DW_LNS_advance_line); EOL("DW_LNS_advance_line");
        EmitSLEB128Bytes(Offset); EOL("Line Offset");
        EmitByte(DW_LNS_copy); EOL("DW_LNS_copy");
      }
    } else {
      // Copy the previous row (different address or source)
      EmitByte(DW_LNS_copy); EOL("DW_LNS_copy");
    }
  }

  // Mark end of matrix.
  EmitByte(0); EOL("DW_LNE_end_sequence");
  EmitULEB128Bytes(1);  O << "\n";
  EmitByte(1); O << "\n";
  
  EmitLabel("line_end", 0);
}

/// ShouldEmitDwarf - Determine if Dwarf declarations should be made.
///
bool DwarfWriter::ShouldEmitDwarf() {
  // Check if debug info is present.
  if (!DebugInfo || !DebugInfo->hasInfo()) return false;
  
  // Make sure initial declarations are made.
  if (!didInitial) {
    EmitInitial();
    didInitial = true;
  }
  
  // Okay to emit.
  return true;
}

//===----------------------------------------------------------------------===//
// Main enties.
//
    
  DwarfWriter::DwarfWriter(std::ostream &o, AsmPrinter *ap)
  : O(o)
  , Asm(ap)
  , DebugInfo(NULL)
  , didInitial(false)
  , CompileUnits()
  , Abbreviations()
  , AddressSize(sizeof(int32_t))
  , hasLEB128(false)
  , hasDotLoc(false)
  , hasDotFile(false)
  , needsSet(false)
  , DwarfAbbrevSection(".debug_abbrev")
  , DwarfInfoSection(".debug_info")
  , DwarfLineSection(".debug_line")
  , TextSection(".text")
  , DataSection(".data")
  {}
  DwarfWriter::~DwarfWriter() {
  }

/// BeginModule - Emit all Dwarf sections that should come prior to the content.
///
void DwarfWriter::BeginModule() {
  if (!ShouldEmitDwarf()) return;
  EOL("Dwarf Begin Module");
}

/// EndModule - Emit all Dwarf sections that should come after the content.
///
void DwarfWriter::EndModule() {
  if (!ShouldEmitDwarf()) return;
  EOL("Dwarf End Module");
  
  // Standard sections final addresses.
  EmitLabel("text_end", 0);
  EmitLabel("data_end", 0);

  // Get directory and source information.
  const UniqueVector<std::string> &Directories = DebugInfo->getDirectories();
  const UniqueVector<SourceFileInfo> &SourceFiles = DebugInfo->getSourceFiles();

  // Construct compile unit DIEs for each source.
  for (unsigned SourceID = 1, NSID = SourceFiles.size();
                SourceID <=  NSID; SourceID++) {
    const SourceFileInfo &SourceFile = SourceFiles[SourceID];
    const std::string &Directory = Directories[SourceFile.getDirectoryID()];
    const std::string &SourceName = SourceFile.getName();
    DIE *CompileUnit = NewCompileUnit(Directory, SourceName);
    CompileUnits.push_back(CompileUnit);
  }

  // Compute DIE offsets and sizes.
  SizeAndOffsets();
  
  // Emit all the DIEs into a debug info section
  EmitDebugInfo();
  
  // Corresponding abbreviations into a abbrev section.
  EmitAbbreviations();
  
  // Emit source line correspondence into a debug line section.
  EmitDebugLines();
}

/// BeginFunction - Emit pre-function debug information.
///
void DwarfWriter::BeginFunction() {
  if (!ShouldEmitDwarf()) return;
  EOL("Dwarf Begin Function");
}

/// EndFunction - Emit post-function debug information.
///
void DwarfWriter::EndFunction() {
  if (!ShouldEmitDwarf()) return;
  EOL("Dwarf End Function");
}
