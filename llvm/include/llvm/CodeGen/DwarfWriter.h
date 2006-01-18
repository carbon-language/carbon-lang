//===-- llvm/CodeGen/DwarfWriter.h - Dwarf Framework ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by James M. Laskey and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing Dwarf debug info into asm files.  For
// Details on the Dwarf 3 specfication see DWARF Debugging Information Format
// V.3 reference manual http://dwarf.freestandards.org ,
//
// The role of the Dwarf Writer class is to extract debug information from the
// MachineDebugInfo object, organize it in Dwarf form and then emit it into asm
// the current asm file using data and high level Dwarf directives.
// 
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_DWARFWRITER_H
#define LLVM_CODEGEN_DWARFWRITER_H

#include "llvm/ADT/UniqueVector.h"

#include <iosfwd>
#include <string>


namespace llvm {

  //===--------------------------------------------------------------------===//
  // Dwarf constants as gleaned from the DWARF Debugging Information Format V.3
  // reference manual http://dwarf.freestandards.org .
  //
  enum dwarf_constants {
    DWARF_VERSION = 2,
    
   // Tags
    DW_TAG_array_type = 0x01,
    DW_TAG_class_type = 0x02,
    DW_TAG_entry_point = 0x03,
    DW_TAG_enumeration_type = 0x04,
    DW_TAG_formal_parameter = 0x05,
    DW_TAG_imported_declaration = 0x08,
    DW_TAG_label = 0x0a,
    DW_TAG_lexical_block = 0x0b,
    DW_TAG_member = 0x0d,
    DW_TAG_pointer_type = 0x0f,
    DW_TAG_reference_type = 0x10,
    DW_TAG_compile_unit = 0x11,
    DW_TAG_string_type = 0x12,
    DW_TAG_structure_type = 0x13,
    DW_TAG_subroutine_type = 0x15,
    DW_TAG_typedef = 0x16,
    DW_TAG_union_type = 0x17,
    DW_TAG_unspecified_parameters = 0x18,
    DW_TAG_variant = 0x19,
    DW_TAG_common_block = 0x1a,
    DW_TAG_common_inclusion = 0x1b,
    DW_TAG_inheritance = 0x1c,
    DW_TAG_inlined_subroutine = 0x1d,
    DW_TAG_module = 0x1e,
    DW_TAG_ptr_to_member_type = 0x1f,
    DW_TAG_set_type = 0x20,
    DW_TAG_subrange_type = 0x21,
    DW_TAG_with_stmt = 0x22,
    DW_TAG_access_declaration = 0x23,
    DW_TAG_base_type = 0x24,
    DW_TAG_catch_block = 0x25,
    DW_TAG_const_type = 0x26,
    DW_TAG_constant = 0x27,
    DW_TAG_enumerator = 0x28,
    DW_TAG_file_type = 0x29,
    DW_TAG_friend = 0x2a,
    DW_TAG_namelist = 0x2b,
    DW_TAG_namelist_item = 0x2c,
    DW_TAG_packed_type = 0x2d,
    DW_TAG_subprogram = 0x2e,
    DW_TAG_template_type_parameter = 0x2f,
    DW_TAG_template_value_parameter = 0x30,
    DW_TAG_thrown_type = 0x31,
    DW_TAG_try_block = 0x32,
    DW_TAG_variant_part = 0x33,
    DW_TAG_variable = 0x34,
    DW_TAG_volatile_type = 0x35,
    DW_TAG_dwarf_procedure = 0x36,
    DW_TAG_restrict_type = 0x37,
    DW_TAG_interface_type = 0x38,
    DW_TAG_namespace = 0x39,
    DW_TAG_imported_module = 0x3a,
    DW_TAG_unspecified_type = 0x3b,
    DW_TAG_partial_unit = 0x3c,
    DW_TAG_imported_unit = 0x3d,
    DW_TAG_condition = 0x3f,
    DW_TAG_shared_type = 0x40,
    DW_TAG_lo_user = 0x4080,
    DW_TAG_hi_user = 0xffff,

    // Children flag
    DW_CHILDREN_no = 0x00,
    DW_CHILDREN_yes = 0x01,

    // Attributes
    DW_AT_sibling = 0x01,
    DW_AT_location = 0x02,
    DW_AT_name = 0x03,
    DW_AT_ordering = 0x09,
    DW_AT_byte_size = 0x0b,
    DW_AT_bit_offset = 0x0c,
    DW_AT_bit_size = 0x0d,
    DW_AT_stmt_list = 0x10,
    DW_AT_low_pc = 0x11,
    DW_AT_high_pc = 0x12,
    DW_AT_language = 0x13,
    DW_AT_discr = 0x15,
    DW_AT_discr_value = 0x16,
    DW_AT_visibility = 0x17,
    DW_AT_import = 0x18,
    DW_AT_string_length = 0x19,
    DW_AT_common_reference = 0x1a,
    DW_AT_comp_dir = 0x1b,
    DW_AT_const_value = 0x1c,
    DW_AT_containing_type = 0x1d,
    DW_AT_default_value = 0x1e,
    DW_AT_inline = 0x20,
    DW_AT_is_optional = 0x21,
    DW_AT_lower_bound = 0x22,
    DW_AT_producer = 0x25,
    DW_AT_prototyped = 0x27,
    DW_AT_return_addr = 0x2a,
    DW_AT_start_scope = 0x2c,
    DW_AT_bit_stride = 0x2e,
    DW_AT_upper_bound = 0x2f,
    DW_AT_abstract_origin = 0x31,
    DW_AT_accessibility = 0x32,
    DW_AT_address_class = 0x33,
    DW_AT_artificial = 0x34,
    DW_AT_base_types = 0x35,
    DW_AT_calling_convention = 0x36,
    DW_AT_count = 0x37,
    DW_AT_data_member_location = 0x38,
    DW_AT_decl_column = 0x39,
    DW_AT_decl_file = 0x3a,
    DW_AT_decl_line = 0x3b,
    DW_AT_declaration = 0x3c,
    DW_AT_discr_list = 0x3d,
    DW_AT_encoding = 0x3e,
    DW_AT_external = 0x3f,
    DW_AT_frame_base = 0x40,
    DW_AT_friend = 0x41,
    DW_AT_identifier_case = 0x42,
    DW_AT_macro_info = 0x43,
    DW_AT_namelist_item = 0x44,
    DW_AT_priority = 0x45,
    DW_AT_segment = 0x46,
    DW_AT_specification = 0x47,
    DW_AT_static_link = 0x48,
    DW_AT_type = 0x49,
    DW_AT_use_location = 0x4a,
    DW_AT_variable_parameter = 0x4b,
    DW_AT_virtuality = 0x4c,
    DW_AT_vtable_elem_location = 0x4d,
    DW_AT_allocated = 0x4e,
    DW_AT_associated = 0x4f,
    DW_AT_data_location = 0x50,
    DW_AT_byte_stride = 0x51,
    DW_AT_entry_pc = 0x52,
    DW_AT_use_UTF8 = 0x53,
    DW_AT_extension = 0x54,
    DW_AT_ranges = 0x55,
    DW_AT_trampoline = 0x56,
    DW_AT_call_column = 0x57,
    DW_AT_call_file = 0x58,
    DW_AT_call_line = 0x59,
    DW_AT_description = 0x5a,
    DW_AT_binary_scale = 0x5b,
    DW_AT_decimal_scale = 0x5c,
    DW_AT_small = 0x5d,
    DW_AT_decimal_sign = 0x5e,
    DW_AT_digit_count = 0x5f,
    DW_AT_picture_string = 0x60,
    DW_AT_mutable = 0x61,
    DW_AT_threads_scaled = 0x62,
    DW_AT_explicit = 0x63,
    DW_AT_object_pointer = 0x64,
    DW_AT_endianity = 0x65,
    DW_AT_elemental = 0x66,
    DW_AT_pure = 0x67,
    DW_AT_recursive = 0x68,
    DW_AT_lo_user = 0x2000,
    DW_AT_hi_user = 0x3fff,

    // Attribute form encodings
    DW_FORM_addr = 0x01,
    DW_FORM_block2 = 0x03,
    DW_FORM_block4 = 0x04,
    DW_FORM_data2 = 0x05,
    DW_FORM_data4 = 0x06,
    DW_FORM_data8 = 0x07,
    DW_FORM_string = 0x08,
    DW_FORM_block = 0x09,
    DW_FORM_block1 = 0x0a,
    DW_FORM_data1 = 0x0b,
    DW_FORM_flag = 0x0c,
    DW_FORM_sdata = 0x0d,
    DW_FORM_strp = 0x0e,
    DW_FORM_udata = 0x0f,
    DW_FORM_ref_addr = 0x10,
    DW_FORM_ref1 = 0x11,
    DW_FORM_ref2 = 0x12,
    DW_FORM_ref4 = 0x13,
    DW_FORM_ref8 = 0x14,
    DW_FORM_ref_udata = 0x15,
    DW_FORM_indirect = 0x16,

    // Operation encodings
    DW_OP_addr = 0x03,
    DW_OP_deref = 0x06,
    DW_OP_const1u = 0x08,
    DW_OP_const1s = 0x09,
    DW_OP_const2u = 0x0a,
    DW_OP_const2s = 0x0b,
    DW_OP_const4u = 0x0c,
    DW_OP_const4s = 0x0d,
    DW_OP_const8u = 0x0e,
    DW_OP_const8s = 0x0f,
    DW_OP_constu = 0x10,
    DW_OP_consts = 0x11,
    DW_OP_dup = 0x12,
    DW_OP_drop = 0x13,
    DW_OP_over = 0x14,
    DW_OP_pick = 0x15,
    DW_OP_swap = 0x16,
    DW_OP_rot = 0x17,
    DW_OP_xderef = 0x18,
    DW_OP_abs = 0x19,
    DW_OP_and = 0x1a,
    DW_OP_div = 0x1b,
    DW_OP_minus = 0x1c,
    DW_OP_mod = 0x1d,
    DW_OP_mul = 0x1e,
    DW_OP_neg = 0x1f,
    DW_OP_not = 0x20,
    DW_OP_or = 0x21,
    DW_OP_plus = 0x22,
    DW_OP_plus_uconst = 0x23,
    DW_OP_shl = 0x24,
    DW_OP_shr = 0x25,
    DW_OP_shra = 0x26,
    DW_OP_xor = 0x27,
    DW_OP_skip = 0x2f,
    DW_OP_bra = 0x28,
    DW_OP_eq = 0x29,
    DW_OP_ge = 0x2a,
    DW_OP_gt = 0x2b,
    DW_OP_le = 0x2c,
    DW_OP_lt = 0x2d,
    DW_OP_ne = 0x2e,
    DW_OP_lit0 = 0x30,
    DW_OP_lit1 = 0x31,
    DW_OP_lit31 = 0x4f,
    DW_OP_reg0 = 0x50,
    DW_OP_reg1 = 0x51,
    DW_OP_reg31 = 0x6f,
    DW_OP_breg0 = 0x70,
    DW_OP_breg1 = 0x71,
    DW_OP_breg31 = 0x8f,
    DW_OP_regx = 0x90,
    DW_OP_fbreg = 0x91,
    DW_OP_bregx = 0x92,
    DW_OP_piece = 0x93,
    DW_OP_deref_size = 0x94,
    DW_OP_xderef_size = 0x95,
    DW_OP_nop = 0x96,
    DW_OP_push_object_address = 0x97,
    DW_OP_call2 = 0x98,
    DW_OP_call4 = 0x99,
    DW_OP_call_ref = 0x9a,
    DW_OP_form_tls_address = 0x9b,
    DW_OP_call_frame_cfa = 0x9c,
    DW_OP_lo_user = 0xe0,
    DW_OP_hi_user = 0xff,

    // Encoding attribute values
    DW_ATE_address = 0x01,
    DW_ATE_boolean = 0x02,
    DW_ATE_complex_float = 0x03,
    DW_ATE_float = 0x04,
    DW_ATE_signed = 0x05,
    DW_ATE_signed_char = 0x06,
    DW_ATE_unsigned = 0x07,
    DW_ATE_unsigned_char = 0x08,
    DW_ATE_imaginary_float = 0x09,
    DW_ATE_packed_decimal = 0x0a,
    DW_ATE_numeric_string = 0x0b,
    DW_ATE_edited = 0x0c,
    DW_ATE_signed_fixed = 0x0d,
    DW_ATE_unsigned_fixed = 0x0e,
    DW_ATE_decimal_float = 0x0f,
    DW_ATE_lo_user = 0x80,
    DW_ATE_hi_user = 0xff,

    // Decimal sign attribute values
    DW_DS_unsigned = 0x01,
    DW_DS_leading_overpunch = 0x02,
    DW_DS_trailing_overpunch = 0x03,
    DW_DS_leading_separate = 0x04,
    DW_DS_trailing_separate = 0x05,

    // Endianity attribute values
    DW_END_default = 0x00,
    DW_END_big = 0x01,
    DW_END_little = 0x02,
    DW_END_lo_user = 0x40,
    DW_END_hi_user = 0xff,

    // Accessibility codes
    DW_ACCESS_public = 0x01,
    DW_ACCESS_protected = 0x02,
    DW_ACCESS_private = 0x03,

    // Visibility codes 
    DW_VIS_local = 0x01,
    DW_VIS_exported = 0x02,
    DW_VIS_qualified = 0x03,

    // Virtuality codes
    DW_VIRTUALITY_none = 0x00,
    DW_VIRTUALITY_virtual = 0x01,
    DW_VIRTUALITY_pure_virtual = 0x02,
    
    // Language names
    DW_LANG_C89 = 0x0001,
    DW_LANG_C = 0x0002,
    DW_LANG_Ada83 = 0x0003,
    DW_LANG_C_plus_plus = 0x0004,
    DW_LANG_Cobol74 = 0x0005,
    DW_LANG_Cobol85 = 0x0006,
    DW_LANG_Fortran77 = 0x0007,
    DW_LANG_Fortran90 = 0x0008,
    DW_LANG_Pascal83 = 0x0009,
    DW_LANG_Modula2 = 0x000a,
    DW_LANG_Java = 0x000b,
    DW_LANG_C99 = 0x000c,
    DW_LANG_Ada95 = 0x000d,
    DW_LANG_Fortran95 = 0x000e,
    DW_LANG_PLI = 0x000f,
    DW_LANG_ObjC = 0x0010,
    DW_LANG_ObjC_plus_plus = 0x0011,
    DW_LANG_UPC = 0x0012,
    DW_LANG_D = 0x0013,
    DW_LANG_lo_user = 0x8000,
    DW_LANG_hi_user = 0xffff,
    
    // Identifier case codes
    DW_ID_case_sensitive = 0x00,
    DW_ID_up_case = 0x01,
    DW_ID_down_case = 0x02,
    DW_ID_case_insensitive = 0x03,

    // Calling convention codes
    DW_CC_normal = 0x01,
    DW_CC_program = 0x02,
    DW_CC_nocall = 0x03,
    DW_CC_lo_user = 0x40,
    DW_CC_hi_user = 0xff,

    // Inline codes
    DW_INL_not_inlined = 0x00,
    DW_INL_inlined = 0x01,
    DW_INL_declared_not_inlined = 0x02,
    DW_INL_declared_inlined = 0x03,

    // Array ordering 
    DW_ORD_row_major = 0x00,
    DW_ORD_col_major = 0x01,

    // Discriminant descriptor values
    DW_DSC_label = 0x00,
    DW_DSC_range = 0x01,

    // Line Number Standard Opcode Encodings
    DW_LNS_copy = 0x01,
    DW_LNS_advance_pc = 0x02,
    DW_LNS_advance_line = 0x03,
    DW_LNS_set_file = 0x04,
    DW_LNS_set_column = 0x05,
    DW_LNS_negate_stmt = 0x06,
    DW_LNS_set_basic_block = 0x07,
    DW_LNS_const_add_pc = 0x08,
    DW_LNS_fixed_advance_pc = 0x09,
    DW_LNS_set_prologue_end = 0x0a,
    DW_LNS_set_epilogue_begin = 0x0b,
    DW_LNS_set_isa = 0x0c,

    // Line Number Extended Opcode Encodings
    DW_LNE_end_sequence = 0x01,
    DW_LNE_set_address = 0x02,
    DW_LNE_define_file = 0x03,
    DW_LNE_lo_user = 0x80,
    DW_LNE_hi_user = 0xff,

    // Macinfo Type Encodings
    DW_MACINFO_define = 0x01,
    DW_MACINFO_undef = 0x02,
    DW_MACINFO_start_file = 0x03,
    DW_MACINFO_end_file = 0x04,
    DW_MACINFO_vendor_ext = 0xff,

    // Call frame instruction encodings
    DW_CFA_advance_loc = 0x40,
    DW_CFA_offset = 0x80,
    DW_CFA_restore = 0xc0,
    DW_CFA_set_loc = 0x01,
    DW_CFA_advance_loc1 = 0x02,
    DW_CFA_advance_loc2 = 0x03,
    DW_CFA_advance_loc4 = 0x04,
    DW_CFA_offset_extended = 0x05,
    DW_CFA_restore_extended = 0x06,
    DW_CFA_undefined = 0x07,
    DW_CFA_same_value = 0x08,
    DW_CFA_register = 0x09,
    DW_CFA_remember_state = 0x0a,
    DW_CFA_restore_state = 0x0b,
    DW_CFA_def_cfa = 0x0c,
    DW_CFA_def_cfa_register = 0x0d,
    DW_CFA_def_cfa_offset = 0x0e,
    DW_CFA_def_cfa_expression = 0x0f,
    DW_CFA_expression = 0x10,
    DW_CFA_offset_extended_sf = 0x11,
    DW_CFA_def_cfa_sf = 0x12,
    DW_CFA_def_cfa_offset_sf = 0x13,
    DW_CFA_val_offset = 0x14,
    DW_CFA_val_offset_sf = 0x15,
    DW_CFA_val_expression = 0x16,
    DW_CFA_lo_user = 0x1c,
    DW_CFA_hi_user = 0x3f
  };
  
  //===--------------------------------------------------------------------===//
  // DWLabel - Labels are used to track locations in the assembler file.
  // Labels appear in the form <prefix>debug_<Tag><Number>, where the tag is a
  // category of label (Ex. location) and number is a value unique in that
  // category.
  struct DWLabel {
    const char *Tag;                    // Label category tag. Should always be
                                        // a staticly declared C string.
    unsigned    Number;                 // Unique number

    DWLabel() : Tag(NULL), Number(0) {}
    DWLabel(const char *T, unsigned N) : Tag(T), Number(N) {}
  };

  //===--------------------------------------------------------------------===//
  // DIEAbbrev - Dwarf abbreviation, describes the organization of a debug
  // information object.
  //
  class DIEAbbrev {
  private:
    const unsigned char *Data;          // Static array of bytes containing the
                                        // image of the raw abbreviation data.

  public:
  
    DIEAbbrev(const unsigned char *D)
    : Data(D)
    {}
    
    /// operator== - Used by UniqueVector to locate entry.
    ///
    bool operator==(const DIEAbbrev &DA) const {
      return Data == DA.Data;
    }

    /// operator< - Used by UniqueVector to locate entry.
    ///
    bool operator<(const DIEAbbrev &DA) const {
      return Data < DA.Data;
    }

    // Accessors
    unsigned getTag()                 const { return Data[0]; }
    unsigned getChildrenFlag()        const { return Data[1]; }
    unsigned getAttribute(unsigned i) const { return Data[2 + 2 * i + 0]; }
    unsigned getForm(unsigned i)      const { return Data[2 + 2 * i + 1]; }
  };

  //===--------------------------------------------------------------------===//
  // DIEValue - A debug information entry value.
  //
  class DwarfWriter; 
  class DIEValue {
  public:
    enum {
      isInteger,
      isString,
      isLabel,
      isDelta
    };
    
    unsigned Type;                      // Type of the value
    
    DIEValue(unsigned T) : Type(T) {}
    virtual ~DIEValue() {}
    
    // Implement isa/cast/dyncast.
    static bool classof(const DIEValue *) { return true; }
    
    /// EmitValue - Emit value via the Dwarf writer.
    ///
    virtual void EmitValue(const DwarfWriter &DW, unsigned Form) const = 0;
    
    /// SizeOf - Return the size of a value in bytes.
    ///
    virtual unsigned SizeOf(const DwarfWriter &DW, unsigned Form) const = 0;
  };

  //===--------------------------------------------------------------------===//
  // DWInteger - An integer value DIE.
  // 
  class DIEInteger : public DIEValue {
  private:
    int Value;
    
  public:
    DIEInteger(int V) : DIEValue(isInteger), Value(V) {}

    // Implement isa/cast/dyncast.
    static bool classof(const DIEInteger *) { return true; }
    static bool classof(const DIEValue *V)  { return V->Type == isInteger; }
    
    /// EmitValue - Emit integer of appropriate size.
    ///
    virtual void EmitValue(const DwarfWriter &DW, unsigned Form) const;
    
    /// SizeOf - Determine size of integer value in bytes.
    ///
    virtual unsigned SizeOf(const DwarfWriter &DW, unsigned Form) const;
  };

  //===--------------------------------------------------------------------===//
  // DIEString - A string value DIE.
  // 
  struct DIEString : public DIEValue {
    const std::string Value;
    
    DIEString(const std::string &V) : DIEValue(isString), Value(V) {}

    // Implement isa/cast/dyncast.
    static bool classof(const DIEString *) { return true; }
    static bool classof(const DIEValue *V) { return V->Type == isString; }
    
    /// EmitValue - Emit string value.
    ///
    virtual void EmitValue(const DwarfWriter &DW, unsigned Form) const;
    
    /// SizeOf - Determine size of string value in bytes.
    ///
    virtual unsigned SizeOf(const DwarfWriter &DW, unsigned Form) const;
  };

  //===--------------------------------------------------------------------===//
  // DIELabel - A simple label expression DIE.
  //
  struct DIELabel : public DIEValue {
    const DWLabel Value;
    
    DIELabel(const DWLabel &V) : DIEValue(isLabel), Value(V) {}

    // Implement isa/cast/dyncast.
    static bool classof(const DWLabel *)   { return true; }
    static bool classof(const DIEValue *V) { return V->Type == isLabel; }
    
    /// EmitValue - Emit label value.
    ///
    virtual void EmitValue(const DwarfWriter &DW, unsigned Form) const;
    
    /// SizeOf - Determine size of label value in bytes.
    ///
    virtual unsigned SizeOf(const DwarfWriter &DW, unsigned Form) const;
  };

  //===--------------------------------------------------------------------===//
  // DIEDelta - A simple label difference DIE.
  // 
  struct DIEDelta : public DIEValue {
    const DWLabel Value1;
    const DWLabel Value2;
    
    DIEDelta(const DWLabel &V1, const DWLabel &V2)
    : DIEValue(isDelta), Value1(V1), Value2(V2) {}

    // Implement isa/cast/dyncast.
    static bool classof(const DIEDelta *)  { return true; }
    static bool classof(const DIEValue *V) { return V->Type == isDelta; }
    
    /// EmitValue - Emit delta value.
    ///
    virtual void EmitValue(const DwarfWriter &DW, unsigned Form) const;
    
    /// SizeOf - Determine size of delta value in bytes.
    ///
    virtual unsigned SizeOf(const DwarfWriter &DW, unsigned Form) const;
  };
  
  //===--------------------------------------------------------------------===//
  // DIE - A structured debug information entry.  Has an abbreviation which
  // describes it's organization.
  class DIE {
  private:
    unsigned AbbrevID;                    // Decribing abbreviation ID.
    unsigned Offset;                      // Offset in debug info section
    unsigned Size;                        // Size of instance + children
    std::vector<DIE *> Children;          // Children DIEs
    std::vector<DIEValue *> Values;       // Attributes values
    
  public:
    DIE(unsigned AbbrevID)
    : AbbrevID(AbbrevID)
    , Offset(0)
    , Size(0)
    , Children()
    , Values()
    {}
    ~DIE() {
      for (unsigned i = 0, N = Children.size(); i < N; i++) {
        delete Children[i];
      }

      for (unsigned j = 0, M = Children.size(); j < M; j++) {
        delete Children[j];
      }
    }
    
    // Accessors
    unsigned getAbbrevID()                     const { return AbbrevID; }
    unsigned getOffset()                       const { return Offset; }
    unsigned getSize()                         const { return Size; }
    const std::vector<DIE *> &getChildren()    const { return Children; }
    const std::vector<DIEValue *> &getValues() const { return Values; }
    void setOffset(unsigned O)                 { Offset = O; }
    void setSize(unsigned S)                   { Size = S; }
    
    /// AddValue - Add an attribute value of appropriate type.
    ///
    void AddValue(int Value) {
      Values.push_back(new DIEInteger(Value));
    }
    void AddValue(const std::string &Value) {
      Values.push_back(new DIEString(Value));
    }
    void AddValue(const DWLabel &Value) {
      Values.push_back(new DIELabel(Value));
    }
    void AddValue(const DWLabel &Value1, const DWLabel &Value2) {
      Values.push_back(new DIEDelta(Value1, Value2));
    }
    
    /// SiblingOffset - Return the offset of the debug information entry's
    /// sibling.
    unsigned SiblingOffset() const { return Offset + Size; }
  };

  //===--------------------------------------------------------------------===//
  // Forward declarations.
  //
  class AsmPrinter;
  class MachineDebugInfo;
  
  //===--------------------------------------------------------------------===//
  // DwarfWriter - emits Dwarf debug and exception handling directives.
  //
  class DwarfWriter {
  
  protected:
  
    //===------------------------------------------------------------------===//
    // Core attributes used by the Dwarf  writer.
    //
    
    //
    /// O - Stream to .s file.
    ///
    std::ostream &O;

    /// Asm - Target of Dwarf emission.
    ///
    AsmPrinter *Asm;
    
    /// DebugInfo - Collected debug information.
    ///
    MachineDebugInfo *DebugInfo;
    
    /// didInitial - Flag to indicate if initial emission has been done.
    ///
    bool didInitial;
    
    //===------------------------------------------------------------------===//
    // Attributes used to construct specific Dwarf sections.
    //
    
    /// CompileUnits - All the compile units involved in this build.  The index
    /// of each entry in this vector corresponds to the sources in DebugInfo.
    std::vector<DIE *> CompileUnits;

    /// Abbreviations - A UniqueVector of TAG structure abbreviations.
    ///
    UniqueVector<DIEAbbrev> Abbreviations;
    
    //===------------------------------------------------------------------===//
    // Properties to be set by the derived class ctor, used to configure the
    // Dwarf writer.
    //
    
    /// AddressSize - Size of addresses used in file.
    ///
    unsigned AddressSize;

    /// hasLEB128 - True if target asm supports leb128 directives.
    ///
    bool hasLEB128; /// Defaults to false.
    
    /// hasDotLoc - True if target asm supports .loc directives.
    ///
    bool hasDotLoc; /// Defaults to false.
    
    /// hasDotFile - True if target asm supports .file directives.
    ///
    bool hasDotFile; /// Defaults to false.
    
    /// needsSet - True if target asm can't compute addresses on data
    /// directives.
    bool needsSet; /// Defaults to false.
    
    /// DwarfAbbrevSection - Section directive for Dwarf abbrev.
    ///
    const char *DwarfAbbrevSection; /// Defaults to ".debug_abbrev".

    /// DwarfInfoSection - Section directive for Dwarf info.
    ///
    const char *DwarfInfoSection; /// Defaults to ".debug_info".

    /// DwarfLineSection - Section directive for Dwarf info.
    ///
    const char *DwarfLineSection; /// Defaults to ".debug_line".
    
    /// DwarfFrameSection - Section directive for Dwarf info.
    ///
    const char *DwarfFrameSection; /// Defaults to ".debug_frame".
    
    /// DwarfPubNamesSection - Section directive for Dwarf info.
    ///
    const char *DwarfPubNamesSection; /// Defaults to ".debug_pubnames".
    
    /// DwarfPubTypesSection - Section directive for Dwarf info.
    ///
    const char *DwarfPubTypesSection; /// Defaults to ".debug_pubtypes".
    
    /// DwarfStrSection - Section directive for Dwarf info.
    ///
    const char *DwarfStrSection; /// Defaults to ".debug_str".

    /// DwarfLocSection - Section directive for Dwarf info.
    ///
    const char *DwarfLocSection; /// Defaults to ".debug_loc".

    /// DwarfARangesSection - Section directive for Dwarf info.
    ///
    const char *DwarfARangesSection; /// Defaults to ".debug_aranges".

    /// DwarfRangesSection - Section directive for Dwarf info.
    ///
    const char *DwarfRangesSection; /// Defaults to ".debug_ranges".

    /// DwarfMacInfoSection - Section directive for Dwarf info.
    ///
    const char *DwarfMacInfoSection; /// Defaults to ".debug_macinfo".

    /// TextSection - Section directive for standard text.
    ///
    const char *TextSection; /// Defaults to ".text".
    
    /// DataSection - Section directive for standard data.
    ///
    const char *DataSection; /// Defaults to ".data".

    //===------------------------------------------------------------------===//
    // Emission and print routines
    //

public:
    /// getAddressSize - Return the size of a target address in bytes.
    ///
    unsigned getAddressSize() const { return AddressSize; }

    /// PrintHex - Print a value as a hexidecimal value.
    ///
    void PrintHex(int Value) const;

    /// EOL - Print a newline character to asm stream.  If a comment is present
    /// then it will be printed first.  Comments should not contain '\n'.
    void EOL(const std::string &Comment) const;
                                          
    /// EmitULEB128Bytes - Emit an assembler byte data directive to compose an
    /// unsigned leb128 value.
    void EmitULEB128Bytes(unsigned Value) const;
    
    /// EmitSLEB128Bytes - print an assembler byte data directive to compose a
    /// signed leb128 value.
    void EmitSLEB128Bytes(int Value) const;
    
    /// PrintULEB128 - Print a series of hexidecimal values (separated by
    /// commas) representing an unsigned leb128 value.
    void PrintULEB128(unsigned Value) const;

    /// SizeULEB128 - Compute the number of bytes required for an unsigned
    /// leb128 value.
    static unsigned SizeULEB128(unsigned Value);
    
    /// PrintSLEB128 - Print a series of hexidecimal values (separated by
    /// commas) representing a signed leb128 value.
    void PrintSLEB128(int Value) const;
    
    /// SizeSLEB128 - Compute the number of bytes required for a signed leb128
    /// value.
    static unsigned SizeSLEB128(int Value);
    
    /// EmitByte - Emit a byte directive and value.
    ///
    void EmitByte(int Value) const;

    /// EmitShort - Emit a short directive and value.
    ///
    void EmitShort(int Value) const;

    /// EmitLong - Emit a long directive and value.
    ///
    void EmitLong(int Value) const;
    
    /// EmitString - Emit a string with quotes and a null terminator.
    /// Special characters are emitted properly. (Eg. '\t')
    void EmitString(const std::string &String) const;

    /// PrintLabelName - Print label name in form used by Dwarf writer.
    ///
    void PrintLabelName(DWLabel Label) const {
      PrintLabelName(Label.Tag, Label.Number);
    }
    void PrintLabelName(const char *Tag, unsigned Number) const;
    
    /// EmitLabel - Emit location label for internal use by Dwarf.
    ///
    void EmitLabel(DWLabel Label) const {
      EmitLabel(Label.Tag, Label.Number);
    }
    void EmitLabel(const char *Tag, unsigned Number) const;
    
    /// EmitReference - Emit a reference to a label.
    ///
    void EmitReference(DWLabel Label) const {
      EmitReference(Label.Tag, Label.Number);
    }
    void EmitReference(const char *Tag, unsigned Number) const;

    /// EmitDifference - Emit the difference between two labels.  Some
    /// assemblers do not behave with absolute expressions with data directives,
    /// so there is an option (needsSet) to use an intermediary set expression.
    void EmitDifference(DWLabel Label1, DWLabel Label2) const {
      EmitDifference(Label1.Tag, Label1.Number, Label2.Tag, Label2.Number);
    }
    void EmitDifference(const char *Tag1, unsigned Number1,
                        const char *Tag2, unsigned Number2) const;
                                   
private:
    /// NewDIE - Construct a new structured debug information entry.
    ///  
    DIE *NewDIE(const unsigned char *AbbrevData);

    /// NewCompileUnit - Create new compile unit information.
    ///
    DIE *NewCompileUnit(const std::string &Directory,
                        const std::string &SourceName);

    /// EmitInitial - Emit initial Dwarf declarations.
    ///
    void EmitInitial() const;
    
    /// EmitDIE - Recusively Emits a debug information entry.
    ///
    void EmitDIE(DIE *Die) const;
    
    /// SizeAndOffsetDie - Compute the size and offset of a DIE.
    ///
    unsigned SizeAndOffsetDie(DIE *Die, unsigned Offset) const;

    /// SizeAndOffsets - Compute the size and offset of all the DIEs.
    ///
    void SizeAndOffsets();
    
    /// EmitDebugInfo - Emit the debug info section.
    ///
    void EmitDebugInfo() const;
    
    /// EmitAbbreviations - Emit the abbreviation section.
    ///
    void EmitAbbreviations() const;
    
    /// EmitDebugLines - Emit source line information.
    ///
    void EmitDebugLines() const;

    /// EmitDebugFrame - Emit info into a debug frame section.
    ///
    void EmitDebugFrame();
    
    /// EmitDebugPubNames - Emit info into a debug pubnames section.
    ///
    void EmitDebugPubNames();
    
    /// EmitDebugPubTypes - Emit info into a debug pubtypes section.
    ///
    void EmitDebugPubTypes();
    
    /// EmitDebugStr - Emit info into a debug str section.
    ///
    void EmitDebugStr();
    
    /// EmitDebugLoc - Emit info into a debug loc section.
    ///
    void EmitDebugLoc();
    
    /// EmitDebugARanges - Emit info into a debug aranges section.
    ///
    void EmitDebugARanges();
    
    /// EmitDebugRanges - Emit info into a debug ranges section.
    ///
    void EmitDebugRanges();
    
    /// EmitDebugMacInfo - Emit info into a debug macinfo section.
    ///
    void EmitDebugMacInfo();
    
    /// ShouldEmitDwarf - Returns true if Dwarf declarations should be made.
    /// When called it also checks to see if debug info is newly available.  if
    /// so the initial Dwarf headers are emitted.
    bool ShouldEmitDwarf();
    
  public:
    
    DwarfWriter(std::ostream &o, AsmPrinter *ap);
    virtual ~DwarfWriter();
    
    /// SetDebugInfo - Set DebugInfo when it's known that pass manager has
    /// created it.  Set by the target AsmPrinter.
    void SetDebugInfo(MachineDebugInfo *di) { DebugInfo = di; }
    
    //===------------------------------------------------------------------===//
    // Main enties.
    //
    
    /// BeginModule - Emit all Dwarf sections that should come prior to the
    /// content.
    void BeginModule();
    
    /// EndModule - Emit all Dwarf sections that should come after the content.
    ///
    void EndModule();
    
    /// BeginFunction - Gather pre-function debug information.
    ///
    void BeginFunction();
    
    /// EndFunction - Gather and emit post-function debug information.
    ///
    void EndFunction();
  };

} // end llvm namespace

#endif
