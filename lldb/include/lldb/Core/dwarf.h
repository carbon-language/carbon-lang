//===-- dwarf.h -------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef DebugBase_dwarf_h_
#define DebugBase_dwarf_h_

#include <stdint.h>
#include <stdbool.h>

typedef uint32_t    dw_uleb128_t;
typedef int32_t     dw_sleb128_t;
typedef uint16_t    dw_attr_t;
typedef uint8_t     dw_form_t;
typedef uint16_t    dw_tag_t;
typedef uint64_t    dw_addr_t;      // Dwarf address define that must be big enough for any addresses in the compile units that get parsed

#ifdef DWARFUTILS_DWARF64
#define DWARF_REF_ADDR_SIZE     8
typedef uint64_t    dw_offset_t;    // Dwarf Debug Information Entry offset for any offset into the file
#else
#define DWARF_REF_ADDR_SIZE     4
typedef uint32_t    dw_offset_t;    // Dwarf Debug Information Entry offset for any offset into the file
#endif

/* Constants */
#define DW_INVALID_ADDRESS                  (~(dw_addr_t)0)
#define DW_INVALID_OFFSET                   (~(dw_offset_t)0)
#define DW_INVALID_INDEX                    0xFFFFFFFFul


/* [7.5.4] Figure 16 "Tag Encodings" (pp. 125-127) in DWARFv3 draft 8 */

#define DW_TAG_array_type                  0x1
#define DW_TAG_class_type                  0x2
#define DW_TAG_entry_point                 0x3
#define DW_TAG_enumeration_type            0x4
#define DW_TAG_formal_parameter            0x5
#define DW_TAG_imported_declaration        0x8
#define DW_TAG_label                       0xA
#define DW_TAG_lexical_block               0xB
#define DW_TAG_member                      0xD
#define DW_TAG_pointer_type                0xF
#define DW_TAG_reference_type             0x10
#define DW_TAG_compile_unit               0x11
#define DW_TAG_string_type                0x12
#define DW_TAG_structure_type             0x13
#define DW_TAG_subroutine_type            0x15
#define DW_TAG_typedef                    0x16
#define DW_TAG_union_type                 0x17
#define DW_TAG_unspecified_parameters     0x18
#define DW_TAG_variant                    0x19
#define DW_TAG_common_block               0x1A
#define DW_TAG_common_inclusion           0x1B
#define DW_TAG_inheritance                0x1C
#define DW_TAG_inlined_subroutine         0x1D
#define DW_TAG_module                     0x1E
#define DW_TAG_ptr_to_member_type         0x1F
#define DW_TAG_set_type                   0x20
#define DW_TAG_subrange_type              0x21
#define DW_TAG_with_stmt                  0x22
#define DW_TAG_access_declaration         0x23
#define DW_TAG_base_type                  0x24
#define DW_TAG_catch_block                0x25
#define DW_TAG_const_type                 0x26
#define DW_TAG_constant                   0x27
#define DW_TAG_enumerator                 0x28
#define DW_TAG_file_type                  0x29
#define DW_TAG_friend                     0x2A
#define DW_TAG_namelist                   0x2B
#define DW_TAG_namelist_item              0x2C
#define DW_TAG_packed_type                0x2D
#define DW_TAG_subprogram                 0x2E
#define DW_TAG_template_type_parameter    0x2F
#define DW_TAG_template_value_parameter   0x30
#define DW_TAG_thrown_type                0x31
#define DW_TAG_try_block                  0x32
#define DW_TAG_variant_part               0x33
#define DW_TAG_variable                   0x34
#define DW_TAG_volatile_type              0x35
#define DW_TAG_dwarf_procedure            0x36
#define DW_TAG_restrict_type              0x37
#define DW_TAG_interface_type             0x38
#define DW_TAG_namespace                  0x39
#define DW_TAG_imported_module            0x3A
#define DW_TAG_unspecified_type           0x3B
#define DW_TAG_partial_unit               0x3C
#define DW_TAG_imported_unit              0x3D
#define DW_TAG_condition                  0x3F
#define DW_TAG_shared_type                0x40
#define DW_TAG_lo_user                  0x4080
#define DW_TAG_hi_user                  0xFFFF

/* [7.5.4] Figure 17 "Child determination encodings" (p. 128) in DWARFv3 draft 8 */

#define DW_CHILDREN_no  0x0
#define DW_CHILDREN_yes 0x1

/* [7.5.4] Figure 18 "Attribute encodings" (pp. 129-132) in DWARFv3 draft 8 */

#define DW_AT_sibling                         0x1
#define DW_AT_location                        0x2
#define DW_AT_name                            0x3
#define DW_AT_ordering                        0x9
#define DW_AT_byte_size                       0xB
#define DW_AT_bit_offset                      0xC
#define DW_AT_bit_size                        0xD
#define DW_AT_stmt_list                      0x10
#define DW_AT_low_pc                         0x11
#define DW_AT_high_pc                        0x12
#define DW_AT_language                       0x13
#define DW_AT_discr                          0x15
#define DW_AT_discr_value                    0x16
#define DW_AT_visibility                     0x17
#define DW_AT_import                         0x18
#define DW_AT_string_length                  0x19
#define DW_AT_common_reference               0x1A
#define DW_AT_comp_dir                       0x1B
#define DW_AT_const_value                    0x1C
#define DW_AT_containing_type                0x1D
#define DW_AT_default_value                  0x1E
#define DW_AT_inline                         0x20
#define DW_AT_is_optional                    0x21
#define DW_AT_lower_bound                    0x22
#define DW_AT_producer                       0x25
#define DW_AT_prototyped                     0x27
#define DW_AT_return_addr                    0x2A
#define DW_AT_start_scope                    0x2C
#define DW_AT_bit_stride                     0x2E
#define DW_AT_upper_bound                    0x2F
#define DW_AT_abstract_origin                0x31
#define DW_AT_accessibility                  0x32
#define DW_AT_address_class                  0x33
#define DW_AT_artificial                     0x34
#define DW_AT_base_types                     0x35
#define DW_AT_calling_convention             0x36
#define DW_AT_count                          0x37
#define DW_AT_data_member_location           0x38
#define DW_AT_decl_column                    0x39
#define DW_AT_decl_file                      0x3A
#define DW_AT_decl_line                      0x3B
#define DW_AT_declaration                    0x3C
#define DW_AT_discr_list                     0x3D
#define DW_AT_encoding                       0x3E
#define DW_AT_external                       0x3F
#define DW_AT_frame_base                     0x40
#define DW_AT_friend                         0x41
#define DW_AT_identifier_case                0x42
#define DW_AT_macro_info                     0x43
#define DW_AT_namelist_item                  0x44
#define DW_AT_priority                       0x45
#define DW_AT_segment                        0x46
#define DW_AT_specification                  0x47
#define DW_AT_static_link                    0x48
#define DW_AT_type                           0x49
#define DW_AT_use_location                   0x4A
#define DW_AT_variable_parameter             0x4B
#define DW_AT_virtuality                     0x4C
#define DW_AT_vtable_elem_location           0x4D
#define DW_AT_allocated                      0x4E
#define DW_AT_associated                     0x4F
#define DW_AT_data_location                  0x50
#define DW_AT_byte_stride                    0x51
#define DW_AT_entry_pc                       0x52
#define DW_AT_use_UTF8                       0x53
#define DW_AT_extension                      0x54
#define DW_AT_ranges                         0x55
#define DW_AT_trampoline                     0x56
#define DW_AT_call_column                    0x57
#define DW_AT_call_file                      0x58
#define DW_AT_call_line                      0x59
#define DW_AT_description                    0x5A
#define DW_AT_binary_scale                   0x5B
#define DW_AT_decimal_scale                  0x5C
#define DW_AT_small                          0x5D
#define DW_AT_decimal_sign                   0x5E
#define DW_AT_digit_count                    0x5F
#define DW_AT_picture_string                 0x60
#define DW_AT_mutable                        0x61
#define DW_AT_threads_scaled                 0x62
#define DW_AT_explicit                       0x63
#define DW_AT_object_pointer                 0x64
#define DW_AT_endianity                      0x65
#define DW_AT_elemental                      0x66
#define DW_AT_pure                           0x67
#define DW_AT_recursive                      0x68
#define DW_AT_lo_user                      0x2000
#define DW_AT_hi_user                      0x3FFF
#define DW_AT_MIPS_fde                     0x2001
#define DW_AT_MIPS_loop_begin              0x2002
#define DW_AT_MIPS_tail_loop_begin         0x2003
#define DW_AT_MIPS_epilog_begin            0x2004
#define DW_AT_MIPS_loop_unroll_factor      0x2005
#define DW_AT_MIPS_software_pipeline_depth 0x2006
#define DW_AT_MIPS_linkage_name            0x2007
#define DW_AT_MIPS_stride                  0x2008
#define DW_AT_MIPS_abstract_name           0x2009
#define DW_AT_MIPS_clone_origin            0x200A
#define DW_AT_MIPS_has_inlines             0x200B
/* GNU extensions.  */
#define DW_AT_sf_names                     0x2101
#define DW_AT_src_info                     0x2102
#define DW_AT_mac_info                     0x2103
#define DW_AT_src_coords                   0x2104
#define DW_AT_body_begin                   0x2105
#define DW_AT_body_end                     0x2106
#define DW_AT_GNU_vector                   0x2107

#define DW_AT_APPLE_repository_file             0x2501
#define DW_AT_APPLE_repository_type             0x2502
#define DW_AT_APPLE_repository_name             0x2503
#define DW_AT_APPLE_repository_specification    0x2504
#define DW_AT_APPLE_repository_import           0x2505
#define DW_AT_APPLE_repository_abstract_origin  0x2506
#define DW_AT_APPLE_optimized                   0x3FE1
#define DW_AT_APPLE_flags                       0x3FE2
#define DW_AT_APPLE_isa                         0x3FE3
#define DW_AT_APPLE_block                       0x3FE4

/* [7.5.4] Figure 19 "Attribute form encodings" (pp. 133-134) in DWARFv3 draft 8 */

#define DW_FORM_addr       0x1
#define DW_FORM_block2     0x3
#define DW_FORM_block4     0x4
#define DW_FORM_data2      0x5
#define DW_FORM_data4      0x6
#define DW_FORM_data8      0x7
#define DW_FORM_string     0x8
#define DW_FORM_block      0x9
#define DW_FORM_block1     0xA
#define DW_FORM_data1      0xB
#define DW_FORM_flag       0xC
#define DW_FORM_sdata      0xD
#define DW_FORM_strp       0xE
#define DW_FORM_udata      0xF
#define DW_FORM_ref_addr  0x10
#define DW_FORM_ref1      0x11
#define DW_FORM_ref2      0x12
#define DW_FORM_ref4      0x13
#define DW_FORM_ref8      0x14
#define DW_FORM_ref_udata 0x15
#define DW_FORM_indirect  0x16 // cf section 7.5.3, "Abbreviations Tables", p. 119 DWARFv3 draft 8
#define DW_FORM_APPLE_db_str    0x50 // read same as udata, but refers to string in repository

/* [7.7.1] Figure 22 "DWARF operation encodings" (pp. 136-139) in DWARFv3 draft 8 */

#define DW_OP_addr                 0x3 // constant address (size target specific)
#define DW_OP_deref                0x6
#define DW_OP_const1u              0x8 // 1-byte constant
#define DW_OP_const1s              0x9 // 1-byte constant
#define DW_OP_const2u              0xA // 2-byte constant
#define DW_OP_const2s              0xB // 2-byte constant
#define DW_OP_const4u              0xC // 4-byte constant
#define DW_OP_const4s              0xD // 4-byte constant
#define DW_OP_const8u              0xE // 8-byte constant
#define DW_OP_const8s              0xF // 8-byte constant
#define DW_OP_constu              0x10 // ULEB128 constant
#define DW_OP_consts              0x11 // SLEB128 constant
#define DW_OP_dup                 0x12
#define DW_OP_drop                0x13
#define DW_OP_over                0x14
#define DW_OP_pick                0x15 // 1-byte stack index
#define DW_OP_swap                0x16
#define DW_OP_rot                 0x17
#define DW_OP_xderef              0x18
#define DW_OP_abs                 0x19
#define DW_OP_and                 0x1A
#define DW_OP_div                 0x1B
#define DW_OP_minus               0x1C
#define DW_OP_mod                 0x1D
#define DW_OP_mul                 0x1E
#define DW_OP_neg                 0x1F
#define DW_OP_not                 0x20
#define DW_OP_or                  0x21
#define DW_OP_plus                0x22
#define DW_OP_plus_uconst         0x23 // ULEB128 addend
#define DW_OP_shl                 0x24
#define DW_OP_shr                 0x25
#define DW_OP_shra                0x26
#define DW_OP_xor                 0x27
#define DW_OP_skip                0x2F // signed 2-byte constant
#define DW_OP_bra                 0x28 // signed 2-byte constant
#define DW_OP_eq                  0x29
#define DW_OP_ge                  0x2A
#define DW_OP_gt                  0x2B
#define DW_OP_le                  0x2C
#define DW_OP_lt                  0x2D
#define DW_OP_ne                  0x2E
#define DW_OP_lit0                0x30 // Literal 0
#define DW_OP_lit1                0x31 // Literal 1
#define DW_OP_lit2                0x32 // Literal 2
#define DW_OP_lit3                0x33 // Literal 3
#define DW_OP_lit4                0x34 // Literal 4
#define DW_OP_lit5                0x35 // Literal 5
#define DW_OP_lit6                0x36 // Literal 6
#define DW_OP_lit7                0x37 // Literal 7
#define DW_OP_lit8                0x38 // Literal 8
#define DW_OP_lit9                0x39 // Literal 9
#define DW_OP_lit10               0x3A // Literal 10
#define DW_OP_lit11               0x3B // Literal 11
#define DW_OP_lit12               0x3C // Literal 12
#define DW_OP_lit13               0x3D // Literal 13
#define DW_OP_lit14               0x3E // Literal 14
#define DW_OP_lit15               0x3F // Literal 15
#define DW_OP_lit16               0x40 // Literal 16
#define DW_OP_lit17               0x41 // Literal 17
#define DW_OP_lit18               0x42 // Literal 18
#define DW_OP_lit19               0x43 // Literal 19
#define DW_OP_lit20               0x44 // Literal 20
#define DW_OP_lit21               0x45 // Literal 21
#define DW_OP_lit22               0x46 // Literal 22
#define DW_OP_lit23               0x47 // Literal 23
#define DW_OP_lit24               0x48 // Literal 24
#define DW_OP_lit25               0x49 // Literal 25
#define DW_OP_lit26               0x4A // Literal 26
#define DW_OP_lit27               0x4B // Literal 27
#define DW_OP_lit28               0x4C // Literal 28
#define DW_OP_lit29               0x4D // Literal 29
#define DW_OP_lit30               0x4E // Literal 30
#define DW_OP_lit31               0x4F // Literal 31
#define DW_OP_reg0                0x50 // Contents of reg0
#define DW_OP_reg1                0x51 // Contents of reg1
#define DW_OP_reg2                0x52 // Contents of reg2
#define DW_OP_reg3                0x53 // Contents of reg3
#define DW_OP_reg4                0x54 // Contents of reg4
#define DW_OP_reg5                0x55 // Contents of reg5
#define DW_OP_reg6                0x56 // Contents of reg6
#define DW_OP_reg7                0x57 // Contents of reg7
#define DW_OP_reg8                0x58 // Contents of reg8
#define DW_OP_reg9                0x59 // Contents of reg9
#define DW_OP_reg10               0x5A // Contents of reg10
#define DW_OP_reg11               0x5B // Contents of reg11
#define DW_OP_reg12               0x5C // Contents of reg12
#define DW_OP_reg13               0x5D // Contents of reg13
#define DW_OP_reg14               0x5E // Contents of reg14
#define DW_OP_reg15               0x5F // Contents of reg15
#define DW_OP_reg16               0x60 // Contents of reg16
#define DW_OP_reg17               0x61 // Contents of reg17
#define DW_OP_reg18               0x62 // Contents of reg18
#define DW_OP_reg19               0x63 // Contents of reg19
#define DW_OP_reg20               0x64 // Contents of reg20
#define DW_OP_reg21               0x65 // Contents of reg21
#define DW_OP_reg22               0x66 // Contents of reg22
#define DW_OP_reg23               0x67 // Contents of reg23
#define DW_OP_reg24               0x68 // Contents of reg24
#define DW_OP_reg25               0x69 // Contents of reg25
#define DW_OP_reg26               0x6A // Contents of reg26
#define DW_OP_reg27               0x6B // Contents of reg27
#define DW_OP_reg28               0x6C // Contents of reg28
#define DW_OP_reg29               0x6D // Contents of reg29
#define DW_OP_reg30               0x6E // Contents of reg30
#define DW_OP_reg31               0x6F // Contents of reg31
#define DW_OP_breg0               0x70 // base register 0 + SLEB128 offset
#define DW_OP_breg1               0x71 // base register 1 + SLEB128 offset
#define DW_OP_breg2               0x72 // base register 2 + SLEB128 offset
#define DW_OP_breg3               0x73 // base register 3 + SLEB128 offset
#define DW_OP_breg4               0x74 // base register 4 + SLEB128 offset
#define DW_OP_breg5               0x75 // base register 5 + SLEB128 offset
#define DW_OP_breg6               0x76 // base register 6 + SLEB128 offset
#define DW_OP_breg7               0x77 // base register 7 + SLEB128 offset
#define DW_OP_breg8               0x78 // base register 8 + SLEB128 offset
#define DW_OP_breg9               0x79 // base register 9 + SLEB128 offset
#define DW_OP_breg10              0x7A // base register 10 + SLEB128 offset
#define DW_OP_breg11              0x7B // base register 11 + SLEB128 offset
#define DW_OP_breg12              0x7C // base register 12 + SLEB128 offset
#define DW_OP_breg13              0x7D // base register 13 + SLEB128 offset
#define DW_OP_breg14              0x7E // base register 14 + SLEB128 offset
#define DW_OP_breg15              0x7F // base register 15 + SLEB128 offset
#define DW_OP_breg16              0x80 // base register 16 + SLEB128 offset
#define DW_OP_breg17              0x81 // base register 17 + SLEB128 offset
#define DW_OP_breg18              0x82 // base register 18 + SLEB128 offset
#define DW_OP_breg19              0x83 // base register 19 + SLEB128 offset
#define DW_OP_breg20              0x84 // base register 20 + SLEB128 offset
#define DW_OP_breg21              0x85 // base register 21 + SLEB128 offset
#define DW_OP_breg22              0x86 // base register 22 + SLEB128 offset
#define DW_OP_breg23              0x87 // base register 23 + SLEB128 offset
#define DW_OP_breg24              0x88 // base register 24 + SLEB128 offset
#define DW_OP_breg25              0x89 // base register 25 + SLEB128 offset
#define DW_OP_breg26              0x8A // base register 26 + SLEB128 offset
#define DW_OP_breg27              0x8B // base register 27 + SLEB128 offset
#define DW_OP_breg28              0x8C // base register 28 + SLEB128 offset
#define DW_OP_breg29              0x8D // base register 29 + SLEB128 offset
#define DW_OP_breg30              0x8E // base register 30 + SLEB128 offset
#define DW_OP_breg31              0x8F // base register 31 + SLEB128 offset
#define DW_OP_regx                0x90 // ULEB128 register
#define DW_OP_fbreg               0x91 // SLEB128 offset
#define DW_OP_bregx               0x92 // ULEB128 register followed by SLEB128 offset
#define DW_OP_piece               0x93 // ULEB128 size of piece addressed
#define DW_OP_deref_size          0x94 // 1-byte size of data retrieved
#define DW_OP_xderef_size         0x95 // 1-byte size of data retrieved
#define DW_OP_nop                 0x96
#define DW_OP_push_object_address 0x97
#define DW_OP_call2               0x98 // 2-byte offset of DIE
#define DW_OP_call4               0x99 // 4-byte offset of DIE
#define DW_OP_call_ref            0x9A // 4- or 8-byte offset of DIE
#define DW_OP_lo_user             0xE0
#define DW_OP_APPLE_array_ref     0xEE // first pops index, then pops array; pushes array[index]
#define DW_OP_APPLE_extern        0xEF // ULEB128 index of external object (i.e., an entity from the program that was used in the expression)
#define DW_OP_APPLE_uninit        0xF0
#define DW_OP_APPLE_assign        0xF1 // pops value off and assigns it to second item on stack (2nd item must have assignable context)
#define DW_OP_APPLE_address_of    0xF2 // gets the address of the top stack item (top item must be a variable, or have value_type that is an address already)
#define DW_OP_APPLE_value_of      0xF3 // pops the value off the stack and pushes the value of that object (top item must be a variable, or expression local)
#define DW_OP_APPLE_deref_type    0xF4 // gets the address of the top stack item (top item must be a variable, or a clang type)
#define DW_OP_APPLE_expr_local    0xF5 // ULEB128 expression local index
#define DW_OP_APPLE_constf        0xF6 // 1 byte float size, followed by constant float data
#define DW_OP_APPLE_scalar_cast   0xF7 // Cast top of stack to 2nd in stack's type leaving all items in place
#define DW_OP_APPLE_clang_cast    0xF8 // pointer size clang::Type * off the stack and cast top stack item to this type
#define DW_OP_APPLE_clear         0xFE // clears the entire expression stack, ok if the stack is empty
#define DW_OP_APPLE_error         0xFF // Stops expression evaluation and returns an error (no args)
#define DW_OP_hi_user             0xFF

/* [7.8] Figure 23 "Base type encoding values" (pp. 140-141) in DWARFv3 draft 8 */

#define DW_ATE_address          0x1
#define DW_ATE_boolean          0x2
#define DW_ATE_complex_float    0x3
#define DW_ATE_float            0x4
#define DW_ATE_signed           0x5
#define DW_ATE_signed_char      0x6
#define DW_ATE_unsigned         0x7
#define DW_ATE_unsigned_char    0x8
#define DW_ATE_imaginary_float  0x9
#define DW_ATE_lo_user         0x80
#define DW_ATE_hi_user         0xFF

/* [7.9] Figure 24 "Accessibility encodings" (p. 141) in DWARFv3 draft 8 */

#define DW_ACCESS_public    0x1
#define DW_ACCESS_protected 0x2
#define DW_ACCESS_private   0x3

/* [7.10] Figure 25 "Visibility encodings" (p. 142) in DWARFv3 draft 8 */

#define DW_VIS_local     0x1
#define DW_VIS_exported  0x2
#define DW_VIS_qualified 0x3

/* [7.11] Figure 26 "Virtuality encodings" (p. 142) in DWARFv3 draft 8 */

#define DW_VIRTUALITY_none         0x0
#define DW_VIRTUALITY_virtual      0x1
#define DW_VIRTUALITY_pure_virtual 0x2


/* [7.12] Figure 27 "Language encodings" (p. 143) in DWARFv3 draft 8 */

#define DW_LANG_C89               0x1
#define DW_LANG_C                 0x2
#define DW_LANG_Ada83             0x3
#define DW_LANG_C_plus_plus       0x4
#define DW_LANG_Cobol74           0x5
#define DW_LANG_Cobol85           0x6
#define DW_LANG_Fortran77         0x7
#define DW_LANG_Fortran90         0x8
#define DW_LANG_Pascal83          0x9
#define DW_LANG_Modula2           0xA
#define DW_LANG_Java              0xB
#define DW_LANG_C99               0xC
#define DW_LANG_Ada95             0xD
#define DW_LANG_Fortran95         0xE
#define DW_LANG_PLI               0xF
#define DW_LANG_lo_user        0x8000
#define DW_LANG_hi_user        0xFFFF
#define DW_LANG_MIPS_Assembler 0x8001

/* [7.13], "Address Class Encodings" (p. 144) in DWARFv3 draft 8 */

#define DW_ADDR_none 0x0

/* [7.14] Figure 28 "Identifier case encodings" (p. 144) in DWARFv3 draft 8 */

#define DW_ID_case_sensitive   0x0
#define DW_ID_up_case          0x1
#define DW_ID_down_case        0x2
#define DW_ID_case_insensitive 0x3

/* [7.15] Figure 29 "Calling convention encodings" (p. 144) in DWARFv3 draft 8 */

#define DW_CC_normal   0x1
#define DW_CC_program  0x2
#define DW_CC_nocall   0x3
#define DW_CC_lo_user 0x40
#define DW_CC_hi_user 0xFF

/* [7.16] Figure 30 "Inline encodings" (p. 145) in DWARFv3 draft 8 */

#define DW_INL_not_inlined          0x0
#define DW_INL_inlined              0x1
#define DW_INL_declared_not_inlined 0x2
#define DW_INL_declared_inlined     0x3

/* [7.17] Figure 31 "Ordering encodings" (p. 145) in DWARFv3 draft 8 */

#define DW_ORD_row_major 0x0
#define DW_ORD_col_major 0x1

/* [7.18] Figure 32 "Discriminant descriptor encodings" (p. 146) in DWARFv3 draft 8 */

#define DW_DSC_label 0x0
#define DW_DSC_range 0x1

/* [7.21] Figure 33 "Line Number Standard Opcode Encodings" (pp. 148-149) in DWARFv3 draft 8 */

#define DW_LNS_copy               0x1
#define DW_LNS_advance_pc         0x2
#define DW_LNS_advance_line       0x3
#define DW_LNS_set_file           0x4
#define DW_LNS_set_column         0x5
#define DW_LNS_negate_stmt        0x6
#define DW_LNS_set_basic_block    0x7
#define DW_LNS_const_add_pc       0x8
#define DW_LNS_fixed_advance_pc   0x9
#define DW_LNS_set_prologue_end   0xA
#define DW_LNS_set_epilogue_begin 0xB
#define DW_LNS_set_isa            0xC

/* [7.21] Figure 34 "Line Number Extended Opcode Encodings" (p. 149) in DWARFv3 draft 8 */

#define DW_LNE_end_sequence  0x1
#define DW_LNE_set_address   0x2
#define DW_LNE_define_file   0x3
#define DW_LNE_lo_user      0x80
#define DW_LNE_hi_user      0xFF

/* [7.22] Figure 35 "Macinfo Type Encodings" (p. 150) in DWARFv3 draft 8 */

#define DW_MACINFO_define      0x1
#define DW_MACINFO_undef       0x2
#define DW_MACINFO_start_file  0x3
#define DW_MACINFO_end_file    0x4
#define DW_MACINFO_vendor_ext 0xFF

/* [7.23] Figure 36 "Call frame instruction encodings" (pp. 151-152) in DWARFv3 draft 8 */

#define DW_CFA_advance_loc        0x40 // high 2 bits are 0x1, lower 6 bits are delta
#define DW_CFA_offset             0x80 // high 2 bits are 0x2, lower 6 bits are register
#define DW_CFA_restore            0xC0 // high 2 bits are 0x3, lower 6 bits are register
#define DW_CFA_nop                 0x0
#define DW_CFA_set_loc             0x1
#define DW_CFA_advance_loc1        0x2
#define DW_CFA_advance_loc2        0x3
#define DW_CFA_advance_loc4        0x4
#define DW_CFA_offset_extended     0x5
#define DW_CFA_restore_extended    0x6
#define DW_CFA_undefined           0x7
#define DW_CFA_same_value          0x8
#define DW_CFA_register            0x9
#define DW_CFA_remember_state      0xA
#define DW_CFA_restore_state       0xB
#define DW_CFA_def_cfa             0xC
#define DW_CFA_def_cfa_register    0xD
#define DW_CFA_def_cfa_offset      0xE
#define DW_CFA_def_cfa_expression  0xF
#define DW_CFA_expression         0x10
#define DW_CFA_offset_extended_sf 0x11
#define DW_CFA_def_cfa_sf         0x12
#define DW_CFA_def_cfa_offset_sf  0x13
#define DW_CFA_val_offset         0x14
#define DW_CFA_val_offset_sf      0x15
#define DW_CFA_val_expression     0x16
#define DW_CFA_lo_user            0x1C
#define DW_CFA_hi_user            0x3F

/* FSF exception handling Pointer-Encoding constants (CFI augmentation) -- "DW_EH_PE_..." in the FSF sources */

#define DW_GNU_EH_PE_absptr    0x0
#define DW_GNU_EH_PE_uleb128   0x1
#define DW_GNU_EH_PE_udata2    0x2
#define DW_GNU_EH_PE_udata4    0x3
#define DW_GNU_EH_PE_udata8    0x4
#define DW_GNU_EH_PE_sleb128   0x9
#define DW_GNU_EH_PE_sdata2    0xA
#define DW_GNU_EH_PE_sdata4    0xB
#define DW_GNU_EH_PE_sdata8    0xC
#define DW_GNU_EH_PE_signed    0x8
#define DW_GNU_EH_PE_MASK_ENCODING 0x0F
#define DW_GNU_EH_PE_pcrel    0x10
#define DW_GNU_EH_PE_textrel  0x20
#define DW_GNU_EH_PE_datarel  0x30
#define DW_GNU_EH_PE_funcrel  0x40
#define DW_GNU_EH_PE_aligned  0x50
#define DW_GNU_EH_PE_indirect 0x80
#define DW_GNU_EH_PE_omit     0xFF

#endif  // DebugBase_dwarf_h_
