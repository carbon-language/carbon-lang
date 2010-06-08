//===-- DWARFDefines.c ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DWARFDefines.h"
#include <stdio.h>

#define DW_TAG_PREFIX   "TAG_"
#define DW_AT_PREFIX    " AT_"
#define DW_FORM_PREFIX  "FORM_"

/* [7.5.4] Figure 16 "Tag Encodings" (pp. 125-127) in DWARFv3 draft 8 */

const char *
DW_TAG_value_to_name (uint32_t val)
{
  static char invalid[100];
  switch (val) {
    case 0x0000: return DW_TAG_PREFIX "NULL";
    case 0x0001: return DW_TAG_PREFIX "array_type";
    case 0x0002: return DW_TAG_PREFIX "class_type";
    case 0x0003: return DW_TAG_PREFIX "entry_point";
    case 0x0004: return DW_TAG_PREFIX "enumeration_type";
    case 0x0005: return DW_TAG_PREFIX "formal_parameter";
    case 0x0008: return DW_TAG_PREFIX "imported_declaration";
    case 0x000a: return DW_TAG_PREFIX "label";
    case 0x000b: return DW_TAG_PREFIX "lexical_block";
    case 0x000d: return DW_TAG_PREFIX "member";
    case 0x000f: return DW_TAG_PREFIX "pointer_type";
    case 0x0010: return DW_TAG_PREFIX "reference_type";
    case 0x0011: return DW_TAG_PREFIX "compile_unit";
    case 0x0012: return DW_TAG_PREFIX "string_type";
    case 0x0013: return DW_TAG_PREFIX "structure_type";
    case 0x0015: return DW_TAG_PREFIX "subroutine_type";
    case 0x0016: return DW_TAG_PREFIX "typedef";
    case 0x0017: return DW_TAG_PREFIX "union_type";
    case 0x0018: return DW_TAG_PREFIX "unspecified_parameters";
    case 0x0019: return DW_TAG_PREFIX "variant";
    case 0x001a: return DW_TAG_PREFIX "common_block";
    case 0x001b: return DW_TAG_PREFIX "common_inclusion";
    case 0x001c: return DW_TAG_PREFIX "inheritance";
    case 0x001d: return DW_TAG_PREFIX "inlined_subroutine";
    case 0x001e: return DW_TAG_PREFIX "module";
    case 0x001f: return DW_TAG_PREFIX "ptr_to_member_type";
    case 0x0020: return DW_TAG_PREFIX "set_type";
    case 0x0021: return DW_TAG_PREFIX "subrange_type";
    case 0x0022: return DW_TAG_PREFIX "with_stmt";
    case 0x0023: return DW_TAG_PREFIX "access_declaration";
    case 0x0024: return DW_TAG_PREFIX "base_type";
    case 0x0025: return DW_TAG_PREFIX "catch_block";
    case 0x0026: return DW_TAG_PREFIX "const_type";
    case 0x0027: return DW_TAG_PREFIX "constant";
    case 0x0028: return DW_TAG_PREFIX "enumerator";
    case 0x0029: return DW_TAG_PREFIX "file_type";
    case 0x002a: return DW_TAG_PREFIX "friend";
    case 0x002b: return DW_TAG_PREFIX "namelist";
    case 0x002c: return DW_TAG_PREFIX "namelist_item";
    case 0x002d: return DW_TAG_PREFIX "packed_type";
    case 0x002e: return DW_TAG_PREFIX "subprogram";
    case 0x002f: return DW_TAG_PREFIX "template_type_parameter";
    case 0x0030: return DW_TAG_PREFIX "template_value_parameter";
    case 0x0031: return DW_TAG_PREFIX "thrown_type";
    case 0x0032: return DW_TAG_PREFIX "try_block";
    case 0x0033: return DW_TAG_PREFIX "variant_part";
    case 0x0034: return DW_TAG_PREFIX "variable";
    case 0x0035: return DW_TAG_PREFIX "volatile_type";
    case 0x0036: return DW_TAG_PREFIX "dwarf_procedure";
    case 0x0037: return DW_TAG_PREFIX "restrict_type";
    case 0x0038: return DW_TAG_PREFIX "interface_type";
    case 0x0039: return DW_TAG_PREFIX "namespace";
    case 0x003a: return DW_TAG_PREFIX "imported_module";
    case 0x003b: return DW_TAG_PREFIX "unspecified_type";
    case 0x003c: return DW_TAG_PREFIX "partial_unit";
    case 0x003d: return DW_TAG_PREFIX "imported_unit";
//  case 0x003d: return DW_TAG_PREFIX "condition";
    case 0x0040: return DW_TAG_PREFIX "shared_type";
    case 0x4080: return DW_TAG_PREFIX "lo_user";
    case 0xffff: return DW_TAG_PREFIX "hi_user";
    default:
       snprintf (invalid, sizeof(invalid), "Unknown DW_TAG constant: 0x%x", val);
       return invalid;
  }
}

const char *
DW_TAG_value_to_englishy_name (uint32_t val)
{
  static char invalid[100];
  switch (val) {
    case 0x0001: return "array type";
    case 0x0002: return "class type";
    case 0x0003: return "entry point";
    case 0x0004: return "enumeration type";
    case 0x0005: return "formal parameter";
    case 0x0008: return "imported declaration";
    case 0x000a: return "label";
    case 0x000b: return "lexical block";
    case 0x000d: return "member";
    case 0x000f: return "pointer type";
    case 0x0010: return "reference type";
    case 0x0011: return "file";
    case 0x0012: return "string type";
    case 0x0013: return "structure type";
    case 0x0015: return "subroutine type";
    case 0x0016: return "typedef";
    case 0x0017: return "union type";
    case 0x0018: return "unspecified parameters";
    case 0x0019: return "variant";
    case 0x001a: return "common block";
    case 0x001b: return "common inclusion";
    case 0x001c: return "inheritance";
    case 0x001d: return "inlined subroutine";
    case 0x001e: return "module";
    case 0x001f: return "ptr to member type";
    case 0x0020: return "set type";
    case 0x0021: return "subrange type";
    case 0x0022: return "with stmt";
    case 0x0023: return "access declaration";
    case 0x0024: return "base type";
    case 0x0025: return "catch block";
    case 0x0026: return "const type";
    case 0x0027: return "constant";
    case 0x0028: return "enumerator";
    case 0x0029: return "file type";
    case 0x002a: return "friend";
    case 0x002b: return "namelist";
    case 0x002c: return "namelist item";
    case 0x002d: return "packed type";
    case 0x002e: return "function";
    case 0x002f: return "template type parameter";
    case 0x0030: return "template value parameter";
    case 0x0031: return "thrown type";
    case 0x0032: return "try block";
    case 0x0033: return "variant part";
    case 0x0034: return "variable";
    case 0x0035: return "volatile type";
    case 0x0036: return "dwarf procedure";
    case 0x0037: return "restrict type";
    case 0x0038: return "interface type";
    case 0x0039: return "namespace";
    case 0x003a: return "imported module";
    case 0x003b: return "unspecified type";
    case 0x003c: return "partial unit";
    case 0x003d: return "imported unit";
//  case 0x003d: return "condition";
    case 0x0040: return "shared type";
    case 0x4080: return "lo user";
    case 0xffff: return "hi user";
    default:
       snprintf (invalid, sizeof(invalid), "Unknown DW_TAG constant: 0x%x", val);
       return invalid;
  }
}

DRC_class
DW_TAG_value_to_class (uint32_t val)
{
  switch (val) {
    case 0x0001: return 0;
    case 0x0002: return 0;
    case 0x0003: return 0;
    case 0x0004: return 0;
    case 0x0005: return 0;
    case 0x0008: return 0;
    case 0x000a: return 0;
    case 0x000b: return 0;
    case 0x000d: return 0;
    case 0x000f: return 0;
    case 0x0010: return 0;
    case 0x0011: return 0;
    case 0x0012: return 0;
    case 0x0013: return 0;
    case 0x0015: return 0;
    case 0x0016: return 0;
    case 0x0017: return 0;
    case 0x0018: return 0;
    case 0x0019: return 0;
    case 0x001a: return 0;
    case 0x001b: return 0;
    case 0x001c: return 0;
    case 0x001d: return 0;
    case 0x001e: return 0;
    case 0x001f: return 0;
    case 0x0020: return 0;
    case 0x0021: return 0;
    case 0x0022: return 0;
    case 0x0023: return 0;
    case 0x0024: return 0;
    case 0x0025: return 0;
    case 0x0026: return 0;
    case 0x0027: return 0;
    case 0x0028: return 0;
    case 0x0029: return 0;
    case 0x002a: return 0;
    case 0x002b: return 0;
    case 0x002c: return 0;
    case 0x002d: return 0;
    case 0x002e: return 0;
    case 0x002f: return 0;
    case 0x0030: return 0;
    case 0x0031: return 0;
    case 0x0032: return 0;
    case 0x0033: return 0;
    case 0x0034: return 0;
    case 0x0035: return 0;
    case 0x0036: return DRC_DWARFv3;
    case 0x0037: return DRC_DWARFv3;
    case 0x0038: return DRC_DWARFv3;
    case 0x0039: return DRC_DWARFv3;
    case 0x003a: return DRC_DWARFv3;
    case 0x003b: return DRC_DWARFv3;
    case 0x003c: return DRC_DWARFv3;
    case 0x003d: return DRC_DWARFv3;
//  case 0x003d: return DRC_DWARFv3;
    case 0x0040: return DRC_DWARFv3;
    case 0x4080: return 0;
    case 0xffff: return 0;
    default: return 0;
  }
}

/* [7.5.4] Figure 17 "Child determination encodings" (p. 128) in DWARFv3 draft 8 */

const char *
DW_CHILDREN_value_to_name (uint8_t val)
{
  static char invalid[100];
  switch (val) {
    case 0x0: return "DW_CHILDREN_no";
    case 0x1: return "DW_CHILDREN_yes";
    default:
       snprintf (invalid, sizeof(invalid), "Unknown DW_CHILDREN constant: 0x%x", val);
       return invalid;
  }
}

const char *
DW_CHILDREN_value_to_englishy_name (uint8_t val)
{
  static char invalid[100];
  switch (val) {
    case 0x0: return "no";
    case 0x1: return "yes";
    default:
       snprintf (invalid, sizeof(invalid), "Unknown DW_CHILDREN constant: 0x%x", val);
       return invalid;
  }
}

DRC_class
DW_CHILDREN_value_to_class (uint32_t val)
{
  switch (val) {
    case 0x0: return 0;
    case 0x1: return 0;
    default: return 0;
  }
}

/* [7.5.4] Figure 18 "Attribute encodings" (pp. 129-132) in DWARFv3 draft 8 */

const char *
DW_AT_value_to_name (uint32_t val)
{
  static char invalid[100];
  switch (val) {
    case 0x0001: return DW_AT_PREFIX "sibling";
    case 0x0002: return DW_AT_PREFIX "location";
    case 0x0003: return DW_AT_PREFIX "name";
    case 0x0009: return DW_AT_PREFIX "ordering";
    case 0x000b: return DW_AT_PREFIX "byte_size";
    case 0x000c: return DW_AT_PREFIX "bit_offset";
    case 0x000d: return DW_AT_PREFIX "bit_size";
    case 0x0010: return DW_AT_PREFIX "stmt_list";
    case 0x0011: return DW_AT_PREFIX "low_pc";
    case 0x0012: return DW_AT_PREFIX "high_pc";
    case 0x0013: return DW_AT_PREFIX "language";
    case 0x0015: return DW_AT_PREFIX "discr";
    case 0x0016: return DW_AT_PREFIX "discr_value";
    case 0x0017: return DW_AT_PREFIX "visibility";
    case 0x0018: return DW_AT_PREFIX "import";
    case 0x0019: return DW_AT_PREFIX "string_length";
    case 0x001a: return DW_AT_PREFIX "common_reference";
    case 0x001b: return DW_AT_PREFIX "comp_dir";
    case 0x001c: return DW_AT_PREFIX "const_value";
    case 0x001d: return DW_AT_PREFIX "containing_type";
    case 0x001e: return DW_AT_PREFIX "default_value";
    case 0x0020: return DW_AT_PREFIX "inline";
    case 0x0021: return DW_AT_PREFIX "is_optional";
    case 0x0022: return DW_AT_PREFIX "lower_bound";
    case 0x0025: return DW_AT_PREFIX "producer";
    case 0x0027: return DW_AT_PREFIX "prototyped";
    case 0x002a: return DW_AT_PREFIX "return_addr";
    case 0x002c: return DW_AT_PREFIX "start_scope";
    case 0x002e: return DW_AT_PREFIX "bit_stride";
    case 0x002f: return DW_AT_PREFIX "upper_bound";
    case 0x0031: return DW_AT_PREFIX "abstract_origin";
    case 0x0032: return DW_AT_PREFIX "accessibility";
    case 0x0033: return DW_AT_PREFIX "address_class";
    case 0x0034: return DW_AT_PREFIX "artificial";
    case 0x0035: return DW_AT_PREFIX "base_types";
    case 0x0036: return DW_AT_PREFIX "calling_convention";
    case 0x0037: return DW_AT_PREFIX "count";
    case 0x0038: return DW_AT_PREFIX "data_member_location";
    case 0x0039: return DW_AT_PREFIX "decl_column";
    case 0x003a: return DW_AT_PREFIX "decl_file";
    case 0x003b: return DW_AT_PREFIX "decl_line";
    case 0x003c: return DW_AT_PREFIX "declaration";
    case 0x003d: return DW_AT_PREFIX "discr_list";
    case 0x003e: return DW_AT_PREFIX "encoding";
    case 0x003f: return DW_AT_PREFIX "external";
    case 0x0040: return DW_AT_PREFIX "frame_base";
    case 0x0041: return DW_AT_PREFIX "friend";
    case 0x0042: return DW_AT_PREFIX "identifier_case";
    case 0x0043: return DW_AT_PREFIX "macro_info";
    case 0x0044: return DW_AT_PREFIX "namelist_item";
    case 0x0045: return DW_AT_PREFIX "priority";
    case 0x0046: return DW_AT_PREFIX "segment";
    case 0x0047: return DW_AT_PREFIX "specification";
    case 0x0048: return DW_AT_PREFIX "static_link";
    case 0x0049: return DW_AT_PREFIX "type";
    case 0x004a: return DW_AT_PREFIX "use_location";
    case 0x004b: return DW_AT_PREFIX "variable_parameter";
    case 0x004c: return DW_AT_PREFIX "virtuality";
    case 0x004d: return DW_AT_PREFIX "vtable_elem_location";
    case 0x004e: return DW_AT_PREFIX "allocated";
    case 0x004f: return DW_AT_PREFIX "associated";
    case 0x0050: return DW_AT_PREFIX "data_location";
    case 0x0051: return DW_AT_PREFIX "byte_stride";
    case 0x0052: return DW_AT_PREFIX "entry_pc";
    case 0x0053: return DW_AT_PREFIX "use_UTF8";
    case 0x0054: return DW_AT_PREFIX "extension";
    case 0x0055: return DW_AT_PREFIX "ranges";
    case 0x0056: return DW_AT_PREFIX "trampoline";
    case 0x0057: return DW_AT_PREFIX "call_column";
    case 0x0058: return DW_AT_PREFIX "call_file";
    case 0x0059: return DW_AT_PREFIX "call_line";
    case 0x005a: return DW_AT_PREFIX "description";
    case 0x005b: return DW_AT_PREFIX "binary_scale";
    case 0x005c: return DW_AT_PREFIX "decimal_scale";
    case 0x005d: return DW_AT_PREFIX "small";
    case 0x005e: return DW_AT_PREFIX "decimal_sign";
    case 0x005f: return DW_AT_PREFIX "digit_count";
    case 0x0060: return DW_AT_PREFIX "picture_string";
    case 0x0061: return DW_AT_PREFIX "mutable";
    case 0x0062: return DW_AT_PREFIX "threads_scaled";
    case 0x0063: return DW_AT_PREFIX "explicit";
    case 0x0064: return DW_AT_PREFIX "object_pointer";
    case 0x0065: return DW_AT_PREFIX "endianity";
    case 0x0066: return DW_AT_PREFIX "elemental";
    case 0x0067: return DW_AT_PREFIX "pure";
    case 0x0068: return DW_AT_PREFIX "recursive";
    case 0x2000: return DW_AT_PREFIX "lo_user";
    case 0x3fff: return DW_AT_PREFIX "hi_user";
    case 0x2001: return DW_AT_PREFIX "MIPS_fde";
    case 0x2002: return DW_AT_PREFIX "MIPS_loop_begin";
    case 0x2003: return DW_AT_PREFIX "MIPS_tail_loop_begin";
    case 0x2004: return DW_AT_PREFIX "MIPS_epilog_begin";
    case 0x2005: return DW_AT_PREFIX "MIPS_loop_unroll_factor";
    case 0x2006: return DW_AT_PREFIX "MIPS_software_pipeline_depth";
    case 0x2007: return DW_AT_PREFIX "MIPS_linkage_name";
    case 0x2008: return DW_AT_PREFIX "MIPS_stride";
    case 0x2009: return DW_AT_PREFIX "MIPS_abstract_name";
    case 0x200a: return DW_AT_PREFIX "MIPS_clone_origin";
    case 0x200b: return DW_AT_PREFIX "MIPS_has_inlines";
    case 0x2101: return DW_AT_PREFIX "sf_names";
    case 0x2102: return DW_AT_PREFIX "src_info";
    case 0x2103: return DW_AT_PREFIX "mac_info";
    case 0x2104: return DW_AT_PREFIX "src_coords";
    case 0x2105: return DW_AT_PREFIX "body_begin";
    case 0x2106: return DW_AT_PREFIX "body_end";
    case 0x2107: return DW_AT_PREFIX "GNU_vector";
    case 0x2501: return DW_AT_PREFIX "APPLE_repository_file";
    case 0x2502: return DW_AT_PREFIX "APPLE_repository_type";
    case 0x2503: return DW_AT_PREFIX "APPLE_repository_name";
    case 0x2504: return DW_AT_PREFIX "APPLE_repository_specification";
    case 0x2505: return DW_AT_PREFIX "APPLE_repository_import";
    case 0x2506: return DW_AT_PREFIX "APPLE_repository_abstract_origin";
    case DW_AT_APPLE_flags: return DW_AT_PREFIX "APPLE_flags";
    case DW_AT_APPLE_optimized: return DW_AT_PREFIX "APPLE_optimized";
    case DW_AT_APPLE_isa: return DW_AT_PREFIX "APPLE_isa";
    case DW_AT_APPLE_block: return DW_AT_PREFIX "APPLE_block";
    default:
       snprintf (invalid, sizeof(invalid), "Unknown DW_AT constant: 0x%x", val);
       return invalid;
  }
}

const char *
DW_AT_value_to_englishy_name (uint32_t val)
{
  static char invalid[100];
  switch (val) {
    case 0x0001: return "sibling";
    case 0x0002: return "location";
    case 0x0003: return "name";
    case 0x0009: return "ordering";
    case 0x000b: return "byte size";
    case 0x000c: return "bit offset";
    case 0x000d: return "bit size";
    case 0x0010: return "stmt list";
    case 0x0011: return "low pc";
    case 0x0012: return "high pc";
    case 0x0013: return "language";
    case 0x0015: return "discr";
    case 0x0016: return "discr value";
    case 0x0017: return "visibility";
    case 0x0018: return "import";
    case 0x0019: return "string length";
    case 0x001a: return "common reference";
    case 0x001b: return "comp dir";
    case 0x001c: return "const value";
    case 0x001d: return "containing type";
    case 0x001e: return "default value";
    case 0x0020: return "inline";
    case 0x0021: return "is optional";
    case 0x0022: return "lower bound";
    case 0x0025: return "producer";
    case 0x0027: return "prototyped";
    case 0x002a: return "return addr";
    case 0x002c: return "start scope";
    case 0x002e: return "bit stride";
    case 0x002f: return "upper bound";
    case 0x0031: return "abstract origin";
    case 0x0032: return "accessibility";
    case 0x0033: return "address class";
    case 0x0034: return "artificial";
    case 0x0035: return "base types";
    case 0x0036: return "calling convention";
    case 0x0037: return "count";
    case 0x0038: return "data member location";
    case 0x0039: return "decl column";
    case 0x003a: return "decl file";
    case 0x003b: return "decl line";
    case 0x003c: return "declaration";
    case 0x003d: return "discr list";
    case 0x003e: return "encoding";
    case 0x003f: return "external";
    case 0x0040: return "frame base";
    case 0x0041: return "friend";
    case 0x0042: return "identifier case";
    case 0x0043: return "macro info";
    case 0x0044: return "namelist item";
    case 0x0045: return "priority";
    case 0x0046: return "segment";
    case 0x0047: return "specification";
    case 0x0048: return "static link";
    case 0x0049: return "type";
    case 0x004a: return "use location";
    case 0x004b: return "variable parameter";
    case 0x004c: return "virtuality";
    case 0x004d: return "vtable elem location";
    case 0x004e: return "allocated";
    case 0x004f: return "associated";
    case 0x0050: return "data location";
    case 0x0051: return "byte stride";
    case 0x0052: return "entry pc";
    case 0x0053: return "use UTF8";
    case 0x0054: return "extension";
    case 0x0055: return "ranges";
    case 0x0056: return "trampoline";
    case 0x0057: return "call column";
    case 0x0058: return "call file";
    case 0x0059: return "call line";
    case 0x005a: return "description";
    case 0x005b: return "binary scale";
    case 0x005c: return "decimal scale";
    case 0x005d: return "small";
    case 0x005e: return "decimal sign";
    case 0x005f: return "digit count";
    case 0x0060: return "picture string";
    case 0x0061: return "mutable";
    case 0x0062: return "threads scaled";
    case 0x0063: return "explicit";
    case 0x0064: return "object pointer";
    case 0x0065: return "endianity";
    case 0x0066: return "elemental";
    case 0x0067: return "pure";
    case 0x0068: return "recursive";
    case 0x2000: return "lo user";
    case 0x3fff: return "hi user";
    case 0x2001: return "MIPS fde";
    case 0x2002: return "MIPS loop begin";
    case 0x2003: return "MIPS tail loop begin";
    case 0x2004: return "MIPS epilog begin";
    case 0x2005: return "MIPS loop unroll factor";
    case 0x2006: return "MIPS software pipeline depth";
    case 0x2007: return "MIPS linkage name";
    case 0x2008: return "MIPS stride";
    case 0x2009: return "MIPS abstract name";
    case 0x200a: return "MIPS clone origin";
    case 0x200b: return "MIPS has inlines";
    case 0x2101: return "source file names";
    case 0x2102: return "source info";
    case 0x2103: return "macro info";
    case 0x2104: return "source coordinates";
    case 0x2105: return "body begin";
    case 0x2106: return "body end";
    case 0x2107: return "GNU vector";
    case 0x2501: return "repository file";
    case 0x2502: return "repository type";
    case 0x2503: return "repository name";
    case 0x2504: return "repository specification";
    case 0x2505: return "repository import";
    case 0x2506: return "repository abstract origin";
    case DW_AT_APPLE_flags: return "Apple gcc compiler flags";
    case DW_AT_APPLE_optimized: return "APPLE optimized";
    case DW_AT_APPLE_isa: return "APPLE instruction set architecture";
    case DW_AT_APPLE_block: return "APPLE block";
   default:
       snprintf (invalid, sizeof(invalid), "Unknown DW_AT constant: 0x%x", val);
       return invalid;
  }
}

DRC_class
DW_AT_value_to_class (uint32_t val)
{
  switch (val) {
    case 0x0001: return DRC_REFERENCE;
    case 0x0002: return DRC_BLOCK | DRC_LOCEXPR | DRC_LOCLISTPTR;
    case 0x0003: return DRC_STRING;
    case 0x0009: return DRC_CONSTANT;
    case 0x000b: return DRC_BLOCK | DRC_CONSTANT | DRC_REFERENCE;
    case 0x000c: return DRC_BLOCK | DRC_CONSTANT | DRC_REFERENCE;
    case 0x000d: return DRC_BLOCK | DRC_CONSTANT | DRC_REFERENCE;
    case 0x0010: return DRC_LINEPTR;
    case 0x0011: return DRC_ADDRESS;
    case 0x0012: return DRC_ADDRESS;
    case 0x0013: return DRC_CONSTANT;
    case 0x0015: return DRC_REFERENCE;
    case 0x0016: return DRC_CONSTANT;
    case 0x0017: return DRC_CONSTANT;
    case 0x0018: return DRC_REFERENCE;
    case 0x0019: return DRC_BLOCK | DRC_LOCEXPR | DRC_LOCLISTPTR;
    case 0x001a: return DRC_REFERENCE;
    case 0x001b: return DRC_STRING;
    case 0x001c: return DRC_BLOCK | DRC_CONSTANT | DRC_STRING;
    case 0x001d: return DRC_REFERENCE;
    case 0x001e: return DRC_REFERENCE;
    case 0x0020: return DRC_CONSTANT;
    case 0x0021: return DRC_FLAG;
    case 0x0022: return DRC_BLOCK | DRC_CONSTANT | DRC_REFERENCE;
    case 0x0025: return DRC_STRING;
    case 0x0027: return DRC_FLAG;
    case 0x002a: return DRC_BLOCK | DRC_LOCEXPR | DRC_LOCLISTPTR;
    case 0x002c: return DRC_CONSTANT;
    case 0x002e: return DRC_CONSTANT;
    case 0x002f: return DRC_BLOCK | DRC_CONSTANT | DRC_REFERENCE;
    case 0x0031: return DRC_REFERENCE;
    case 0x0032: return DRC_CONSTANT;
    case 0x0033: return DRC_CONSTANT;
    case 0x0034: return DRC_FLAG;
    case 0x0035: return DRC_REFERENCE;
    case 0x0036: return DRC_CONSTANT;
    case 0x0037: return DRC_BLOCK | DRC_CONSTANT | DRC_REFERENCE;
    case 0x0038: return DRC_BLOCK | DRC_CONSTANT | DRC_LOCEXPR | DRC_LOCLISTPTR;
    case 0x0039: return DRC_CONSTANT;
    case 0x003a: return DRC_CONSTANT;
    case 0x003b: return DRC_CONSTANT;
    case 0x003c: return DRC_FLAG;
    case 0x003d: return DRC_BLOCK;
    case 0x003e: return DRC_CONSTANT;
    case 0x003f: return DRC_FLAG;
    case 0x0040: return DRC_BLOCK | DRC_LOCEXPR | DRC_LOCLISTPTR;
    case 0x0041: return DRC_REFERENCE;
    case 0x0042: return DRC_CONSTANT;
    case 0x0043: return DRC_MACPTR;
    case 0x0044: return DRC_BLOCK;
    case 0x0045: return DRC_REFERENCE;
    case 0x0046: return DRC_BLOCK | DRC_CONSTANT;
    case 0x0047: return DRC_REFERENCE;
    case 0x0048: return DRC_BLOCK | DRC_LOCEXPR | DRC_LOCLISTPTR;
    case 0x0049: return DRC_REFERENCE;
    case 0x004a: return DRC_BLOCK | DRC_LOCEXPR | DRC_LOCLISTPTR;
    case 0x004b: return DRC_FLAG;
    case 0x004c: return DRC_CONSTANT;
    case 0x004d: return DRC_BLOCK | DRC_LOCEXPR | DRC_LOCLISTPTR;
    case 0x004e: return DRC_BLOCK | DRC_CONSTANT | DRC_DWARFv3 | DRC_REFERENCE;
    case 0x004f: return DRC_BLOCK | DRC_CONSTANT | DRC_DWARFv3 | DRC_REFERENCE;
    case 0x0050: return DRC_BLOCK | DRC_DWARFv3;
    case 0x0051: return DRC_BLOCK | DRC_CONSTANT | DRC_DWARFv3 | DRC_REFERENCE;
    case 0x0052: return DRC_ADDRESS | DRC_DWARFv3;
    case 0x0053: return DRC_DWARFv3 | DRC_FLAG;
    case 0x0054: return DRC_DWARFv3 | DRC_REFERENCE;
    case 0x0055: return DRC_DWARFv3 | DRC_RANGELISTPTR;
    case 0x0056: return DRC_ADDRESS | DRC_DWARFv3 | DRC_FLAG | DRC_REFERENCE | DRC_STRING;
    case 0x0057: return DRC_CONSTANT | DRC_DWARFv3;
    case 0x0058: return DRC_CONSTANT | DRC_DWARFv3;
    case 0x0059: return DRC_CONSTANT | DRC_DWARFv3;
    case 0x005a: return DRC_DWARFv3 | DRC_STRING;
    case 0x005b: return DRC_CONSTANT | DRC_DWARFv3;
    case 0x005c: return DRC_CONSTANT | DRC_DWARFv3;
    case 0x005d: return DRC_DWARFv3 | DRC_REFERENCE;
    case 0x005e: return DRC_CONSTANT | DRC_DWARFv3;
    case 0x005f: return DRC_CONSTANT | DRC_DWARFv3;
    case 0x0060: return DRC_DWARFv3 | DRC_STRING;
    case 0x0061: return DRC_DWARFv3 | DRC_FLAG;
    case 0x0062: return DRC_DWARFv3 | DRC_FLAG;
    case 0x0063: return DRC_DWARFv3 | DRC_FLAG;
    case 0x0064: return DRC_DWARFv3 | DRC_REFERENCE;
    case 0x0065: return DRC_0x65 | DRC_CONSTANT | DRC_DWARFv3;
    case 0x0066: return DRC_DWARFv3 | DRC_FLAG;
    case 0x0067: return DRC_DWARFv3 | DRC_FLAG;
    case 0x0068: return DRC_DWARFv3 | DRC_FLAG;
    case 0x2000: return 0;
    case 0x3fff: return 0;
    case 0x2001: return DRC_VENDOR_MIPS;
    case 0x2002: return DRC_VENDOR_MIPS;
    case 0x2003: return DRC_VENDOR_MIPS;
    case 0x2004: return DRC_VENDOR_MIPS;
    case 0x2005: return DRC_VENDOR_MIPS;
    case 0x2006: return DRC_VENDOR_MIPS;
    case 0x2007: return DRC_STRING | DRC_VENDOR_MIPS;
    case 0x2008: return DRC_VENDOR_MIPS;
    case 0x2009: return DRC_VENDOR_MIPS;
    case 0x200a: return DRC_VENDOR_MIPS;
    case 0x200b: return DRC_VENDOR_MIPS;
    default: return 0;
  }
}

/* [7.5.4] Figure 19 "Attribute form encodings" (pp. 133-134) in DWARFv3 draft 8 */

const char *
DW_FORM_value_to_name (uint32_t val)
{
  static char invalid[100];
  switch (val) {
    case 0x01: return DW_FORM_PREFIX "addr";
    case 0x03: return DW_FORM_PREFIX "block2";
    case 0x04: return DW_FORM_PREFIX "block4";
    case 0x05: return DW_FORM_PREFIX "data2";
    case 0x06: return DW_FORM_PREFIX "data4";
    case 0x07: return DW_FORM_PREFIX "data8";
    case 0x08: return DW_FORM_PREFIX "string";
    case 0x09: return DW_FORM_PREFIX "block";
    case 0x0a: return DW_FORM_PREFIX "block1";
    case 0x0b: return DW_FORM_PREFIX "data1";
    case 0x0c: return DW_FORM_PREFIX "flag";
    case 0x0d: return DW_FORM_PREFIX "sdata";
    case 0x0e: return DW_FORM_PREFIX "strp";
    case 0x0f: return DW_FORM_PREFIX "udata";
    case 0x10: return DW_FORM_PREFIX "ref_addr";
    case 0x11: return DW_FORM_PREFIX "ref1";
    case 0x12: return DW_FORM_PREFIX "ref2";
    case 0x13: return DW_FORM_PREFIX "ref4";
    case 0x14: return DW_FORM_PREFIX "ref8";
    case 0x15: return DW_FORM_PREFIX "ref_udata";
    case 0x16: return DW_FORM_PREFIX "indirect";
//  case DW_FORM_APPLE_db_str: return DW_FORM_PREFIX "APPLE_db_str";
    default:
       snprintf (invalid, sizeof(invalid), "Unknown DW_FORM constant: 0x%x", val);
       return invalid;
  }
}

const char *
DW_FORM_value_to_englishy_name (uint32_t val)
{
  static char invalid[100];
  switch (val) {
    case 0x01: return "addr";
    case 0x03: return "block2";
    case 0x04: return "block4";
    case 0x05: return "data2";
    case 0x06: return "data4";
    case 0x07: return "data8";
    case 0x08: return "string";
    case 0x09: return "block";
    case 0x0a: return "block1";
    case 0x0b: return "data1";
    case 0x0c: return "flag";
    case 0x0d: return "sdata";
    case 0x0e: return "strp";
    case 0x0f: return "udata";
    case 0x10: return "ref addr";
    case 0x11: return "ref1";
    case 0x12: return "ref2";
    case 0x13: return "ref4";
    case 0x14: return "ref8";
    case 0x15: return "ref udata";
    case 0x16: return "indirect";
//  case DW_FORM_APPLE_db_str: return "repository str";
    default:
       snprintf (invalid, sizeof(invalid), "Unknown DW_FORM constant: 0x%x", val);
       return invalid;
  }
}

DRC_class
DW_FORM_value_to_class (uint32_t val)
{
  switch (val) {
    case 0x01: return DRC_ADDRESS;
    case 0x03: return DRC_BLOCK | DRC_LOCEXPR;
    case 0x04: return DRC_BLOCK | DRC_LOCEXPR;
    case 0x05: return DRC_CONSTANT;
    case 0x06: return DRC_CONSTANT | DRC_LINEPTR | DRC_LOCLISTPTR | DRC_MACPTR | DRC_RANGELISTPTR;
    case 0x07: return DRC_CONSTANT | DRC_LINEPTR | DRC_LOCLISTPTR | DRC_MACPTR | DRC_RANGELISTPTR;
    case 0x08: return DRC_STRING;
    case 0x09: return DRC_BLOCK | DRC_LOCEXPR;
    case 0x0a: return DRC_BLOCK | DRC_LOCEXPR;
    case 0x0b: return DRC_CONSTANT;
    case 0x0c: return DRC_FLAG;
    case 0x0d: return DRC_CONSTANT;
    case 0x0e: return DRC_STRING;
    case 0x0f: return DRC_CONSTANT;
    case 0x10: return DRC_REFERENCE;
    case 0x11: return DRC_REFERENCE;
    case 0x12: return DRC_REFERENCE;
    case 0x13: return DRC_REFERENCE;
    case 0x14: return DRC_REFERENCE;
    case 0x15: return DRC_REFERENCE;
    case 0x16: return DRC_INDIRECT_SPECIAL;
    default: return 0;
  }
}

/* [7.7.1] Figure 22 "DWARF operation encodings" (pp. 136-139) in DWARFv3 draft 8 */

const char *
DW_OP_value_to_name (uint32_t val)
{
  static char invalid[100];
  switch (val) {
    case 0x03: return "DW_OP_addr";
    case 0x06: return "DW_OP_deref";
    case 0x08: return "DW_OP_const1u";
    case 0x09: return "DW_OP_const1s";
    case 0x0a: return "DW_OP_const2u";
    case 0x0b: return "DW_OP_const2s";
    case 0x0c: return "DW_OP_const4u";
    case 0x0d: return "DW_OP_const4s";
    case 0x0e: return "DW_OP_const8u";
    case 0x0f: return "DW_OP_const8s";
    case 0x10: return "DW_OP_constu";
    case 0x11: return "DW_OP_consts";
    case 0x12: return "DW_OP_dup";
    case 0x13: return "DW_OP_drop";
    case 0x14: return "DW_OP_over";
    case 0x15: return "DW_OP_pick";
    case 0x16: return "DW_OP_swap";
    case 0x17: return "DW_OP_rot";
    case 0x18: return "DW_OP_xderef";
    case 0x19: return "DW_OP_abs";
    case 0x1a: return "DW_OP_and";
    case 0x1b: return "DW_OP_div";
    case 0x1c: return "DW_OP_minus";
    case 0x1d: return "DW_OP_mod";
    case 0x1e: return "DW_OP_mul";
    case 0x1f: return "DW_OP_neg";
    case 0x20: return "DW_OP_not";
    case 0x21: return "DW_OP_or";
    case 0x22: return "DW_OP_plus";
    case 0x23: return "DW_OP_plus_uconst";
    case 0x24: return "DW_OP_shl";
    case 0x25: return "DW_OP_shr";
    case 0x26: return "DW_OP_shra";
    case 0x27: return "DW_OP_xor";
    case 0x2f: return "DW_OP_skip";
    case 0x28: return "DW_OP_bra";
    case 0x29: return "DW_OP_eq";
    case 0x2a: return "DW_OP_ge";
    case 0x2b: return "DW_OP_gt";
    case 0x2c: return "DW_OP_le";
    case 0x2d: return "DW_OP_lt";
    case 0x2e: return "DW_OP_ne";
    case 0x30: return "DW_OP_lit0";
    case 0x31: return "DW_OP_lit1";
    case 0x32: return "DW_OP_lit2";
    case 0x33: return "DW_OP_lit3";
    case 0x34: return "DW_OP_lit4";
    case 0x35: return "DW_OP_lit5";
    case 0x36: return "DW_OP_lit6";
    case 0x37: return "DW_OP_lit7";
    case 0x38: return "DW_OP_lit8";
    case 0x39: return "DW_OP_lit9";
    case 0x3a: return "DW_OP_lit10";
    case 0x3b: return "DW_OP_lit11";
    case 0x3c: return "DW_OP_lit12";
    case 0x3d: return "DW_OP_lit13";
    case 0x3e: return "DW_OP_lit14";
    case 0x3f: return "DW_OP_lit15";
    case 0x40: return "DW_OP_lit16";
    case 0x41: return "DW_OP_lit17";
    case 0x42: return "DW_OP_lit18";
    case 0x43: return "DW_OP_lit19";
    case 0x44: return "DW_OP_lit20";
    case 0x45: return "DW_OP_lit21";
    case 0x46: return "DW_OP_lit22";
    case 0x47: return "DW_OP_lit23";
    case 0x48: return "DW_OP_lit24";
    case 0x49: return "DW_OP_lit25";
    case 0x4a: return "DW_OP_lit26";
    case 0x4b: return "DW_OP_lit27";
    case 0x4c: return "DW_OP_lit28";
    case 0x4d: return "DW_OP_lit29";
    case 0x4e: return "DW_OP_lit30";
    case 0x4f: return "DW_OP_lit31";
    case 0x50: return "DW_OP_reg0";
    case 0x51: return "DW_OP_reg1";
    case 0x52: return "DW_OP_reg2";
    case 0x53: return "DW_OP_reg3";
    case 0x54: return "DW_OP_reg4";
    case 0x55: return "DW_OP_reg5";
    case 0x56: return "DW_OP_reg6";
    case 0x57: return "DW_OP_reg7";
    case 0x58: return "DW_OP_reg8";
    case 0x59: return "DW_OP_reg9";
    case 0x5a: return "DW_OP_reg10";
    case 0x5b: return "DW_OP_reg11";
    case 0x5c: return "DW_OP_reg12";
    case 0x5d: return "DW_OP_reg13";
    case 0x5e: return "DW_OP_reg14";
    case 0x5f: return "DW_OP_reg15";
    case 0x60: return "DW_OP_reg16";
    case 0x61: return "DW_OP_reg17";
    case 0x62: return "DW_OP_reg18";
    case 0x63: return "DW_OP_reg19";
    case 0x64: return "DW_OP_reg20";
    case 0x65: return "DW_OP_reg21";
    case 0x66: return "DW_OP_reg22";
    case 0x67: return "DW_OP_reg23";
    case 0x68: return "DW_OP_reg24";
    case 0x69: return "DW_OP_reg25";
    case 0x6a: return "DW_OP_reg26";
    case 0x6b: return "DW_OP_reg27";
    case 0x6c: return "DW_OP_reg28";
    case 0x6d: return "DW_OP_reg29";
    case 0x6e: return "DW_OP_reg30";
    case 0x6f: return "DW_OP_reg31";
    case 0x70: return "DW_OP_breg0";
    case 0x71: return "DW_OP_breg1";
    case 0x72: return "DW_OP_breg2";
    case 0x73: return "DW_OP_breg3";
    case 0x74: return "DW_OP_breg4";
    case 0x75: return "DW_OP_breg5";
    case 0x76: return "DW_OP_breg6";
    case 0x77: return "DW_OP_breg7";
    case 0x78: return "DW_OP_breg8";
    case 0x79: return "DW_OP_breg9";
    case 0x7a: return "DW_OP_breg10";
    case 0x7b: return "DW_OP_breg11";
    case 0x7c: return "DW_OP_breg12";
    case 0x7d: return "DW_OP_breg13";
    case 0x7e: return "DW_OP_breg14";
    case 0x7f: return "DW_OP_breg15";
    case 0x80: return "DW_OP_breg16";
    case 0x81: return "DW_OP_breg17";
    case 0x82: return "DW_OP_breg18";
    case 0x83: return "DW_OP_breg19";
    case 0x84: return "DW_OP_breg20";
    case 0x85: return "DW_OP_breg21";
    case 0x86: return "DW_OP_breg22";
    case 0x87: return "DW_OP_breg23";
    case 0x88: return "DW_OP_breg24";
    case 0x89: return "DW_OP_breg25";
    case 0x8a: return "DW_OP_breg26";
    case 0x8b: return "DW_OP_breg27";
    case 0x8c: return "DW_OP_breg28";
    case 0x8d: return "DW_OP_breg29";
    case 0x8e: return "DW_OP_breg30";
    case 0x8f: return "DW_OP_breg31";
    case 0x90: return "DW_OP_regx";
    case 0x91: return "DW_OP_fbreg";
    case 0x92: return "DW_OP_bregx";
    case 0x93: return "DW_OP_piece";
    case 0x94: return "DW_OP_deref_size";
    case 0x95: return "DW_OP_xderef_size";
    case 0x96: return "DW_OP_nop";
    case 0x97: return "DW_OP_push_object_address";
    case 0x98: return "DW_OP_call2";
    case 0x99: return "DW_OP_call4";
    case 0x9a: return "DW_OP_call_ref";
    case 0xf0: return "DW_OP_APPLE_uninit";
    case 0xe0: return "DW_OP_lo_user";
    case 0xff: return "DW_OP_hi_user";
    default:
       snprintf (invalid, sizeof(invalid), "Unknown DW_OP constant: 0x%x", val);
       return invalid;
  }
}

const char *
DW_OP_value_to_englishy_name (uint32_t val)
{
  static char invalid[100];
  switch (val) {
    case 0x03: return "addr";
    case 0x06: return "deref";
    case 0x08: return "const1u";
    case 0x09: return "const1s";
    case 0x0a: return "const2u";
    case 0x0b: return "const2s";
    case 0x0c: return "const4u";
    case 0x0d: return "const4s";
    case 0x0e: return "const8u";
    case 0x0f: return "const8s";
    case 0x10: return "constu";
    case 0x11: return "consts";
    case 0x12: return "dup";
    case 0x13: return "drop";
    case 0x14: return "over";
    case 0x15: return "pick";
    case 0x16: return "swap";
    case 0x17: return "rot";
    case 0x18: return "xderef";
    case 0x19: return "abs";
    case 0x1a: return "and";
    case 0x1b: return "div";
    case 0x1c: return "minus";
    case 0x1d: return "mod";
    case 0x1e: return "mul";
    case 0x1f: return "neg";
    case 0x20: return "not";
    case 0x21: return "or";
    case 0x22: return "plus";
    case 0x23: return "plus uconst";
    case 0x24: return "shl";
    case 0x25: return "shr";
    case 0x26: return "shra";
    case 0x27: return "xor";
    case 0x2f: return "skip";
    case 0x28: return "bra";
    case 0x29: return "eq";
    case 0x2a: return "ge";
    case 0x2b: return "gt";
    case 0x2c: return "le";
    case 0x2d: return "lt";
    case 0x2e: return "ne";
    case 0x30: return "lit0";
    case 0x31: return "lit1";
    case 0x32: return "lit2";
    case 0x33: return "lit3";
    case 0x34: return "lit4";
    case 0x35: return "lit5";
    case 0x36: return "lit6";
    case 0x37: return "lit7";
    case 0x38: return "lit8";
    case 0x39: return "lit9";
    case 0x3a: return "lit10";
    case 0x3b: return "lit11";
    case 0x3c: return "lit12";
    case 0x3d: return "lit13";
    case 0x3e: return "lit14";
    case 0x3f: return "lit15";
    case 0x40: return "lit16";
    case 0x41: return "lit17";
    case 0x42: return "lit18";
    case 0x43: return "lit19";
    case 0x44: return "lit20";
    case 0x45: return "lit21";
    case 0x46: return "lit22";
    case 0x47: return "lit23";
    case 0x48: return "lit24";
    case 0x49: return "lit25";
    case 0x4a: return "lit26";
    case 0x4b: return "lit27";
    case 0x4c: return "lit28";
    case 0x4d: return "lit29";
    case 0x4e: return "lit30";
    case 0x4f: return "lit31";
    case 0x50: return "reg0";
    case 0x51: return "reg1";
    case 0x52: return "reg2";
    case 0x53: return "reg3";
    case 0x54: return "reg4";
    case 0x55: return "reg5";
    case 0x56: return "reg6";
    case 0x57: return "reg7";
    case 0x58: return "reg8";
    case 0x59: return "reg9";
    case 0x5a: return "reg10";
    case 0x5b: return "reg11";
    case 0x5c: return "reg12";
    case 0x5d: return "reg13";
    case 0x5e: return "reg14";
    case 0x5f: return "reg15";
    case 0x60: return "reg16";
    case 0x61: return "reg17";
    case 0x62: return "reg18";
    case 0x63: return "reg19";
    case 0x64: return "reg20";
    case 0x65: return "reg21";
    case 0x66: return "reg22";
    case 0x67: return "reg23";
    case 0x68: return "reg24";
    case 0x69: return "reg25";
    case 0x6a: return "reg26";
    case 0x6b: return "reg27";
    case 0x6c: return "reg28";
    case 0x6d: return "reg29";
    case 0x6e: return "reg30";
    case 0x6f: return "reg31";
    case 0x70: return "breg0";
    case 0x71: return "breg1";
    case 0x72: return "breg2";
    case 0x73: return "breg3";
    case 0x74: return "breg4";
    case 0x75: return "breg5";
    case 0x76: return "breg6";
    case 0x77: return "breg7";
    case 0x78: return "breg8";
    case 0x79: return "breg9";
    case 0x7a: return "breg10";
    case 0x7b: return "breg11";
    case 0x7c: return "breg12";
    case 0x7d: return "breg13";
    case 0x7e: return "breg14";
    case 0x7f: return "breg15";
    case 0x80: return "breg16";
    case 0x81: return "breg17";
    case 0x82: return "breg18";
    case 0x83: return "breg19";
    case 0x84: return "breg20";
    case 0x85: return "breg21";
    case 0x86: return "breg22";
    case 0x87: return "breg23";
    case 0x88: return "breg24";
    case 0x89: return "breg25";
    case 0x8a: return "breg26";
    case 0x8b: return "breg27";
    case 0x8c: return "breg28";
    case 0x8d: return "breg29";
    case 0x8e: return "breg30";
    case 0x8f: return "breg31";
    case 0x90: return "regx";
    case 0x91: return "fbreg";
    case 0x92: return "bregx";
    case 0x93: return "piece";
    case 0x94: return "deref size";
    case 0x95: return "xderef size";
    case 0x96: return "nop";
    case 0x97: return "push object address";
    case 0x98: return "call2";
    case 0x99: return "call4";
    case 0x9a: return "call ref";
    case 0xf0: return "uninitialized";
    case 0xe0: return "lo user";
    case 0xff: return "hi user";
    default:
       snprintf (invalid, sizeof(invalid), "Unknown DW_OP constant: 0x%x", val);
       return invalid;
  }
}

DRC_class
DW_OP_value_to_class (uint32_t val)
{
  switch (val) {
    case 0x03: return DRC_ONEOPERAND;
    case 0x06: return DRC_ZEROOPERANDS;
    case 0x08: return DRC_ONEOPERAND;
    case 0x09: return DRC_ONEOPERAND;
    case 0x0a: return DRC_ONEOPERAND;
    case 0x0b: return DRC_ONEOPERAND;
    case 0x0c: return DRC_ONEOPERAND;
    case 0x0d: return DRC_ONEOPERAND;
    case 0x0e: return DRC_ONEOPERAND;
    case 0x0f: return DRC_ONEOPERAND;
    case 0x10: return DRC_ONEOPERAND;
    case 0x11: return DRC_ONEOPERAND;
    case 0x12: return DRC_ZEROOPERANDS;
    case 0x13: return DRC_ZEROOPERANDS;
    case 0x14: return DRC_ZEROOPERANDS;
    case 0x15: return DRC_ONEOPERAND;
    case 0x16: return DRC_ZEROOPERANDS;
    case 0x17: return DRC_ZEROOPERANDS;
    case 0x18: return DRC_ZEROOPERANDS;
    case 0x19: return DRC_ZEROOPERANDS;
    case 0x1a: return DRC_ZEROOPERANDS;
    case 0x1b: return DRC_ZEROOPERANDS;
    case 0x1c: return DRC_ZEROOPERANDS;
    case 0x1d: return DRC_ZEROOPERANDS;
    case 0x1e: return DRC_ZEROOPERANDS;
    case 0x1f: return DRC_ZEROOPERANDS;
    case 0x20: return DRC_ZEROOPERANDS;
    case 0x21: return DRC_ZEROOPERANDS;
    case 0x22: return DRC_ZEROOPERANDS;
    case 0x23: return DRC_ONEOPERAND;
    case 0x24: return DRC_ZEROOPERANDS;
    case 0x25: return DRC_ZEROOPERANDS;
    case 0x26: return DRC_ZEROOPERANDS;
    case 0x27: return DRC_ZEROOPERANDS;
    case 0x2f: return DRC_ONEOPERAND;
    case 0x28: return DRC_ONEOPERAND;
    case 0x29: return DRC_ZEROOPERANDS;
    case 0x2a: return DRC_ZEROOPERANDS;
    case 0x2b: return DRC_ZEROOPERANDS;
    case 0x2c: return DRC_ZEROOPERANDS;
    case 0x2d: return DRC_ZEROOPERANDS;
    case 0x2e: return DRC_ZEROOPERANDS;
    case 0x30: return DRC_ZEROOPERANDS;
    case 0x31: return DRC_ZEROOPERANDS;
    case 0x32: return DRC_ZEROOPERANDS;
    case 0x33: return DRC_ZEROOPERANDS;
    case 0x34: return DRC_ZEROOPERANDS;
    case 0x35: return DRC_ZEROOPERANDS;
    case 0x36: return DRC_ZEROOPERANDS;
    case 0x37: return DRC_ZEROOPERANDS;
    case 0x38: return DRC_ZEROOPERANDS;
    case 0x39: return DRC_ZEROOPERANDS;
    case 0x3a: return DRC_ZEROOPERANDS;
    case 0x3b: return DRC_ZEROOPERANDS;
    case 0x3c: return DRC_ZEROOPERANDS;
    case 0x3d: return DRC_ZEROOPERANDS;
    case 0x3e: return DRC_ZEROOPERANDS;
    case 0x3f: return DRC_ZEROOPERANDS;
    case 0x40: return DRC_ZEROOPERANDS;
    case 0x41: return DRC_ZEROOPERANDS;
    case 0x42: return DRC_ZEROOPERANDS;
    case 0x43: return DRC_ZEROOPERANDS;
    case 0x44: return DRC_ZEROOPERANDS;
    case 0x45: return DRC_ZEROOPERANDS;
    case 0x46: return DRC_ZEROOPERANDS;
    case 0x47: return DRC_ZEROOPERANDS;
    case 0x48: return DRC_ZEROOPERANDS;
    case 0x49: return DRC_ZEROOPERANDS;
    case 0x4a: return DRC_ZEROOPERANDS;
    case 0x4b: return DRC_ZEROOPERANDS;
    case 0x4c: return DRC_ZEROOPERANDS;
    case 0x4d: return DRC_ZEROOPERANDS;
    case 0x4e: return DRC_ZEROOPERANDS;
    case 0x4f: return DRC_ZEROOPERANDS;
    case 0x50: return DRC_ZEROOPERANDS;
    case 0x51: return DRC_ZEROOPERANDS;
    case 0x52: return DRC_ZEROOPERANDS;
    case 0x53: return DRC_ZEROOPERANDS;
    case 0x54: return DRC_ZEROOPERANDS;
    case 0x55: return DRC_ZEROOPERANDS;
    case 0x56: return DRC_ZEROOPERANDS;
    case 0x57: return DRC_ZEROOPERANDS;
    case 0x58: return DRC_ZEROOPERANDS;
    case 0x59: return DRC_ZEROOPERANDS;
    case 0x5a: return DRC_ZEROOPERANDS;
    case 0x5b: return DRC_ZEROOPERANDS;
    case 0x5c: return DRC_ZEROOPERANDS;
    case 0x5d: return DRC_ZEROOPERANDS;
    case 0x5e: return DRC_ZEROOPERANDS;
    case 0x5f: return DRC_ZEROOPERANDS;
    case 0x60: return DRC_ZEROOPERANDS;
    case 0x61: return DRC_ZEROOPERANDS;
    case 0x62: return DRC_ZEROOPERANDS;
    case 0x63: return DRC_ZEROOPERANDS;
    case 0x64: return DRC_ZEROOPERANDS;
    case 0x65: return DRC_ZEROOPERANDS;
    case 0x66: return DRC_ZEROOPERANDS;
    case 0x67: return DRC_ZEROOPERANDS;
    case 0x68: return DRC_ZEROOPERANDS;
    case 0x69: return DRC_ZEROOPERANDS;
    case 0x6a: return DRC_ZEROOPERANDS;
    case 0x6b: return DRC_ZEROOPERANDS;
    case 0x6c: return DRC_ZEROOPERANDS;
    case 0x6d: return DRC_ZEROOPERANDS;
    case 0x6e: return DRC_ZEROOPERANDS;
    case 0x6f: return DRC_ZEROOPERANDS;
    case 0x70: return DRC_ONEOPERAND;
    case 0x71: return DRC_ONEOPERAND;
    case 0x72: return DRC_ONEOPERAND;
    case 0x73: return DRC_ONEOPERAND;
    case 0x74: return DRC_ONEOPERAND;
    case 0x75: return DRC_ONEOPERAND;
    case 0x76: return DRC_ONEOPERAND;
    case 0x77: return DRC_ONEOPERAND;
    case 0x78: return DRC_ONEOPERAND;
    case 0x79: return DRC_ONEOPERAND;
    case 0x7a: return DRC_ONEOPERAND;
    case 0x7b: return DRC_ONEOPERAND;
    case 0x7c: return DRC_ONEOPERAND;
    case 0x7d: return DRC_ONEOPERAND;
    case 0x7e: return DRC_ONEOPERAND;
    case 0x7f: return DRC_ONEOPERAND;
    case 0x80: return DRC_ONEOPERAND;
    case 0x81: return DRC_ONEOPERAND;
    case 0x82: return DRC_ONEOPERAND;
    case 0x83: return DRC_ONEOPERAND;
    case 0x84: return DRC_ONEOPERAND;
    case 0x85: return DRC_ONEOPERAND;
    case 0x86: return DRC_ONEOPERAND;
    case 0x87: return DRC_ONEOPERAND;
    case 0x88: return DRC_ONEOPERAND;
    case 0x89: return DRC_ONEOPERAND;
    case 0x8a: return DRC_ONEOPERAND;
    case 0x8b: return DRC_ONEOPERAND;
    case 0x8c: return DRC_ONEOPERAND;
    case 0x8d: return DRC_ONEOPERAND;
    case 0x8e: return DRC_ONEOPERAND;
    case 0x8f: return DRC_ONEOPERAND;
    case 0x90: return DRC_ONEOPERAND;
    case 0x91: return DRC_ONEOPERAND;
    case 0x92: return DRC_TWOOPERANDS;
    case 0x93: return DRC_ONEOPERAND;
    case 0x94: return DRC_ONEOPERAND;
    case 0x95: return DRC_ONEOPERAND;
    case 0x96: return DRC_ZEROOPERANDS;
    case 0x97: return DRC_DWARFv3 | DRC_ZEROOPERANDS;
    case 0x98: return DRC_DWARFv3 | DRC_ONEOPERAND;
    case 0x99: return DRC_DWARFv3 | DRC_ONEOPERAND;
    case 0x9a: return DRC_DWARFv3 | DRC_ONEOPERAND;
    case 0xf0: return DRC_ZEROOPERANDS; /* DW_OP_APPLE_uninit */
    case 0xe0: return 0;
    case 0xff: return 0;
    default: return 0;
  }
}

/* [7.8] Figure 23 "Base type encoding values" (pp. 140-141) in DWARFv3 draft 8 */

const char *
DW_ATE_value_to_name (uint32_t val)
{
  static char invalid[100];
  switch (val) {
    case 0x01: return "DW_ATE_address";
    case 0x02: return "DW_ATE_boolean";
    case 0x03: return "DW_ATE_complex_float";
    case 0x04: return "DW_ATE_float";
    case 0x05: return "DW_ATE_signed";
    case 0x06: return "DW_ATE_signed_char";
    case 0x07: return "DW_ATE_unsigned";
    case 0x08: return "DW_ATE_unsigned_char";
    case 0x09: return "DW_ATE_imaginary_float";
    case 0x80: return "DW_ATE_lo_user";
    case 0xff: return "DW_ATE_hi_user";
    default:
       snprintf (invalid, sizeof(invalid), "Unknown DW_ATE constant: 0x%x", val);
       return invalid;
  }
}

const char *
DW_ATE_value_to_englishy_name (uint32_t val)
{
  static char invalid[100];
  switch (val) {
    case 0x01: return "address";
    case 0x02: return "boolean";
    case 0x03: return "complex float";
    case 0x04: return "float";
    case 0x05: return "signed";
    case 0x06: return "signed char";
    case 0x07: return "unsigned";
    case 0x08: return "unsigned char";
    case 0x09: return "imaginary float";
    case 0x80: return "lo user";
    case 0xff: return "hi user";
    default:
       snprintf (invalid, sizeof(invalid), "Unknown DW_ATE constant: 0x%x", val);
       return invalid;
  }
}

DRC_class
DW_ATE_value_to_class (uint32_t val)
{
  switch (val) {
    case 0x01: return 0;
    case 0x02: return 0;
    case 0x03: return 0;
    case 0x04: return 0;
    case 0x05: return 0;
    case 0x06: return 0;
    case 0x07: return 0;
    case 0x08: return 0;
    case 0x09: return DRC_DWARFv3;
    case 0x80: return 0;
    case 0xff: return 0;
    default: return 0;
  }
}

/* [7.9] Figure 24 "Accessibility encodings" (p. 141) in DWARFv3 draft 8 */

const char *
DW_ACCESS_value_to_name (uint32_t val)
{
  static char invalid[100];
  switch (val) {
    case 0x1: return "DW_ACCESS_public";
    case 0x2: return "DW_ACCESS_protected";
    case 0x3: return "DW_ACCESS_private";
    default:
       snprintf (invalid, sizeof(invalid), "Unknown DW_ACCESS constant: 0x%x", val);
       return invalid;
  }
}

const char *
DW_ACCESS_value_to_englishy_name (uint32_t val)
{
  static char invalid[100];
  switch (val) {
    case 0x1: return "public";
    case 0x2: return "protected";
    case 0x3: return "private";
    default:
       snprintf (invalid, sizeof(invalid), "Unknown DW_ACCESS constant: 0x%x", val);
       return invalid;
  }
}

DRC_class
DW_ACCESS_value_to_class (uint32_t val)
{
  switch (val) {
    case 0x1: return 0;
    case 0x2: return 0;
    case 0x3: return 0;
    default: return 0;
  }
}

/* [7.10] Figure 25 "Visibility encodings" (p. 142) in DWARFv3 draft 8 */

const char *
DW_VIS_value_to_name (uint32_t val)
{
  static char invalid[100];
  switch (val) {
    case 0x1: return "DW_VIS_local";
    case 0x2: return "DW_VIS_exported";
    case 0x3: return "DW_VIS_qualified";
    default:
       snprintf (invalid, sizeof(invalid), "Unknown DW_VIS constant: 0x%x", val);
       return invalid;
  }
}

const char *
DW_VIS_value_to_englishy_name (uint32_t val)
{
  static char invalid[100];
  switch (val) {
    case 0x1: return "local";
    case 0x2: return "exported";
    case 0x3: return "qualified";
    default:
       snprintf (invalid, sizeof(invalid), "Unknown DW_VIS constant: 0x%x", val);
       return invalid;
  }
}

DRC_class
DW_VIS_value_to_class (uint32_t val)
{
  switch (val) {
    case 0x1: return 0;
    case 0x2: return 0;
    case 0x3: return 0;
    default: return 0;
  }
}

/* [7.11] Figure 26 "Virtuality encodings" (p. 142) in DWARFv3 draft 8 */

const char *
DW_VIRTUALITY_value_to_name (uint32_t val)
{
  static char invalid[100];
  switch (val) {
    case 0x0: return "DW_VIRTUALITY_none";
    case 0x1: return "DW_VIRTUALITY_virtual";
    case 0x2: return "DW_VIRTUALITY_pure_virtual";
    default:
       snprintf (invalid, sizeof(invalid), "Unknown DW_VIRTUALITY constant: 0x%x", val);
       return invalid;
  }
}

const char *
DW_VIRTUALITY_value_to_englishy_name (uint32_t val)
{
  static char invalid[100];
  switch (val) {
    case 0x0: return "none";
    case 0x1: return "virtual";
    case 0x2: return "pure virtual";
    default:
       snprintf (invalid, sizeof(invalid), "Unknown DW_VIRTUALITY constant: 0x%x", val);
       return invalid;
  }
}

DRC_class
DW_VIRTUALITY_value_to_class (uint32_t val)
{
  switch (val) {
    case 0x0: return 0;
    case 0x1: return 0;
    case 0x2: return 0;
    default: return 0;
  }
}

/* [7.12] Figure 27 "Language encodings" (p. 143) in DWARFv3 draft 8 */

const char *
DW_LANG_value_to_name (uint32_t val)
{
  static char invalid[100];
  switch (val) {
    case 0x0001: return "DW_LANG_C89";
    case 0x0002: return "DW_LANG_C";
    case 0x0003: return "DW_LANG_Ada83";
    case 0x0004: return "DW_LANG_C_plus_plus";
    case 0x0005: return "DW_LANG_Cobol74";
    case 0x0006: return "DW_LANG_Cobol85";
    case 0x0007: return "DW_LANG_Fortran77";
    case 0x0008: return "DW_LANG_Fortran90";
    case 0x0009: return "DW_LANG_Pascal83";
    case 0x000a: return "DW_LANG_Modula2";
    case 0x000b: return "DW_LANG_Java";
    case 0x000c: return "DW_LANG_C99";
    case 0x000d: return "DW_LANG_Ada95";
    case 0x000e: return "DW_LANG_Fortran95";
    case 0x000f: return "DW_LANG_PLI";
    case 0x0010: return "DW_LANG_ObjC";
    case 0x0011: return "DW_LANG_ObjC_plus_plus";
    case 0x0012: return "DW_LANG_UPC";
    case 0x0013: return "DW_LANG_D";
    case 0x8000: return "DW_LANG_lo_user";
    case 0x8001: return "DW_LANG_Mips_Assembler";
    case 0x8765: return "DW_LANG_Upc";
    case 0xffff: return "DW_LANG_hi_user";
    default:
       snprintf (invalid, sizeof(invalid), "Unknown DW_LANG constant: 0x%x", val);
       return invalid;
  }
}

const char *
DW_LANG_value_to_englishy_name (uint32_t val)
{
  static char invalid[100];
  switch (val) {
    case 0x0001: return "C89";
    case 0x0002: return "C";
    case 0x0003: return "Ada83";
    case 0x0004: return "C++";
    case 0x0005: return "Cobol74";
    case 0x0006: return "Cobol85";
    case 0x0007: return "Fortran77";
    case 0x0008: return "Fortran90";
    case 0x0009: return "Pascal83";
    case 0x000a: return "Modula2";
    case 0x000b: return "Java";
    case 0x000c: return "C99";
    case 0x000d: return "Ada95";
    case 0x000e: return "Fortran95";
    case 0x000f: return "PLI";
    case 0x0010: return "Objective C";
    case 0x0011: return "Objective C++";
    case 0x0012: return "UPC";
    case 0x0013: return "D";
    case 0x8000: return "lo user";
    case 0x8001: return "MIPS Assembler";
    case 0x8765: return "UPC";
    case 0xffff: return "hi user";
    default:
       snprintf (invalid, sizeof(invalid), "Unknown DW_LANG constant: 0x%x", val);
       return invalid;
  }
}

DRC_class
DW_LANG_value_to_class (uint32_t val)
{
  switch (val) {
    case 0x0001: return 0;
    case 0x0002: return 0;
    case 0x0003: return 0;
    case 0x0004: return 0;
    case 0x0005: return 0;
    case 0x0006: return 0;
    case 0x0007: return 0;
    case 0x0008: return 0;
    case 0x0009: return 0;
    case 0x000a: return 0;
    case 0x000b: return DRC_DWARFv3;
    case 0x000c: return DRC_DWARFv3;
    case 0x000d: return DRC_DWARFv3;
    case 0x000e: return DRC_DWARFv3;
    case 0x000f: return DRC_DWARFv3;
    case 0x0010: return DRC_DWARFv3;
    case 0x0011: return DRC_DWARFv3;
    case 0x0012: return DRC_DWARFv3;
    case 0x0013: return DRC_DWARFv3;
    case 0x8000: return 0;
    case 0x8001: return 0;
    case 0x8765: return 0;
    case 0xffff: return 0;
    default: return 0;
  }
}

/* [7.13], "Address Class Encodings" (p. 144) in DWARFv3 draft 8 */

const char *
DW_ADDR_value_to_name (uint32_t val)
{
  static char invalid[100];
  switch (val) {
    case 0x0: return "DW_ADDR_none";
    default:
       snprintf (invalid, sizeof(invalid), "Unknown DW_ADDR constant: 0x%x", val);
       return invalid;
  }
}

const char *
DW_ADDR_value_to_englishy_name (uint32_t val)
{
  static char invalid[100];
  switch (val) {
    case 0x0: return "none";
    default:
       snprintf (invalid, sizeof(invalid), "Unknown DW_ADDR constant: 0x%x", val);
       return invalid;
  }
}

DRC_class
DW_ADDR_value_to_class (uint32_t val)
{
  switch (val) {
    case 0x0: return 0;
    default: return 0;
  }
}

/* [7.14] Figure 28 "Identifier case encodings" (p. 144) in DWARFv3 draft 8 */

const char *
DW_ID_value_to_name (uint32_t val)
{
  static char invalid[100];
  switch (val) {
    case 0x0: return "DW_ID_case_sensitive";
    case 0x1: return "DW_ID_up_case";
    case 0x2: return "DW_ID_down_case";
    case 0x3: return "DW_ID_case_insensitive";
    default:
       snprintf (invalid, sizeof(invalid), "Unknown DW_ID constant: 0x%x", val);
       return invalid;
  }
}

const char *
DW_ID_value_to_englishy_name (uint32_t val)
{
  static char invalid[100];
  switch (val) {
    case 0x0: return "case sensitive";
    case 0x1: return "up case";
    case 0x2: return "down case";
    case 0x3: return "case insensitive";
    default:
       snprintf (invalid, sizeof(invalid), "Unknown DW_ID constant: 0x%x", val);
       return invalid;
  }
}

DRC_class
DW_ID_value_to_class (uint32_t val)
{
  switch (val) {
    case 0x0: return 0;
    case 0x1: return 0;
    case 0x2: return 0;
    case 0x3: return 0;
    default: return 0;
  }
}

/* [7.15] Figure 29 "Calling convention encodings" (p. 144) in DWARFv3 draft 8 */

const char *
DW_CC_value_to_name (uint32_t val)
{
  static char invalid[100];
  switch (val) {
    case 0x01: return "DW_CC_normal";
    case 0x02: return "DW_CC_program";
    case 0x03: return "DW_CC_nocall";
    case 0x40: return "DW_CC_lo_user";
    case 0xff: return "DW_CC_hi_user";
    default:
       snprintf (invalid, sizeof(invalid), "Unknown DW_CC constant: 0x%x", val);
       return invalid;
  }
}

const char *
DW_CC_value_to_englishy_name (uint32_t val)
{
  static char invalid[100];
  switch (val) {
    case 0x01: return "normal";
    case 0x02: return "program";
    case 0x03: return "nocall";
    case 0x40: return "lo user";
    case 0xff: return "hi user";
    default:
       snprintf (invalid, sizeof(invalid), "Unknown DW_CC constant: 0x%x", val);
       return invalid;
  }
}

DRC_class
DW_CC_value_to_class (uint32_t val)
{
  switch (val) {
    case 0x01: return 0;
    case 0x02: return 0;
    case 0x03: return 0;
    case 0x40: return 0;
    case 0xff: return 0;
    default: return 0;
  }
}

/* [7.16] Figure 30 "Inline encodings" (p. 145) in DWARFv3 draft 8 */

const char *
DW_INL_value_to_name (uint32_t val)
{
  static char invalid[100];
  switch (val) {
    case 0x0: return "DW_INL_not_inlined";
    case 0x1: return "DW_INL_inlined";
    case 0x2: return "DW_INL_declared_not_inlined";
    case 0x3: return "DW_INL_declared_inlined";
    default:
       snprintf (invalid, sizeof(invalid), "Unknown DW_INL constant: 0x%x", val);
       return invalid;
  }
}

const char *
DW_INL_value_to_englishy_name (uint32_t val)
{
  static char invalid[100];
  switch (val) {
    case 0x0: return "not inlined";
    case 0x1: return "inlined";
    case 0x2: return "declared not inlined";
    case 0x3: return "declared inlined";
    default:
       snprintf (invalid, sizeof(invalid), "Unknown DW_INL constant: 0x%x", val);
       return invalid;
  }
}

DRC_class
DW_INL_value_to_class (uint32_t val)
{
  switch (val) {
    case 0x0: return 0;
    case 0x1: return 0;
    case 0x2: return 0;
    case 0x3: return 0;
    default: return 0;
  }
}

/* [7.17] Figure 31 "Ordering encodings" (p. 145) in DWARFv3 draft 8 */

const char *
DW_ORD_value_to_name (uint32_t val)
{
  static char invalid[100];
  switch (val) {
    case 0x0: return "DW_ORD_row_major";
    case 0x1: return "DW_ORD_col_major";
    default:
       snprintf (invalid, sizeof(invalid), "Unknown DW_ORD constant: 0x%x", val);
       return invalid;
  }
}

const char *
DW_ORD_value_to_englishy_name (uint32_t val)
{
  static char invalid[100];
  switch (val) {
    case 0x0: return "row major";
    case 0x1: return "col major";
    default:
       snprintf (invalid, sizeof(invalid), "Unknown DW_ORD constant: 0x%x", val);
       return invalid;
  }
}

DRC_class
DW_ORD_value_to_class (uint32_t val)
{
  switch (val) {
    case 0x0: return 0;
    case 0x1: return 0;
    default: return 0;
  }
}

/* [7.18] Figure 32 "Discriminant descriptor encodings" (p. 146) in DWARFv3 draft 8 */

const char *
DW_DSC_value_to_name (uint32_t val)
{
  static char invalid[100];
  switch (val) {
    case 0x0: return "DW_DSC_label";
    case 0x1: return "DW_DSC_range";
    default:
       snprintf (invalid, sizeof(invalid), "Unknown DW_DSC constant: 0x%x", val);
       return invalid;
  }
}

const char *
DW_DSC_value_to_englishy_name (uint32_t val)
{
  static char invalid[100];
  switch (val) {
    case 0x0: return "label";
    case 0x1: return "range";
    default:
       snprintf (invalid, sizeof(invalid), "Unknown DW_DSC constant: 0x%x", val);
       return invalid;
  }
}

DRC_class
DW_DSC_value_to_class (uint32_t val)
{
  switch (val) {
    case 0x0: return 0;
    case 0x1: return 0;
    default: return 0;
  }
}

/* [7.21] Figure 33 "Line Number Standard Opcode Encodings" (pp. 148-149) in DWARFv3 draft 8 */

const char *
DW_LNS_value_to_name (uint32_t val)
{
  static char invalid[100];
  switch (val) {
    case 0x1: return "DW_LNS_copy";
    case 0x2: return "DW_LNS_advance_pc";
    case 0x3: return "DW_LNS_advance_line";
    case 0x4: return "DW_LNS_set_file";
    case 0x5: return "DW_LNS_set_column";
    case 0x6: return "DW_LNS_negate_stmt";
    case 0x7: return "DW_LNS_set_basic_block";
    case 0x8: return "DW_LNS_const_add_pc";
    case 0x9: return "DW_LNS_fixed_advance_pc";
    case 0xa: return "DW_LNS_set_prologue_end";
    case 0xb: return "DW_LNS_set_epilogue_begin";
    case 0xc: return "DW_LNS_set_isa";
    default:
       snprintf (invalid, sizeof(invalid), "Unknown DW_LNS constant: 0x%x", val);
       return invalid;
  }
}

const char *
DW_LNS_value_to_englishy_name (uint32_t val)
{
  static char invalid[100];
  switch (val) {
    case 0x1: return "copy";
    case 0x2: return "advance pc";
    case 0x3: return "advance line";
    case 0x4: return "set file";
    case 0x5: return "set column";
    case 0x6: return "negate stmt";
    case 0x7: return "set basic block";
    case 0x8: return "const add pc";
    case 0x9: return "fixed advance pc";
    case 0xa: return "set prologue end";
    case 0xb: return "set epilogue begin";
    case 0xc: return "set isa";
    default:
       snprintf (invalid, sizeof(invalid), "Unknown DW_LNS constant: 0x%x", val);
       return invalid;
  }
}

DRC_class
DW_LNS_value_to_class (uint32_t val)
{
  switch (val) {
    case 0x1: return 0;
    case 0x2: return 0;
    case 0x3: return 0;
    case 0x4: return 0;
    case 0x5: return 0;
    case 0x6: return 0;
    case 0x7: return 0;
    case 0x8: return 0;
    case 0x9: return 0;
    case 0xa: return DRC_DWARFv3;
    case 0xb: return DRC_DWARFv3;
    case 0xc: return DRC_DWARFv3;
    default: return 0;
  }
}

/* [7.21] Figure 34 "Line Number Extended Opcode Encodings" (p. 149) in DWARFv3 draft 8 */

const char *
DW_LNE_value_to_name (uint32_t val)
{
  static char invalid[100];
  switch (val) {
    case 0x01: return "DW_LNE_end_sequence";
    case 0x02: return "DW_LNE_set_address";
    case 0x03: return "DW_LNE_define_file";
    case 0x80: return "DW_LNE_lo_user";
    case 0xff: return "DW_LNE_hi_user";
    default:
       snprintf (invalid, sizeof(invalid), "Unknown DW_LNE constant: 0x%x", val);
       return invalid;
  }
}

const char *
DW_LNE_value_to_englishy_name (uint32_t val)
{
  static char invalid[100];
  switch (val) {
    case 0x01: return "end sequence";
    case 0x02: return "set address";
    case 0x03: return "define file";
    case 0x80: return "lo user";
    case 0xff: return "hi user";
    default:
       snprintf (invalid, sizeof(invalid), "Unknown DW_LNE constant: 0x%x", val);
       return invalid;
  }
}

DRC_class
DW_LNE_value_to_class (uint32_t val)
{
  switch (val) {
    case 0x01: return 0;
    case 0x02: return 0;
    case 0x03: return 0;
    case 0x80: return DRC_DWARFv3;
    case 0xff: return DRC_DWARFv3;
    default: return 0;
  }
}

/* [7.22] Figure 35 "Macinfo Type Encodings" (p. 150) in DWARFv3 draft 8 */

const char *
DW_MACINFO_value_to_name (uint32_t val)
{
  static char invalid[100];
  switch (val) {
    case 0x01: return "DW_MACINFO_define";
    case 0x02: return "DW_MACINFO_undef";
    case 0x03: return "DW_MACINFO_start_file";
    case 0x04: return "DW_MACINFO_end_file";
    case 0xff: return "DW_MACINFO_vendor_ext";
    default:
       snprintf (invalid, sizeof(invalid), "Unknown DW_MACINFO constant: 0x%x", val);
       return invalid;
  }
}

const char *
DW_MACINFO_value_to_englishy_name (uint32_t val)
{
  static char invalid[100];
  switch (val) {
    case 0x01: return "define";
    case 0x02: return "undef";
    case 0x03: return "start file";
    case 0x04: return "end file";
    case 0xff: return "vendor ext";
    default:
       snprintf (invalid, sizeof(invalid), "Unknown DW_MACINFO constant: 0x%x", val);
       return invalid;
  }
}

DRC_class
DW_MACINFO_value_to_class (uint32_t val)
{
  switch (val) {
    case 0x01: return 0;
    case 0x02: return 0;
    case 0x03: return 0;
    case 0x04: return 0;
    case 0xff: return 0;
    default: return 0;
  }
}

/* [7.23] Figure 36 "Call frame instruction encodings" (pp. 151-152) in DWARFv3 draft 8 */

const char *
DW_CFA_value_to_name (uint32_t val)
{
  static char invalid[100];
  switch (val) {
    case 0x40: return "DW_CFA_advance_loc";
    case 0x80: return "DW_CFA_offset";
    case 0xc0: return "DW_CFA_restore";
    case 0x00: return "DW_CFA_nop";
    case 0x01: return "DW_CFA_set_loc";
    case 0x02: return "DW_CFA_advance_loc1";
    case 0x03: return "DW_CFA_advance_loc2";
    case 0x04: return "DW_CFA_advance_loc4";
    case 0x05: return "DW_CFA_offset_extended";
    case 0x06: return "DW_CFA_restore_extended";
    case 0x07: return "DW_CFA_undefined";
    case 0x08: return "DW_CFA_same_value";
    case 0x09: return "DW_CFA_register";
    case 0x0a: return "DW_CFA_remember_state";
    case 0x0b: return "DW_CFA_restore_state";
    case 0x0c: return "DW_CFA_def_cfa";
    case 0x0d: return "DW_CFA_def_cfa_register";
    case 0x0e: return "DW_CFA_def_cfa_offset";
    case 0x0f: return "DW_CFA_def_cfa_expression";
    case 0x10: return "DW_CFA_expression";
    case 0x11: return "DW_CFA_offset_extended_sf";
    case 0x12: return "DW_CFA_def_cfa_sf";
    case 0x13: return "DW_CFA_def_cfa_offset_sf";
    case 0x1c: return "DW_CFA_lo_user";
    case 0x3f: return "DW_CFA_hi_user";
    default:
       snprintf (invalid, sizeof(invalid), "Unknown DW_CFA constant: 0x%x", val);
       return invalid;
  }
}

const char *
DW_CFA_value_to_englishy_name (uint32_t val)
{
  static char invalid[100];
  switch (val) {
    case 0x40: return "advance loc";
    case 0x80: return "offset";
    case 0xc0: return "restore";
    case 0x00: return "nop";
    case 0x01: return "set loc";
    case 0x02: return "advance loc1";
    case 0x03: return "advance loc2";
    case 0x04: return "advance loc4";
    case 0x05: return "offset extended";
    case 0x06: return "restore extended";
    case 0x07: return "undefined";
    case 0x08: return "same value";
    case 0x09: return "register";
    case 0x0a: return "remember state";
    case 0x0b: return "restore state";
    case 0x0c: return "def cfa";
    case 0x0d: return "def cfa register";
    case 0x0e: return "def cfa offset";
    case 0x0f: return "def cfa expression";
    case 0x10: return "expression";
    case 0x11: return "offset extended sf";
    case 0x12: return "def cfa sf";
    case 0x13: return "def cfa offset sf";
    case 0x1c: return "lo user";
    case 0x3f: return "hi user";
    default:
       snprintf (invalid, sizeof(invalid), "Unknown DW_CFA constant: 0x%x", val);
       return invalid;
  }
}

DRC_class
DW_CFA_value_to_class (uint32_t val)
{
  switch (val) {
    case 0x40: return DRC_ZEROOPERANDS;
    case 0x80: return DRC_ONEOPERAND | DRC_OPERANDONE_ULEB128_OFFSET;
    case 0xc0: return DRC_ZEROOPERANDS;
    case 0x00: return DRC_ZEROOPERANDS;
    case 0x01: return DRC_ONEOPERAND | DRC_OPERANDONE_ADDRESS;
    case 0x02: return DRC_ONEOPERAND | DRC_OPERANDONE_1BYTE_DELTA;
    case 0x03: return DRC_ONEOPERAND | DRC_OPERANDONE_2BYTE_DELTA;
    case 0x04: return DRC_ONEOPERAND | DRC_OPERANDONE_4BYTE_DELTA;
    case 0x05: return DRC_OPERANDTWO_ULEB128_OFFSET | DRC_OPERNADONE_ULEB128_REGISTER | DRC_TWOOPERANDS;
    case 0x06: return DRC_ONEOPERAND | DRC_OPERANDONE_ULEB128_REGISTER;
    case 0x07: return DRC_ONEOPERAND | DRC_OPERANDONE_ULEB128_REGISTER;
    case 0x08: return DRC_ONEOPERAND | DRC_OPERANDONE_ULEB128_REGISTER;
    case 0x09: return DRC_OPERANDONE_ULEB128_REGISTER | DRC_OPERANDTWO_ULEB128_REGISTER | DRC_TWOOPERANDS;
    case 0x0a: return DRC_ZEROOPERANDS;
    case 0x0b: return DRC_ZEROOPERANDS;
    case 0x0c: return DRC_OPERANDONE_ULEB128_REGISTER | DRC_OPERANDTWO_ULEB128_OFFSET | DRC_TWOOPERANDS;
    case 0x0d: return DRC_ONEOPERAND | DRC_OPERANDONE_ULEB128_REGISTER;
    case 0x0e: return DRC_ONEOPERAND | DRC_OPERANDONE_ULEB128_OFFSET;
    case 0x0f: return DRC_DWARFv3 | DRC_ONEOPERAND | DRC_OPERANDONE_BLOCK;
    case 0x10: return DRC_DWARFv3 | DRC_OPERANDONE_ULEB128_REGISTER | DRC_OPERANDTWO_BLOCK | DRC_TWOOPERANDS;
    case 0x11: return DRC_DWARFv3 | DRC_OPERANDONE_ULEB128_REGISTER | DRC_OPERANDTWO_SLEB128_OFFSET | DRC_TWOOPERANDS;
    case 0x12: return DRC_DWARFv3 | DRC_OPERANDONE_ULEB128_REGISTER | DRC_OPERANDTWO_SLEB128_OFFSET | DRC_TWOOPERANDS;
    case 0x13: return DRC_DWARFv3 | DRC_ONEOPERAND | DRC_OPERANDONE_SLEB128_OFFSET;
    case 0x1c: return 0;
    case 0x3f: return 0;
    default: return 0;
  }
}

/* FSF exception handling Pointer-Encoding constants (CFI augmentation) -- "DW_EH_PE_..." in the FSF sources */

const char *
DW_GNU_EH_PE_value_to_name (uint32_t val)
{
  static char invalid[100];
  switch (val) {
    case 0x00: return "DW_GNU_EH_PE_absptr";
    case 0x01: return "DW_GNU_EH_PE_uleb128";
    case 0x02: return "DW_GNU_EH_PE_udata2";
    case 0x03: return "DW_GNU_EH_PE_udata4";
    case 0x04: return "DW_GNU_EH_PE_udata8";
    case 0x09: return "DW_GNU_EH_PE_sleb128";
    case 0x0a: return "DW_GNU_EH_PE_sdata2";
    case 0x0b: return "DW_GNU_EH_PE_sdata4";
    case 0x0c: return "DW_GNU_EH_PE_sdata8";
    case 0x08: return "DW_GNU_EH_PE_signed";
    case 0x10: return "DW_GNU_EH_PE_pcrel";
    case 0x20: return "DW_GNU_EH_PE_textrel";
    case 0x30: return "DW_GNU_EH_PE_datarel";
    case 0x40: return "DW_GNU_EH_PE_funcrel";
    case 0x50: return "DW_GNU_EH_PE_aligned";
    case 0x80: return "DW_GNU_EH_PE_indirect";
    case 0xff: return "DW_GNU_EH_PE_omit";
    default:
       snprintf (invalid, sizeof(invalid), "Unknown DW_GNU_EH_PE constant: 0x%x", val);
       return invalid;
  }
}

const char *
DW_GNU_EH_PE_value_to_englishy_name (uint32_t val)
{
  static char invalid[100];
  switch (val) {
    case 0x00: return "absptr";
    case 0x01: return "uleb128";
    case 0x02: return "udata2";
    case 0x03: return "udata4";
    case 0x04: return "udata8";
    case 0x09: return "sleb128";
    case 0x0a: return "sdata2";
    case 0x0b: return "sdata4";
    case 0x0c: return "sdata8";
    case 0x08: return "signed";
    case 0x10: return "pcrel";
    case 0x20: return "textrel";
    case 0x30: return "datarel";
    case 0x40: return "funcrel";
    case 0x50: return "aligned";
    case 0x80: return "indirect";
    case 0xff: return "omit";
    default:
       snprintf (invalid, sizeof(invalid), "Unknown DW_GNU_EH_PE constant: 0x%x", val);
       return invalid;
  }
}

DRC_class
DW_GNU_EH_PE_value_to_class (uint32_t val)
{
  switch (val) {
    case 0x00: return DRC_VENDOR_GNU;
    case 0x01: return DRC_VENDOR_GNU;
    case 0x02: return DRC_VENDOR_GNU;
    case 0x03: return DRC_VENDOR_GNU;
    case 0x04: return DRC_VENDOR_GNU;
    case 0x09: return DRC_VENDOR_GNU;
    case 0x0a: return DRC_VENDOR_GNU;
    case 0x0b: return DRC_VENDOR_GNU;
    case 0x0c: return DRC_VENDOR_GNU;
    case 0x08: return DRC_VENDOR_GNU;
    case 0x10: return DRC_VENDOR_GNU;
    case 0x20: return DRC_VENDOR_GNU;
    case 0x30: return DRC_VENDOR_GNU;
    case 0x40: return DRC_VENDOR_GNU;
    case 0x50: return DRC_VENDOR_GNU;
    case 0x80: return DRC_VENDOR_GNU;
    case 0xff: return DRC_VENDOR_GNU;
    default: return 0;
  }
}

bool
is_type_tag (uint16_t tag)
{
  switch (tag)
    {
      case DW_TAG_array_type:
      case DW_TAG_base_type:
      case DW_TAG_class_type:
      case DW_TAG_const_type:
      case DW_TAG_enumeration_type:
      case DW_TAG_file_type:
      case DW_TAG_interface_type:
      case DW_TAG_packed_type:
      case DW_TAG_pointer_type:
      case DW_TAG_ptr_to_member_type:
      case DW_TAG_reference_type:
      case DW_TAG_restrict_type:
      case DW_TAG_set_type:
      case DW_TAG_shared_type:
      case DW_TAG_string_type:
      case DW_TAG_structure_type:
      case DW_TAG_subrange_type:
      case DW_TAG_subroutine_type:
      case DW_TAG_thrown_type:
      case DW_TAG_union_type:
      case DW_TAG_unspecified_type:
      case DW_TAG_volatile_type:
        return true;
      default:
        return false;
    }
}

bool
is_pubtype_tag (uint16_t tag)
{
  switch (tag)
    {
      case DW_TAG_array_type:
      case DW_TAG_class_type:
      case DW_TAG_enumeration_type:
      case DW_TAG_file_type:
      case DW_TAG_interface_type:
      case DW_TAG_set_type:
      case DW_TAG_string_type:
      case DW_TAG_structure_type:
      case DW_TAG_subrange_type:
      case DW_TAG_subroutine_type:
      case DW_TAG_thrown_type:
      case DW_TAG_typedef:
      case DW_TAG_union_type:
      case DW_TAG_unspecified_type:
        return true;
      default:
        break;
    }
  return false;
}

DW_TAG_CategoryEnum
get_tag_category (uint16_t tag)
{
  switch (tag)
    {
      case DW_TAG_array_type                 : return TagCategoryType;
      case DW_TAG_class_type                 : return TagCategoryType;
      case DW_TAG_entry_point                : return TagCategoryProgram;
      case DW_TAG_enumeration_type           : return TagCategoryType;
      case DW_TAG_formal_parameter           : return TagCategoryVariable;
      case DW_TAG_imported_declaration       : return TagCategoryProgram;
      case DW_TAG_label                      : return TagCategoryProgram;
      case DW_TAG_lexical_block              : return TagCategoryProgram;
      case DW_TAG_member                     : return TagCategoryType;
      case DW_TAG_pointer_type               : return TagCategoryType;
      case DW_TAG_reference_type             : return TagCategoryType;
      case DW_TAG_compile_unit               : return TagCategoryProgram;
      case DW_TAG_string_type                : return TagCategoryType;
      case DW_TAG_structure_type             : return TagCategoryType;
      case DW_TAG_subroutine_type            : return TagCategoryType;
      case DW_TAG_typedef                    : return TagCategoryType;
      case DW_TAG_union_type                 : return TagCategoryType;
      case DW_TAG_unspecified_parameters     : return TagCategoryVariable;
      case DW_TAG_variant                    : return TagCategoryType;
      case DW_TAG_common_block               : return TagCategoryProgram;
      case DW_TAG_common_inclusion           : return TagCategoryProgram;
      case DW_TAG_inheritance                : return TagCategoryType;
      case DW_TAG_inlined_subroutine         : return TagCategoryProgram;
      case DW_TAG_module                     : return TagCategoryProgram;
      case DW_TAG_ptr_to_member_type         : return TagCategoryType;
      case DW_TAG_set_type                   : return TagCategoryType;
      case DW_TAG_subrange_type              : return TagCategoryType;
      case DW_TAG_with_stmt                  : return TagCategoryProgram;
      case DW_TAG_access_declaration         : return TagCategoryProgram;
      case DW_TAG_base_type                  : return TagCategoryType;
      case DW_TAG_catch_block                : return TagCategoryProgram;
      case DW_TAG_const_type                 : return TagCategoryType;
      case DW_TAG_constant                   : return TagCategoryVariable;
      case DW_TAG_enumerator                 : return TagCategoryType;
      case DW_TAG_file_type                  : return TagCategoryType;
      case DW_TAG_friend                     : return TagCategoryType;
      case DW_TAG_namelist                   : return TagCategoryVariable;
      case DW_TAG_namelist_item              : return TagCategoryVariable;
      case DW_TAG_packed_type                : return TagCategoryType;
      case DW_TAG_subprogram                 : return TagCategoryProgram;
      case DW_TAG_template_type_parameter    : return TagCategoryType;
      case DW_TAG_template_value_parameter   : return TagCategoryType;
      case DW_TAG_thrown_type                : return TagCategoryType;
      case DW_TAG_try_block                  : return TagCategoryProgram;
      case DW_TAG_variant_part               : return TagCategoryType;
      case DW_TAG_variable                   : return TagCategoryVariable;
      case DW_TAG_volatile_type              : return TagCategoryType;
      case DW_TAG_dwarf_procedure            : return TagCategoryProgram;
      case DW_TAG_restrict_type              : return TagCategoryType;
      case DW_TAG_interface_type             : return TagCategoryType;
      case DW_TAG_namespace                  : return TagCategoryProgram;
      case DW_TAG_imported_module            : return TagCategoryProgram;
      case DW_TAG_unspecified_type           : return TagCategoryType;
      case DW_TAG_partial_unit               : return TagCategoryProgram;
      case DW_TAG_imported_unit              : return TagCategoryProgram;
      case DW_TAG_shared_type                : return TagCategoryType;
      default: break;
    }
    return TagCategoryProgram;
}

