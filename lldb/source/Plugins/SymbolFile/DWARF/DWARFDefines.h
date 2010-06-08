//===-- DWARFDefines.h ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_DWARFDefines_h_
#define liblldb_DWARFDefines_h_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>
#include "lldb/Core/dwarf.h"

/* DWARF constants generated on Wed Sep  7 16:41:50 2005 */

typedef uint32_t DRC_class;          // Holds DRC_* class bitfields

/* [7.5.4] Figure 16 "Tag Encodings" (pp. 125-127) in DWARFv3 draft 8 */


enum DW_TAG_Category
{
    TagCategoryVariable,
    TagCategoryType,
    TagCategoryProgram,
    kNumTagCategories
};

typedef enum DW_TAG_Category DW_TAG_CategoryEnum;
const char *DW_TAG_value_to_name (uint32_t val);
const char *DW_TAG_value_to_englishy_name (uint32_t val);
DRC_class DW_TAG_value_to_class (uint32_t val);
DW_TAG_CategoryEnum get_tag_category (uint16_t tag);
#define DW_TAG_MAX_NAME_LENGTH 31


/* [7.5.4] Figure 17 "Child determination encodings" (p. 128) in DWARFv3 draft 8 */

const char *DW_CHILDREN_value_to_name (uint8_t val);
const char *DW_CHILDREN_value_to_englishy_name (uint8_t val);
DRC_class DW_CHILDREN_value_to_class (uint32_t val);
#define DW_CHILDREN_MAX_NAME_LENGTH 15


/* [7.5.4] Figure 18 "Attribute encodings" (pp. 129-132) in DWARFv3 draft 8 */


const char *DW_AT_value_to_name (uint32_t val);
const char *DW_AT_value_to_englishy_name (uint32_t val);
DRC_class DW_AT_value_to_class (uint32_t val);
#define DW_AT_MAX_NAME_LENGTH 34


/* [7.5.4] Figure 19 "Attribute form encodings" (pp. 133-134) in DWARFv3 draft 8 */

const char *DW_FORM_value_to_name (uint32_t val);
const char *DW_FORM_value_to_englishy_name (uint32_t val);
DRC_class DW_FORM_value_to_class (uint32_t val);
#define DW_FORM_MAX_NAME_LENGTH 17


/* [7.7.1] Figure 22 "DWARF operation encodings" (pp. 136-139) in DWARFv3 draft 8 */

const char *DW_OP_value_to_name (uint32_t val);
const char *DW_OP_value_to_englishy_name (uint32_t val);
DRC_class DW_OP_value_to_class (uint32_t val);
#define DW_OP_MAX_NAME_LENGTH 25


/* [7.8] Figure 23 "Base type encoding values" (pp. 140-141) in DWARFv3 draft 8 */

const char *DW_ATE_value_to_name (uint32_t val);
const char *DW_ATE_value_to_englishy_name (uint32_t val);
DRC_class DW_ATE_value_to_class (uint32_t val);
#define DW_ATE_MAX_NAME_LENGTH 22


/* [7.9] Figure 24 "Accessibility encodings" (p. 141) in DWARFv3 draft 8 */

const char *DW_ACCESS_value_to_name (uint32_t val);
const char *DW_ACCESS_value_to_englishy_name (uint32_t val);
DRC_class DW_ACCESS_value_to_class (uint32_t val);
#define DW_ACCESS_MAX_NAME_LENGTH 19


/* [7.10] Figure 25 "Visibility encodings" (p. 142) in DWARFv3 draft 8 */

const char *DW_VIS_value_to_name (uint32_t val);
const char *DW_VIS_value_to_englishy_name (uint32_t val);
DRC_class DW_VIS_value_to_class (uint32_t val);
#define DW_VIS_MAX_NAME_LENGTH 16


/* [7.11] Figure 26 "Virtuality encodings" (p. 142) in DWARFv3 draft 8 */

const char *DW_VIRTUALITY_value_to_name (uint32_t val);
const char *DW_VIRTUALITY_value_to_englishy_name (uint32_t val);
DRC_class DW_VIRTUALITY_value_to_class (uint32_t val);
#define DW_VIRTUALITY_MAX_NAME_LENGTH 26


/* [7.12] Figure 27 "Language encodings" (p. 143) in DWARFv3 draft 8 */

const char *DW_LANG_value_to_name (uint32_t val);
const char *DW_LANG_value_to_englishy_name (uint32_t val);
DRC_class DW_LANG_value_to_class (uint32_t val);
#define DW_LANG_MAX_NAME_LENGTH 19


/* [7.13], "Address Class Encodings" (p. 144) in DWARFv3 draft 8 */

const char *DW_ADDR_value_to_name (uint32_t val);
const char *DW_ADDR_value_to_englishy_name (uint32_t val);
DRC_class DW_ADDR_value_to_class (uint32_t val);
#define DW_ADDR_MAX_NAME_LENGTH 12


/* [7.14] Figure 28 "Identifier case encodings" (p. 144) in DWARFv3 draft 8 */

const char *DW_ID_value_to_name (uint32_t val);
const char *DW_ID_value_to_englishy_name (uint32_t val);
DRC_class DW_ID_value_to_class (uint32_t val);
#define DW_ID_MAX_NAME_LENGTH 22


/* [7.15] Figure 29 "Calling convention encodings" (p. 144) in DWARFv3 draft 8 */

const char *DW_CC_value_to_name (uint32_t val);
const char *DW_CC_value_to_englishy_name (uint32_t val);
DRC_class DW_CC_value_to_class (uint32_t val);
#define DW_CC_MAX_NAME_LENGTH 13


/* [7.16] Figure 30 "Inline encodings" (p. 145) in DWARFv3 draft 8 */

const char *DW_INL_value_to_name (uint32_t val);
const char *DW_INL_value_to_englishy_name (uint32_t val);
DRC_class DW_INL_value_to_class (uint32_t val);
#define DW_INL_MAX_NAME_LENGTH 27


/* [7.17] Figure 31 "Ordering encodings" (p. 145) in DWARFv3 draft 8 */

const char *DW_ORD_value_to_name (uint32_t val);
const char *DW_ORD_value_to_englishy_name (uint32_t val);
DRC_class DW_ORD_value_to_class (uint32_t val);
#define DW_ORD_MAX_NAME_LENGTH 16


/* [7.18] Figure 32 "Discriminant descriptor encodings" (p. 146) in DWARFv3 draft 8 */

const char *DW_DSC_value_to_name (uint32_t val);
const char *DW_DSC_value_to_englishy_name (uint32_t val);
DRC_class DW_DSC_value_to_class (uint32_t val);
#define DW_DSC_MAX_NAME_LENGTH 12


/* [7.21] Figure 33 "Line Number Standard Opcode Encodings" (pp. 148-149) in DWARFv3 draft 8 */

const char *DW_LNS_value_to_name (uint32_t val);
const char *DW_LNS_value_to_englishy_name (uint32_t val);
DRC_class DW_LNS_value_to_class (uint32_t val);
#define DW_LNS_MAX_NAME_LENGTH 25


/* [7.21] Figure 34 "Line Number Extended Opcode Encodings" (p. 149) in DWARFv3 draft 8 */

const char *DW_LNE_value_to_name (uint32_t val);
const char *DW_LNE_value_to_englishy_name (uint32_t val);
DRC_class DW_LNE_value_to_class (uint32_t val);
#define DW_LNE_MAX_NAME_LENGTH 19


/* [7.22] Figure 35 "Macinfo Type Encodings" (p. 150) in DWARFv3 draft 8 */

const char *DW_MACINFO_value_to_name (uint32_t val);
const char *DW_MACINFO_value_to_englishy_name (uint32_t val);
DRC_class DW_MACINFO_value_to_class (uint32_t val);
#define DW_MACINFO_MAX_NAME_LENGTH 21


/* [7.23] Figure 36 "Call frame instruction encodings" (pp. 151-152) in DWARFv3 draft 8 */

const char *DW_CFA_value_to_name (uint32_t val);
const char *DW_CFA_value_to_englishy_name (uint32_t val);
DRC_class DW_CFA_value_to_class (uint32_t val);
#define DW_CFA_MAX_NAME_LENGTH 25


/* FSF exception handling Pointer-Encoding constants (CFI augmentation) -- "DW_EH_PE_..." in the FSF sources */

const char *DW_GNU_EH_PE_value_to_name (uint32_t val);
const char *DW_GNU_EH_PE_value_to_englishy_name (uint32_t val);
DRC_class DW_GNU_EH_PE_value_to_class (uint32_t val);
#define DW_GNU_EH_PE_MAX_NAME_LENGTH 21


/* These DRC are entirely our own construction,
    although they are derived from various comments in the DWARF standard.
    Most of these are not useful to the parser, but the DW_AT and DW_FORM
    classes should prove to be usable in some fashion.  */

#define DRC_0x65                               0x1
#define DRC_ADDRESS                            0x2
#define DRC_BLOCK                              0x4
#define DRC_CONSTANT                           0x8
#define DRC_DWARFv3                           0x10
#define DRC_FLAG                              0x20
#define DRC_INDIRECT_SPECIAL                  0x40
#define DRC_LINEPTR                           0x80
#define DRC_LOCEXPR                          0x100
#define DRC_LOCLISTPTR                       0x200
#define DRC_MACPTR                           0x400
#define DRC_ONEOPERAND                       0x800
#define DRC_OPERANDONE_1BYTE_DELTA          0x1000
#define DRC_OPERANDONE_2BYTE_DELTA          0x2000
#define DRC_OPERANDONE_4BYTE_DELTA          0x4000
#define DRC_OPERANDONE_ADDRESS              0x8000
#define DRC_OPERANDONE_BLOCK               0x10000
#define DRC_OPERANDONE_SLEB128_OFFSET      0x20000
#define DRC_OPERANDONE_ULEB128_OFFSET      0x40000
#define DRC_OPERANDONE_ULEB128_REGISTER    0x80000
#define DRC_OPERANDTWO_BLOCK              0x100000
#define DRC_OPERANDTWO_SLEB128_OFFSET     0x200000
#define DRC_OPERANDTWO_ULEB128_OFFSET     0x400000
#define DRC_OPERANDTWO_ULEB128_REGISTER   0x800000
#define DRC_OPERNADONE_ULEB128_REGISTER  0x1000000
#define DRC_RANGELISTPTR                 0x2000000
#define DRC_REFERENCE                    0x4000000
#define DRC_STRING                       0x8000000
#define DRC_TWOOPERANDS                 0x10000000
#define DRC_VENDOR_GNU                  0x20000000
#define DRC_VENDOR_MIPS                 0x40000000
#define DRC_ZEROOPERANDS                0x80000000

bool is_type_tag (uint16_t tag);
bool is_pubtype_tag (uint16_t tag);


#ifdef __cplusplus
}
#endif


#endif  // liblldb_DWARFDefines_h_
