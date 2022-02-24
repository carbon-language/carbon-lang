//===-- DWARFDebugInfoEntry.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DWARFDebugInfoEntry.h"

#include <cassert>

#include <algorithm>

#include "llvm/Support/LEB128.h"

#include "lldb/Core/Module.h"
#include "lldb/Expression/DWARFExpression.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Utility/Stream.h"

#include "DWARFCompileUnit.h"
#include "DWARFDebugAbbrev.h"
#include "DWARFDebugAranges.h"
#include "DWARFDebugInfo.h"
#include "DWARFDebugRanges.h"
#include "DWARFDeclContext.h"
#include "DWARFFormValue.h"
#include "DWARFUnit.h"
#include "SymbolFileDWARF.h"
#include "SymbolFileDWARFDwo.h"

using namespace lldb_private;
using namespace std;
extern int g_verbose;

// Extract a debug info entry for a given DWARFUnit from the data
// starting at the offset in offset_ptr
bool DWARFDebugInfoEntry::Extract(const DWARFDataExtractor &data,
                                  const DWARFUnit *cu,
                                  lldb::offset_t *offset_ptr) {
  m_offset = *offset_ptr;
  m_parent_idx = 0;
  m_sibling_idx = 0;
  const uint64_t abbr_idx = data.GetULEB128(offset_ptr);
  lldbassert(abbr_idx <= UINT16_MAX);
  m_abbr_idx = abbr_idx;

  // assert (fixed_form_sizes);  // For best performance this should be
  // specified!

  if (m_abbr_idx == 0) {
    m_tag = llvm::dwarf::DW_TAG_null;
    m_has_children = false;
    return true; // NULL debug tag entry
  }

  lldb::offset_t offset = *offset_ptr;
  const auto *abbrevDecl = GetAbbreviationDeclarationPtr(cu);
  if (abbrevDecl == nullptr) {
    cu->GetSymbolFileDWARF().GetObjectFile()->GetModule()->ReportError(
        "{0x%8.8x}: invalid abbreviation code %u, please file a bug and "
        "attach the file at the start of this error message",
        m_offset, (unsigned)abbr_idx);
    // WE can't parse anymore if the DWARF is borked...
    *offset_ptr = UINT32_MAX;
    return false;
  }
  m_tag = abbrevDecl->Tag();
  m_has_children = abbrevDecl->HasChildren();
  // Skip all data in the .debug_info or .debug_types for the attributes
  const uint32_t numAttributes = abbrevDecl->NumAttributes();
  uint32_t i;
  dw_form_t form;
  for (i = 0; i < numAttributes; ++i) {
    form = abbrevDecl->GetFormByIndexUnchecked(i);
    llvm::Optional<uint8_t> fixed_skip_size =
        DWARFFormValue::GetFixedSize(form, cu);
    if (fixed_skip_size)
      offset += *fixed_skip_size;
    else {
      bool form_is_indirect = false;
      do {
        form_is_indirect = false;
        uint32_t form_size = 0;
        switch (form) {
        // Blocks if inlined data that have a length field and the data bytes
        // inlined in the .debug_info/.debug_types
        case DW_FORM_exprloc:
        case DW_FORM_block:
          form_size = data.GetULEB128(&offset);
          break;
        case DW_FORM_block1:
          form_size = data.GetU8_unchecked(&offset);
          break;
        case DW_FORM_block2:
          form_size = data.GetU16_unchecked(&offset);
          break;
        case DW_FORM_block4:
          form_size = data.GetU32_unchecked(&offset);
          break;

        // Inlined NULL terminated C-strings
        case DW_FORM_string:
          data.GetCStr(&offset);
          break;

        // Compile unit address sized values
        case DW_FORM_addr:
          form_size = cu->GetAddressByteSize();
          break;
        case DW_FORM_ref_addr:
          if (cu->GetVersion() <= 2)
            form_size = cu->GetAddressByteSize();
          else
            form_size = 4;
          break;

        // 0 sized form
        case DW_FORM_flag_present:
          form_size = 0;
          break;

        // 1 byte values
        case DW_FORM_addrx1:
        case DW_FORM_data1:
        case DW_FORM_flag:
        case DW_FORM_ref1:
        case DW_FORM_strx1:
          form_size = 1;
          break;

        // 2 byte values
        case DW_FORM_addrx2:
        case DW_FORM_data2:
        case DW_FORM_ref2:
        case DW_FORM_strx2:
          form_size = 2;
          break;

        // 3 byte values
        case DW_FORM_addrx3:
        case DW_FORM_strx3:
          form_size = 3;
          break;

        // 4 byte values
        case DW_FORM_addrx4:
        case DW_FORM_data4:
        case DW_FORM_ref4:
        case DW_FORM_strx4:
          form_size = 4;
          break;

        // 8 byte values
        case DW_FORM_data8:
        case DW_FORM_ref8:
        case DW_FORM_ref_sig8:
          form_size = 8;
          break;

        // signed or unsigned LEB 128 values
        case DW_FORM_addrx:
        case DW_FORM_loclistx:
        case DW_FORM_rnglistx:
        case DW_FORM_sdata:
        case DW_FORM_udata:
        case DW_FORM_ref_udata:
        case DW_FORM_GNU_addr_index:
        case DW_FORM_GNU_str_index:
        case DW_FORM_strx:
          data.Skip_LEB128(&offset);
          break;

        case DW_FORM_indirect:
          form_is_indirect = true;
          form = data.GetULEB128(&offset);
          break;

        case DW_FORM_strp:
        case DW_FORM_line_strp:
        case DW_FORM_sec_offset:
          data.GetU32(&offset);
          break;

        case DW_FORM_implicit_const:
          form_size = 0;
          break;

        default:
          cu->GetSymbolFileDWARF().GetObjectFile()->GetModule()->ReportError(
              "{0x%8.8x}: Unsupported DW_FORM_0x%x, please file a bug and "
              "attach the file at the start of this error message",
              m_offset, (unsigned)form);
          *offset_ptr = m_offset;
          return false;
        }
        offset += form_size;

      } while (form_is_indirect);
    }
  }
  *offset_ptr = offset;
  return true;
}

static DWARFRangeList GetRangesOrReportError(DWARFUnit &unit,
                                             const DWARFDebugInfoEntry &die,
                                             const DWARFFormValue &value) {
  llvm::Expected<DWARFRangeList> expected_ranges =
      (value.Form() == DW_FORM_rnglistx)
          ? unit.FindRnglistFromIndex(value.Unsigned())
          : unit.FindRnglistFromOffset(value.Unsigned());
  if (expected_ranges)
    return std::move(*expected_ranges);
  unit.GetSymbolFileDWARF().GetObjectFile()->GetModule()->ReportError(
      "{0x%8.8x}: DIE has DW_AT_ranges(%s 0x%" PRIx64 ") attribute, but "
      "range extraction failed (%s), please file a bug "
      "and attach the file at the start of this error message",
      die.GetOffset(),
      llvm::dwarf::FormEncodingString(value.Form()).str().c_str(),
      value.Unsigned(), toString(expected_ranges.takeError()).c_str());
  return DWARFRangeList();
}

// GetDIENamesAndRanges
//
// Gets the valid address ranges for a given DIE by looking for a
// DW_AT_low_pc/DW_AT_high_pc pair, DW_AT_entry_pc, or DW_AT_ranges attributes.
bool DWARFDebugInfoEntry::GetDIENamesAndRanges(
    DWARFUnit *cu, const char *&name, const char *&mangled,
    DWARFRangeList &ranges, int &decl_file, int &decl_line, int &decl_column,
    int &call_file, int &call_line, int &call_column,
    DWARFExpression *frame_base) const {
  dw_addr_t lo_pc = LLDB_INVALID_ADDRESS;
  dw_addr_t hi_pc = LLDB_INVALID_ADDRESS;
  std::vector<DWARFDIE> dies;
  bool set_frame_base_loclist_addr = false;

  const auto *abbrevDecl = GetAbbreviationDeclarationPtr(cu);

  SymbolFileDWARF &dwarf = cu->GetSymbolFileDWARF();
  lldb::ModuleSP module = dwarf.GetObjectFile()->GetModule();

  if (abbrevDecl) {
    const DWARFDataExtractor &data = cu->GetData();
    lldb::offset_t offset = GetFirstAttributeOffset();

    if (!data.ValidOffset(offset))
      return false;

    const uint32_t numAttributes = abbrevDecl->NumAttributes();
    bool do_offset = false;

    for (uint32_t i = 0; i < numAttributes; ++i) {
      DWARFFormValue form_value(cu);
      dw_attr_t attr;
      abbrevDecl->GetAttrAndFormValueByIndex(i, attr, form_value);

      if (form_value.ExtractValue(data, &offset)) {
        switch (attr) {
        case DW_AT_low_pc:
          lo_pc = form_value.Address();

          if (do_offset)
            hi_pc += lo_pc;
          do_offset = false;
          break;

        case DW_AT_entry_pc:
          lo_pc = form_value.Address();
          break;

        case DW_AT_high_pc:
          if (form_value.Form() == DW_FORM_addr ||
              form_value.Form() == DW_FORM_addrx ||
              form_value.Form() == DW_FORM_GNU_addr_index) {
            hi_pc = form_value.Address();
          } else {
            hi_pc = form_value.Unsigned();
            if (lo_pc == LLDB_INVALID_ADDRESS)
              do_offset = hi_pc != LLDB_INVALID_ADDRESS;
            else
              hi_pc += lo_pc; // DWARF 4 introduces <offset-from-lo-pc> to save
                              // on relocations
          }
          break;

        case DW_AT_ranges:
          ranges = GetRangesOrReportError(*cu, *this, form_value);
          break;

        case DW_AT_name:
          if (name == nullptr)
            name = form_value.AsCString();
          break;

        case DW_AT_MIPS_linkage_name:
        case DW_AT_linkage_name:
          if (mangled == nullptr)
            mangled = form_value.AsCString();
          break;

        case DW_AT_abstract_origin:
          dies.push_back(form_value.Reference());
          break;

        case DW_AT_specification:
          dies.push_back(form_value.Reference());
          break;

        case DW_AT_decl_file:
          if (decl_file == 0)
            decl_file = form_value.Unsigned();
          break;

        case DW_AT_decl_line:
          if (decl_line == 0)
            decl_line = form_value.Unsigned();
          break;

        case DW_AT_decl_column:
          if (decl_column == 0)
            decl_column = form_value.Unsigned();
          break;

        case DW_AT_call_file:
          if (call_file == 0)
            call_file = form_value.Unsigned();
          break;

        case DW_AT_call_line:
          if (call_line == 0)
            call_line = form_value.Unsigned();
          break;

        case DW_AT_call_column:
          if (call_column == 0)
            call_column = form_value.Unsigned();
          break;

        case DW_AT_frame_base:
          if (frame_base) {
            if (form_value.BlockData()) {
              uint32_t block_offset =
                  form_value.BlockData() - data.GetDataStart();
              uint32_t block_length = form_value.Unsigned();
              *frame_base = DWARFExpression(
                  module, DataExtractor(data, block_offset, block_length), cu);
            } else {
              DataExtractor data = cu->GetLocationData();
              const dw_offset_t offset = form_value.Unsigned();
              if (data.ValidOffset(offset)) {
                data = DataExtractor(data, offset, data.GetByteSize() - offset);
                *frame_base = DWARFExpression(module, data, cu);
                if (lo_pc != LLDB_INVALID_ADDRESS) {
                  assert(lo_pc >= cu->GetBaseAddress());
                  frame_base->SetLocationListAddresses(cu->GetBaseAddress(),
                                                       lo_pc);
                } else {
                  set_frame_base_loclist_addr = true;
                }
              }
            }
          }
          break;

        default:
          break;
        }
      }
    }
  }

  if (ranges.IsEmpty()) {
    if (lo_pc != LLDB_INVALID_ADDRESS) {
      if (hi_pc != LLDB_INVALID_ADDRESS && hi_pc > lo_pc)
        ranges.Append(DWARFRangeList::Entry(lo_pc, hi_pc - lo_pc));
      else
        ranges.Append(DWARFRangeList::Entry(lo_pc, 0));
    }
  }

  if (set_frame_base_loclist_addr) {
    dw_addr_t lowest_range_pc = ranges.GetMinRangeBase(0);
    assert(lowest_range_pc >= cu->GetBaseAddress());
    frame_base->SetLocationListAddresses(cu->GetBaseAddress(), lowest_range_pc);
  }

  if (ranges.IsEmpty() || name == nullptr || mangled == nullptr) {
    for (const DWARFDIE &die : dies) {
      if (die) {
        die.GetDIE()->GetDIENamesAndRanges(die.GetCU(), name, mangled, ranges,
                                           decl_file, decl_line, decl_column,
                                           call_file, call_line, call_column);
      }
    }
  }
  return !ranges.IsEmpty();
}

// Get all attribute values for a given DIE, including following any
// specification or abstract origin attributes and including those in the
// results. Any duplicate attributes will have the first instance take
// precedence (this can happen for declaration attributes).
size_t DWARFDebugInfoEntry::GetAttributes(DWARFUnit *cu,
                                          DWARFAttributes &attributes,
                                          Recurse recurse,
                                          uint32_t curr_depth) const {
  const auto *abbrevDecl = GetAbbreviationDeclarationPtr(cu);
  if (abbrevDecl) {
    const DWARFDataExtractor &data = cu->GetData();
    lldb::offset_t offset = GetFirstAttributeOffset();

    const uint32_t num_attributes = abbrevDecl->NumAttributes();
    for (uint32_t i = 0; i < num_attributes; ++i) {
      DWARFFormValue form_value(cu);
      dw_attr_t attr;
      abbrevDecl->GetAttrAndFormValueByIndex(i, attr, form_value);
      const dw_form_t form = form_value.Form();

      // If we are tracking down DW_AT_specification or DW_AT_abstract_origin
      // attributes, the depth will be non-zero. We need to omit certain
      // attributes that don't make sense.
      switch (attr) {
      case DW_AT_sibling:
      case DW_AT_declaration:
        if (curr_depth > 0) {
          // This attribute doesn't make sense when combined with the DIE that
          // references this DIE. We know a DIE is referencing this DIE because
          // curr_depth is not zero
          break;
        }
        LLVM_FALLTHROUGH;
      default:
        attributes.Append(form_value, offset, attr);
        break;
      }

      if (recurse == Recurse::yes &&
          ((attr == DW_AT_specification) || (attr == DW_AT_abstract_origin))) {
        if (form_value.ExtractValue(data, &offset)) {
          DWARFDIE spec_die = form_value.Reference();
          if (spec_die)
            spec_die.GetDIE()->GetAttributes(spec_die.GetCU(), attributes,
                                             recurse, curr_depth + 1);
        }
      } else {
        llvm::Optional<uint8_t> fixed_skip_size = DWARFFormValue::GetFixedSize(form, cu);
        if (fixed_skip_size)
          offset += *fixed_skip_size;
        else
          DWARFFormValue::SkipValue(form, data, &offset, cu);
      }
    }
  } else {
    attributes.Clear();
  }
  return attributes.Size();
}

// GetAttributeValue
//
// Get the value of an attribute and return the .debug_info or .debug_types
// offset of the attribute if it was properly extracted into form_value,
// or zero if we fail since an offset of zero is invalid for an attribute (it
// would be a compile unit header).
dw_offset_t DWARFDebugInfoEntry::GetAttributeValue(
    const DWARFUnit *cu, const dw_attr_t attr, DWARFFormValue &form_value,
    dw_offset_t *end_attr_offset_ptr,
    bool check_specification_or_abstract_origin) const {
  if (const auto *abbrevDecl = GetAbbreviationDeclarationPtr(cu)) {
    uint32_t attr_idx = abbrevDecl->FindAttributeIndex(attr);

    if (attr_idx != DW_INVALID_INDEX) {
      const DWARFDataExtractor &data = cu->GetData();
      lldb::offset_t offset = GetFirstAttributeOffset();

      uint32_t idx = 0;
      while (idx < attr_idx)
        DWARFFormValue::SkipValue(abbrevDecl->GetFormByIndex(idx++),
                                  data, &offset, cu);

      const dw_offset_t attr_offset = offset;
      form_value.SetUnit(cu);
      form_value.SetForm(abbrevDecl->GetFormByIndex(idx));
      if (form_value.ExtractValue(data, &offset)) {
        if (end_attr_offset_ptr)
          *end_attr_offset_ptr = offset;
        return attr_offset;
      }
    }
  }

  if (check_specification_or_abstract_origin) {
    if (GetAttributeValue(cu, DW_AT_specification, form_value)) {
      DWARFDIE die = form_value.Reference();
      if (die) {
        dw_offset_t die_offset = die.GetDIE()->GetAttributeValue(
            die.GetCU(), attr, form_value, end_attr_offset_ptr, false);
        if (die_offset)
          return die_offset;
      }
    }

    if (GetAttributeValue(cu, DW_AT_abstract_origin, form_value)) {
      DWARFDIE die = form_value.Reference();
      if (die) {
        dw_offset_t die_offset = die.GetDIE()->GetAttributeValue(
            die.GetCU(), attr, form_value, end_attr_offset_ptr, false);
        if (die_offset)
          return die_offset;
      }
    }
  }
  return 0;
}

// GetAttributeValueAsString
//
// Get the value of an attribute as a string return it. The resulting pointer
// to the string data exists within the supplied SymbolFileDWARF and will only
// be available as long as the SymbolFileDWARF is still around and it's content
// doesn't change.
const char *DWARFDebugInfoEntry::GetAttributeValueAsString(
    const DWARFUnit *cu, const dw_attr_t attr, const char *fail_value,
    bool check_specification_or_abstract_origin) const {
  DWARFFormValue form_value;
  if (GetAttributeValue(cu, attr, form_value, nullptr,
                        check_specification_or_abstract_origin))
    return form_value.AsCString();
  return fail_value;
}

// GetAttributeValueAsUnsigned
//
// Get the value of an attribute as unsigned and return it.
uint64_t DWARFDebugInfoEntry::GetAttributeValueAsUnsigned(
    const DWARFUnit *cu, const dw_attr_t attr, uint64_t fail_value,
    bool check_specification_or_abstract_origin) const {
  DWARFFormValue form_value;
  if (GetAttributeValue(cu, attr, form_value, nullptr,
                        check_specification_or_abstract_origin))
    return form_value.Unsigned();
  return fail_value;
}

// GetAttributeValueAsReference
//
// Get the value of an attribute as reference and fix up and compile unit
// relative offsets as needed.
DWARFDIE DWARFDebugInfoEntry::GetAttributeValueAsReference(
    const DWARFUnit *cu, const dw_attr_t attr,
    bool check_specification_or_abstract_origin) const {
  DWARFFormValue form_value;
  if (GetAttributeValue(cu, attr, form_value, nullptr,
                        check_specification_or_abstract_origin))
    return form_value.Reference();
  return {};
}

uint64_t DWARFDebugInfoEntry::GetAttributeValueAsAddress(
    const DWARFUnit *cu, const dw_attr_t attr, uint64_t fail_value,
    bool check_specification_or_abstract_origin) const {
  DWARFFormValue form_value;
  if (GetAttributeValue(cu, attr, form_value, nullptr,
                        check_specification_or_abstract_origin))
    return form_value.Address();
  return fail_value;
}

// GetAttributeHighPC
//
// Get the hi_pc, adding hi_pc to lo_pc when specified as an <offset-from-low-
// pc>.
//
// Returns the hi_pc or fail_value.
dw_addr_t DWARFDebugInfoEntry::GetAttributeHighPC(
    const DWARFUnit *cu, dw_addr_t lo_pc, uint64_t fail_value,
    bool check_specification_or_abstract_origin) const {
  DWARFFormValue form_value;
  if (GetAttributeValue(cu, DW_AT_high_pc, form_value, nullptr,
                        check_specification_or_abstract_origin)) {
    dw_form_t form = form_value.Form();
    if (form == DW_FORM_addr || form == DW_FORM_addrx ||
        form == DW_FORM_GNU_addr_index)
      return form_value.Address();

    // DWARF4 can specify the hi_pc as an <offset-from-lowpc>
    return lo_pc + form_value.Unsigned();
  }
  return fail_value;
}

// GetAttributeAddressRange
//
// Get the lo_pc and hi_pc, adding hi_pc to lo_pc when specified as an <offset-
// from-low-pc>.
//
// Returns true or sets lo_pc and hi_pc to fail_value.
bool DWARFDebugInfoEntry::GetAttributeAddressRange(
    const DWARFUnit *cu, dw_addr_t &lo_pc, dw_addr_t &hi_pc,
    uint64_t fail_value, bool check_specification_or_abstract_origin) const {
  lo_pc = GetAttributeValueAsAddress(cu, DW_AT_low_pc, fail_value,
                                     check_specification_or_abstract_origin);
  if (lo_pc != fail_value) {
    hi_pc = GetAttributeHighPC(cu, lo_pc, fail_value,
                               check_specification_or_abstract_origin);
    if (hi_pc != fail_value)
      return true;
  }
  lo_pc = fail_value;
  hi_pc = fail_value;
  return false;
}

size_t DWARFDebugInfoEntry::GetAttributeAddressRanges(
    DWARFUnit *cu, DWARFRangeList &ranges, bool check_hi_lo_pc,
    bool check_specification_or_abstract_origin) const {
  ranges.Clear();

  DWARFFormValue form_value;
  if (GetAttributeValue(cu, DW_AT_ranges, form_value)) {
    ranges = GetRangesOrReportError(*cu, *this, form_value);
  } else if (check_hi_lo_pc) {
    dw_addr_t lo_pc = LLDB_INVALID_ADDRESS;
    dw_addr_t hi_pc = LLDB_INVALID_ADDRESS;
    if (GetAttributeAddressRange(cu, lo_pc, hi_pc, LLDB_INVALID_ADDRESS,
                                 check_specification_or_abstract_origin)) {
      if (lo_pc < hi_pc)
        ranges.Append(DWARFRangeList::Entry(lo_pc, hi_pc - lo_pc));
    }
  }
  return ranges.GetSize();
}

// GetName
//
// Get value of the DW_AT_name attribute and return it if one exists, else
// return NULL.
const char *DWARFDebugInfoEntry::GetName(const DWARFUnit *cu) const {
  return GetAttributeValueAsString(cu, DW_AT_name, nullptr, true);
}

// GetMangledName
//
// Get value of the DW_AT_MIPS_linkage_name attribute and return it if one
// exists, else return the value of the DW_AT_name attribute
const char *
DWARFDebugInfoEntry::GetMangledName(const DWARFUnit *cu,
                                    bool substitute_name_allowed) const {
  const char *name = nullptr;

  name = GetAttributeValueAsString(cu, DW_AT_MIPS_linkage_name, nullptr, true);
  if (name)
    return name;

  name = GetAttributeValueAsString(cu, DW_AT_linkage_name, nullptr, true);
  if (name)
    return name;

  if (!substitute_name_allowed)
    return nullptr;

  name = GetAttributeValueAsString(cu, DW_AT_name, nullptr, true);
  return name;
}

// GetPubname
//
// Get value the name for a DIE as it should appear for a .debug_pubnames or
// .debug_pubtypes section.
const char *DWARFDebugInfoEntry::GetPubname(const DWARFUnit *cu) const {
  const char *name = nullptr;
  if (!cu)
    return name;

  name = GetAttributeValueAsString(cu, DW_AT_MIPS_linkage_name, nullptr, true);
  if (name)
    return name;

  name = GetAttributeValueAsString(cu, DW_AT_linkage_name, nullptr, true);
  if (name)
    return name;

  name = GetAttributeValueAsString(cu, DW_AT_name, nullptr, true);
  return name;
}

/// This function is builds a table very similar to the standard .debug_aranges
/// table, except that the actual DIE offset for the function is placed in the
/// table instead of the compile unit offset.
void DWARFDebugInfoEntry::BuildFunctionAddressRangeTable(
    DWARFUnit *cu, DWARFDebugAranges *debug_aranges) const {
  if (m_tag) {
    if (m_tag == DW_TAG_subprogram) {
      DWARFRangeList ranges;
      GetAttributeAddressRanges(cu, ranges,
                                /*check_hi_lo_pc=*/true);
      for (const auto &r : ranges) {
        debug_aranges->AppendRange(GetOffset(), r.GetRangeBase(),
                                   r.GetRangeEnd());
      }
    }

    const DWARFDebugInfoEntry *child = GetFirstChild();
    while (child) {
      child->BuildFunctionAddressRangeTable(cu, debug_aranges);
      child = child->GetSibling();
    }
  }
}

DWARFDeclContext
DWARFDebugInfoEntry::GetDWARFDeclContextStatic(const DWARFDebugInfoEntry *die,
                                               DWARFUnit *cu) {
  DWARFDeclContext dwarf_decl_ctx;
  for (;;) {
    const dw_tag_t tag = die->Tag();
    if (tag == DW_TAG_compile_unit || tag == DW_TAG_partial_unit)
      return dwarf_decl_ctx;
    dwarf_decl_ctx.AppendDeclContext(tag, die->GetName(cu));
    DWARFDIE parent_decl_ctx_die = die->GetParentDeclContextDIE(cu);
    if (!parent_decl_ctx_die || parent_decl_ctx_die.GetDIE() == die)
      return dwarf_decl_ctx;
    if (parent_decl_ctx_die.Tag() == DW_TAG_compile_unit ||
        parent_decl_ctx_die.Tag() == DW_TAG_partial_unit)
      return dwarf_decl_ctx;
    die = parent_decl_ctx_die.GetDIE();
    cu = parent_decl_ctx_die.GetCU();
  }
}

DWARFDeclContext DWARFDebugInfoEntry::GetDWARFDeclContext(DWARFUnit *cu) const {
  return GetDWARFDeclContextStatic(this, cu);
}

DWARFDIE
DWARFDebugInfoEntry::GetParentDeclContextDIE(DWARFUnit *cu) const {
  DWARFAttributes attributes;
  GetAttributes(cu, attributes, Recurse::yes);
  return GetParentDeclContextDIE(cu, attributes);
}

DWARFDIE
DWARFDebugInfoEntry::GetParentDeclContextDIE(
    DWARFUnit *cu, const DWARFAttributes &attributes) const {
  DWARFDIE die(cu, const_cast<DWARFDebugInfoEntry *>(this));

  while (die) {
    // If this is the original DIE that we are searching for a declaration for,
    // then don't look in the cache as we don't want our own decl context to be
    // our decl context...
    if (die.GetDIE() != this) {
      switch (die.Tag()) {
      case DW_TAG_compile_unit:
      case DW_TAG_partial_unit:
      case DW_TAG_namespace:
      case DW_TAG_structure_type:
      case DW_TAG_union_type:
      case DW_TAG_class_type:
        return die;

      default:
        break;
      }
    }

    DWARFDIE spec_die = attributes.FormValueAsReference(DW_AT_specification);
    if (spec_die) {
      DWARFDIE decl_ctx_die = spec_die.GetParentDeclContextDIE();
      if (decl_ctx_die)
        return decl_ctx_die;
    }

    DWARFDIE abs_die = attributes.FormValueAsReference(DW_AT_abstract_origin);
    if (abs_die) {
      DWARFDIE decl_ctx_die = abs_die.GetParentDeclContextDIE();
      if (decl_ctx_die)
        return decl_ctx_die;
    }

    die = die.GetParent();
  }
  return DWARFDIE();
}

const char *DWARFDebugInfoEntry::GetQualifiedName(DWARFUnit *cu,
                                                  std::string &storage) const {
  DWARFAttributes attributes;
  GetAttributes(cu, attributes, Recurse::yes);
  return GetQualifiedName(cu, attributes, storage);
}

const char *
DWARFDebugInfoEntry::GetQualifiedName(DWARFUnit *cu,
                                      const DWARFAttributes &attributes,
                                      std::string &storage) const {

  const char *name = GetName(cu);

  if (name) {
    DWARFDIE parent_decl_ctx_die = GetParentDeclContextDIE(cu);
    storage.clear();
    // TODO: change this to get the correct decl context parent....
    while (parent_decl_ctx_die) {
      const dw_tag_t parent_tag = parent_decl_ctx_die.Tag();
      switch (parent_tag) {
      case DW_TAG_namespace: {
        const char *namespace_name = parent_decl_ctx_die.GetName();
        if (namespace_name) {
          storage.insert(0, "::");
          storage.insert(0, namespace_name);
        } else {
          storage.insert(0, "(anonymous namespace)::");
        }
        parent_decl_ctx_die = parent_decl_ctx_die.GetParentDeclContextDIE();
      } break;

      case DW_TAG_class_type:
      case DW_TAG_structure_type:
      case DW_TAG_union_type: {
        const char *class_union_struct_name = parent_decl_ctx_die.GetName();

        if (class_union_struct_name) {
          storage.insert(0, "::");
          storage.insert(0, class_union_struct_name);
        }
        parent_decl_ctx_die = parent_decl_ctx_die.GetParentDeclContextDIE();
      } break;

      default:
        parent_decl_ctx_die.Clear();
        break;
      }
    }

    if (storage.empty())
      storage.append("::");

    storage.append(name);
  }
  if (storage.empty())
    return nullptr;
  return storage.c_str();
}

lldb::offset_t DWARFDebugInfoEntry::GetFirstAttributeOffset() const {
  return GetOffset() + llvm::getULEB128Size(m_abbr_idx);
}

const DWARFAbbreviationDeclaration *
DWARFDebugInfoEntry::GetAbbreviationDeclarationPtr(const DWARFUnit *cu) const {
  if (cu) {
    const DWARFAbbreviationDeclarationSet *abbrev_set = cu->GetAbbreviations();
    if (abbrev_set)
      return abbrev_set->GetAbbreviationDeclaration(m_abbr_idx);
  }
  return nullptr;
}

bool DWARFDebugInfoEntry::IsGlobalOrStaticScopeVariable() const {
  if (Tag() != DW_TAG_variable)
    return false;
  const DWARFDebugInfoEntry *parent_die = GetParent();
  while (parent_die != nullptr) {
    switch (parent_die->Tag()) {
    case DW_TAG_subprogram:
    case DW_TAG_lexical_block:
    case DW_TAG_inlined_subroutine:
      return false;

    case DW_TAG_compile_unit:
    case DW_TAG_partial_unit:
      return true;

    default:
      break;
    }
    parent_die = parent_die->GetParent();
  }
  return false;
}

bool DWARFDebugInfoEntry::operator==(const DWARFDebugInfoEntry &rhs) const {
  return m_offset == rhs.m_offset && m_parent_idx == rhs.m_parent_idx &&
         m_sibling_idx == rhs.m_sibling_idx &&
         m_abbr_idx == rhs.m_abbr_idx && m_has_children == rhs.m_has_children &&
         m_tag == rhs.m_tag;
}

bool DWARFDebugInfoEntry::operator!=(const DWARFDebugInfoEntry &rhs) const {
  return !(*this == rhs);
}
