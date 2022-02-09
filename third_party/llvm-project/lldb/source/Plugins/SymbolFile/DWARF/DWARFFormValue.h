//===-- DWARFFormValue.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_DWARFFORMVALUE_H
#define LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_DWARFFORMVALUE_H

#include "DWARFDataExtractor.h"
#include "llvm/ADT/Optional.h"
#include <cstddef>

class DWARFUnit;
class SymbolFileDWARF;
class DWARFDIE;

class DWARFFormValue {
public:
  typedef struct ValueTypeTag {
    ValueTypeTag() : value() { value.uval = 0; }

    union {
      uint64_t uval;
      int64_t sval;
      const char *cstr;
    } value;
    const uint8_t *data = nullptr;
  } ValueType;

  enum {
    eValueTypeInvalid = 0,
    eValueTypeUnsigned,
    eValueTypeSigned,
    eValueTypeCStr,
    eValueTypeBlock
  };

  DWARFFormValue() = default;
  DWARFFormValue(const DWARFUnit *unit) : m_unit(unit) {}
  DWARFFormValue(const DWARFUnit *unit, dw_form_t form)
      : m_unit(unit), m_form(form) {}
  const DWARFUnit *GetUnit() const { return m_unit; }
  void SetUnit(const DWARFUnit *unit) { m_unit = unit; }
  dw_form_t Form() const { return m_form; }
  dw_form_t& FormRef() { return m_form; }
  void SetForm(dw_form_t form) { m_form = form; }
  const ValueType &Value() const { return m_value; }
  ValueType &ValueRef() { return m_value; }
  void SetValue(const ValueType &val) { m_value = val; }

  void Dump(lldb_private::Stream &s) const;
  bool ExtractValue(const lldb_private::DWARFDataExtractor &data,
                    lldb::offset_t *offset_ptr);
  const uint8_t *BlockData() const;
  static llvm::Optional<uint8_t> GetFixedSize(dw_form_t form,
                                              const DWARFUnit *u);
  llvm::Optional<uint8_t> GetFixedSize() const;
  DWARFDIE Reference() const;
  uint64_t Reference(dw_offset_t offset) const;
  bool Boolean() const { return m_value.value.uval != 0; }
  uint64_t Unsigned() const { return m_value.value.uval; }
  void SetUnsigned(uint64_t uval) { m_value.value.uval = uval; }
  int64_t Signed() const { return m_value.value.sval; }
  void SetSigned(int64_t sval) { m_value.value.sval = sval; }
  const char *AsCString() const;
  dw_addr_t Address() const;
  bool IsValid() const { return m_form != 0; }
  bool SkipValue(const lldb_private::DWARFDataExtractor &debug_info_data,
                 lldb::offset_t *offset_ptr) const;
  static bool SkipValue(const dw_form_t form,
                        const lldb_private::DWARFDataExtractor &debug_info_data,
                        lldb::offset_t *offset_ptr, const DWARFUnit *unit);
  static bool IsBlockForm(const dw_form_t form);
  static bool IsDataForm(const dw_form_t form);
  static int Compare(const DWARFFormValue &a, const DWARFFormValue &b);
  void Clear();
  static bool FormIsSupported(dw_form_t form);

protected:
  // Compile unit where m_value was located.
  // It may be different from compile unit where m_value refers to.
  const DWARFUnit *m_unit = nullptr; // Unit for this form
  dw_form_t m_form = 0;              // Form for this value
  ValueType m_value;            // Contains all data for the form
};

#endif // LLDB_SOURCE_PLUGINS_SYMBOLFILE_DWARF_DWARFFORMVALUE_H
