//===-- DWARFFormValue.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_DWARFFORMVALUE_H
#define LLVM_DEBUGINFO_DWARFFORMVALUE_H

#include "llvm/Support/DataExtractor.h"

namespace llvm {

class DWARFUnit;
class raw_ostream;

class DWARFFormValue {
public:
  struct ValueType {
    ValueType() : data(NULL), IsDWOIndex(false) {
      uval = 0;
    }

    union {
      uint64_t uval;
      int64_t sval;
      const char* cstr;
    };
    const uint8_t* data;
    bool IsDWOIndex;
  };

private:
  uint16_t Form;   // Form for this value.
  ValueType Value; // Contains all data for the form.

public:
  DWARFFormValue(uint16_t form = 0) : Form(form) {}
  uint16_t getForm() const { return Form; }
  const ValueType& value() const { return Value; }
  void dump(raw_ostream &OS, const DWARFUnit *U) const;
  bool extractValue(DataExtractor data, uint32_t *offset_ptr,
                    const DWARFUnit *u);
  bool isInlinedCStr() const {
    return Value.data != NULL && Value.data == (const uint8_t*)Value.cstr;
  }

  uint64_t getReference(const DWARFUnit *U) const;
  uint64_t getUnsigned() const { return Value.uval; }
  int64_t getSigned() const { return Value.sval; }
  const char *getAsCString(const DWARFUnit *U) const;
  uint64_t getAsAddress(const DWARFUnit *U) const;

  bool skipValue(DataExtractor debug_info_data, uint32_t *offset_ptr,
                 const DWARFUnit *u) const;
  static bool skipValue(uint16_t form, DataExtractor debug_info_data,
                        uint32_t *offset_ptr, const DWARFUnit *u);

  static const uint8_t *getFixedFormSizes(uint8_t AddrSize, uint16_t Version);
};

}

#endif
