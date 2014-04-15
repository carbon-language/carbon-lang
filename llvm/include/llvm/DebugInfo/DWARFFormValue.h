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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/Support/DataExtractor.h"

namespace llvm {

class DWARFUnit;
class raw_ostream;

class DWARFFormValue {
public:
  enum FormClass {
    FC_Unknown,
    FC_Address,
    FC_Block,
    FC_Constant,
    FC_String,
    FC_Flag,
    FC_Reference,
    FC_Indirect,
    FC_SectionOffset,
    FC_Exprloc
  };

private:
  struct ValueType {
    ValueType() : data(nullptr) {
      uval = 0;
    }

    union {
      uint64_t uval;
      int64_t sval;
      const char* cstr;
    };
    const uint8_t* data;
  };

  uint16_t Form;   // Form for this value.
  ValueType Value; // Contains all data for the form.

public:
  DWARFFormValue(uint16_t Form = 0) : Form(Form) {}
  uint16_t getForm() const { return Form; }
  bool isFormClass(FormClass FC) const;

  void dump(raw_ostream &OS, const DWARFUnit *U) const;
  bool extractValue(DataExtractor data, uint32_t *offset_ptr,
                    const DWARFUnit *u);
  bool isInlinedCStr() const {
    return Value.data != nullptr && Value.data == (const uint8_t*)Value.cstr;
  }

  /// getAsFoo functions below return the extracted value as Foo if only
  /// DWARFFormValue has form class is suitable for representing Foo.
  Optional<uint64_t> getAsReference(const DWARFUnit *U) const;
  Optional<uint64_t> getAsUnsignedConstant() const;
  Optional<const char *> getAsCString(const DWARFUnit *U) const;
  Optional<uint64_t> getAsAddress(const DWARFUnit *U) const;
  Optional<uint64_t> getAsSectionOffset() const;

  bool skipValue(DataExtractor debug_info_data, uint32_t *offset_ptr,
                 const DWARFUnit *u) const;
  static bool skipValue(uint16_t form, DataExtractor debug_info_data,
                        uint32_t *offset_ptr, const DWARFUnit *u);

  static ArrayRef<uint8_t> getFixedFormSizes(uint8_t AddrSize,
                                             uint16_t Version);
};

}

#endif
