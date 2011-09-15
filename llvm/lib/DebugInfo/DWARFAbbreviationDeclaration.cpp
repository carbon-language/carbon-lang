//===-- DWARFAbbreviationDeclaration.cpp ----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DWARFAbbreviationDeclaration.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;
using namespace dwarf;

bool
DWARFAbbreviationDeclaration::extract(DataExtractor data, uint32_t* offset_ptr){
  return extract(data, offset_ptr, data.getULEB128(offset_ptr));
}

bool
DWARFAbbreviationDeclaration::extract(DataExtractor data, uint32_t* offset_ptr,
                                      uint32_t code) {
  Code = code;
  Attributes.clear();
  if (Code) {
    Tag = data.getULEB128(offset_ptr);
    HasChildren = data.getU8(offset_ptr);

    while (data.isValidOffset(*offset_ptr)) {
      uint16_t attr = data.getULEB128(offset_ptr);
      uint16_t form = data.getULEB128(offset_ptr);

      if (attr && form)
        Attributes.push_back(DWARFAttribute(attr, form));
      else
        break;
    }

    return Tag != 0;
  } else {
    Tag = 0;
    HasChildren = false;
  }

  return false;
}

void DWARFAbbreviationDeclaration::dump(raw_ostream &OS) const {
  const char *tagString = TagString(getTag());
  OS << '[' << getCode() << "] " << (tagString ? tagString : "DW_TAG_Unknown")
     << "\tDW_CHILDREN_"
     << (hasChildren() ? "yes" : "no") << '\n';
  for (unsigned i = 0, e = Attributes.size(); i != e; ++i) {
    const char *attrString = AttributeString(Attributes[i].getAttribute());
    const char *formString = FormEncodingString(Attributes[i].getForm());
    OS << '\t' << (attrString ? attrString : "DW_AT_Unknown")
       << '\t' << (formString ? formString : "DW_FORM_Unknown") << '\n';
  }
  OS << '\n';
}

uint32_t
DWARFAbbreviationDeclaration::findAttributeIndex(uint16_t attr) const {
  for (uint32_t i = 0, e = Attributes.size(); i != e; ++i) {
    if (Attributes[i].getAttribute() == attr)
      return i;
  }
  return -1U;
}
