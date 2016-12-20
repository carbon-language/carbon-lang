//===- yaml2dwarf - Convert YAML to DWARF binary data ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief The DWARF component of yaml2obj.
///
//===----------------------------------------------------------------------===//

#include "llvm/ObjectYAML/DWARFYAML.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>

using namespace llvm;

void ZeroFillBytes(raw_ostream &OS, size_t Size) {
  std::vector<uint8_t> FillData;
  FillData.insert(FillData.begin(), Size, 0);
  OS.write(reinterpret_cast<char *>(FillData.data()), Size);
}

void yaml2debug_str(raw_ostream &OS, const DWARFYAML::Data &DI) {
  for (auto Str : DI.DebugStrings) {
    OS.write(Str.data(), Str.size());
    OS.write('\0');
  }
}

void yaml2debug_abbrev(raw_ostream &OS, const DWARFYAML::Data &DI) {
  for (auto AbbrevDecl : DI.AbbrevDecls) {
    encodeULEB128(AbbrevDecl.Code, OS);
    encodeULEB128(AbbrevDecl.Tag, OS);
    OS.write(AbbrevDecl.Children);
    for (auto Attr : AbbrevDecl.Attributes) {
      encodeULEB128(Attr.Attribute, OS);
      encodeULEB128(Attr.Form, OS);
    }
    encodeULEB128(0, OS);
    encodeULEB128(0, OS);
  }
}

void yaml2debug_aranges(raw_ostream &OS, const DWARFYAML::Data &DI) {
  for (auto Range : DI.ARanges) {
    auto HeaderStart = OS.tell();
    OS.write(reinterpret_cast<char *>(&Range.Length), 4);
    OS.write(reinterpret_cast<char *>(&Range.Version), 2);
    OS.write(reinterpret_cast<char *>(&Range.CuOffset), 4);
    OS.write(reinterpret_cast<char *>(&Range.AddrSize), 1);
    OS.write(reinterpret_cast<char *>(&Range.SegSize), 1);

    auto HeaderSize = OS.tell() - HeaderStart;
    auto FirstDescriptor = alignTo(HeaderSize, Range.AddrSize * 2);
    ZeroFillBytes(OS, FirstDescriptor - HeaderSize);

    for (auto Descriptor : Range.Descriptors) {
      OS.write(reinterpret_cast<char *>(&Descriptor.Address), Range.AddrSize);
      OS.write(reinterpret_cast<char *>(&Descriptor.Length), Range.AddrSize);
    }
    ZeroFillBytes(OS, Range.AddrSize * 2);
  }
}

void yaml2pubsection(raw_ostream &OS, const DWARFYAML::PubSection &Sect) {
  OS.write(reinterpret_cast<const char *>(&Sect.Length), 4);
  OS.write(reinterpret_cast<const char *>(&Sect.Version), 2);
  OS.write(reinterpret_cast<const char *>(&Sect.UnitOffset), 4);
  OS.write(reinterpret_cast<const char *>(&Sect.UnitSize), 4);
  for (auto Entry : Sect.Entries) {
    OS.write(reinterpret_cast<const char *>(&Entry.DieOffset), 4);
    if (Sect.IsGNUStyle)
      OS.write(reinterpret_cast<const char *>(&Entry.Descriptor), 4);
    OS.write(Entry.Name.data(), Entry.Name.size());
    OS.write('\0');
  }
}

void yaml2debug_info(raw_ostream &OS, const DWARFYAML::Data &DI) {

  for (auto CU : DI.CompileUnits) {
    OS.write(reinterpret_cast<char *>(&CU.Length), 4);
    OS.write(reinterpret_cast<char *>(&CU.Version), 2);
    OS.write(reinterpret_cast<char *>(&CU.AbbrOffset), 4);
    OS.write(reinterpret_cast<char *>(&CU.AddrSize), 1);

    auto FirstAbbrevCode = CU.Entries[0].AbbrCode;

    for (auto Entry : CU.Entries) {
      encodeULEB128(Entry.AbbrCode, OS);
      if(Entry.AbbrCode == 0u)
        continue;
      bool Indirect = false;
      assert(Entry.AbbrCode-FirstAbbrevCode < DI.AbbrevDecls.size() &&
        "Out of range AbbCode");
      auto &Abbrev = DI.AbbrevDecls[Entry.AbbrCode-FirstAbbrevCode];

      auto FormVal = Entry.Values.begin();
      auto AbbrForm = Abbrev.Attributes.begin();
      for (;
           FormVal != Entry.Values.end() && AbbrForm != Abbrev.Attributes.end();
           ++FormVal, ++AbbrForm) {
        dwarf::Form Form = AbbrForm->Form;
        do {
          bool Indirect = false;
          switch (Form) {
          case dwarf::DW_FORM_addr:
            OS.write(reinterpret_cast<char *>(&FormVal->Value), CU.AddrSize);
            break;
          case dwarf::DW_FORM_ref_addr: {
            // TODO: Handle DWARF32/DWARF64 after Line Table data is done
            auto writeSize = CU.Version == 2 ? CU.AddrSize : 4;
            OS.write(reinterpret_cast<char *>(&FormVal->Value), writeSize);
            break;
          }
          case dwarf::DW_FORM_exprloc:
          case dwarf::DW_FORM_block:
            encodeULEB128(FormVal->BlockData.size(), OS);
            OS.write(reinterpret_cast<char *>(&FormVal->BlockData[0]),
                     FormVal->BlockData.size());
            break;
          case dwarf::DW_FORM_block1: {
            auto writeSize = FormVal->BlockData.size();
            OS.write(reinterpret_cast<char *>(&writeSize), 1);
            OS.write(reinterpret_cast<char *>(&FormVal->BlockData[0]),
                     FormVal->BlockData.size());
            break;
          }
          case dwarf::DW_FORM_block2: {
            auto writeSize = FormVal->BlockData.size();
            OS.write(reinterpret_cast<char *>(&writeSize), 2);
            OS.write(reinterpret_cast<char *>(&FormVal->BlockData[0]),
                     FormVal->BlockData.size());
            break;
          }
          case dwarf::DW_FORM_block4: {
            auto writeSize = FormVal->BlockData.size();
            OS.write(reinterpret_cast<char *>(&writeSize), 4);
            OS.write(reinterpret_cast<char *>(&FormVal->BlockData[0]),
                     FormVal->BlockData.size());
            break;
          }
          case dwarf::DW_FORM_data1:
          case dwarf::DW_FORM_ref1:
          case dwarf::DW_FORM_flag:
            OS.write(reinterpret_cast<char *>(&FormVal->Value), 1);
            break;
          case dwarf::DW_FORM_data2:
          case dwarf::DW_FORM_ref2:
            OS.write(reinterpret_cast<char *>(&FormVal->Value), 2);
            break;
          case dwarf::DW_FORM_data4:
          case dwarf::DW_FORM_ref4:
            OS.write(reinterpret_cast<char *>(&FormVal->Value), 4);
            break;
          case dwarf::DW_FORM_data8:
          case dwarf::DW_FORM_ref8:
            OS.write(reinterpret_cast<char *>(&FormVal->Value), 8);
            break;
          case dwarf::DW_FORM_sdata:
            encodeSLEB128(FormVal->Value, OS);
            break;
          case dwarf::DW_FORM_udata:
          case dwarf::DW_FORM_ref_udata:
            encodeULEB128(FormVal->Value, OS);
            break;
          case dwarf::DW_FORM_string:
            OS.write(FormVal->CStr.data(), FormVal->CStr.size());
            OS.write('\0');
            break;
          case dwarf::DW_FORM_indirect:
            encodeULEB128(FormVal->Value, OS);
            Indirect = true;
            Form = static_cast<dwarf::Form>((uint64_t)FormVal->Value);
            ++FormVal;
            break;
          case dwarf::DW_FORM_strp:
          case dwarf::DW_FORM_sec_offset:
          case dwarf::DW_FORM_GNU_ref_alt:
          case dwarf::DW_FORM_GNU_strp_alt:
          case dwarf::DW_FORM_line_strp:
          case dwarf::DW_FORM_strp_sup:
          case dwarf::DW_FORM_ref_sup:
            // TODO: Handle DWARF32/64
            OS.write(reinterpret_cast<char *>(&FormVal->Value), 4);
            break;
          case dwarf::DW_FORM_ref_sig8:
            OS.write(reinterpret_cast<char *>(&FormVal->Value), 8);
            break;
          case dwarf::DW_FORM_GNU_addr_index:
          case dwarf::DW_FORM_GNU_str_index:
            encodeULEB128(FormVal->Value, OS);
            break;
          default:
            break;
          }
        } while (Indirect);
      }
    }
  }
}
