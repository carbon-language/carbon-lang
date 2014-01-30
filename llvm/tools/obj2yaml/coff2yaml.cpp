//===------ utils/obj2yaml.cpp - obj2yaml conversion tool -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "obj2yaml.h"
#include "llvm/Object/COFF.h"
#include "llvm/Object/COFFYAML.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/YAMLTraits.h"

using namespace llvm;

namespace {

class COFFDumper {
  const object::COFFObjectFile &Obj;
  COFFYAML::Object YAMLObj;
  void dumpHeader(const object::coff_file_header *Header);
  void dumpSections(unsigned numSections);
  void dumpSymbols(unsigned numSymbols);

public:
  COFFDumper(const object::COFFObjectFile &Obj);
  COFFYAML::Object &getYAMLObj();
};

}

static void check(error_code ec) {
  if (ec)
    report_fatal_error(ec.message());
}

COFFDumper::COFFDumper(const object::COFFObjectFile &Obj) : Obj(Obj) {
  const object::coff_file_header *Header;
  check(Obj.getCOFFHeader(Header));
  dumpHeader(Header);
  dumpSections(Header->NumberOfSections);
  dumpSymbols(Header->NumberOfSymbols);
}

void COFFDumper::dumpHeader(const object::coff_file_header *Header) {
  YAMLObj.Header.Machine = Header->Machine;
  YAMLObj.Header.Characteristics = Header->Characteristics;
}

void COFFDumper::dumpSections(unsigned NumSections) {
  std::vector<COFFYAML::Section> &Sections = YAMLObj.Sections;
  for (object::section_iterator iter = Obj.begin_sections();
       iter != Obj.end_sections(); ++iter) {
    const object::coff_section *Sect = Obj.getCOFFSection(iter);
    COFFYAML::Section Sec;
    Sec.Name = Sect->Name; // FIXME: check the null termination!
    uint32_t Characteristics = Sect->Characteristics;
    Sec.Header.Characteristics = Characteristics;
    Sec.Alignment = 1 << (((Characteristics >> 20) & 0xf) - 1);

    ArrayRef<uint8_t> sectionData;
    Obj.getSectionContents(Sect, sectionData);
    Sec.SectionData = object::yaml::BinaryRef(sectionData);

    std::vector<COFFYAML::Relocation> Relocations;
    for (object::relocation_iterator rIter = iter->begin_relocations();
                       rIter != iter->end_relocations(); ++rIter) {
      const object::coff_relocation *reloc = Obj.getCOFFRelocation(rIter);
      COFFYAML::Relocation Rel;
      object::symbol_iterator Sym = rIter->getSymbol();
      Sym->getName(Rel.SymbolName);
      Rel.VirtualAddress = reloc->VirtualAddress;
      Rel.Type = reloc->Type;
      Relocations.push_back(Rel);
    }
    Sec.Relocations = Relocations;
    Sections.push_back(Sec);
  }
}

void COFFDumper::dumpSymbols(unsigned NumSymbols) {
  std::vector<COFFYAML::Symbol> &Symbols = YAMLObj.Symbols;
  for (object::symbol_iterator iter = Obj.begin_symbols();
       iter != Obj.end_symbols(); ++iter) {
    const object::coff_symbol *Symbol = Obj.getCOFFSymbol(iter);
    COFFYAML::Symbol Sym;
    Obj.getSymbolName(Symbol, Sym.Name);
    Sym.SimpleType = COFF::SymbolBaseType(Symbol->getBaseType());
    Sym.ComplexType = COFF::SymbolComplexType(Symbol->getComplexType());
    Sym.Header.StorageClass = Symbol->StorageClass;
    Sym.Header.Value = Symbol->Value;
    Sym.Header.SectionNumber = Symbol->SectionNumber;
    Sym.Header.NumberOfAuxSymbols = Symbol->NumberOfAuxSymbols;
    Sym.AuxiliaryData = object::yaml::BinaryRef(Obj.getSymbolAuxData(Symbol));
    Symbols.push_back(Sym);
  }
}

COFFYAML::Object &COFFDumper::getYAMLObj() {
  return YAMLObj;
}

error_code coff2yaml(raw_ostream &Out, MemoryBuffer *Buff) {
  error_code ec;
  object::COFFObjectFile Obj(Buff, ec);
  check(ec);
  COFFDumper Dumper(Obj);

  yaml::Output Yout(Out);
  Yout << Dumper.getYAMLObj();

  return object::object_error::success;
}
