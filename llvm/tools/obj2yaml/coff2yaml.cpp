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
  for (const auto &Section : Obj.sections()) {
    const object::coff_section *Sect = Obj.getCOFFSection(Section);
    COFFYAML::Section Sec;
    Sec.Name = Sect->Name; // FIXME: check the null termination!
    uint32_t Characteristics = Sect->Characteristics;
    Sec.Header.Characteristics = Characteristics;
    Sec.Alignment = 1 << (((Characteristics >> 20) & 0xf) - 1);

    ArrayRef<uint8_t> sectionData;
    Obj.getSectionContents(Sect, sectionData);
    Sec.SectionData = object::yaml::BinaryRef(sectionData);

    std::vector<COFFYAML::Relocation> Relocations;
    for (const auto &Reloc : Section.relocations()) {
      const object::coff_relocation *reloc = Obj.getCOFFRelocation(Reloc);
      COFFYAML::Relocation Rel;
      object::symbol_iterator Sym = Reloc.getSymbol();
      Sym->getName(Rel.SymbolName);
      Rel.VirtualAddress = reloc->VirtualAddress;
      Rel.Type = reloc->Type;
      Relocations.push_back(Rel);
    }
    Sec.Relocations = Relocations;
    Sections.push_back(Sec);
  }
}

static void
dumpFunctionDefinition(COFFYAML::Symbol *Sym,
                       const object::coff_aux_function_definition *ObjFD) {
  COFF::AuxiliaryFunctionDefinition YAMLFD;
  YAMLFD.TagIndex = ObjFD->TagIndex;
  YAMLFD.TotalSize = ObjFD->TotalSize;
  YAMLFD.PointerToLinenumber = ObjFD->PointerToLinenumber;
  YAMLFD.PointerToNextFunction = ObjFD->PointerToNextFunction;

  Sym->FunctionDefinition = YAMLFD;
}

static void
dumpbfAndEfLineInfo(COFFYAML::Symbol *Sym,
                    const object::coff_aux_bf_and_ef_symbol *ObjBES) {
  COFF::AuxiliarybfAndefSymbol YAMLAAS;
  YAMLAAS.Linenumber = ObjBES->Linenumber;
  YAMLAAS.PointerToNextFunction = ObjBES->PointerToNextFunction;

  Sym->bfAndefSymbol = YAMLAAS;
}

static void dumpWeakExternal(COFFYAML::Symbol *Sym,
                             const object::coff_aux_weak_external *ObjWE) {
  COFF::AuxiliaryWeakExternal YAMLWE;
  YAMLWE.TagIndex = ObjWE->TagIndex;
  YAMLWE.Characteristics = ObjWE->Characteristics;

  Sym->WeakExternal = YAMLWE;
}

static void
dumpSectionDefinition(COFFYAML::Symbol *Sym,
                      const object::coff_aux_section_definition *ObjSD) {
  COFF::AuxiliarySectionDefinition YAMLASD;
  YAMLASD.Length = ObjSD->Length;
  YAMLASD.NumberOfRelocations = ObjSD->NumberOfRelocations;
  YAMLASD.NumberOfLinenumbers = ObjSD->NumberOfLinenumbers;
  YAMLASD.CheckSum = ObjSD->CheckSum;
  YAMLASD.Number = ObjSD->Number;
  YAMLASD.Selection = ObjSD->Selection;

  Sym->SectionDefinition = YAMLASD;
}

static void
dumpCLRTokenDefinition(COFFYAML::Symbol *Sym,
                       const object::coff_aux_clr_token *ObjCLRToken) {
  COFF::AuxiliaryCLRToken YAMLCLRToken;
  YAMLCLRToken.AuxType = ObjCLRToken->AuxType;
  YAMLCLRToken.SymbolTableIndex = ObjCLRToken->SymbolTableIndex;

  Sym->CLRToken = YAMLCLRToken;
}

void COFFDumper::dumpSymbols(unsigned NumSymbols) {
  std::vector<COFFYAML::Symbol> &Symbols = YAMLObj.Symbols;
  for (const auto &S : Obj.symbols()) {
    const object::coff_symbol *Symbol = Obj.getCOFFSymbol(S);
    COFFYAML::Symbol Sym;
    Obj.getSymbolName(Symbol, Sym.Name);
    Sym.SimpleType = COFF::SymbolBaseType(Symbol->getBaseType());
    Sym.ComplexType = COFF::SymbolComplexType(Symbol->getComplexType());
    Sym.Header.StorageClass = Symbol->StorageClass;
    Sym.Header.Value = Symbol->Value;
    Sym.Header.SectionNumber = Symbol->SectionNumber;
    Sym.Header.NumberOfAuxSymbols = Symbol->NumberOfAuxSymbols;

    if (Symbol->NumberOfAuxSymbols > 0) {
      ArrayRef<uint8_t> AuxData = Obj.getSymbolAuxData(Symbol);
      if (Symbol->isFunctionDefinition()) {
        // This symbol represents a function definition.
        assert(Symbol->NumberOfAuxSymbols == 1 &&
               "Expected a single aux symbol to describe this function!");

        const object::coff_aux_function_definition *ObjFD =
            reinterpret_cast<const object::coff_aux_function_definition *>(
                AuxData.data());
        dumpFunctionDefinition(&Sym, ObjFD);
      } else if (Symbol->isFunctionLineInfo()) {
        // This symbol describes function line number information.
        assert(Symbol->NumberOfAuxSymbols == 1 &&
               "Exepected a single aux symbol to describe this section!");

        const object::coff_aux_bf_and_ef_symbol *ObjBES =
            reinterpret_cast<const object::coff_aux_bf_and_ef_symbol *>(
                AuxData.data());
        dumpbfAndEfLineInfo(&Sym, ObjBES);
      } else if (Symbol->isWeakExternal()) {
        // This symbol represents a weak external definition.
        assert(Symbol->NumberOfAuxSymbols == 1 &&
               "Exepected a single aux symbol to describe this section!");

        const object::coff_aux_weak_external *ObjWE =
            reinterpret_cast<const object::coff_aux_weak_external *>(
                AuxData.data());
        dumpWeakExternal(&Sym, ObjWE);
      } else if (Symbol->isFileRecord()) {
        // This symbol represents a file record.
        Sym.File = StringRef(reinterpret_cast<const char *>(AuxData.data()),
                             Symbol->NumberOfAuxSymbols * COFF::SymbolSize);
      } else if (Symbol->isSectionDefinition()) {
        // This symbol represents a section definition.
        assert(Symbol->NumberOfAuxSymbols == 1 &&
               "Expected a single aux symbol to describe this section!");

        const object::coff_aux_section_definition *ObjSD =
            reinterpret_cast<const object::coff_aux_section_definition *>(
                AuxData.data());
        dumpSectionDefinition(&Sym, ObjSD);
      } else if (Symbol->isCLRToken()) {
        // This symbol represents a CLR token definition.
        assert(Symbol->NumberOfAuxSymbols == 1 &&
               "Expected a single aux symbol to describe this CLR Token");

        const object::coff_aux_clr_token *ObjCLRToken =
            reinterpret_cast<const object::coff_aux_clr_token *>(
                AuxData.data());
        dumpCLRTokenDefinition(&Sym, ObjCLRToken);
      } else {
        llvm_unreachable("Unhandled auxiliary symbol!");
      }
    }
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
