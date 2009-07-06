//===-- ELFWriter.h - Target-independent ELF writer support -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the ELFWriter class.
//
//===----------------------------------------------------------------------===//

#ifndef ELFWRITER_H
#define ELFWRITER_H

#include "llvm/ADT/SetVector.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include <list>
#include <map>

namespace llvm {
  class BinaryObject;
  class Constant;
  class ConstantStruct;
  class ELFCodeEmitter;
  class ELFRelocation;
  class ELFSection;
  class ELFSym;
  class GlobalVariable;
  class Mangler;
  class MachineCodeEmitter;
  class ObjectCodeEmitter;
  class TargetAsmInfo;
  class TargetELFWriterInfo;
  class raw_ostream;

  /// ELFWriter - This class implements the common target-independent code for
  /// writing ELF files.  Targets should derive a class from this to
  /// parameterize the output format.
  ///
  class ELFWriter : public MachineFunctionPass {
    friend class ELFCodeEmitter;
  public:
    static char ID;

    /// Return the ELFCodeEmitter as an instance of ObjectCodeEmitter
    ObjectCodeEmitter *getObjectCodeEmitter() {
      return reinterpret_cast<ObjectCodeEmitter*>(ElfCE);
    }

    ELFWriter(raw_ostream &O, TargetMachine &TM);
    ~ELFWriter();

  protected:
    /// Output stream to send the resultant object file to.
    raw_ostream &O;

    /// Target machine description.
    TargetMachine &TM;

    /// Target Elf Writer description.
    const TargetELFWriterInfo *TEW;

    /// Mang - The object used to perform name mangling for this module.
    Mangler *Mang;

    /// MCE - The MachineCodeEmitter object that we are exposing to emit machine
    /// code for functions to the .o file.
    ELFCodeEmitter *ElfCE;

    /// TAI - Target Asm Info, provide information about section names for
    /// globals and other target specific stuff.
    const TargetAsmInfo *TAI;

    //===------------------------------------------------------------------===//
    // Properties inferred automatically from the target machine.
    //===------------------------------------------------------------------===//

    /// is64Bit/isLittleEndian - This information is inferred from the target
    /// machine directly, indicating whether to emit a 32- or 64-bit ELF file.
    bool is64Bit, isLittleEndian;

    /// doInitialization - Emit the file header and all of the global variables
    /// for the module to the ELF file.
    bool doInitialization(Module &M);
    bool runOnMachineFunction(MachineFunction &MF);

    /// doFinalization - Now that the module has been completely processed, emit
    /// the ELF file to 'O'.
    bool doFinalization(Module &M);

  private:
    /// Blob containing the Elf header
    BinaryObject ElfHdr;

    /// SectionList - This is the list of sections that we have emitted to the
    /// file.  Once the file has been completely built, the section header table
    /// is constructed from this info.
    std::list<ELFSection> SectionList;
    unsigned NumSections;   // Always = SectionList.size()

    /// SectionLookup - This is a mapping from section name to section number in
    /// the SectionList.
    std::map<std::string, ELFSection*> SectionLookup;

    /// GblSymLookup - This is a mapping from global value to a symbol index
    /// in the symbol table. This is useful since relocations symbol references
    /// must be quickly mapped to a symbol table index
    std::map<const GlobalValue*, uint32_t> GblSymLookup;

    /// SymbolList - This is the list of symbols emitted to the symbol table
    /// Local symbols go to the front and Globals to the back.
    std::list<ELFSym> SymbolList;

    /// PendingGlobals - List of externally defined symbols that we have been
    /// asked to emit, but have not seen a reference to.  When a reference
    /// is seen, the symbol will move from this list to the SymbolList.
    SetVector<GlobalValue*> PendingGlobals;

    // Remove tab from section name prefix. This is necessary becase TAI 
    // sometimes return a section name prefixed with a "\t" char. This is
    // a little bit dirty. FIXME: find a better approach, maybe add more
    // methods to TAI to get the clean name?
    void fixNameForSection(std::string &Name) {
      size_t Pos = Name.find("\t");
      if (Pos != std::string::npos)
        Name.erase(Pos, 1);

      Pos = Name.find(".section ");
      if (Pos != std::string::npos)
        Name.erase(Pos, 9);

      Pos = Name.find("\n");
      if (Pos != std::string::npos)
        Name.erase(Pos, 1);
    }

    /// getSection - Return the section with the specified name, creating a new
    /// section if one does not already exist.
    ELFSection &getSection(const std::string &Name, unsigned Type,
                           unsigned Flags = 0, unsigned Align = 0) {
      std::string SectionName(Name);
      fixNameForSection(SectionName);

      ELFSection *&SN = SectionLookup[SectionName];
      if (SN) return *SN;

      SectionList.push_back(ELFSection(SectionName, isLittleEndian, is64Bit));
      SN = &SectionList.back();
      SN->SectionIdx = NumSections++;
      SN->Type = Type;
      SN->Flags = Flags;
      SN->Link = ELFSection::SHN_UNDEF;
      SN->Align = Align;
      return *SN;
    }

    /// TODO: support mangled names here to emit the right .text section
    /// for c++ object files.
    ELFSection &getTextSection() {
      return getSection(".text", ELFSection::SHT_PROGBITS,
                        ELFSection::SHF_EXECINSTR | ELFSection::SHF_ALLOC);
    }

    /// Get jump table section on the section name returned by TAI
    ELFSection &getJumpTableSection(std::string SName, unsigned Align) {
      return getSection(SName, ELFSection::SHT_PROGBITS,
                        ELFSection::SHF_ALLOC, Align);
    }

    /// Get a constant pool section based on the section name returned by TAI
    ELFSection &getConstantPoolSection(std::string SName, unsigned Align) {
      return getSection(SName, ELFSection::SHT_PROGBITS,
                        ELFSection::SHF_MERGE | ELFSection::SHF_ALLOC, Align);
    }

    /// Return the relocation section of section 'S'. 'RelA' is true
    /// if the relocation section contains entries with addends.
    ELFSection &getRelocSection(std::string SName, bool RelA, unsigned Align) {
      std::string RelSName(".rel");
      unsigned SHdrTy = RelA ? ELFSection::SHT_RELA : ELFSection::SHT_REL;

      if (RelA) RelSName.append("a");
      RelSName.append(SName);

      return getSection(RelSName, SHdrTy, 0, Align);
    }

    ELFSection &getNonExecStackSection() {
      return getSection(".note.GNU-stack", ELFSection::SHT_PROGBITS, 0, 1);
    }

    ELFSection &getSymbolTableSection() {
      return getSection(".symtab", ELFSection::SHT_SYMTAB, 0);
    }

    ELFSection &getStringTableSection() {
      return getSection(".strtab", ELFSection::SHT_STRTAB, 0, 1);
    }

    ELFSection &getSectionHeaderStringTableSection() {
      return getSection(".shstrtab", ELFSection::SHT_STRTAB, 0, 1);
    }

    ELFSection &getDataSection() {
      return getSection(".data", ELFSection::SHT_PROGBITS,
                        ELFSection::SHF_WRITE | ELFSection::SHF_ALLOC, 4);
    }

    ELFSection &getBSSSection() {
      return getSection(".bss", ELFSection::SHT_NOBITS,
                        ELFSection::SHF_WRITE | ELFSection::SHF_ALLOC, 4);
    }

    ELFSection &getNullSection() {
      return getSection("", ELFSection::SHT_NULL, 0);
    }

    // Helpers for obtaining ELF specific info.
    unsigned getGlobalELFLinkage(const GlobalValue *GV);
    unsigned getGlobalELFVisibility(const GlobalValue *GV);
    unsigned getElfSectionFlags(unsigned Flags);

    // As we complete the ELF file, we need to update fields in the ELF header
    // (e.g. the location of the section table).  These members keep track of
    // the offset in ELFHeader of these various pieces to update and other
    // locations in the file.
    unsigned ELFHdr_e_shoff_Offset;     // e_shoff    in ELF header.
    unsigned ELFHdr_e_shstrndx_Offset;  // e_shstrndx in ELF header.
    unsigned ELFHdr_e_shnum_Offset;     // e_shnum    in ELF header.

  private:
    void EmitFunctionDeclaration(const Function *F);
    void EmitGlobalVar(const GlobalVariable *GV);
    void EmitGlobalConstant(const Constant *C, ELFSection &GblS);
    void EmitGlobalConstantStruct(const ConstantStruct *CVS,
                                  ELFSection &GblS);
    ELFSection &getGlobalSymELFSection(const GlobalVariable *GV, ELFSym &Sym);
    void EmitRelocations();
    void EmitRelocation(BinaryObject &RelSec, ELFRelocation &Rel, bool HasRelA);
    void EmitSectionHeader(BinaryObject &SHdrTab, const ELFSection &SHdr);
    void EmitSectionTableStringTable();
    void EmitSymbol(BinaryObject &SymbolTable, ELFSym &Sym);
    void EmitSymbolTable();
    void EmitStringTable();
    void OutputSectionsAndSectionTable();
  };
}

#endif
