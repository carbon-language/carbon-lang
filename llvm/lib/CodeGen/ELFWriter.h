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
#include <map>

namespace llvm {
  class BinaryObject;
  class Constant;
  class ConstantInt;
  class ConstantStruct;
  class ELFCodeEmitter;
  class ELFRelocation;
  class ELFSection;
  struct ELFSym;
  class GlobalVariable;
  class Mangler;
  class MachineCodeEmitter;
  class MachineConstantPoolEntry;
  class ObjectCodeEmitter;
  class TargetAsmInfo;
  class TargetELFWriterInfo;
  class TargetLoweringObjectFile;
  class raw_ostream;
  class SectionKind;
  class MCContext;

  typedef std::vector<ELFSym*>::iterator ELFSymIter;
  typedef std::vector<ELFSection*>::iterator ELFSectionIter;
  typedef SetVector<const GlobalValue*>::const_iterator PendingGblsIter;
  typedef SetVector<const char *>::const_iterator PendingExtsIter;
  typedef std::pair<const Constant *, int64_t> CstExprResTy;

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

    /// Context object for machine code objects.
    MCContext &OutContext;
    
    /// Target Elf Writer description.
    const TargetELFWriterInfo *TEW;

    /// Mang - The object used to perform name mangling for this module.
    Mangler *Mang;

    /// MCE - The MachineCodeEmitter object that we are exposing to emit machine
    /// code for functions to the .o file.
    ELFCodeEmitter *ElfCE;

    /// TLOF - Target Lowering Object File, provide section names for globals 
    /// and other object file specific stuff
    const TargetLoweringObjectFile &TLOF;

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
    /// file. Once the file has been completely built, the section header table
    /// is constructed from this info.
    std::vector<ELFSection*> SectionList;
    unsigned NumSections;   // Always = SectionList.size()

    /// SectionLookup - This is a mapping from section name to section number in
    /// the SectionList. Used to quickly gather the Section Index from TAI names
    std::map<std::string, ELFSection*> SectionLookup;

    /// PendingGlobals - Globals not processed as symbols yet.
    SetVector<const GlobalValue*> PendingGlobals;

    /// GblSymLookup - This is a mapping from global value to a symbol index
    /// in the symbol table or private symbols list. This is useful since reloc
    /// symbol references must be quickly mapped to their indices on the lists.
    std::map<const GlobalValue*, uint32_t> GblSymLookup;

    /// PendingExternals - Externals not processed as symbols yet.
    SetVector<const char *> PendingExternals;

    /// ExtSymLookup - This is a mapping from externals to a symbol index
    /// in the symbol table list. This is useful since reloc symbol references
    /// must be quickly mapped to their symbol table indices.
    std::map<const char *, uint32_t> ExtSymLookup;

    /// SymbolList - This is the list of symbols emitted to the symbol table.
    /// When the SymbolList is finally built, local symbols must be placed in
    /// the beginning while non-locals at the end.
    std::vector<ELFSym*> SymbolList;

    /// PrivateSyms - Record private symbols, every symbol here must never be
    /// present in the SymbolList.
    std::vector<ELFSym*> PrivateSyms;

    // Remove tab from section name prefix. This is necessary becase TAI
    // sometimes return a section name prefixed with elf unused chars. This is
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
      std::string SName(Name);
      fixNameForSection(SName);

      ELFSection *&SN = SectionLookup[SName];
      if (SN) return *SN;

      SectionList.push_back(new ELFSection(SName, isLittleEndian, is64Bit));
      SN = SectionList.back();
      SN->SectionIdx = NumSections++;
      SN->Type = Type;
      SN->Flags = Flags;
      SN->Link = ELFSection::SHN_UNDEF;
      SN->Align = Align;
      return *SN;
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

    ELFSection &getCtorSection();
    ELFSection &getDtorSection();
    ELFSection &getJumpTableSection();
    ELFSection &getConstantPoolSection(MachineConstantPoolEntry &CPE);
    ELFSection &getTextSection(Function *F);
    ELFSection &getRelocSection(ELFSection &S);

    // Helpers for obtaining ELF specific info.
    unsigned getGlobalELFBinding(const GlobalValue *GV);
    unsigned getGlobalELFType(const GlobalValue *GV);
    unsigned getGlobalELFVisibility(const GlobalValue *GV);
    unsigned getElfSectionFlags(SectionKind Kind, bool IsAlloc = true);

    // addGlobalSymbol - Add a global to be processed and to 
    // the global symbol lookup, use a zero index because the table 
    // index will be determined later.
    void addGlobalSymbol(const GlobalValue *GV, bool AddToLookup = false);
    
    // addExternalSymbol - Add the external to be processed and to the
    // external symbol lookup, use a zero index because the symbol
    // table index will be determined later
    void addExternalSymbol(const char *External);

    // As we complete the ELF file, we need to update fields in the ELF header
    // (e.g. the location of the section table).  These members keep track of
    // the offset in ELFHeader of these various pieces to update and other
    // locations in the file.
    unsigned ELFHdr_e_shoff_Offset;     // e_shoff    in ELF header.
    unsigned ELFHdr_e_shstrndx_Offset;  // e_shstrndx in ELF header.
    unsigned ELFHdr_e_shnum_Offset;     // e_shnum    in ELF header.

  private:
    void EmitGlobal(const GlobalValue *GV);
    void EmitGlobalConstant(const Constant *C, ELFSection &GblS);
    void EmitGlobalConstantStruct(const ConstantStruct *CVS,
                                  ELFSection &GblS);
    void EmitGlobalConstantLargeInt(const ConstantInt *CI, ELFSection &S);
    void EmitGlobalDataRelocation(const GlobalValue *GV, unsigned Size, 
                                  ELFSection &GblS, int64_t Offset = 0);
    bool EmitSpecialLLVMGlobal(const GlobalVariable *GV);
    void EmitXXStructorList(Constant *List, ELFSection &Xtor);
    void EmitRelocations();
    void EmitRelocation(BinaryObject &RelSec, ELFRelocation &Rel, bool HasRelA);
    void EmitSectionHeader(BinaryObject &SHdrTab, const ELFSection &SHdr);
    void EmitSectionTableStringTable();
    void EmitSymbol(BinaryObject &SymbolTable, ELFSym &Sym);
    void EmitSymbolTable();
    void EmitStringTable(const std::string &ModuleName);
    void OutputSectionsAndSectionTable();
    void RelocateField(BinaryObject &BO, uint32_t Offset, int64_t Value,
                       unsigned Size);
    unsigned SortSymbols();
    CstExprResTy ResolveConstantExpr(const Constant *CV);
  };
}

#endif
