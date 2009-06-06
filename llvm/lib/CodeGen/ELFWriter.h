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

#include "llvm/CodeGen/MachineFunctionPass.h"
#include "ELF.h"
#include <list>
#include <map>

namespace llvm {
  class GlobalVariable;
  class Mangler;
  class MachineCodeEmitter;
  class ELFCodeEmitter;
  class raw_ostream;

  /// ELFWriter - This class implements the common target-independent code for
  /// writing ELF files.  Targets should derive a class from this to
  /// parameterize the output format.
  ///
  class ELFWriter : public MachineFunctionPass {
    friend class ELFCodeEmitter;
  public:
    static char ID;

    MachineCodeEmitter &getMachineCodeEmitter() const {
      return *(MachineCodeEmitter*)MCE;
    }

    ELFWriter(raw_ostream &O, TargetMachine &TM);
    ~ELFWriter();

    typedef std::vector<unsigned char> DataBuffer;

  protected:
    /// Output stream to send the resultant object file to.
    ///
    raw_ostream &O;

    /// Target machine description.
    ///
    TargetMachine &TM;

    /// Mang - The object used to perform name mangling for this module.
    ///
    Mangler *Mang;

    /// MCE - The MachineCodeEmitter object that we are exposing to emit machine
    /// code for functions to the .o file.
    ELFCodeEmitter *MCE;

    //===------------------------------------------------------------------===//
    // Properties to be set by the derived class ctor, used to configure the
    // ELFWriter.

    // e_machine - This field is the target specific value to emit as the
    // e_machine member of the ELF header.
    unsigned short e_machine;

    // e_flags - The machine flags for the target.  This defaults to zero.
    unsigned e_flags;

    //===------------------------------------------------------------------===//
    // Properties inferred automatically from the target machine.
    //

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
    // The buffer we accumulate the file header into.  Note that this should be
    // changed into something much more efficient later (and the bitcode writer
    // as well!).
    DataBuffer FileHeader;

    /// SectionList - This is the list of sections that we have emitted to the
    /// file.  Once the file has been completely built, the section header table
    /// is constructed from this info.
    std::list<ELFSection> SectionList;
    unsigned NumSections;   // Always = SectionList.size()

    /// SectionLookup - This is a mapping from section name to section number in
    /// the SectionList.
    std::map<std::string, ELFSection*> SectionLookup;

    /// getSection - Return the section with the specified name, creating a new
    /// section if one does not already exist.
    ELFSection &getSection(const std::string &Name,
                           unsigned Type, unsigned Flags = 0) {
      ELFSection *&SN = SectionLookup[Name];
      if (SN) return *SN;

      SectionList.push_back(Name);
      SN = &SectionList.back();
      SN->SectionIdx = NumSections++;
      SN->Type = Type;
      SN->Flags = Flags;
      SN->Link = ELFSection::SHN_UNDEF;
      return *SN;
    }

    ELFSection &getTextSection() {
      return getSection(".text", ELFSection::SHT_PROGBITS,
                        ELFSection::SHF_EXECINSTR | ELFSection::SHF_ALLOC);
    }

    ELFSection &getDataSection() {
      return getSection(".data", ELFSection::SHT_PROGBITS,
                        ELFSection::SHF_WRITE | ELFSection::SHF_ALLOC);
    }
    ELFSection &getBSSSection() {
      return getSection(".bss", ELFSection::SHT_NOBITS,
                        ELFSection::SHF_WRITE | ELFSection::SHF_ALLOC);
    }

    /// SymbolTable - This is the list of symbols we have emitted to the file.
    /// This actually gets rearranged before emission to the file (to put the
    /// local symbols first in the list).
    std::vector<ELFSym> SymbolTable;

    // As we complete the ELF file, we need to update fields in the ELF header
    // (e.g. the location of the section table).  These members keep track of
    // the offset in ELFHeader of these various pieces to update and other
    // locations in the file.
    unsigned ELFHdr_e_shoff_Offset;     // e_shoff    in ELF header.
    unsigned ELFHdr_e_shstrndx_Offset;  // e_shstrndx in ELF header.
    unsigned ELFHdr_e_shnum_Offset;     // e_shnum    in ELF header.
  private:
    void EmitGlobal(GlobalVariable *GV);

    void EmitSymbolTable();

    void EmitSectionTableStringTable();
    void OutputSectionsAndSectionTable();
  };
}

#endif
