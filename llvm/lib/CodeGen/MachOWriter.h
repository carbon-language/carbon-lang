//=== MachOWriter.h - Target-independent Mach-O writer support --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the MachOWriter class.
//
//===----------------------------------------------------------------------===//

#ifndef MACHOWRITER_H
#define MACHOWRITER_H

#include "llvm/CodeGen/MachineFunctionPass.h"
#include <vector>
#include <map>

namespace llvm {
  class Constant;
  class GlobalVariable;
  class Mangler;
  class MachineBasicBlock;
  class MachineRelocation;
  class MachOCodeEmitter;
  struct MachODySymTab;
  struct MachOHeader;
  struct MachOSection;
  struct MachOSym;
  class TargetData;
  class TargetMachine;
  class MCAsmInfo;
  class ObjectCodeEmitter;
  class OutputBuffer;
  class raw_ostream;

  /// MachOWriter - This class implements the common target-independent code for
  /// writing Mach-O files.  Targets should derive a class from this to
  /// parameterize the output format.
  ///
  class MachOWriter : public MachineFunctionPass {
    friend class MachOCodeEmitter;
  public:
    static char ID;

    ObjectCodeEmitter *getObjectCodeEmitter() {
      return reinterpret_cast<ObjectCodeEmitter*>(MachOCE);
    }

    MachOWriter(raw_ostream &O, TargetMachine &TM);
    virtual ~MachOWriter();

    virtual const char *getPassName() const {
      return "Mach-O Writer";
    }

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

    /// MachOCE - The MachineCodeEmitter object that we are exposing to emit
    /// machine code for functions to the .o file.
    MachOCodeEmitter *MachOCE;

    /// is64Bit/isLittleEndian - This information is inferred from the target
    /// machine directly, indicating what header values and flags to set.
    bool is64Bit, isLittleEndian;

    // Target Asm Info
    const MCAsmInfo *TAI;

    /// Header - An instance of MachOHeader that we will update while we build
    /// the file, and then emit during finalization.
    MachOHeader Header;

    /// doInitialization - Emit the file header and all of the global variables
    /// for the module to the Mach-O file.
    bool doInitialization(Module &M);

    bool runOnMachineFunction(MachineFunction &MF);

    /// doFinalization - Now that the module has been completely processed, emit
    /// the Mach-O file to 'O'.
    bool doFinalization(Module &M);

  private:

    /// SectionList - This is the list of sections that we have emitted to the
    /// file.  Once the file has been completely built, the segment load command
    /// SectionCommands are constructed from this info.
    std::vector<MachOSection*> SectionList;

    /// SectionLookup - This is a mapping from section name to SectionList entry
    std::map<std::string, MachOSection*> SectionLookup;

    /// GVSection - This is a mapping from a GlobalValue to a MachOSection,
    /// to aid in emitting relocations.
    std::map<GlobalValue*, MachOSection*> GVSection;

    /// GVOffset - This is a mapping from a GlobalValue to an offset from the
    /// start of the section in which the GV resides, to aid in emitting
    /// relocations.
    std::map<GlobalValue*, intptr_t> GVOffset;

    /// getSection - Return the section with the specified name, creating a new
    /// section if one does not already exist.
    MachOSection *getSection(const std::string &seg, const std::string &sect,
                             unsigned Flags = 0);

    /// getTextSection - Return text section with different flags for code/data
    MachOSection *getTextSection(bool isCode = true);

    MachOSection *getDataSection() {
      return getSection("__DATA", "__data");
    }

    MachOSection *getBSSSection();
    MachOSection *getConstSection(Constant *C);
    MachOSection *getJumpTableSection();

    /// MachOSymTab - This struct contains information about the offsets and
    /// size of symbol table information.
    /// segment.
    struct MachOSymTab {
      uint32_t cmd;     // LC_SYMTAB
      uint32_t cmdsize; // sizeof( MachOSymTab )
      uint32_t symoff;  // symbol table offset
      uint32_t nsyms;   // number of symbol table entries
      uint32_t stroff;  // string table offset
      uint32_t strsize; // string table size in bytes

      // Constants for the cmd field
      // see <mach-o/loader.h>
      enum { LC_SYMTAB = 0x02  // link-edit stab symbol table info
      };

      MachOSymTab() : cmd(LC_SYMTAB), cmdsize(6 * sizeof(uint32_t)), symoff(0),
        nsyms(0), stroff(0), strsize(0) { }
    };

    /// SymTab - The "stab" style symbol table information
    MachOSymTab SymTab;
    /// DySymTab - symbol table info for the dynamic link editor
    MachODySymTab DySymTab;

  protected:

    /// SymbolTable - This is the list of symbols we have emitted to the file.
    /// This actually gets rearranged before emission to the file (to put the
    /// local symbols first in the list).
    std::vector<MachOSym> SymbolTable;

    /// SymT - A buffer to hold the symbol table before we write it out at the
    /// appropriate location in the file.
    std::vector<unsigned char> SymT;

    /// StrT - A buffer to hold the string table before we write it out at the
    /// appropriate location in the file.
    std::vector<unsigned char> StrT;

    /// PendingSyms - This is a list of externally defined symbols that we have
    /// been asked to emit, but have not seen a reference to.  When a reference
    /// is seen, the symbol will move from this list to the SymbolTable.
    std::vector<GlobalValue*> PendingGlobals;

    /// DynamicSymbolTable - This is just a vector of indices into
    /// SymbolTable to aid in emitting the DYSYMTAB load command.
    std::vector<unsigned> DynamicSymbolTable;

    static void InitMem(const Constant *C, uintptr_t Offset,
                        const TargetData *TD, MachOSection* mos);

  private:
    void AddSymbolToSection(MachOSection *MOS, GlobalVariable *GV);
    void EmitGlobal(GlobalVariable *GV);
    void EmitHeaderAndLoadCommands();
    void EmitSections();
    void EmitRelocations();
    void BufferSymbolAndStringTable();
    void CalculateRelocations(MachOSection &MOS);

    // GetJTRelocation - Get a relocation a new BB relocation based
    // on target information.
    MachineRelocation GetJTRelocation(unsigned Offset,
                                      MachineBasicBlock *MBB) const;

    /// GetTargetRelocation - Returns the number of relocations.
    unsigned GetTargetRelocation(MachineRelocation &MR, unsigned FromIdx,
                                 unsigned ToAddr, unsigned ToIndex,
                                 OutputBuffer &RelocOut, OutputBuffer &SecOut,
                                 bool Scattered, bool Extern);
  };
}

#endif
