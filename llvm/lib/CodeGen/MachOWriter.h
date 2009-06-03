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

#include "MachO.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetMachOWriterInfo.h"
#include <map>

namespace llvm {
  class GlobalVariable;
  class Mangler;
  class MachineCodeEmitter;
  class MachOCodeEmitter;
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
    MachineCodeEmitter &getMachineCodeEmitter() const {
      return *(MachineCodeEmitter*)MCE;
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
    
    /// MCE - The MachineCodeEmitter object that we are exposing to emit machine
    /// code for functions to the .o file.

    MachOCodeEmitter *MCE;

    /// is64Bit/isLittleEndian - This information is inferred from the target
    /// machine directly, indicating what header values and flags to set.

    bool is64Bit, isLittleEndian;

    // Target Asm Info

    const TargetAsmInfo *TAI;

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
                             unsigned Flags = 0) {
      MachOSection *MOS = SectionLookup[seg+sect];
      if (MOS) return MOS;

      MOS = new MachOSection(seg, sect);
      SectionList.push_back(MOS);
      MOS->Index = SectionList.size();
      MOS->flags = MachOSection::S_REGULAR | Flags;
      SectionLookup[seg+sect] = MOS;
      return MOS;
    }
    MachOSection *getTextSection(bool isCode = true) {
      if (isCode)
        return getSection("__TEXT", "__text", 
                          MachOSection::S_ATTR_PURE_INSTRUCTIONS |
                          MachOSection::S_ATTR_SOME_INSTRUCTIONS);
      else
        return getSection("__TEXT", "__text");
    }
    MachOSection *getBSSSection() {
      return getSection("__DATA", "__bss", MachOSection::S_ZEROFILL);
    }
    MachOSection *getDataSection() {
      return getSection("__DATA", "__data");
    }
    MachOSection *getConstSection(Constant *C) {
      const ConstantArray *CVA = dyn_cast<ConstantArray>(C);
      if (CVA && CVA->isCString())
        return getSection("__TEXT", "__cstring", 
                          MachOSection::S_CSTRING_LITERALS);
      
      const Type *Ty = C->getType();
      if (Ty->isPrimitiveType() || Ty->isInteger()) {
        unsigned Size = TM.getTargetData()->getTypeAllocSize(Ty);
        switch(Size) {
        default: break; // Fall through to __TEXT,__const
        case 4:
          return getSection("__TEXT", "__literal4",
                            MachOSection::S_4BYTE_LITERALS);
        case 8:
          return getSection("__TEXT", "__literal8",
                            MachOSection::S_8BYTE_LITERALS);
        case 16:
          return getSection("__TEXT", "__literal16",
                            MachOSection::S_16BYTE_LITERALS);
        }
      }
      return getSection("__TEXT", "__const");
    }
    MachOSection *getJumpTableSection() {
      if (TM.getRelocationModel() == Reloc::PIC_)
        return getTextSection(false);
      else
        return getSection("__TEXT", "__const");
    }
    
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
    MachOSymTab   SymTab;     
    /// DySymTab - symbol table info for the dynamic link editor
    MachODySymTab DySymTab;

  protected:
  
    /// SymbolTable - This is the list of symbols we have emitted to the file.
    /// This actually gets rearranged before emission to the file (to put the
    /// local symbols first in the list).
    std::vector<MachOSym> SymbolTable;
    
    /// SymT - A buffer to hold the symbol table before we write it out at the
    /// appropriate location in the file.
    DataBuffer SymT;
    
    /// StrT - A buffer to hold the string table before we write it out at the
    /// appropriate location in the file.
    DataBuffer StrT;
    
    /// PendingSyms - This is a list of externally defined symbols that we have
    /// been asked to emit, but have not seen a reference to.  When a reference
    /// is seen, the symbol will move from this list to the SymbolTable.
    std::vector<GlobalValue*> PendingGlobals;
    
    /// DynamicSymbolTable - This is just a vector of indices into
    /// SymbolTable to aid in emitting the DYSYMTAB load command.
    std::vector<unsigned> DynamicSymbolTable;
    
    static void InitMem(const Constant *C, void *Addr, intptr_t Offset,
                        const TargetData *TD, 
                        std::vector<MachineRelocation> &MRs);

  private:
    void AddSymbolToSection(MachOSection *MOS, GlobalVariable *GV);
    void EmitGlobal(GlobalVariable *GV);
    void EmitHeaderAndLoadCommands();
    void EmitSections();
    void EmitRelocations();
    void BufferSymbolAndStringTable();
    void CalculateRelocations(MachOSection &MOS);

    MachineRelocation GetJTRelocation(unsigned Offset,
                                      MachineBasicBlock *MBB) const {
      return TM.getMachOWriterInfo()->GetJTRelocation(Offset, MBB);
    }

    /// GetTargetRelocation - Returns the number of relocations.
    unsigned GetTargetRelocation(MachineRelocation &MR,
                                 unsigned FromIdx,
                                 unsigned ToAddr,
                                 unsigned ToIndex,
                                 OutputBuffer &RelocOut,
                                 OutputBuffer &SecOut,
                                 bool Scattered,
                                 bool Extern) {
      return TM.getMachOWriterInfo()->GetTargetRelocation(MR, FromIdx, ToAddr,
                                                          ToIndex, RelocOut,
                                                          SecOut, Scattered,
                                                          Extern);
    }
  };
}

#endif
