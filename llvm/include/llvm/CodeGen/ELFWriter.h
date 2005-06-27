//===-- ELFWriter.h - Target-independent ELF writer support -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the ELFWriter class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_ELFWRITER_H
#define LLVM_CODEGEN_ELFWRITER_H

#include "llvm/CodeGen/MachineFunctionPass.h"

namespace llvm {
  class GlobalVariable;

  /// ELFWriter - This class implements the common target-independent code for
  /// writing ELF files.  Targets should derive a class from this to
  /// parameterize the output format.
  ///
  class ELFWriter : public MachineFunctionPass {
  protected:
    ELFWriter(std::ostream &O, TargetMachine &TM);

    /// Output stream to send the resultant object file to.
    ///
    std::ostream &O;

    /// Target machine description.
    ///
    TargetMachine &TM;

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
    // The buffer we are accumulating the file into.  Note that this should be
    // changed into something much more efficient later (and the bytecode writer
    // as well!).
    std::vector<unsigned char> OutputBuffer;

    /// ELFSection - This struct contains information about each section that is
    /// emitted to the OutputBuffer.  This is eventually turned into the section
    /// header table at the end of the file.
    struct ELFSection {
      std::string Name;       // Name of the section.
      unsigned NameIdx;       // Index in .shstrtab of name, once emitted.
      unsigned Type;
      unsigned Flags;
      uint64_t Addr;
      unsigned Offset;
      unsigned Size;
      unsigned Link;
      unsigned Info;
      unsigned Align;
      unsigned EntSize;
      ELFSection() : Type(0), Flags(0), Addr(0), Offset(0), Size(0), Link(0),
                     Info(0), Align(0), EntSize(0) {
      }
      ELFSection(const char *name, unsigned offset)
        : Name(name), Type(0), Flags(0), Addr(0), Offset(offset), Size(0),
          Link(0), Info(0), Align(0), EntSize(0) {
      }
    };

    /// SectionList - This is the list of sections that we have emitted to the
    /// file.  Once the file has been completely built, the section header table
    /// is constructed from this info.
    std::vector<ELFSection> SectionList;

    // As we accumulate the ELF file into OutputBuffer, we occasionally need to
    // keep track of locations to update later (e.g. the location of the section
    // table in the ELF header.  These members keep track of the offset in
    // OffsetBuffer of these various pieces to update and other locations in the
    // file.
    unsigned ELFHeader_e_shoff_Offset;     // e_shoff    in ELF header.
    unsigned ELFHeader_e_shstrndx_Offset;  // e_shstrndx in ELF header.
    unsigned ELFHeader_e_shnum_Offset;     // e_shnum    in ELF header.

    void outbyte(unsigned char X) { OutputBuffer.push_back(X); }
    void outhalf(unsigned short X) {
      if (isLittleEndian) {
        OutputBuffer.push_back(X&255);
        OutputBuffer.push_back(X >> 8);
      } else {
        OutputBuffer.push_back(X >> 8);
        OutputBuffer.push_back(X&255);
      }
    }
    void outword(unsigned X) {
      if (isLittleEndian) {
        OutputBuffer.push_back((X >>  0) & 255);
        OutputBuffer.push_back((X >>  8) & 255);
        OutputBuffer.push_back((X >> 16) & 255);
        OutputBuffer.push_back((X >> 24) & 255);
      } else {
        OutputBuffer.push_back((X >> 24) & 255);
        OutputBuffer.push_back((X >> 16) & 255);
        OutputBuffer.push_back((X >>  8) & 255);
        OutputBuffer.push_back((X >>  0) & 255);
      }
    }
    void outaddr(uint64_t X) {
      if (!is64Bit)
        outword((unsigned)X);
      else
        assert(0 && "Emission of 64-bit data not implemented yet!");
    }

    // fix functions - Replace an existing entry at an offset.
    void fixhalf(unsigned short X, unsigned Offset) {
      unsigned char *P = &OutputBuffer[Offset];
      P[0] = (X >> (isLittleEndian ?  0 : 8)) & 255;
      P[1] = (X >> (isLittleEndian ?  8 : 0)) & 255;
    }

    void fixword(unsigned X, unsigned Offset) {
      unsigned char *P = &OutputBuffer[Offset];
      P[0] = (X >> (isLittleEndian ?  0 : 24)) & 255;
      P[1] = (X >> (isLittleEndian ?  8 : 16)) & 255;
      P[2] = (X >> (isLittleEndian ? 16 :  8)) & 255;
      P[3] = (X >> (isLittleEndian ? 24 :  0)) & 255;
    }

    void fixaddr(uint64_t X, unsigned Offset) {
      if (!is64Bit)
        fixword((unsigned)X, Offset);
      else
        assert(0 && "Emission of 64-bit data not implemented yet!");
    }

  private:
    void EmitDATASectionGlobal(GlobalVariable *GV);
    void EmitBSSSectionGlobal(GlobalVariable *GV);
    void EmitGlobal(GlobalVariable *GV);

    void EmitSectionTableStringTable();
    void EmitSectionTable();
  };
}

#endif
