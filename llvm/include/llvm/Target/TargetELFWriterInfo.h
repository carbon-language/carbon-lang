//===-- llvm/Target/TargetELFWriterInfo.h - ELF Writer Info -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the TargetELFWriterInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETELFWRITERINFO_H
#define LLVM_TARGET_TARGETELFWRITERINFO_H

#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Function.h"

namespace llvm {

  //===--------------------------------------------------------------------===//
  //                          TargetELFWriterInfo
  //===--------------------------------------------------------------------===//

  class TargetELFWriterInfo {
  protected:
    // EMachine - This field is the target specific value to emit as the
    // e_machine member of the ELF header.
    unsigned short EMachine;
    TargetMachine &TM;
  public:

    // Machine architectures
    enum MachineType {
      EM_NONE = 0,     // No machine
      EM_M32 = 1,      // AT&T WE 32100
      EM_SPARC = 2,    // SPARC
      EM_386 = 3,      // Intel 386
      EM_68K = 4,      // Motorola 68000
      EM_88K = 5,      // Motorola 88000
      EM_486 = 6,      // Intel 486 (deprecated)
      EM_860 = 7,      // Intel 80860
      EM_MIPS = 8,     // MIPS R3000
      EM_PPC = 20,     // PowerPC
      EM_ARM = 40,     // ARM
      EM_ALPHA = 41,   // DEC Alpha
      EM_SPARCV9 = 43, // SPARC V9
      EM_X86_64 = 62   // AMD64
    };

    explicit TargetELFWriterInfo(TargetMachine &tm) : TM(tm) {}
    virtual ~TargetELFWriterInfo() {}

    unsigned short getEMachine() const { return EMachine; }

    /// getFunctionAlignment - Returns the alignment for function 'F', targets
    /// with different alignment constraints should overload this method
    virtual unsigned getFunctionAlignment(const Function *F) const {
      const TargetData *TD = TM.getTargetData();
      unsigned FnAlign = F->getAlignment();
      unsigned TDAlign = TD->getPointerABIAlignment();
      unsigned Align = std::max(FnAlign, TDAlign);
      assert(!(Align & (Align-1)) && "Alignment is not a power of two!");
      return Align;
    }
  };

} // end llvm namespace

#endif // LLVM_TARGET_TARGETELFWRITERINFO_H
