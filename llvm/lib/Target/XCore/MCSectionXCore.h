//===- MCSectionXCore.h - XCore-specific section representation -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the MCSectionXCore class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MCSECTION_XCORE_H
#define LLVM_MCSECTION_XCORE_H

#include "llvm/MC/MCSectionELF.h"

namespace llvm {
  
class MCSectionXCore : public MCSectionELF {
  MCSectionXCore(const StringRef &Section, unsigned Type, unsigned Flags,
                 SectionKind K, bool isExplicit)
    : MCSectionELF(Section, Type, Flags, K, isExplicit) {}
  
public:
  
  enum {
    /// SHF_CP_SECTION - All sections with the "c" flag are grouped together
    /// by the linker to form the constant pool and the cp register is set to
    /// the start of the constant pool by the boot code.
    SHF_CP_SECTION = FIRST_TARGET_DEP_FLAG,
    
    /// SHF_DP_SECTION - All sections with the "d" flag are grouped together
    /// by the linker to form the data section and the dp register is set to
    /// the start of the section by the boot code.
    SHF_DP_SECTION = FIRST_TARGET_DEP_FLAG << 1
  };
  
  static MCSectionXCore *Create(const StringRef &Section, unsigned Type,
                                unsigned Flags, SectionKind K,
                                bool isExplicit, MCContext &Ctx);
  
  
  /// PrintTargetSpecificSectionFlags - This handles the XCore-specific cp/dp
  /// section flags.
  virtual void PrintTargetSpecificSectionFlags(const MCAsmInfo &MAI,
                                               raw_ostream &OS) const;

};
  
} // end namespace llvm

#endif
