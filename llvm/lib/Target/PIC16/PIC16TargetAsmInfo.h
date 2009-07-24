//=====-- PIC16TargetAsmInfo.h - PIC16 asm properties ---------*- C++ -*--====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source 
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the PIC16TargetAsmInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef PIC16TARGETASMINFO_H
#define PIC16TARGETASMINFO_H

#include "PIC16.h"
#include "llvm/Target/TargetAsmInfo.h"
#include <vector>
#include "llvm/Module.h"

namespace llvm {

  enum { DataBankSize = 80 };

  // Forward declaration.
  class PIC16TargetMachine;
  class GlobalVariable;

  /// PIC16 Splits the global data into mulitple udata and idata sections.
  /// Each udata and idata section needs to contain a list of globals that
  /// they contain, in order to avoid scanning over all the global values 
  /// again and printing only those that match the current section. 
  /// Keeping values inside the sections make printing a section much easier.
  struct PIC16Section {
    const Section *S_; // Connection to actual Section.
    unsigned Size;  // Total size of the objects contained.
    bool SectionPrinted;
    std::vector<const GlobalVariable*> Items;
   
    PIC16Section(const Section *s) {
      S_ = s;
      Size = 0;
      SectionPrinted = false;
    }
    bool isPrinted() const { return SectionPrinted; }
    void setPrintedStatus(bool status) { SectionPrinted = status; } 
  };
      
  struct PIC16TargetAsmInfo : public TargetAsmInfo {
    std::string getSectionNameForSym(const std::string &Sym) const;
    PIC16TargetAsmInfo(const PIC16TargetMachine &TM);
    mutable std::vector<PIC16Section *> BSSSections;
    mutable std::vector<PIC16Section *> IDATASections;
    mutable std::vector<PIC16Section *> AutosSections;
    mutable std::vector<PIC16Section *> ROSections;
    mutable PIC16Section *ExternalVarDecls;
    mutable PIC16Section *ExternalVarDefs;
    virtual ~PIC16TargetAsmInfo();

  private:
    const char *RomData8bitsDirective;
    const char *RomData16bitsDirective;
    const char *RomData32bitsDirective;
    const char *getRomDirective(unsigned size) const;
    virtual const char *getDataASDirective(unsigned size, unsigned AS) const;
    const Section *getBSSSectionForGlobal(const GlobalVariable *GV) const;
    const Section *getIDATASectionForGlobal(const GlobalVariable *GV) const;
    const Section *getSectionForAuto(const GlobalVariable *GV) const;
    const Section *CreateBSSSectionForGlobal(const GlobalVariable *GV,
                                             std::string Addr = "") const;
    const Section *CreateIDATASectionForGlobal(const GlobalVariable *GV,
                                               std::string Addr = "") const;
    const Section *getROSectionForGlobal(const GlobalVariable *GV) const;
    const Section *CreateROSectionForGlobal(const GlobalVariable *GV,
                                            std::string Addr = "") const;
    virtual const Section *SelectSectionForGlobal(const GlobalValue *GV) const;
    const Section *CreateSectionForGlobal(const GlobalVariable *GV,
                                          const std::string &Addr = "") const;
  public:
    void SetSectionForGVs(Module &M);
    const std::vector<PIC16Section*> &getBSSSections() const {
      return BSSSections;
    }
    const std::vector<PIC16Section*> &getIDATASections() const {
      return IDATASections;
    }
    const std::vector<PIC16Section*> &getAutosSections() const {
      return AutosSections;
    }
    const std::vector<PIC16Section*> &getROSections() const {
      return ROSections;
    }
    
    /// getSpecialCasedSectionGlobals - Allow the target to completely override
    /// section assignment of a global.
    virtual const Section *
    getSpecialCasedSectionGlobals(const GlobalValue *GV,
                                  SectionKind::Kind Kind) const;
    
  };

} // namespace llvm

#endif
