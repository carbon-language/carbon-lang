//===-- PIC16TargetObjectFile.h - PIC16 Object Info -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_PIC16_TARGETOBJECTFILE_H
#define LLVM_TARGET_PIC16_TARGETOBJECTFILE_H

#include "llvm/Target/TargetLoweringObjectFile.h"
#include <vector>

namespace llvm {
  class GlobalVariable;
  class Module;
  class PIC16TargetMachine;
  
  enum { DataBankSize = 80 };

  /// PIC16 Splits the global data into mulitple udata and idata sections.
  /// Each udata and idata section needs to contain a list of globals that
  /// they contain, in order to avoid scanning over all the global values 
  /// again and printing only those that match the current section. 
  /// Keeping values inside the sections make printing a section much easier.
  ///
  /// FIXME: Reimplement by inheriting from MCSection.
  ///
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
  
  class PIC16TargetObjectFile : public TargetLoweringObjectFile {
    const PIC16TargetMachine &TM;
  public:
    mutable std::vector<PIC16Section*> BSSSections;
    mutable std::vector<PIC16Section*> IDATASections;
    mutable std::vector<PIC16Section*> AutosSections;
    mutable std::vector<PIC16Section*> ROSections;
    mutable PIC16Section *ExternalVarDecls;
    mutable PIC16Section *ExternalVarDefs;
    
    PIC16TargetObjectFile(const PIC16TargetMachine &TM);
    ~PIC16TargetObjectFile();
    
    /// getSpecialCasedSectionGlobals - Allow the target to completely override
    /// section assignment of a global.
    virtual const Section *
    getSpecialCasedSectionGlobals(const GlobalValue *GV, Mangler *Mang,
                                  SectionKind Kind) const;
    virtual const Section *SelectSectionForGlobal(const GlobalValue *GV,
                                                  SectionKind Kind,
                                                  Mangler *Mang,
                                                  const TargetMachine&) const;
  private:
    std::string getSectionNameForSym(const std::string &Sym) const;

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
    const Section *CreateSectionForGlobal(const GlobalVariable *GV,
                                          Mangler *Mang,
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
    
  };
} // end namespace llvm

#endif
