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
  class MCSectionPIC16;
  
  enum { DataBankSize = 80 };

  /// PIC16 Splits the global data into mulitple udata and idata sections.
  /// Each udata and idata section needs to contain a list of globals that
  /// they contain, in order to avoid scanning over all the global values 
  /// again and printing only those that match the current section. 
  /// Keeping values inside the sections make printing a section much easier.
  ///
  /// FIXME: MOVE ALL THIS STUFF TO MCSectionPIC16.
  ///
  struct PIC16Section {
    const MCSectionPIC16 *S_; // Connection to actual Section.
    unsigned Size;  // Total size of the objects contained.
    bool SectionPrinted;
    std::vector<const GlobalVariable*> Items;
    
    PIC16Section(const MCSectionPIC16 *s) {
      S_ = s;
      Size = 0;
      SectionPrinted = false;
    }
    bool isPrinted() const { return SectionPrinted; }
    void setPrintedStatus(bool status) { SectionPrinted = status; } 
  };
  
  class PIC16TargetObjectFile : public TargetLoweringObjectFile {
    const TargetMachine *TM;
    
    const MCSectionPIC16 *getPIC16Section(const char *Name,
                                          SectionKind K) const;
  public:
    mutable std::vector<PIC16Section*> BSSSections;
    mutable std::vector<PIC16Section*> IDATASections;
    mutable std::vector<PIC16Section*> AutosSections;
    mutable std::vector<PIC16Section*> ROSections;
    mutable PIC16Section *ExternalVarDecls;
    mutable PIC16Section *ExternalVarDefs;

    PIC16TargetObjectFile();
    ~PIC16TargetObjectFile();
    
    void Initialize(MCContext &Ctx, const TargetMachine &TM);

    
    virtual const MCSection *
    getExplicitSectionGlobal(const GlobalValue *GV, SectionKind Kind, 
                             Mangler *Mang, const TargetMachine &TM) const;
    
    virtual const MCSection *SelectSectionForGlobal(const GlobalValue *GV,
                                                    SectionKind Kind,
                                                    Mangler *Mang,
                                                    const TargetMachine&) const;

    const MCSection *getSectionForFunction(const std::string &FnName) const;
    const MCSection *getSectionForFunctionFrame(const std::string &FnName)const;
    
    
  private:
    std::string getSectionNameForSym(const std::string &Sym) const;

    const MCSection *getBSSSectionForGlobal(const GlobalVariable *GV) const;
    const MCSection *getIDATASectionForGlobal(const GlobalVariable *GV) const;
    const MCSection *getSectionForAuto(const GlobalVariable *GV) const;
    const MCSection *CreateBSSSectionForGlobal(const GlobalVariable *GV,
                                               std::string Addr = "") const;
    const MCSection *CreateIDATASectionForGlobal(const GlobalVariable *GV,
                                                 std::string Addr = "") const;
    const MCSection *getROSectionForGlobal(const GlobalVariable *GV) const;
    const MCSection *CreateROSectionForGlobal(const GlobalVariable *GV,
                                              std::string Addr = "") const;
    const MCSection *CreateSectionForGlobal(const GlobalVariable *GV,
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
