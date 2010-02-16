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

#include "PIC16.h"
#include "PIC16ABINames.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/ADT/StringMap.h"
#include <vector>
#include <string>

namespace llvm {
  class GlobalVariable;
  class Module;
  class PIC16TargetMachine;
  class PIC16Section;
  
  enum { DataBankSize = 80 };

  /// PIC16 Splits the global data into mulitple udata and idata sections.
  /// Each udata and idata section needs to contain a list of globals that
  /// they contain, in order to avoid scanning over all the global values 
  /// again and printing only those that match the current section. 
  /// Keeping values inside the sections make printing a section much easier.
  ///
  /// FIXME: MOVE ALL THIS STUFF TO PIC16Section.
  ///

  /// PIC16TargetObjectFile - PIC16 Object file. Contains data and code
  /// sections. 
  // PIC16 Object File has two types of sections.
  // 1. Standard Sections
  //    1.1 un-initialized global data 
  //    1.2 initialized global data
  //    1.3 program memory data
  //    1.4 local variables of functions.
  // 2. User defined sections
  //    2.1 Objects placed in a specific section. (By _Section() macro)
  //    2.2 Objects placed at a specific address. (By _Address() macro)
  class PIC16TargetObjectFile : public TargetLoweringObjectFile {
    /// SectionsByName - Bindings of names to allocated sections.
    mutable StringMap<PIC16Section*> SectionsByName;

    const TargetMachine *TM;
    
    /// Lists of sections.
    /// Standard Data Sections.
    mutable std::vector<PIC16Section *> UDATASections_;
    mutable std::vector<PIC16Section *> IDATASections_;
    mutable PIC16Section * ROMDATASection_;
    mutable PIC16Section * SHAREDUDATASection_;

    /// Standard Auto Sections.
    mutable std::vector<PIC16Section *> AUTOSections_;
 
    /// User specified sections.
    mutable std::vector<PIC16Section *> USERSections_;

    
    /// Find or Create a PIC16 Section, without adding it to any
    /// section list.
    PIC16Section *getPIC16Section(const std::string &Name,
                                   PIC16SectionType Ty, 
                                   const std::string &Address = "", 
                                   int Color = -1) const;

    /// Convenience functions. These wrappers also take care of adding 
    /// the newly created section to the appropriate sections list.

    /// Find or Create PIC16 Standard Data Section.
    PIC16Section *getPIC16DataSection(const std::string &Name,
                                       PIC16SectionType Ty, 
                                       const std::string &Address = "", 
                                       int Color = -1) const;

    /// Find or Create PIC16 Standard Auto Section.
    PIC16Section *getPIC16AutoSection(const std::string &Name,
                                       PIC16SectionType Ty = UDATA_OVR,
                                       const std::string &Address = "", 
                                       int Color = -1) const;

    /// Find or Create PIC16 Standard Auto Section.
    PIC16Section *getPIC16UserSection(const std::string &Name,
                                       PIC16SectionType Ty, 
                                       const std::string &Address = "", 
                                       int Color = -1) const;

    /// Allocate Un-initialized data to a standard UDATA section. 
    const MCSection *allocateUDATA(const GlobalVariable *GV) const;

    /// Allocate Initialized data to a standard IDATA section. 
    const MCSection *allocateIDATA(const GlobalVariable *GV) const;

    /// Allocate ROM data to the standard ROMDATA section. 
    const MCSection *allocateROMDATA(const GlobalVariable *GV) const;

    /// Allocate an AUTO variable to an AUTO section.
    const MCSection *allocateAUTO(const GlobalVariable *GV) const;
    
    /// Allocate DATA in user specified section.
    const MCSection *allocateInGivenSection(const GlobalVariable *GV) const;

    /// Allocate DATA at user specified address.
    const MCSection *allocateAtGivenAddress(const GlobalVariable *GV,
                                            const std::string &Addr) const;

    /// Allocate a shared variable to SHARED section.
    const MCSection *allocateSHARED(const GlobalVariable *GV,
                                    Mangler *Mang) const;
   
    public:
    PIC16TargetObjectFile();
    ~PIC16TargetObjectFile();
    void Initialize(MCContext &Ctx, const TargetMachine &TM);

    /// Return the section with the given Name. Null if not found.
    PIC16Section *findPIC16Section(const std::string &Name);

    /// Override section allocations for user specified sections.
    virtual const MCSection *
    getExplicitSectionGlobal(const GlobalValue *GV, SectionKind Kind, 
                             Mangler *Mang, const TargetMachine &TM) const;
    
    /// Select sections for Data and Auto variables(globals).
    virtual const MCSection *SelectSectionForGlobal(const GlobalValue *GV,
                                                    SectionKind Kind,
                                                    Mangler *Mang,
                                                    const TargetMachine&) const;


    /// Return a code section for a function.
    const PIC16Section *SectionForCode (const std::string &FnName,
                                        bool isISR) const;

    /// Return a frame section for a function.
    const PIC16Section *SectionForFrame (const std::string &FnName) const;

    /// Accessors for various section lists.
    const std::vector<PIC16Section *> &UDATASections() const {
      return UDATASections_;
    }
    const std::vector<PIC16Section *> &IDATASections() const {
      return IDATASections_;
    }
    const PIC16Section *ROMDATASection() const {
      return ROMDATASection_;
    }
    const PIC16Section *SHAREDUDATASection() const {
      return SHAREDUDATASection_;
    }
    const std::vector<PIC16Section *> &AUTOSections() const {
      return AUTOSections_;
    }
    const std::vector<PIC16Section *> &USERSections() const {
      return USERSections_;
    }
  };
} // end namespace llvm

#endif
