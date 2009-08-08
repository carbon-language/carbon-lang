//===- MCSection.h - Machine Code Sections ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the MCSection class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCSECTION_H
#define LLVM_MC_MCSECTION_H

#include <string>
#include "llvm/ADT/StringRef.h"
#include "llvm/MC/SectionKind.h"

namespace llvm {
  class MCContext;
  class TargetAsmInfo;
  class raw_ostream;
  
  /// MCSection - Instances of this class represent a uniqued identifier for a
  /// section in the current translation unit.  The MCContext class uniques and
  /// creates these.
  class MCSection {
    MCSection(const MCSection&);      // DO NOT IMPLEMENT
    void operator=(const MCSection&); // DO NOT IMPLEMENT
  protected:
    MCSection(SectionKind K) : Kind(K) {}
    SectionKind Kind;
  public:
    virtual ~MCSection();

    SectionKind getKind() const { return Kind; }
    
    virtual void PrintSwitchToSection(const TargetAsmInfo &TAI,
                                      raw_ostream &OS) const = 0;
  };

  
  class MCSectionELF : public MCSection {
    std::string Name;
    
    /// IsDirective - This is true if the section name is a directive, not
    /// something that should be printed with ".section".
    ///
    /// FIXME: This is a hack.  Switch to a semantic view of the section instead
    /// of a syntactic one.
    bool IsDirective;
    
    MCSectionELF(const StringRef &Name, bool IsDirective, SectionKind K,
                 MCContext &Ctx);
  public:
    
    static MCSectionELF *Create(const StringRef &Name, bool IsDirective, 
                                SectionKind K, MCContext &Ctx);

    const std::string &getName() const { return Name; }
    bool isDirective() const { return IsDirective; }
    
    
    virtual void PrintSwitchToSection(const TargetAsmInfo &TAI,
                                      raw_ostream &OS) const;
  };

  class MCSectionMachO : public MCSection {
    std::string Name;
    
    /// IsDirective - This is true if the section name is a directive, not
    /// something that should be printed with ".section".
    ///
    /// FIXME: This is a hack.  Switch to a semantic view of the section instead
    /// of a syntactic one.
    bool IsDirective;
    
    MCSectionMachO(const StringRef &Name, bool IsDirective, SectionKind K,
                   MCContext &Ctx);
  public:
    
    static MCSectionMachO *Create(const StringRef &Name, bool IsDirective, 
                                  SectionKind K, MCContext &Ctx);

    const std::string &getName() const { return Name; }
    bool isDirective() const { return IsDirective; }
    
    virtual void PrintSwitchToSection(const TargetAsmInfo &TAI,
                                      raw_ostream &OS) const;
  };
  
  class MCSectionCOFF : public MCSection {
    std::string Name;
    
    /// IsDirective - This is true if the section name is a directive, not
    /// something that should be printed with ".section".
    ///
    /// FIXME: This is a hack.  Switch to a semantic view of the section instead
    /// of a syntactic one.
    bool IsDirective;
    
    MCSectionCOFF(const StringRef &Name, bool IsDirective, SectionKind K,
                  MCContext &Ctx);
  public:
    
    static MCSectionCOFF *Create(const StringRef &Name, bool IsDirective, 
                                   SectionKind K, MCContext &Ctx);

    const std::string &getName() const { return Name; }
    bool isDirective() const { return IsDirective; }
    
    virtual void PrintSwitchToSection(const TargetAsmInfo &TAI,
                                      raw_ostream &OS) const;
  };
  
} // end namespace llvm

#endif
