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
  
  /// MCSection - Instances of this class represent a uniqued identifier for a
  /// section in the current translation unit.  The MCContext class uniques and
  /// creates these.
  class MCSection {
    std::string Name;
    
    /// IsDirective - This is true if the section name is a directive, not
    /// something that should be printed with ".section".
    ///
    /// FIXME: This is a hack.  Switch to a semantic view of the section instead
    /// of a syntactic one.
    bool IsDirective;
    
    MCSection(const MCSection&);      // DO NOT IMPLEMENT
    void operator=(const MCSection&); // DO NOT IMPLEMENT
  protected:
    MCSection(const StringRef &Name, bool IsDirective, SectionKind K,
              MCContext &Ctx);
    SectionKind Kind;
  public:
    virtual ~MCSection();

    static MCSection *Create(const StringRef &Name, bool IsDirective, 
                             SectionKind K, MCContext &Ctx);
    
    const std::string &getName() const { return Name; }
    bool isDirective() const { return IsDirective; }
    
    SectionKind getKind() const { return Kind; }
  };

  
  class MCSectionELF : public MCSection {
    MCSectionELF(const StringRef &Name, bool IsDirective, SectionKind K,
                 MCContext &Ctx) : MCSection(Name, IsDirective, K, Ctx) {}
  public:
    
    static MCSectionELF *Create(const StringRef &Name, bool IsDirective, 
                                SectionKind K, MCContext &Ctx);
    
  };

  class MCSectionMachO : public MCSection {
    MCSectionMachO(const StringRef &Name, bool IsDirective, SectionKind K,
                   MCContext &Ctx) : MCSection(Name, IsDirective, K, Ctx) {}
  public:
    
    static MCSectionMachO *Create(const StringRef &Name, bool IsDirective, 
                                  SectionKind K, MCContext &Ctx);
  };
  
  class MCSectionCOFF : public MCSection {
    MCSectionCOFF(const StringRef &Name, bool IsDirective, SectionKind K,
                  MCContext &Ctx) : MCSection(Name, IsDirective, K, Ctx) {}
  public:
    
    static MCSectionCOFF *Create(const StringRef &Name, bool IsDirective, 
                                   SectionKind K, MCContext &Ctx);
  };
  
} // end namespace llvm

#endif
