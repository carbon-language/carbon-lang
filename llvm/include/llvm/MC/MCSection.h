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

  
  /// MCSectionMachO - This represents a section on a Mach-O system (used by
  /// Mac OS X).  On a Mac system, these are also described in
  /// /usr/include/mach-o/loader.h.
  class MCSectionMachO : public MCSection {
    char SegmentName[16];  // Not necessarily null terminated!
    char SectionName[16];  // Not necessarily null terminated!
    
    /// TypeAndAttributes - This is the SECTION_TYPE and SECTION_ATTRIBUTES
    /// field of a section, drawn from the enums below.
    unsigned TypeAndAttributes;
    
    /// Reserved2 - The 'reserved2' field of a section, used to represent the
    /// size of stubs, for example.
    unsigned Reserved2;
    
    MCSectionMachO(const StringRef &Segment, const StringRef &Section,
                   unsigned TAA, unsigned reserved2, SectionKind K)
      : MCSection(K), TypeAndAttributes(TAA), Reserved2(reserved2) {
      assert(Segment.size() <= 16 && Section.size() <= 16 &&
             "Segment or section string too long");
      for (unsigned i = 0; i != 16; ++i) {
        if (i < Segment.size())
          SegmentName[i] = Segment[i];
        else
          SegmentName[i] = 0;
        
        if (i < Section.size())
          SectionName[i] = Section[i];
        else
          SectionName[i] = 0;
      }        
    }
  public:
    
    static MCSectionMachO *Create(const StringRef &Segment,
                                  const StringRef &Section,
                                  unsigned TypeAndAttributes,
                                  unsigned Reserved2,
                                  SectionKind K, MCContext &Ctx);
    
    /// These are the section type and attributes fields.  A MachO section can
    /// have only one Type, but can have any of the attributes specified.
    enum {
      // TypeAndAttributes bitmasks.
      SECTION_TYPE       = 0x000000FFU,
      SECTION_ATTRIBUTES = 0xFFFFFF00U,
      
      // Valid section types.
      
      /// S_REGULAR - Regular section.
      S_REGULAR                    = 0x00U,
      /// S_ZEROFILL - Zero fill on demand section.
      S_ZEROFILL                   = 0x01U,
      /// S_CSTRING_LITERALS - Section with literal C strings.
      S_CSTRING_LITERALS           = 0x02U,
      /// S_4BYTE_LITERALS - Section with 4 byte literals.
      S_4BYTE_LITERALS             = 0x03U,
      /// S_8BYTE_LITERALS - Section with 8 byte literals.
      S_8BYTE_LITERALS             = 0x04U,
      /// S_LITERAL_POINTERS - Section with pointers to literals.
      S_LITERAL_POINTERS           = 0x05U,
      /// S_NON_LAZY_SYMBOL_POINTERS - Section with non-lazy symbol pointers.
      S_NON_LAZY_SYMBOL_POINTERS   = 0x06U,
      /// S_LAZY_SYMBOL_POINTERS - Section with lazy symbol pointers.
      S_LAZY_SYMBOL_POINTERS       = 0x07U,
      /// S_SYMBOL_STUBS - Section with symbol stubs, byte size of stub in
      /// the Reserved2 field.
      S_SYMBOL_STUBS               = 0x08U,
      /// S_SYMBOL_STUBS - Section with only function pointers for
      /// initialization.
      S_MOD_INIT_FUNC_POINTERS     = 0x09U,
      /// S_MOD_INIT_FUNC_POINTERS - Section with only function pointers for
      /// termination.
      S_MOD_TERM_FUNC_POINTERS     = 0x0AU,
      /// S_COALESCED - Section contains symbols that are to be coalesced.
      S_COALESCED                  = 0x0BU,
      /// S_GB_ZEROFILL - Zero fill on demand section (that can be larger than 4
      /// gigabytes).
      S_GB_ZEROFILL                = 0x0CU,
      /// S_INTERPOSING - Section with only pairs of function pointers for
      /// interposing.
      S_INTERPOSING                = 0x0DU,
      /// S_16BYTE_LITERALS - Section with only 16 byte literals.
      S_16BYTE_LITERALS            = 0x0EU,
      /// S_DTRACE_DOF - Section contains DTrace Object Format.
      S_DTRACE_DOF                 = 0x0FU,
      /// S_LAZY_DYLIB_SYMBOL_POINTERS - Section with lazy symbol pointers to
      /// lazy loaded dylibs.
      S_LAZY_DYLIB_SYMBOL_POINTERS = 0x10U,

      LAST_KNOWN_SECTION_TYPE = S_LAZY_DYLIB_SYMBOL_POINTERS,
      

      // Valid section attributes.
      
      /// S_ATTR_PURE_INSTRUCTIONS - Section contains only true machine
      /// instructions.
      S_ATTR_PURE_INSTRUCTIONS   = 1U << 31,
      /// S_ATTR_NO_TOC - Section contains coalesced symbols that are not to be
      /// in a ranlib table of contents.
      S_ATTR_NO_TOC              = 1U << 30,
      /// S_ATTR_STRIP_STATIC_SYMS - Ok to strip static symbols in this section
      /// in files with the MY_DYLDLINK flag.
      S_ATTR_STRIP_STATIC_SYMS   = 1U << 29,
      /// S_ATTR_NO_DEAD_STRIP - No dead stripping.
      S_ATTR_NO_DEAD_STRIP       = 1U << 28,
      /// S_ATTR_LIVE_SUPPORT - Blocks are live if they reference live blocks.
      S_ATTR_LIVE_SUPPORT        = 1U << 27,
      /// S_ATTR_SELF_MODIFYING_CODE - Used with i386 code stubs written on by
      /// dyld.
      S_ATTR_SELF_MODIFYING_CODE = 1U << 26,
      /// S_ATTR_DEBUG - A debug section.
      S_ATTR_DEBUG               = 1U << 25,
      /// S_ATTR_SOME_INSTRUCTIONS - Section contains some machine instructions.
      S_ATTR_SOME_INSTRUCTIONS   = 1U << 10,
      /// S_ATTR_EXT_RELOC - Section has external relocation entries.
      S_ATTR_EXT_RELOC           = 1U << 9,
      /// S_ATTR_LOC_RELOC - Section has local relocation entries.
      S_ATTR_LOC_RELOC           = 1U << 8
    };

    StringRef getSegmentName() const {
      // SegmentName is not necessarily null terminated!
      if (SegmentName[15])
        return StringRef(SegmentName, 16);
      return StringRef(SegmentName);
    }
    StringRef getSectionName() const {
      // SectionName is not necessarily null terminated!
      if (SectionName[15])
        return StringRef(SectionName, 16);
      return StringRef(SectionName);
    }
    
    unsigned getTypeAndAttributes() const { return TypeAndAttributes; }
    
    
    /// ParseSectionSpecifier - Parse the section specifier indicated by "Spec".
    /// This is a string that can appear after a .section directive in a mach-o
    /// flavored .s file.  If successful, this fills in the specified Out
    /// parameters and returns an empty string.  When an invalid section
    /// specifier is present, this returns a string indicating the problem.
    static std::string ParseSectionSpecifier(StringRef Spec,       // In.
                                             StringRef &Segment,   // Out.
                                             StringRef &Section,   // Out.
                                             unsigned  &TAA,       // Out.
                                             unsigned  &StubSize); // Out.
    
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
