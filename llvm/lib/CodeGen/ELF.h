//===-- lib/CodeGen/ELF.h - ELF constants and data structures ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This header contains common, non-processor-specific data structures and
// constants for the ELF file format.
//
// The details of the ELF32 bits in this file are largely based on the Tool
// Interface Standard (TIS) Executable and Linking Format (ELF) Specification
// Version 1.2, May 1995. The ELF64 is based on HP/Intel definition of the
// ELF-64 object file format document, Version 1.5 Draft 2 May 27, 1998
//
//===----------------------------------------------------------------------===//

#ifndef CODEGEN_ELF_H
#define CODEGEN_ELF_H

#include "llvm/CodeGen/BinaryObject.h"
#include "llvm/CodeGen/MachineRelocation.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/DataTypes.h"

namespace llvm {
  class GlobalValue;

  /// ELFSym - This struct contains information about each symbol that is
  /// added to logical symbol table for the module.  This is eventually
  /// turned into a real symbol table in the file.
  struct ELFSym {

    // ELF symbols are related to llvm ones by being one of the two llvm
    // types, for the other ones (section, file, func) a null pointer is
    // assumed by default.
    union {
      const GlobalValue *GV;  // If this is a pointer to a GV
      const char *Ext;        // If this is a pointer to a named symbol
    } Source;

    // Describes from which source type this ELF symbol comes from,
    // they can be GlobalValue, ExternalSymbol or neither.
    enum {
      isGV,      // The Source.GV field is valid.
      isExtSym,  // The Source.ExtSym field is valid.
      isOther    // Not a GlobalValue or External Symbol
    };
    unsigned SourceType;

    bool isGlobalValue() const { return SourceType == isGV; }
    bool isExternalSym() const { return SourceType == isExtSym; }

    // getGlobalValue - If this is a global value which originated the
    // elf symbol, return a reference to it.
    const GlobalValue *getGlobalValue() const {
      assert(SourceType == isGV && "This is not a global value");
      return Source.GV;
    }

    // getExternalSym - If this is an external symbol which originated the
    // elf symbol, return a reference to it.
    const char *getExternalSymbol() const {
      assert(SourceType == isExtSym && "This is not an external symbol");
      return Source.Ext;
    }

    // getGV - From a global value return a elf symbol to represent it
    static ELFSym *getGV(const GlobalValue *GV, unsigned Bind,
                         unsigned Type, unsigned Visibility) {
      ELFSym *Sym = new ELFSym();
      Sym->Source.GV = GV;
      Sym->setBind(Bind);
      Sym->setType(Type);
      Sym->setVisibility(Visibility);
      Sym->SourceType = isGV;
      return Sym;
    }

    // getExtSym - Create and return an elf symbol to represent an
    // external symbol
    static ELFSym *getExtSym(const char *Ext) {
      ELFSym *Sym = new ELFSym();
      Sym->Source.Ext = Ext;
      Sym->setBind(ELF::STB_GLOBAL);
      Sym->setType(ELF::STT_NOTYPE);
      Sym->setVisibility(ELF::STV_DEFAULT);
      Sym->SourceType = isExtSym;
      return Sym;
    }

    // getSectionSym - Returns a elf symbol to represent an elf section
    static ELFSym *getSectionSym() {
      ELFSym *Sym = new ELFSym();
      Sym->setBind(ELF::STB_LOCAL);
      Sym->setType(ELF::STT_SECTION);
      Sym->setVisibility(ELF::STV_DEFAULT);
      Sym->SourceType = isOther;
      return Sym;
    }

    // getFileSym - Returns a elf symbol to represent the module identifier
    static ELFSym *getFileSym() {
      ELFSym *Sym = new ELFSym();
      Sym->setBind(ELF::STB_LOCAL);
      Sym->setType(ELF::STT_FILE);
      Sym->setVisibility(ELF::STV_DEFAULT);
      Sym->SectionIdx = 0xfff1;  // ELFSection::SHN_ABS;
      Sym->SourceType = isOther;
      return Sym;
    }

    // getUndefGV - Returns a STT_NOTYPE symbol
    static ELFSym *getUndefGV(const GlobalValue *GV, unsigned Bind) {
      ELFSym *Sym = new ELFSym();
      Sym->Source.GV = GV;
      Sym->setBind(Bind);
      Sym->setType(ELF::STT_NOTYPE);
      Sym->setVisibility(ELF::STV_DEFAULT);
      Sym->SectionIdx = 0;  //ELFSection::SHN_UNDEF;
      Sym->SourceType = isGV;
      return Sym;
    }

    // ELF specific fields
    unsigned NameIdx;         // Index in .strtab of name, once emitted.
    uint64_t Value;
    unsigned Size;
    uint8_t Info;
    uint8_t Other;
    unsigned short SectionIdx;

    // Symbol index into the Symbol table
    unsigned SymTabIdx;

    ELFSym() : SourceType(isOther), NameIdx(0), Value(0),
               Size(0), Info(0), Other(ELF::STV_DEFAULT), SectionIdx(0),
               SymTabIdx(0) {}

    unsigned getBind() const { return (Info >> 4) & 0xf; }
    unsigned getType() const { return Info & 0xf; }
    bool isLocalBind() const { return getBind() == ELF::STB_LOCAL; }
    bool isFileType() const { return getType() == ELF::STT_FILE; }

    void setBind(unsigned X) {
      assert(X == (X & 0xF) && "Bind value out of range!");
      Info = (Info & 0x0F) | (X << 4);
    }

    void setType(unsigned X) {
      assert(X == (X & 0xF) && "Type value out of range!");
      Info = (Info & 0xF0) | X;
    }

    void setVisibility(unsigned V) {
      assert(V == (V & 0x3) && "Visibility value out of range!");
      Other = V;
    }
  };

  /// ELFSection - This struct contains information about each section that is
  /// emitted to the file.  This is eventually turned into the section header
  /// table at the end of the file.
  class ELFSection : public BinaryObject {
    public:
    // ELF specific fields
    unsigned NameIdx;   // sh_name - .shstrtab idx of name, once emitted.
    unsigned Type;      // sh_type - Section contents & semantics 
    unsigned Flags;     // sh_flags - Section flags.
    uint64_t Addr;      // sh_addr - The mem addr this section is in.
    unsigned Offset;    // sh_offset - Offset from the file start
    unsigned Size;      // sh_size - The section size.
    unsigned Link;      // sh_link - Section header table index link.
    unsigned Info;      // sh_info - Auxiliary information.
    unsigned Align;     // sh_addralign - Alignment of section.
    unsigned EntSize;   // sh_entsize - Size of entries in the section e

    /// SectionIdx - The number of the section in the Section Table.
    unsigned short SectionIdx;

    /// Sym - The symbol to represent this section if it has one.
    ELFSym *Sym;

    /// getSymIndex - Returns the symbol table index of the symbol
    /// representing this section.
    unsigned getSymbolTableIndex() const {
      assert(Sym && "section not present in the symbol table");
      return Sym->SymTabIdx;
    }

    ELFSection(const std::string &name, bool isLittleEndian, bool is64Bit)
      : BinaryObject(name, isLittleEndian, is64Bit), Type(0), Flags(0), Addr(0),
        Offset(0), Size(0), Link(0), Info(0), Align(0), EntSize(0), Sym(0) {}
  };

  /// ELFRelocation - This class contains all the information necessary to
  /// to generate any 32-bit or 64-bit ELF relocation entry.
  class ELFRelocation {
    uint64_t r_offset;    // offset in the section of the object this applies to
    uint32_t r_symidx;    // symbol table index of the symbol to use
    uint32_t r_type;      // machine specific relocation type
    int64_t  r_add;       // explicit relocation addend
    bool     r_rela;      // if true then the addend is part of the entry
                          // otherwise the addend is at the location specified
                          // by r_offset
  public:
    uint64_t getInfo(bool is64Bit) const {
      if (is64Bit)
        return ((uint64_t)r_symidx << 32) + ((uint64_t)r_type & 0xFFFFFFFFL);
      else
        return (r_symidx << 8)  + (r_type & 0xFFL);
    }

    uint64_t getOffset() const { return r_offset; }
    int64_t getAddend() const { return r_add; }

    ELFRelocation(uint64_t off, uint32_t sym, uint32_t type,
                  bool rela = true, int64_t addend = 0) :
      r_offset(off), r_symidx(sym), r_type(type),
      r_add(addend), r_rela(rela) {}
  };

} // end namespace llvm

#endif
