//===- lib/MC/ELFObjectWriter.h - ELF File Writer -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements ELF object file writer information.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_ELFOBJECTWRITER_H
#define LLVM_MC_ELFOBJECTWRITER_H

#include "MCELF.h"
#include "llvm/ADT/OwningPtr.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCELFObjectWriter.h"
#include "llvm/MC/MCELFSymbolFlags.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCSymbol.h"

#include <vector>

namespace llvm {

class MCSection;
class MCDataFragment;
class MCSectionELF;

class ELFObjectWriter : public MCObjectWriter {
  protected:

    static bool isFixupKindPCRel(const MCAssembler &Asm, unsigned Kind);
    static bool RelocNeedsGOT(MCSymbolRefExpr::VariantKind Variant);
    static uint64_t SymbolValue(MCSymbolData &Data, const MCAsmLayout &Layout);
    static bool isInSymtab(const MCAssembler &Asm, const MCSymbolData &Data,
                           bool Used, bool Renamed);
    static bool isLocal(const MCSymbolData &Data, bool isSignature,
                        bool isUsedInReloc);
    static bool IsELFMetaDataSection(const MCSectionData &SD);
    static uint64_t DataSectionSize(const MCSectionData &SD);
    static uint64_t GetSectionFileSize(const MCAsmLayout &Layout,
                                       const MCSectionData &SD);
    static uint64_t GetSectionAddressSize(const MCAsmLayout &Layout,
                                          const MCSectionData &SD);

    void WriteDataSectionData(MCAssembler &Asm,
                              const MCAsmLayout &Layout,
                              const MCSectionELF &Section);

    /*static bool isFixupKindX86RIPRel(unsigned Kind) {
      return Kind == X86::reloc_riprel_4byte ||
        Kind == X86::reloc_riprel_4byte_movq_load;
    }*/

    /// ELFSymbolData - Helper struct for containing some precomputed
    /// information on symbols.
    struct ELFSymbolData {
      MCSymbolData *SymbolData;
      uint64_t StringIndex;
      uint32_t SectionIndex;

      // Support lexicographic sorting.
      bool operator<(const ELFSymbolData &RHS) const {
        if (MCELF::GetType(*SymbolData) == ELF::STT_FILE)
          return true;
        if (MCELF::GetType(*RHS.SymbolData) == ELF::STT_FILE)
          return false;
        return SymbolData->getSymbol().getName() <
               RHS.SymbolData->getSymbol().getName();
      }
    };

    /// @name Relocation Data
    /// @{

    struct ELFRelocationEntry {
      // Make these big enough for both 32-bit and 64-bit
      uint64_t r_offset;
      int Index;
      unsigned Type;
      const MCSymbol *Symbol;
      uint64_t r_addend;

      ELFRelocationEntry()
        : r_offset(0), Index(0), Type(0), Symbol(0), r_addend(0) {}

      ELFRelocationEntry(uint64_t RelocOffset, int Idx,
                         unsigned RelType, const MCSymbol *Sym,
                         uint64_t Addend)
        : r_offset(RelocOffset), Index(Idx), Type(RelType),
          Symbol(Sym), r_addend(Addend) {}

      // Support lexicographic sorting.
      bool operator<(const ELFRelocationEntry &RE) const {
        return RE.r_offset < r_offset;
      }
    };

    /// The target specific ELF writer instance.
    llvm::OwningPtr<MCELFObjectTargetWriter> TargetObjectWriter;

    SmallPtrSet<const MCSymbol *, 16> UsedInReloc;
    SmallPtrSet<const MCSymbol *, 16> WeakrefUsedInReloc;
    DenseMap<const MCSymbol *, const MCSymbol *> Renames;

    llvm::DenseMap<const MCSectionData*,
                   std::vector<ELFRelocationEntry> > Relocations;
    DenseMap<const MCSection*, uint64_t> SectionStringTableIndex;

    /// @}
    /// @name Symbol Table Data
    /// @{

    SmallString<256> StringTable;
    std::vector<ELFSymbolData> LocalSymbolData;
    std::vector<ELFSymbolData> ExternalSymbolData;
    std::vector<ELFSymbolData> UndefinedSymbolData;

    /// @}

    bool NeedsGOT;

    bool NeedsSymtabShndx;

    // This holds the symbol table index of the last local symbol.
    unsigned LastLocalSymbolIndex;
    // This holds the .strtab section index.
    unsigned StringTableIndex;
    // This holds the .symtab section index.
    unsigned SymbolTableIndex;

    unsigned ShstrtabIndex;


    virtual const MCSymbol *SymbolToReloc(const MCAssembler &Asm,
                                          const MCValue &Target,
                                          const MCFragment &F,
                                          const MCFixup &Fixup,
                                          bool IsPCRel) const;

    // For arch-specific emission of explicit reloc symbol
    virtual const MCSymbol *ExplicitRelSym(const MCAssembler &Asm,
                                           const MCValue &Target,
                                           const MCFragment &F,
                                           const MCFixup &Fixup,
                                           bool IsPCRel) const {
      return NULL;
    }

    bool is64Bit() const { return TargetObjectWriter->is64Bit(); }
    bool hasRelocationAddend() const {
      return TargetObjectWriter->hasRelocationAddend();
    }

  public:
    ELFObjectWriter(MCELFObjectTargetWriter *MOTW,
                    raw_ostream &_OS, bool IsLittleEndian)
      : MCObjectWriter(_OS, IsLittleEndian),
        TargetObjectWriter(MOTW),
        NeedsGOT(false), NeedsSymtabShndx(false){
    }

    virtual ~ELFObjectWriter();

    void WriteWord(uint64_t W) {
      if (is64Bit())
        Write64(W);
      else
        Write32(W);
    }

    void StringLE16(char *buf, uint16_t Value) {
      buf[0] = char(Value >> 0);
      buf[1] = char(Value >> 8);
    }

    void StringLE32(char *buf, uint32_t Value) {
      StringLE16(buf, uint16_t(Value >> 0));
      StringLE16(buf + 2, uint16_t(Value >> 16));
    }

    void StringLE64(char *buf, uint64_t Value) {
      StringLE32(buf, uint32_t(Value >> 0));
      StringLE32(buf + 4, uint32_t(Value >> 32));
    }

    void StringBE16(char *buf ,uint16_t Value) {
      buf[0] = char(Value >> 8);
      buf[1] = char(Value >> 0);
    }

    void StringBE32(char *buf, uint32_t Value) {
      StringBE16(buf, uint16_t(Value >> 16));
      StringBE16(buf + 2, uint16_t(Value >> 0));
    }

    void StringBE64(char *buf, uint64_t Value) {
      StringBE32(buf, uint32_t(Value >> 32));
      StringBE32(buf + 4, uint32_t(Value >> 0));
    }

    void String8(MCDataFragment &F, uint8_t Value) {
      char buf[1];
      buf[0] = Value;
      F.getContents() += StringRef(buf, 1);
    }

    void String16(MCDataFragment &F, uint16_t Value) {
      char buf[2];
      if (isLittleEndian())
        StringLE16(buf, Value);
      else
        StringBE16(buf, Value);
      F.getContents() += StringRef(buf, 2);
    }

    void String32(MCDataFragment &F, uint32_t Value) {
      char buf[4];
      if (isLittleEndian())
        StringLE32(buf, Value);
      else
        StringBE32(buf, Value);
      F.getContents() += StringRef(buf, 4);
    }

    void String64(MCDataFragment &F, uint64_t Value) {
      char buf[8];
      if (isLittleEndian())
        StringLE64(buf, Value);
      else
        StringBE64(buf, Value);
      F.getContents() += StringRef(buf, 8);
    }

    virtual void WriteHeader(uint64_t SectionDataSize, unsigned NumberOfSections);

    /// Default e_flags = 0
    virtual void WriteEFlags() { Write32(0); }

    virtual void WriteSymbolEntry(MCDataFragment *SymtabF, MCDataFragment *ShndxF,
                          uint64_t name, uint8_t info,
                          uint64_t value, uint64_t size,
                          uint8_t other, uint32_t shndx,
                          bool Reserved);

    virtual void WriteSymbol(MCDataFragment *SymtabF,  MCDataFragment *ShndxF,
                     ELFSymbolData &MSD,
                     const MCAsmLayout &Layout);

    typedef DenseMap<const MCSectionELF*, uint32_t> SectionIndexMapTy;
    virtual void WriteSymbolTable(MCDataFragment *SymtabF, MCDataFragment *ShndxF,
                          const MCAssembler &Asm,
                          const MCAsmLayout &Layout,
                          const SectionIndexMapTy &SectionIndexMap);

    virtual void RecordRelocation(const MCAssembler &Asm, const MCAsmLayout &Layout,
                                  const MCFragment *Fragment, const MCFixup &Fixup,
                                  MCValue Target, uint64_t &FixedValue);

    virtual uint64_t getSymbolIndexInSymbolTable(const MCAssembler &Asm,
                                         const MCSymbol *S);

    // Map from a group section to the signature symbol
    typedef DenseMap<const MCSectionELF*, const MCSymbol*> GroupMapTy;
    // Map from a signature symbol to the group section
    typedef DenseMap<const MCSymbol*, const MCSectionELF*> RevGroupMapTy;
    // Map from a section to the section with the relocations
    typedef DenseMap<const MCSectionELF*, const MCSectionELF*> RelMapTy;
    // Map from a section to its offset
    typedef DenseMap<const MCSectionELF*, uint64_t> SectionOffsetMapTy;

    /// ComputeSymbolTable - Compute the symbol table data
    ///
    /// \param StringTable [out] - The string table data.
    /// \param StringIndexMap [out] - Map from symbol names to offsets in the
    /// string table.
    virtual void ComputeSymbolTable(MCAssembler &Asm,
                            const SectionIndexMapTy &SectionIndexMap,
                                    RevGroupMapTy RevGroupMap,
                                    unsigned NumRegularSections);

    virtual void ComputeIndexMap(MCAssembler &Asm,
                                 SectionIndexMapTy &SectionIndexMap,
                                 const RelMapTy &RelMap);

    void CreateRelocationSections(MCAssembler &Asm, MCAsmLayout &Layout,
                                  RelMapTy &RelMap);

    void WriteRelocations(MCAssembler &Asm, MCAsmLayout &Layout,
                          const RelMapTy &RelMap);

    virtual void CreateMetadataSections(MCAssembler &Asm, MCAsmLayout &Layout,
                                        SectionIndexMapTy &SectionIndexMap,
                                        const RelMapTy &RelMap);

    // Create the sections that show up in the symbol table. Currently
    // those are the .note.GNU-stack section and the group sections.
    virtual void CreateIndexedSections(MCAssembler &Asm, MCAsmLayout &Layout,
                                       GroupMapTy &GroupMap,
                                       RevGroupMapTy &RevGroupMap,
                                       SectionIndexMapTy &SectionIndexMap,
                                       const RelMapTy &RelMap);

    virtual void ExecutePostLayoutBinding(MCAssembler &Asm,
                                          const MCAsmLayout &Layout);

    void WriteSectionHeader(MCAssembler &Asm, const GroupMapTy &GroupMap,
                            const MCAsmLayout &Layout,
                            const SectionIndexMapTy &SectionIndexMap,
                            const SectionOffsetMapTy &SectionOffsetMap);

    void ComputeSectionOrder(MCAssembler &Asm,
                             std::vector<const MCSectionELF*> &Sections);

    virtual void WriteSecHdrEntry(uint32_t Name, uint32_t Type, uint64_t Flags,
                          uint64_t Address, uint64_t Offset,
                          uint64_t Size, uint32_t Link, uint32_t Info,
                          uint64_t Alignment, uint64_t EntrySize);

    virtual void WriteRelocationsFragment(const MCAssembler &Asm,
                                          MCDataFragment *F,
                                          const MCSectionData *SD);

    virtual bool
    IsSymbolRefDifferenceFullyResolvedImpl(const MCAssembler &Asm,
                                           const MCSymbolData &DataA,
                                           const MCFragment &FB,
                                           bool InSet,
                                           bool IsPCRel) const;

    virtual void WriteObject(MCAssembler &Asm, const MCAsmLayout &Layout);
    virtual void WriteSection(MCAssembler &Asm,
                      const SectionIndexMapTy &SectionIndexMap,
                      uint32_t GroupSymbolIndex,
                      uint64_t Offset, uint64_t Size, uint64_t Alignment,
                      const MCSectionELF &Section);

  protected:
    virtual unsigned GetRelocType(const MCValue &Target, const MCFixup &Fixup,
                                  bool IsPCRel, bool IsRelocWithSymbol,
                                  int64_t Addend) = 0;
  };

  //===- X86ELFObjectWriter -------------------------------------------===//

  class X86ELFObjectWriter : public ELFObjectWriter {
  public:
    X86ELFObjectWriter(MCELFObjectTargetWriter *MOTW,
                       raw_ostream &_OS,
                       bool IsLittleEndian);

    virtual ~X86ELFObjectWriter();
  protected:
    virtual unsigned GetRelocType(const MCValue &Target, const MCFixup &Fixup,
                                  bool IsPCRel, bool IsRelocWithSymbol,
                                  int64_t Addend);
  };


  //===- ARMELFObjectWriter -------------------------------------------===//

  class ARMELFObjectWriter : public ELFObjectWriter {
  public:
    // FIXME: MCAssembler can't yet return the Subtarget,
    enum { DefaultEABIVersion = 0x05000000U };

    ARMELFObjectWriter(MCELFObjectTargetWriter *MOTW,
                       raw_ostream &_OS,
                       bool IsLittleEndian);

    virtual ~ARMELFObjectWriter();

    virtual void WriteEFlags();
  protected:
    virtual const MCSymbol *ExplicitRelSym(const MCAssembler &Asm,
                                           const MCValue &Target,
                                           const MCFragment &F,
                                           const MCFixup &Fixup,
                                           bool IsPCRel) const;

    virtual unsigned GetRelocType(const MCValue &Target, const MCFixup &Fixup,
                                  bool IsPCRel, bool IsRelocWithSymbol,
                                  int64_t Addend);
  private:
    unsigned GetRelocTypeInner(const MCValue &Target,
                               const MCFixup &Fixup, bool IsPCRel) const;
    
  };

  //===- PPCELFObjectWriter -------------------------------------------===//

  class PPCELFObjectWriter : public ELFObjectWriter {
  public:
    PPCELFObjectWriter(MCELFObjectTargetWriter *MOTW,
                          raw_ostream &_OS,
                          bool IsLittleEndian);

    virtual ~PPCELFObjectWriter();
  protected:
    virtual unsigned GetRelocType(const MCValue &Target, const MCFixup &Fixup,
                                  bool IsPCRel, bool IsRelocWithSymbol,
                                  int64_t Addend);
  };

  //===- MBlazeELFObjectWriter -------------------------------------------===//

  class MBlazeELFObjectWriter : public ELFObjectWriter {
  public:
    MBlazeELFObjectWriter(MCELFObjectTargetWriter *MOTW,
                          raw_ostream &_OS,
                          bool IsLittleEndian);

    virtual ~MBlazeELFObjectWriter();
  protected:
    virtual unsigned GetRelocType(const MCValue &Target, const MCFixup &Fixup,
                                  bool IsPCRel, bool IsRelocWithSymbol,
                                  int64_t Addend);
  };
}

#endif
