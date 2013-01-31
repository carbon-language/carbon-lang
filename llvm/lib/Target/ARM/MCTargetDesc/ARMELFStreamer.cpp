//===- lib/MC/ARMELFStreamer.cpp - ELF Object Output for ARM --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file assembles .s files and emits ARM ELF .o object files. Different
// from generic ELF streamer in emitting mapping symbols ($a, $t and $d) to
// delimit regions of data and code.
//
//===----------------------------------------------------------------------===//

#include "ARMUnwindOp.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCELF.h"
#include "llvm/MC/MCELFStreamer.h"
#include "llvm/MC/MCELFSymbolFlags.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCObjectStreamer.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {

/// Extend the generic ELFStreamer class so that it can emit mapping symbols at
/// the appropriate points in the object files. These symbols are defined in the
/// ARM ELF ABI: infocenter.arm.com/help/topic/com.arm.../IHI0044D_aaelf.pdf.
///
/// In brief: $a, $t or $d should be emitted at the start of each contiguous
/// region of ARM code, Thumb code or data in a section. In practice, this
/// emission does not rely on explicit assembler directives but on inherent
/// properties of the directives doing the emission (e.g. ".byte" is data, "add
/// r0, r0, r0" an instruction).
///
/// As a result this system is orthogonal to the DataRegion infrastructure used
/// by MachO. Beware!
class ARMELFStreamer : public MCELFStreamer {
public:
  ARMELFStreamer(MCContext &Context, MCAsmBackend &TAB, raw_ostream &OS,
                 MCCodeEmitter *Emitter, bool IsThumb)
      : MCELFStreamer(SK_ARMELFStreamer, Context, TAB, OS, Emitter),
        IsThumb(IsThumb), MappingSymbolCounter(0), LastEMS(EMS_None), ExTab(0),
        FnStart(0), Personality(0), CantUnwind(false) {}

  ~ARMELFStreamer() {}

  // ARM exception handling directives
  virtual void EmitFnStart();
  virtual void EmitFnEnd();
  virtual void EmitCantUnwind();
  virtual void EmitPersonality(const MCSymbol *Per);
  virtual void EmitHandlerData();
  virtual void EmitSetFP(unsigned NewFpReg,
                         unsigned NewSpReg,
                         int64_t Offset = 0);
  virtual void EmitPad(int64_t Offset);
  virtual void EmitRegSave(const SmallVectorImpl<unsigned> &RegList,
                           bool isVector);

  virtual void ChangeSection(const MCSection *Section) {
    // We have to keep track of the mapping symbol state of any sections we
    // use. Each one should start off as EMS_None, which is provided as the
    // default constructor by DenseMap::lookup.
    LastMappingSymbols[getPreviousSection()] = LastEMS;
    LastEMS = LastMappingSymbols.lookup(Section);

    MCELFStreamer::ChangeSection(Section);
  }

  /// This function is the one used to emit instruction data into the ELF
  /// streamer. We override it to add the appropriate mapping symbol if
  /// necessary.
  virtual void EmitInstruction(const MCInst& Inst) {
    if (IsThumb)
      EmitThumbMappingSymbol();
    else
      EmitARMMappingSymbol();

    MCELFStreamer::EmitInstruction(Inst);
  }

  /// This is one of the functions used to emit data into an ELF section, so the
  /// ARM streamer overrides it to add the appropriate mapping symbol ($d) if
  /// necessary.
  virtual void EmitBytes(StringRef Data, unsigned AddrSpace) {
    EmitDataMappingSymbol();
    MCELFStreamer::EmitBytes(Data, AddrSpace);
  }

  /// This is one of the functions used to emit data into an ELF section, so the
  /// ARM streamer overrides it to add the appropriate mapping symbol ($d) if
  /// necessary.
  virtual void EmitValueImpl(const MCExpr *Value, unsigned Size,
                             unsigned AddrSpace) {
    EmitDataMappingSymbol();
    MCELFStreamer::EmitValueImpl(Value, Size, AddrSpace);
  }

  virtual void EmitAssemblerFlag(MCAssemblerFlag Flag) {
    MCELFStreamer::EmitAssemblerFlag(Flag);

    switch (Flag) {
    case MCAF_SyntaxUnified:
      return; // no-op here.
    case MCAF_Code16:
      IsThumb = true;
      return; // Change to Thumb mode
    case MCAF_Code32:
      IsThumb = false;
      return; // Change to ARM mode
    case MCAF_Code64:
      return;
    case MCAF_SubsectionsViaSymbols:
      return;
    }
  }

  static bool classof(const MCStreamer *S) {
    return S->getKind() == SK_ARMELFStreamer;
  }

private:
  enum ElfMappingSymbol {
    EMS_None,
    EMS_ARM,
    EMS_Thumb,
    EMS_Data
  };

  void EmitDataMappingSymbol() {
    if (LastEMS == EMS_Data) return;
    EmitMappingSymbol("$d");
    LastEMS = EMS_Data;
  }

  void EmitThumbMappingSymbol() {
    if (LastEMS == EMS_Thumb) return;
    EmitMappingSymbol("$t");
    LastEMS = EMS_Thumb;
  }

  void EmitARMMappingSymbol() {
    if (LastEMS == EMS_ARM) return;
    EmitMappingSymbol("$a");
    LastEMS = EMS_ARM;
  }

  void EmitMappingSymbol(StringRef Name) {
    MCSymbol *Start = getContext().CreateTempSymbol();
    EmitLabel(Start);

    MCSymbol *Symbol =
      getContext().GetOrCreateSymbol(Name + "." +
                                     Twine(MappingSymbolCounter++));

    MCSymbolData &SD = getAssembler().getOrCreateSymbolData(*Symbol);
    MCELF::SetType(SD, ELF::STT_NOTYPE);
    MCELF::SetBinding(SD, ELF::STB_LOCAL);
    SD.setExternal(false);
    Symbol->setSection(*getCurrentSection());

    const MCExpr *Value = MCSymbolRefExpr::Create(Start, getContext());
    Symbol->setVariableValue(Value);
  }

  void EmitThumbFunc(MCSymbol *Func) {
    // FIXME: Anything needed here to flag the function as thumb?

    getAssembler().setIsThumbFunc(Func);

    MCSymbolData &SD = getAssembler().getOrCreateSymbolData(*Func);
    SD.setFlags(SD.getFlags() | ELF_Other_ThumbFunc);
  }

  // Helper functions for ARM exception handling directives
  void Reset();

  void EmitPersonalityFixup(StringRef Name);

  void SwitchToEHSection(const char *Prefix, unsigned Type, unsigned Flags,
                         SectionKind Kind, const MCSymbol &Fn);
  void SwitchToExTabSection(const MCSymbol &FnStart);
  void SwitchToExIdxSection(const MCSymbol &FnStart);

  bool IsThumb;
  int64_t MappingSymbolCounter;

  DenseMap<const MCSection *, ElfMappingSymbol> LastMappingSymbols;
  ElfMappingSymbol LastEMS;

  // ARM Exception Handling Frame Information
  MCSymbol *ExTab;
  MCSymbol *FnStart;
  const MCSymbol *Personality;
  bool CantUnwind;
};
}

inline void ARMELFStreamer::SwitchToEHSection(const char *Prefix,
                                              unsigned Type,
                                              unsigned Flags,
                                              SectionKind Kind,
                                              const MCSymbol &Fn) {
  const MCSectionELF &FnSection =
    static_cast<const MCSectionELF &>(Fn.getSection());

  // Create the name for new section
  StringRef FnSecName(FnSection.getSectionName());
  SmallString<128> EHSecName(Prefix);
  if (FnSecName != ".text") {
    EHSecName += FnSecName;
  }

  // Get .ARM.extab or .ARM.exidx section
  const MCSectionELF *EHSection = NULL;
  if (const MCSymbol *Group = FnSection.getGroup()) {
    EHSection = getContext().getELFSection(
      EHSecName, Type, Flags | ELF::SHF_GROUP, Kind,
      FnSection.getEntrySize(), Group->getName());
  } else {
    EHSection = getContext().getELFSection(EHSecName, Type, Flags, Kind);
  }
  assert(EHSection);

  // Switch to .ARM.extab or .ARM.exidx section
  SwitchSection(EHSection);
  EmitCodeAlignment(4, 0);
}

inline void ARMELFStreamer::SwitchToExTabSection(const MCSymbol &FnStart) {
  SwitchToEHSection(".ARM.extab",
                    ELF::SHT_PROGBITS,
                    ELF::SHF_ALLOC,
                    SectionKind::getDataRel(),
                    FnStart);
}

inline void ARMELFStreamer::SwitchToExIdxSection(const MCSymbol &FnStart) {
  SwitchToEHSection(".ARM.exidx",
                    ELF::SHT_ARM_EXIDX,
                    ELF::SHF_ALLOC | ELF::SHF_LINK_ORDER,
                    SectionKind::getDataRel(),
                    FnStart);
}

void ARMELFStreamer::Reset() {
  ExTab = NULL;
  FnStart = NULL;
  Personality = NULL;
  CantUnwind = false;
}

// Add the R_ARM_NONE fixup at the same position
void ARMELFStreamer::EmitPersonalityFixup(StringRef Name) {
  const MCSymbol *PersonalitySym = getContext().GetOrCreateSymbol(Name);

  const MCSymbolRefExpr *PersonalityRef =
    MCSymbolRefExpr::Create(PersonalitySym,
                            MCSymbolRefExpr::VK_ARM_NONE,
                            getContext());

  AddValueSymbols(PersonalityRef);
  MCDataFragment *DF = getOrCreateDataFragment();
  DF->getFixups().push_back(
    MCFixup::Create(DF->getContents().size(), PersonalityRef,
                    MCFixup::getKindForSize(4, false)));
}

void ARMELFStreamer::EmitFnStart() {
  assert(FnStart == 0);
  FnStart = getContext().CreateTempSymbol();
  EmitLabel(FnStart);
}

void ARMELFStreamer::EmitFnEnd() {
  assert(FnStart && ".fnstart must preceeds .fnend");

  // Emit unwind opcodes if there is no .handlerdata directive
  int PersonalityIndex = -1;
  if (!ExTab && !CantUnwind) {
    // For __aeabi_unwind_cpp_pr1, we have to emit opcodes in .ARM.extab.
    SwitchToExTabSection(*FnStart);

    // Create .ARM.extab label for offset in .ARM.exidx
    ExTab = getContext().CreateTempSymbol();
    EmitLabel(ExTab);

    PersonalityIndex = 1;

    uint32_t Entry = 0;
    uint32_t NumExtraEntryWords = 0;
    Entry |= NumExtraEntryWords << 24;
    Entry |= (EHT_COMPACT | PersonalityIndex) << 16;

    // TODO: This should be generated according to .save, .vsave, .setfp
    // directives.  Currently, we are simply generating FINISH opcode.
    Entry |= UNWIND_OPCODE_FINISH << 8;
    Entry |= UNWIND_OPCODE_FINISH;

    EmitIntValue(Entry, 4, 0);
  }

  // Emit the exception index table entry
  SwitchToExIdxSection(*FnStart);

  if (PersonalityIndex == 1)
    EmitPersonalityFixup("__aeabi_unwind_cpp_pr1");

  const MCSymbolRefExpr *FnStartRef =
    MCSymbolRefExpr::Create(FnStart,
                            MCSymbolRefExpr::VK_ARM_PREL31,
                            getContext());

  EmitValue(FnStartRef, 4, 0);

  if (CantUnwind) {
    EmitIntValue(EXIDX_CANTUNWIND, 4, 0);
  } else {
    const MCSymbolRefExpr *ExTabEntryRef =
      MCSymbolRefExpr::Create(ExTab,
                              MCSymbolRefExpr::VK_ARM_PREL31,
                              getContext());
    EmitValue(ExTabEntryRef, 4, 0);
  }

  // Clean exception handling frame information
  Reset();
}

void ARMELFStreamer::EmitCantUnwind() {
  CantUnwind = true;
}

void ARMELFStreamer::EmitHandlerData() {
  SwitchToExTabSection(*FnStart);

  // Create .ARM.extab label for offset in .ARM.exidx
  assert(!ExTab);
  ExTab = getContext().CreateTempSymbol();
  EmitLabel(ExTab);

  // Emit Personality
  assert(Personality && ".personality directive must preceed .handlerdata");

  const MCSymbolRefExpr *PersonalityRef =
    MCSymbolRefExpr::Create(Personality,
                            MCSymbolRefExpr::VK_ARM_PREL31,
                            getContext());

  EmitValue(PersonalityRef, 4, 0);

  // Emit unwind opcodes
  uint32_t Entry = 0;
  uint32_t NumExtraEntryWords = 0;

  // TODO: This should be generated according to .save, .vsave, .setfp
  // directives.  Currently, we are simply generating FINISH opcode.
  Entry |= NumExtraEntryWords << 24;
  Entry |= UNWIND_OPCODE_FINISH << 16;
  Entry |= UNWIND_OPCODE_FINISH << 8;
  Entry |= UNWIND_OPCODE_FINISH;

  EmitIntValue(Entry, 4, 0);
}

void ARMELFStreamer::EmitPersonality(const MCSymbol *Per) {
  Personality = Per;
}

void ARMELFStreamer::EmitSetFP(unsigned NewFpReg,
                               unsigned NewSpReg,
                               int64_t Offset) {
  // TODO: Not implemented
}

void ARMELFStreamer::EmitPad(int64_t Offset) {
  // TODO: Not implemented
}

void ARMELFStreamer::EmitRegSave(const SmallVectorImpl<unsigned> &RegList,
                                 bool IsVector) {
  // TODO: Not implemented
}

namespace llvm {
  MCELFStreamer* createARMELFStreamer(MCContext &Context, MCAsmBackend &TAB,
                                      raw_ostream &OS, MCCodeEmitter *Emitter,
                                      bool RelaxAll, bool NoExecStack,
                                      bool IsThumb) {
    ARMELFStreamer *S = new ARMELFStreamer(Context, TAB, OS, Emitter, IsThumb);
    if (RelaxAll)
      S->getAssembler().setRelaxAll(true);
    if (NoExecStack)
      S->getAssembler().setNoExecStack(true);
    return S;
  }

}


