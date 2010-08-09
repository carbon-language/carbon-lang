//===- lib/MC/MCMachOStreamer.cpp - Mach-O Object Output ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCStreamer.h"

#include "llvm/MC/MCAssembler.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCObjectStreamer.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCMachOSymbolFlags.h"
#include "llvm/MC/MCSectionMachO.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetAsmBackend.h"

using namespace llvm;

namespace {

class MCMachOStreamer : public MCObjectStreamer {
private:
  void EmitInstToFragment(const MCInst &Inst);
  void EmitInstToData(const MCInst &Inst);
  // FIXME: These will likely moved to a better place.
  const MCExpr * MakeStartMinusEndExpr(MCSymbol *Start, MCSymbol *End,
                                                        int IntVal);
  void EmitDwarfFileTable(void);

public:
  MCMachOStreamer(MCContext &Context, TargetAsmBackend &TAB,
                  raw_ostream &OS, MCCodeEmitter *Emitter)
    : MCObjectStreamer(Context, TAB, OS, Emitter) {}

  /// @name MCStreamer Interface
  /// @{

  virtual void EmitLabel(MCSymbol *Symbol);
  virtual void EmitAssemblerFlag(MCAssemblerFlag Flag);
  virtual void EmitAssignment(MCSymbol *Symbol, const MCExpr *Value);
  virtual void EmitSymbolAttribute(MCSymbol *Symbol, MCSymbolAttr Attribute);
  virtual void EmitSymbolDesc(MCSymbol *Symbol, unsigned DescValue);
  virtual void EmitCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                                unsigned ByteAlignment);
  virtual void BeginCOFFSymbolDef(const MCSymbol *Symbol) {
    assert(0 && "macho doesn't support this directive");
  }
  virtual void EmitCOFFSymbolStorageClass(int StorageClass) {
    assert(0 && "macho doesn't support this directive");
  }
  virtual void EmitCOFFSymbolType(int Type) {
    assert(0 && "macho doesn't support this directive");
  }
  virtual void EndCOFFSymbolDef() {
    assert(0 && "macho doesn't support this directive");
  }
  virtual void EmitELFSize(MCSymbol *Symbol, const MCExpr *Value) {
    assert(0 && "macho doesn't support this directive");
  }
  virtual void EmitLocalCommonSymbol(MCSymbol *Symbol, uint64_t Size) {
    assert(0 && "macho doesn't support this directive");
  }
  virtual void EmitZerofill(const MCSection *Section, MCSymbol *Symbol = 0,
                            unsigned Size = 0, unsigned ByteAlignment = 0);
  virtual void EmitTBSSSymbol(const MCSection *Section, MCSymbol *Symbol,
                              uint64_t Size, unsigned ByteAlignment = 0);
  virtual void EmitBytes(StringRef Data, unsigned AddrSpace);
  virtual void EmitValue(const MCExpr *Value, unsigned Size,unsigned AddrSpace);
  virtual void EmitGPRel32Value(const MCExpr *Value) {
    assert(0 && "macho doesn't support this directive");
  }
  virtual void EmitValueToAlignment(unsigned ByteAlignment, int64_t Value = 0,
                                    unsigned ValueSize = 1,
                                    unsigned MaxBytesToEmit = 0);
  virtual void EmitCodeAlignment(unsigned ByteAlignment,
                                 unsigned MaxBytesToEmit = 0);
  virtual void EmitValueToOffset(const MCExpr *Offset,
                                 unsigned char Value = 0);

  virtual void EmitFileDirective(StringRef Filename) {
    // FIXME: Just ignore the .file; it isn't important enough to fail the
    // entire assembly.

    //report_fatal_error("unsupported directive: '.file'");
  }
  virtual void EmitDwarfFileDirective(unsigned FileNo, StringRef Filename) {
    // FIXME: Just ignore the .file; it isn't important enough to fail the
    // entire assembly.

    //report_fatal_error("unsupported directive: '.file'");
  }

  virtual void EmitInstruction(const MCInst &Inst);

  virtual void Finish();

  /// @}
};

} // end anonymous namespace.

void MCMachOStreamer::EmitLabel(MCSymbol *Symbol) {
  // TODO: This is almost exactly the same as WinCOFFStreamer. Consider merging
  // into MCObjectStreamer.
  assert(Symbol->isUndefined() && "Cannot define a symbol twice!");
  assert(!Symbol->isVariable() && "Cannot emit a variable symbol!");
  assert(CurSection && "Cannot emit before setting section!");

  Symbol->setSection(*CurSection);

  MCSymbolData &SD = getAssembler().getOrCreateSymbolData(*Symbol);

  // We have to create a new fragment if this is an atom defining symbol,
  // fragments cannot span atoms.
  if (getAssembler().isSymbolLinkerVisible(SD.getSymbol()))
    new MCDataFragment(getCurrentSectionData());

  // FIXME: This is wasteful, we don't necessarily need to create a data
  // fragment. Instead, we should mark the symbol as pointing into the data
  // fragment if it exists, otherwise we should just queue the label and set its
  // fragment pointer when we emit the next fragment.
  MCDataFragment *F = getOrCreateDataFragment();
  assert(!SD.getFragment() && "Unexpected fragment on symbol data!");
  SD.setFragment(F);
  SD.setOffset(F->getContents().size());

  // This causes the reference type flag to be cleared. Darwin 'as' was "trying"
  // to clear the weak reference and weak definition bits too, but the
  // implementation was buggy. For now we just try to match 'as', for
  // diffability.
  //
  // FIXME: Cleanup this code, these bits should be emitted based on semantic
  // properties, not on the order of definition, etc.
  SD.setFlags(SD.getFlags() & ~SF_ReferenceTypeMask);
}

void MCMachOStreamer::EmitAssemblerFlag(MCAssemblerFlag Flag) {
  switch (Flag) {
  case MCAF_SubsectionsViaSymbols:
    getAssembler().setSubsectionsViaSymbols(true);
    return;
  }

  assert(0 && "invalid assembler flag!");
}

void MCMachOStreamer::EmitAssignment(MCSymbol *Symbol, const MCExpr *Value) {
  // TODO: This is exactly the same as WinCOFFStreamer. Consider merging into
  // MCObjectStreamer.
  // FIXME: Lift context changes into super class.
  getAssembler().getOrCreateSymbolData(*Symbol);
  Symbol->setVariableValue(AddValueSymbols(Value));
}

void MCMachOStreamer::EmitSymbolAttribute(MCSymbol *Symbol,
                                          MCSymbolAttr Attribute) {
  // Indirect symbols are handled differently, to match how 'as' handles
  // them. This makes writing matching .o files easier.
  if (Attribute == MCSA_IndirectSymbol) {
    // Note that we intentionally cannot use the symbol data here; this is
    // important for matching the string table that 'as' generates.
    IndirectSymbolData ISD;
    ISD.Symbol = Symbol;
    ISD.SectionData = getCurrentSectionData();
    getAssembler().getIndirectSymbols().push_back(ISD);
    return;
  }

  // Adding a symbol attribute always introduces the symbol, note that an
  // important side effect of calling getOrCreateSymbolData here is to register
  // the symbol with the assembler.
  MCSymbolData &SD = getAssembler().getOrCreateSymbolData(*Symbol);

  // The implementation of symbol attributes is designed to match 'as', but it
  // leaves much to desired. It doesn't really make sense to arbitrarily add and
  // remove flags, but 'as' allows this (in particular, see .desc).
  //
  // In the future it might be worth trying to make these operations more well
  // defined.
  switch (Attribute) {
  case MCSA_Invalid:
  case MCSA_ELF_TypeFunction:
  case MCSA_ELF_TypeIndFunction:
  case MCSA_ELF_TypeObject:
  case MCSA_ELF_TypeTLS:
  case MCSA_ELF_TypeCommon:
  case MCSA_ELF_TypeNoType:
  case MCSA_IndirectSymbol:
  case MCSA_Hidden:
  case MCSA_Internal:
  case MCSA_Protected:
  case MCSA_Weak:
  case MCSA_Local:
    assert(0 && "Invalid symbol attribute for Mach-O!");
    break;

  case MCSA_Global:
    SD.setExternal(true);
    // This effectively clears the undefined lazy bit, in Darwin 'as', although
    // it isn't very consistent because it implements this as part of symbol
    // lookup.
    //
    // FIXME: Cleanup this code, these bits should be emitted based on semantic
    // properties, not on the order of definition, etc.
    SD.setFlags(SD.getFlags() & ~SF_ReferenceTypeUndefinedLazy);
    break;

  case MCSA_LazyReference:
    // FIXME: This requires -dynamic.
    SD.setFlags(SD.getFlags() | SF_NoDeadStrip);
    if (Symbol->isUndefined())
      SD.setFlags(SD.getFlags() | SF_ReferenceTypeUndefinedLazy);
    break;

    // Since .reference sets the no dead strip bit, it is equivalent to
    // .no_dead_strip in practice.
  case MCSA_Reference:
  case MCSA_NoDeadStrip:
    SD.setFlags(SD.getFlags() | SF_NoDeadStrip);
    break;

  case MCSA_PrivateExtern:
    SD.setExternal(true);
    SD.setPrivateExtern(true);
    break;

  case MCSA_WeakReference:
    // FIXME: This requires -dynamic.
    if (Symbol->isUndefined())
      SD.setFlags(SD.getFlags() | SF_WeakReference);
    break;

  case MCSA_WeakDefinition:
    // FIXME: 'as' enforces that this is defined and global. The manual claims
    // it has to be in a coalesced section, but this isn't enforced.
    SD.setFlags(SD.getFlags() | SF_WeakDefinition);
    break;

  case MCSA_WeakDefAutoPrivate:
    SD.setFlags(SD.getFlags() | SF_WeakDefinition | SF_WeakReference);
    break;
  }
}

void MCMachOStreamer::EmitSymbolDesc(MCSymbol *Symbol, unsigned DescValue) {
  // Encode the 'desc' value into the lowest implementation defined bits.
  assert(DescValue == (DescValue & SF_DescFlagsMask) &&
         "Invalid .desc value!");
  getAssembler().getOrCreateSymbolData(*Symbol).setFlags(
    DescValue & SF_DescFlagsMask);
}

void MCMachOStreamer::EmitCommonSymbol(MCSymbol *Symbol, uint64_t Size,
                                       unsigned ByteAlignment) {
  // FIXME: Darwin 'as' does appear to allow redef of a .comm by itself.
  assert(Symbol->isUndefined() && "Cannot define a symbol twice!");

  MCSymbolData &SD = getAssembler().getOrCreateSymbolData(*Symbol);
  SD.setExternal(true);
  SD.setCommon(Size, ByteAlignment);
}

void MCMachOStreamer::EmitZerofill(const MCSection *Section, MCSymbol *Symbol,
                                   unsigned Size, unsigned ByteAlignment) {
  MCSectionData &SectData = getAssembler().getOrCreateSectionData(*Section);

  // The symbol may not be present, which only creates the section.
  if (!Symbol)
    return;

  // FIXME: Assert that this section has the zerofill type.

  assert(Symbol->isUndefined() && "Cannot define a symbol twice!");

  MCSymbolData &SD = getAssembler().getOrCreateSymbolData(*Symbol);

  // Emit an align fragment if necessary.
  if (ByteAlignment != 1)
    new MCAlignFragment(ByteAlignment, 0, 0, ByteAlignment, &SectData);

  MCFragment *F = new MCFillFragment(0, 0, Size, &SectData);
  SD.setFragment(F);

  Symbol->setSection(*Section);

  // Update the maximum alignment on the zero fill section if necessary.
  if (ByteAlignment > SectData.getAlignment())
    SectData.setAlignment(ByteAlignment);
}

// This should always be called with the thread local bss section.  Like the
// .zerofill directive this doesn't actually switch sections on us.
void MCMachOStreamer::EmitTBSSSymbol(const MCSection *Section, MCSymbol *Symbol,
                                     uint64_t Size, unsigned ByteAlignment) {
  EmitZerofill(Section, Symbol, Size, ByteAlignment);
  return;
}

void MCMachOStreamer::EmitBytes(StringRef Data, unsigned AddrSpace) {
  // TODO: This is exactly the same as WinCOFFStreamer. Consider merging into
  // MCObjectStreamer.
  getOrCreateDataFragment()->getContents().append(Data.begin(), Data.end());
}

void MCMachOStreamer::EmitValue(const MCExpr *Value, unsigned Size,
                                unsigned AddrSpace) {
  // TODO: This is exactly the same as WinCOFFStreamer. Consider merging into
  // MCObjectStreamer.
  MCDataFragment *DF = getOrCreateDataFragment();

  // Avoid fixups when possible.
  int64_t AbsValue;
  if (AddValueSymbols(Value)->EvaluateAsAbsolute(AbsValue)) {
    // FIXME: Endianness assumption.
    for (unsigned i = 0; i != Size; ++i)
      DF->getContents().push_back(uint8_t(AbsValue >> (i * 8)));
  } else {
    DF->addFixup(MCFixup::Create(DF->getContents().size(),
                                 AddValueSymbols(Value),
                                 MCFixup::getKindForSize(Size)));
    DF->getContents().resize(DF->getContents().size() + Size, 0);
  }
}

void MCMachOStreamer::EmitValueToAlignment(unsigned ByteAlignment,
                                           int64_t Value, unsigned ValueSize,
                                           unsigned MaxBytesToEmit) {
  // TODO: This is exactly the same as WinCOFFStreamer. Consider merging into
  // MCObjectStreamer.
  if (MaxBytesToEmit == 0)
    MaxBytesToEmit = ByteAlignment;
  new MCAlignFragment(ByteAlignment, Value, ValueSize, MaxBytesToEmit,
                      getCurrentSectionData());

  // Update the maximum alignment on the current section if necessary.
  if (ByteAlignment > getCurrentSectionData()->getAlignment())
    getCurrentSectionData()->setAlignment(ByteAlignment);
}

void MCMachOStreamer::EmitCodeAlignment(unsigned ByteAlignment,
                                        unsigned MaxBytesToEmit) {
  // TODO: This is exactly the same as WinCOFFStreamer. Consider merging into
  // MCObjectStreamer.
  if (MaxBytesToEmit == 0)
    MaxBytesToEmit = ByteAlignment;
  MCAlignFragment *F = new MCAlignFragment(ByteAlignment, 0, 1, MaxBytesToEmit,
                                           getCurrentSectionData());
  F->setEmitNops(true);

  // Update the maximum alignment on the current section if necessary.
  if (ByteAlignment > getCurrentSectionData()->getAlignment())
    getCurrentSectionData()->setAlignment(ByteAlignment);
}

void MCMachOStreamer::EmitValueToOffset(const MCExpr *Offset,
                                        unsigned char Value) {
  new MCOrgFragment(*Offset, Value, getCurrentSectionData());
}

void MCMachOStreamer::EmitInstToFragment(const MCInst &Inst) {
  MCInstFragment *IF = new MCInstFragment(Inst, getCurrentSectionData());

  // Add the fixups and data.
  //
  // FIXME: Revisit this design decision when relaxation is done, we may be
  // able to get away with not storing any extra data in the MCInst.
  SmallVector<MCFixup, 4> Fixups;
  SmallString<256> Code;
  raw_svector_ostream VecOS(Code);
  getAssembler().getEmitter().EncodeInstruction(Inst, VecOS, Fixups);
  VecOS.flush();

  IF->getCode() = Code;
  IF->getFixups() = Fixups;
}

void MCMachOStreamer::EmitInstToData(const MCInst &Inst) {
  MCDataFragment *DF = getOrCreateDataFragment();

  SmallVector<MCFixup, 4> Fixups;
  SmallString<256> Code;
  raw_svector_ostream VecOS(Code);
  getAssembler().getEmitter().EncodeInstruction(Inst, VecOS, Fixups);
  VecOS.flush();

  // Add the fixups and data.
  for (unsigned i = 0, e = Fixups.size(); i != e; ++i) {
    Fixups[i].setOffset(Fixups[i].getOffset() + DF->getContents().size());
    DF->addFixup(Fixups[i]);
  }
  DF->getContents().append(Code.begin(), Code.end());
}

void MCMachOStreamer::EmitInstruction(const MCInst &Inst) {
  // Scan for values.
  for (unsigned i = Inst.getNumOperands(); i--; )
    if (Inst.getOperand(i).isExpr())
      AddValueSymbols(Inst.getOperand(i).getExpr());

  getCurrentSectionData()->setHasInstructions(true);

  // If this instruction doesn't need relaxation, just emit it as data.
  if (!getAssembler().getBackend().MayNeedRelaxation(Inst)) {
    EmitInstToData(Inst);
    return;
  }

  // Otherwise, if we are relaxing everything, relax the instruction as much as
  // possible and emit it as data.
  if (getAssembler().getRelaxAll()) {
    MCInst Relaxed;
    getAssembler().getBackend().RelaxInstruction(Inst, Relaxed);
    while (getAssembler().getBackend().MayNeedRelaxation(Relaxed))
      getAssembler().getBackend().RelaxInstruction(Relaxed, Relaxed);
    EmitInstToData(Relaxed);
    return;
  }

  // Otherwise emit to a separate fragment.
  EmitInstToFragment(Inst);
}

//
// This helper routine returns an expression of End - Start + IntVal for use
// by EmitDwarfFileTable() below.
// 
const MCExpr * MCMachOStreamer::MakeStartMinusEndExpr(MCSymbol *Start,
                                                      MCSymbol *End,
                                                      int IntVal) {
  MCSymbolRefExpr::VariantKind Variant = MCSymbolRefExpr::VK_None;
  const MCExpr *Res =
    MCSymbolRefExpr::Create(End, Variant, getContext());
  const MCExpr *RHS =
    MCSymbolRefExpr::Create(Start, Variant, getContext());
  const MCExpr *Res1 =
    MCBinaryExpr::Create(MCBinaryExpr::Sub, Res, RHS,getContext());
  const MCExpr *Res2 =
    MCConstantExpr::Create(IntVal, getContext());
  const MCExpr *Res3 =
    MCBinaryExpr::Create(MCBinaryExpr::Sub, Res1, Res2, getContext());
  return Res3;
}

//
// This emits the Dwarf file (and eventually the line) table.
//
void MCMachOStreamer::EmitDwarfFileTable(void) {
  // For now make sure we don't put out the Dwarf file table if no .file
  // directives were seen.
  const std::vector<MCDwarfFile *> &MCDwarfFiles =
    getContext().getMCDwarfFiles();
  if (MCDwarfFiles.size() == 0)
    return;

  // This is the Mach-O section, for ELF it is the .debug_line section.
  SwitchSection(getContext().getMachOSection("__DWARF", "__debug_line",
                                         MCSectionMachO::S_ATTR_DEBUG,
                                         0, SectionKind::getDataRelLocal()));

  // Create a symbol at the beginning of this section.
  MCSymbol *LineStartSym = getContext().CreateTempSymbol();
  // Set the value of the symbol, as we are at the start of the section.
  EmitLabel(LineStartSym);

  // Create a symbol for the end of the section (to be set when we get there).
  MCSymbol *LineEndSym = getContext().CreateTempSymbol();

  // The first 4 bytes is the total length of the information for this
  // compilation unit (not including these 4 bytes for the length).
  EmitValue(MakeStartMinusEndExpr(LineStartSym, LineEndSym, 4), 4, 0);

  // Next 2 bytes is the Version, which is Dwarf 2.
  EmitIntValue(2, 2);

  // Create a symbol for the end of the prologue (to be set when we get there).
  MCSymbol *ProEndSym = getContext().CreateTempSymbol(); // Lprologue_end

  // Length of the prologue, is the next 4 bytes.  Which is the start of the
  // section to the end of the prologue.  Not including the 4 bytes for the
  // total length, the 2 bytes for the version, and these 4 bytes for the
  // length of the prologue.
  EmitValue(MakeStartMinusEndExpr(LineStartSym, ProEndSym, (4 + 2 + 4)), 4, 0);

  // Parameters of the state machine, are next.
  //  Define the architecture-dependent minimum instruction length (in
  //  bytes).  This value should be rather too small than too big.  */
  //  DWARF2_LINE_MIN_INSN_LENGTH
  EmitIntValue(1, 1);
  //  Flag that indicates the initial value of the is_stmt_start flag.
  //  DWARF2_LINE_DEFAULT_IS_STMT
  EmitIntValue(1, 1);
  //  Minimum line offset in a special line info. opcode.  This value
  //  was chosen to give a reasonable range of values.  */
  //  DWARF2_LINE_BASE
  EmitIntValue(-5, 1);
  //  Range of line offsets in a special line info. opcode.
  //  DWARF2_LINE_RANGE
  EmitIntValue(14, 1);
  //  First special line opcode - leave room for the standard opcodes.
  //  DWARF2_LINE_OPCODE_BASE
  EmitIntValue(13, 1);

  // Standard opcode lengths
  EmitIntValue(0, 1); // length of DW_LNS_copy
  EmitIntValue(1, 1); // length of DW_LNS_advance_pc
  EmitIntValue(1, 1); // length of DW_LNS_advance_line
  EmitIntValue(1, 1); // length of DW_LNS_set_file
  EmitIntValue(1, 1); // length of DW_LNS_set_column
  EmitIntValue(0, 1); // length of DW_LNS_negate_stmt
  EmitIntValue(0, 1); // length of DW_LNS_set_basic_block
  EmitIntValue(0, 1); // length of DW_LNS_const_add_pc
  EmitIntValue(1, 1); // length of DW_LNS_fixed_advance_pc
  EmitIntValue(0, 1); // length of DW_LNS_set_prologue_end
  EmitIntValue(0, 1); // length of DW_LNS_set_epilogue_begin
  EmitIntValue(1, 1); // DW_LNS_set_isa

  // Put out the directory and file tables.

  // First the directory table.
  const std::vector<StringRef> &MCDwarfDirs =
    getContext().getMCDwarfDirs();
  for (unsigned i = 0; i < MCDwarfDirs.size(); i++) {
    EmitBytes(MCDwarfDirs[i], 0); // the DirectoryName
    EmitBytes(StringRef("\0", 1), 0); // the null termination of the string
  }
  EmitIntValue(0, 1); // Terminate the directory list

  // Second the file table.
  for (unsigned i = 1; i < MCDwarfFiles.size(); i++) {
    EmitBytes(MCDwarfFiles[i]->getName(), 0); // FileName
    EmitBytes(StringRef("\0", 1), 0); // the null termination of the string
    // FIXME the Directory number should be a .uleb128 not a .byte
    EmitIntValue(MCDwarfFiles[i]->getDirIndex(), 1);
    EmitIntValue(0, 1); // last modification timestamp (always 0)
    EmitIntValue(0, 1); // filesize (always 0)
  }
  EmitIntValue(0, 1); // Terminate the file list

  // This is the end of the prologue, so set the value of the symbol at the
  // end of the prologue (that was used in a previous expression).
  EmitLabel(ProEndSym);

  // TODO: This is the point where the line tables would be emitted.

  // If there are no line tables emited then we emit:
  // The following DW_LNE_set_address sequence to set the address to zero
  //   TODO test for 32-bit or 64-bit output
  //     This is the sequence for 32-bit code
  EmitIntValue(0, 1);
  EmitIntValue(5, 1);
  EmitIntValue(2, 1);
  EmitIntValue(0, 1);
  EmitIntValue(0, 1);
  EmitIntValue(0, 1);
  EmitIntValue(0, 1);

  // Lastly emit the DW_LNE_end_sequence which consists of 3 bytes '00 01 01'
  // (00 is the code for extended opcodes, followed by a ULEB128 length of the
  // extended opcode (01), and the DW_LNE_end_sequence (01).
  EmitIntValue(0, 1); // DW_LNS_extended_op
  EmitIntValue(1, 1); // ULEB128 length of the extended opcode
  EmitIntValue(1, 1); // DW_LNE_end_sequence

  // This is the end of the section, so set the value of the symbol at the end
  // of this section (that was used in a previous expression).
  EmitLabel(LineEndSym);
}

void MCMachOStreamer::Finish() {
  // Dump out the dwarf file and directory tables (soon to include line table)
  EmitDwarfFileTable();

  // We have to set the fragment atom associations so we can relax properly for
  // Mach-O.

  // First, scan the symbol table to build a lookup table from fragments to
  // defining symbols.
  DenseMap<const MCFragment*, MCSymbolData*> DefiningSymbolMap;
  for (MCAssembler::symbol_iterator it = getAssembler().symbol_begin(),
         ie = getAssembler().symbol_end(); it != ie; ++it) {
    if (getAssembler().isSymbolLinkerVisible(it->getSymbol()) &&
        it->getFragment()) {
      // An atom defining symbol should never be internal to a fragment.
      assert(it->getOffset() == 0 && "Invalid offset in atom defining symbol!");
      DefiningSymbolMap[it->getFragment()] = it;
    }
  }

  // Set the fragment atom associations by tracking the last seen atom defining
  // symbol.
  for (MCAssembler::iterator it = getAssembler().begin(),
         ie = getAssembler().end(); it != ie; ++it) {
    MCSymbolData *CurrentAtom = 0;
    for (MCSectionData::iterator it2 = it->begin(),
           ie2 = it->end(); it2 != ie2; ++it2) {
      if (MCSymbolData *SD = DefiningSymbolMap.lookup(it2))
        CurrentAtom = SD;
      it2->setAtom(CurrentAtom);
    }
  }

  this->MCObjectStreamer::Finish();
}

MCStreamer *llvm::createMachOStreamer(MCContext &Context, TargetAsmBackend &TAB,
                                      raw_ostream &OS, MCCodeEmitter *CE,
                                      bool RelaxAll) {
  MCMachOStreamer *S = new MCMachOStreamer(Context, TAB, OS, CE);
  if (RelaxAll)
    S->getAssembler().setRelaxAll(true);
  return S;
}
