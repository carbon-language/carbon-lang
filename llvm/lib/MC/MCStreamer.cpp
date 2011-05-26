//===- lib/MC/MCStreamer.cpp - Streaming Machine Code Output --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include <cstdlib>
using namespace llvm;

MCStreamer::MCStreamer(MCContext &Ctx) : Context(Ctx), EmitEHFrame(true),
                                         EmitDebugFrame(false),
                                         CurrentW64UnwindInfo(0) {
  const MCSection *section = NULL;
  SectionStack.push_back(std::make_pair(section, section));
}

MCStreamer::~MCStreamer() {
}

const MCExpr *MCStreamer::BuildSymbolDiff(MCContext &Context,
                                          const MCSymbol *A,
                                          const MCSymbol *B) {
  MCSymbolRefExpr::VariantKind Variant = MCSymbolRefExpr::VK_None;
  const MCExpr *ARef =
    MCSymbolRefExpr::Create(A, Variant, Context);
  const MCExpr *BRef =
    MCSymbolRefExpr::Create(B, Variant, Context);
  const MCExpr *AddrDelta =
    MCBinaryExpr::Create(MCBinaryExpr::Sub, ARef, BRef, Context);
  return AddrDelta;
}

const MCExpr *MCStreamer::ForceExpAbs(const MCExpr* Expr) {
  if (Context.getAsmInfo().hasAggressiveSymbolFolding() ||
      isa<MCSymbolRefExpr>(Expr))
    return Expr;

  MCSymbol *ABS = Context.CreateTempSymbol();
  EmitAssignment(ABS, Expr);
  return MCSymbolRefExpr::Create(ABS, Context);
}

raw_ostream &MCStreamer::GetCommentOS() {
  // By default, discard comments.
  return nulls();
}

void MCStreamer::EmitDwarfSetLineAddr(int64_t LineDelta,
                                      const MCSymbol *Label, int PointerSize) {
  // emit the sequence to set the address
  EmitIntValue(dwarf::DW_LNS_extended_op, 1);
  EmitULEB128IntValue(PointerSize + 1);
  EmitIntValue(dwarf::DW_LNE_set_address, 1);
  EmitSymbolValue(Label, PointerSize);

  // emit the sequence for the LineDelta (from 1) and a zero address delta.
  MCDwarfLineAddr::Emit(this, LineDelta, 0);
}

/// EmitIntValue - Special case of EmitValue that avoids the client having to
/// pass in a MCExpr for constant integers.
void MCStreamer::EmitIntValue(uint64_t Value, unsigned Size,
                              unsigned AddrSpace) {
  assert(Size <= 8 && "Invalid size");
  assert((isUIntN(8 * Size, Value) || isIntN(8 * Size, Value)) &&
         "Invalid size");
  char buf[8];
  // FIXME: Endianness assumption.
  for (unsigned i = 0; i != Size; ++i)
    buf[i] = uint8_t(Value >> (i * 8));
  EmitBytes(StringRef(buf, Size), AddrSpace);
}

/// EmitULEB128Value - Special case of EmitULEB128Value that avoids the
/// client having to pass in a MCExpr for constant integers.
void MCStreamer::EmitULEB128IntValue(uint64_t Value, unsigned AddrSpace) {
  SmallString<32> Tmp;
  raw_svector_ostream OSE(Tmp);
  MCObjectWriter::EncodeULEB128(Value, OSE);
  EmitBytes(OSE.str(), AddrSpace);
}

/// EmitSLEB128Value - Special case of EmitSLEB128Value that avoids the
/// client having to pass in a MCExpr for constant integers.
void MCStreamer::EmitSLEB128IntValue(int64_t Value, unsigned AddrSpace) {
  SmallString<32> Tmp;
  raw_svector_ostream OSE(Tmp);
  MCObjectWriter::EncodeSLEB128(Value, OSE);
  EmitBytes(OSE.str(), AddrSpace);
}

void MCStreamer::EmitAbsValue(const MCExpr *Value, unsigned Size,
                              unsigned AddrSpace) {
  const MCExpr *ABS = ForceExpAbs(Value);
  EmitValue(ABS, Size, AddrSpace);
}


void MCStreamer::EmitValue(const MCExpr *Value, unsigned Size,
                           unsigned AddrSpace) {
  EmitValueImpl(Value, Size, AddrSpace);
}

void MCStreamer::EmitSymbolValue(const MCSymbol *Sym, unsigned Size,
                                  unsigned AddrSpace) {
  EmitValueImpl(MCSymbolRefExpr::Create(Sym, getContext()), Size,
                AddrSpace);
}

void MCStreamer::EmitGPRel32Value(const MCExpr *Value) {
  report_fatal_error("unsupported directive in streamer");
}

/// EmitFill - Emit NumBytes bytes worth of the value specified by
/// FillValue.  This implements directives such as '.space'.
void MCStreamer::EmitFill(uint64_t NumBytes, uint8_t FillValue,
                          unsigned AddrSpace) {
  const MCExpr *E = MCConstantExpr::Create(FillValue, getContext());
  for (uint64_t i = 0, e = NumBytes; i != e; ++i)
    EmitValue(E, 1, AddrSpace);
}

bool MCStreamer::EmitDwarfFileDirective(unsigned FileNo,
                                        StringRef Filename) {
  return getContext().GetDwarfFile(Filename, FileNo) == 0;
}

void MCStreamer::EmitDwarfLocDirective(unsigned FileNo, unsigned Line,
                                       unsigned Column, unsigned Flags,
                                       unsigned Isa,
                                       unsigned Discriminator,
                                       StringRef FileName) {
  getContext().setCurrentDwarfLoc(FileNo, Line, Column, Flags, Isa,
                                  Discriminator);
}

MCDwarfFrameInfo *MCStreamer::getCurrentFrameInfo() {
  if (FrameInfos.empty())
    return NULL;
  return &FrameInfos.back();
}

void MCStreamer::EnsureValidFrame() {
  MCDwarfFrameInfo *CurFrame = getCurrentFrameInfo();
  if (!CurFrame || CurFrame->End)
    report_fatal_error("No open frame");
}

void MCStreamer::EmitEHSymAttributes(const MCSymbol *Symbol,
                                     MCSymbol *EHSymbol) {
}

void MCStreamer::EmitLabel(MCSymbol *Symbol) {
  assert(!Symbol->isVariable() && "Cannot emit a variable symbol!");
  assert(getCurrentSection() && "Cannot emit before setting section!");
  Symbol->setSection(*getCurrentSection());

  StringRef Prefix = getContext().getAsmInfo().getPrivateGlobalPrefix();
  if (!Symbol->getName().startswith(Prefix))
    LastNonPrivate = Symbol;
}

void MCStreamer::EmitCFISections(bool EH, bool Debug) {
  assert(EH || Debug);
  EmitEHFrame = EH;
  EmitDebugFrame = Debug;
}

void MCStreamer::EmitCFIStartProc() {
  MCDwarfFrameInfo *CurFrame = getCurrentFrameInfo();
  if (CurFrame && !CurFrame->End)
    report_fatal_error("Starting a frame before finishing the previous one!");
  MCDwarfFrameInfo Frame;
  Frame.Begin = getContext().CreateTempSymbol();
  Frame.Function = LastNonPrivate;
  EmitLabel(Frame.Begin);
  FrameInfos.push_back(Frame);
}

void MCStreamer::EmitCFIEndProc() {
  EnsureValidFrame();
  MCDwarfFrameInfo *CurFrame = getCurrentFrameInfo();
  CurFrame->End = getContext().CreateTempSymbol();
  EmitLabel(CurFrame->End);
}

void MCStreamer::EmitCFIDefCfa(int64_t Register, int64_t Offset) {
  EnsureValidFrame();
  MCDwarfFrameInfo *CurFrame = getCurrentFrameInfo();
  MCSymbol *Label = getContext().CreateTempSymbol();
  EmitLabel(Label);
  MachineLocation Dest(MachineLocation::VirtualFP);
  MachineLocation Source(Register, -Offset);
  MCCFIInstruction Instruction(Label, Dest, Source);
  CurFrame->Instructions.push_back(Instruction);
}

void MCStreamer::EmitCFIDefCfaOffset(int64_t Offset) {
  EnsureValidFrame();
  MCDwarfFrameInfo *CurFrame = getCurrentFrameInfo();
  MCSymbol *Label = getContext().CreateTempSymbol();
  EmitLabel(Label);
  MachineLocation Dest(MachineLocation::VirtualFP);
  MachineLocation Source(MachineLocation::VirtualFP, -Offset);
  MCCFIInstruction Instruction(Label, Dest, Source);
  CurFrame->Instructions.push_back(Instruction);
}

void MCStreamer::EmitCFIAdjustCfaOffset(int64_t Adjustment) {
  EnsureValidFrame();
  MCDwarfFrameInfo *CurFrame = getCurrentFrameInfo();
  MCSymbol *Label = getContext().CreateTempSymbol();
  EmitLabel(Label);
  MachineLocation Dest(MachineLocation::VirtualFP);
  MachineLocation Source(MachineLocation::VirtualFP, Adjustment);
  MCCFIInstruction Instruction(MCCFIInstruction::RelMove, Label, Dest, Source);
  CurFrame->Instructions.push_back(Instruction);
}

void MCStreamer::EmitCFIDefCfaRegister(int64_t Register) {
  EnsureValidFrame();
  MCDwarfFrameInfo *CurFrame = getCurrentFrameInfo();
  MCSymbol *Label = getContext().CreateTempSymbol();
  EmitLabel(Label);
  MachineLocation Dest(Register);
  MachineLocation Source(MachineLocation::VirtualFP);
  MCCFIInstruction Instruction(Label, Dest, Source);
  CurFrame->Instructions.push_back(Instruction);
}

void MCStreamer::EmitCFIOffset(int64_t Register, int64_t Offset) {
  EnsureValidFrame();
  MCDwarfFrameInfo *CurFrame = getCurrentFrameInfo();
  MCSymbol *Label = getContext().CreateTempSymbol();
  EmitLabel(Label);
  MachineLocation Dest(Register, Offset);
  MachineLocation Source(Register, Offset);
  MCCFIInstruction Instruction(Label, Dest, Source);
  CurFrame->Instructions.push_back(Instruction);
}

void MCStreamer::EmitCFIRelOffset(int64_t Register, int64_t Offset) {
  EnsureValidFrame();
  MCDwarfFrameInfo *CurFrame = getCurrentFrameInfo();
  MCSymbol *Label = getContext().CreateTempSymbol();
  EmitLabel(Label);
  MachineLocation Dest(Register, Offset);
  MachineLocation Source(Register, Offset);
  MCCFIInstruction Instruction(MCCFIInstruction::RelMove, Label, Dest, Source);
  CurFrame->Instructions.push_back(Instruction);
}

void MCStreamer::EmitCFIPersonality(const MCSymbol *Sym,
                                    unsigned Encoding) {
  EnsureValidFrame();
  MCDwarfFrameInfo *CurFrame = getCurrentFrameInfo();
  CurFrame->Personality = Sym;
  CurFrame->PersonalityEncoding = Encoding;
}

void MCStreamer::EmitCFILsda(const MCSymbol *Sym, unsigned Encoding) {
  EnsureValidFrame();
  MCDwarfFrameInfo *CurFrame = getCurrentFrameInfo();
  CurFrame->Lsda = Sym;
  CurFrame->LsdaEncoding = Encoding;
}

void MCStreamer::EmitCFIRememberState() {
  EnsureValidFrame();
  MCDwarfFrameInfo *CurFrame = getCurrentFrameInfo();
  MCSymbol *Label = getContext().CreateTempSymbol();
  EmitLabel(Label);
  MCCFIInstruction Instruction(MCCFIInstruction::Remember, Label);
  CurFrame->Instructions.push_back(Instruction);
}

void MCStreamer::EmitCFIRestoreState() {
  // FIXME: Error if there is no matching cfi_remember_state.
  EnsureValidFrame();
  MCDwarfFrameInfo *CurFrame = getCurrentFrameInfo();
  MCSymbol *Label = getContext().CreateTempSymbol();
  EmitLabel(Label);
  MCCFIInstruction Instruction(MCCFIInstruction::Restore, Label);
  CurFrame->Instructions.push_back(Instruction);
}

void MCStreamer::EmitCFISameValue(int64_t Register) {
  EnsureValidFrame();
  MCDwarfFrameInfo *CurFrame = getCurrentFrameInfo();
  MCSymbol *Label = getContext().CreateTempSymbol();
  EmitLabel(Label);
  MCCFIInstruction Instruction(MCCFIInstruction::SameValue, Label, Register);
  CurFrame->Instructions.push_back(Instruction);
}

void MCStreamer::setCurrentW64UnwindInfo(MCWin64EHUnwindInfo *Frame) {
  W64UnwindInfos.push_back(Frame);
  CurrentW64UnwindInfo = W64UnwindInfos.back();
}

void MCStreamer::EnsureValidW64UnwindInfo() {
  MCWin64EHUnwindInfo *CurFrame = CurrentW64UnwindInfo;
  if (!CurFrame || CurFrame->End)
    report_fatal_error("No open Win64 EH frame function!");
}

void MCStreamer::EmitWin64EHStartProc(const MCSymbol *Symbol) {
  MCWin64EHUnwindInfo *CurFrame = CurrentW64UnwindInfo;
  if (CurFrame && !CurFrame->End)
    report_fatal_error("Starting a function before ending the previous one!");
  MCWin64EHUnwindInfo *Frame = new (getContext()) MCWin64EHUnwindInfo;
  Frame->Begin = getContext().CreateTempSymbol();
  Frame->Function = Symbol;
  EmitLabel(Frame->Begin);
  setCurrentW64UnwindInfo(Frame);
}

void MCStreamer::EmitWin64EHEndProc() {
  EnsureValidW64UnwindInfo();
  MCWin64EHUnwindInfo *CurFrame = CurrentW64UnwindInfo;
  if (CurFrame->ChainedParent)
    report_fatal_error("Not all chained regions terminated!");
  CurFrame->End = getContext().CreateTempSymbol();
  EmitLabel(CurFrame->End);
}

void MCStreamer::EmitWin64EHStartChained() {
  EnsureValidW64UnwindInfo();
  MCWin64EHUnwindInfo *Frame = new (getContext()) MCWin64EHUnwindInfo;
  MCWin64EHUnwindInfo *CurFrame = CurrentW64UnwindInfo;
  Frame->Begin = getContext().CreateTempSymbol();
  Frame->Function = CurFrame->Function;
  Frame->ChainedParent = CurFrame;
  EmitLabel(Frame->Begin);
  setCurrentW64UnwindInfo(Frame);
}

void MCStreamer::EmitWin64EHEndChained() {
  EnsureValidW64UnwindInfo();
  MCWin64EHUnwindInfo *CurFrame = CurrentW64UnwindInfo;
  if (!CurFrame->ChainedParent)
    report_fatal_error("End of a chained region outside a chained region!");
  CurFrame->End = getContext().CreateTempSymbol();
  EmitLabel(CurFrame->End);
  CurrentW64UnwindInfo = CurFrame->ChainedParent;
}

void MCStreamer::EmitWin64EHHandler(const MCSymbol *Sym, bool Unwind,
                                    bool Except) {
  EnsureValidW64UnwindInfo();
  MCWin64EHUnwindInfo *CurFrame = CurrentW64UnwindInfo;
  if (CurFrame->ChainedParent)
    report_fatal_error("Chained unwind areas can't have handlers!");
  CurFrame->ExceptionHandler = Sym;
  if (!Except && !Unwind)
    report_fatal_error("Don't know what kind of handler this is!");
  if (Unwind)
    CurFrame->HandlesUnwind = true;
  if (Except)
    CurFrame->HandlesExceptions = true;
}

void MCStreamer::EmitWin64EHHandlerData() {
  EnsureValidW64UnwindInfo();
  MCWin64EHUnwindInfo *CurFrame = CurrentW64UnwindInfo;
  if (CurFrame->ChainedParent)
    report_fatal_error("Chained unwind areas can't have handlers!");
}

void MCStreamer::EmitWin64EHPushReg(unsigned Register) {
  EnsureValidW64UnwindInfo();
  MCWin64EHUnwindInfo *CurFrame = CurrentW64UnwindInfo;
  MCWin64EHInstruction Inst(Win64EH::UOP_PushNonVol, Register);
  CurFrame->Instructions.push_back(Inst);
}

void MCStreamer::EmitWin64EHSetFrame(unsigned Register, unsigned Offset) {
  EnsureValidW64UnwindInfo();
  if (Offset & 0x0F)
    report_fatal_error("Misaligned frame pointer offset!");
  MCWin64EHUnwindInfo *CurFrame = CurrentW64UnwindInfo;
  MCWin64EHInstruction Inst(Win64EH::UOP_SetFPReg, Register, Offset);
  CurFrame->LastFrameInst = CurFrame->Instructions.size();
  CurFrame->Instructions.push_back(Inst);
}

void MCStreamer::EmitWin64EHAllocStack(unsigned Size) {
  EnsureValidW64UnwindInfo();
  if (Size & 7)
    report_fatal_error("Misaligned stack allocation!");
  MCWin64EHUnwindInfo *CurFrame = CurrentW64UnwindInfo;
  MCWin64EHInstruction Inst(Size);
  CurFrame->Instructions.push_back(Inst);
}

void MCStreamer::EmitWin64EHSaveReg(unsigned Register, unsigned Offset) {
  EnsureValidW64UnwindInfo();
  if (Offset & 7)
    report_fatal_error("Misaligned saved register offset!");
  MCWin64EHUnwindInfo *CurFrame = CurrentW64UnwindInfo;
  MCWin64EHInstruction Inst(
     Offset > 512*1024-8 ? Win64EH::UOP_SaveNonVol : Win64EH::UOP_SaveNonVolBig,
                            Register, Offset);
  CurFrame->Instructions.push_back(Inst);
}

void MCStreamer::EmitWin64EHSaveXMM(unsigned Register, unsigned Offset) {
  EnsureValidW64UnwindInfo();
  if (Offset & 0x0F)
    report_fatal_error("Misaligned saved vector register offset!");
  MCWin64EHUnwindInfo *CurFrame = CurrentW64UnwindInfo;
  MCWin64EHInstruction Inst(
    Offset > 512*1024-16 ? Win64EH::UOP_SaveXMM128 : Win64EH::UOP_SaveXMM128Big,
                            Register, Offset);
  CurFrame->Instructions.push_back(Inst);
}

void MCStreamer::EmitWin64EHPushFrame(bool Code) {
  EnsureValidW64UnwindInfo();
  MCWin64EHUnwindInfo *CurFrame = CurrentW64UnwindInfo;
  MCWin64EHInstruction Inst(Win64EH::UOP_PushMachFrame, Code);
  CurFrame->Instructions.push_back(Inst);
}

void MCStreamer::EmitWin64EHEndProlog() {
  EnsureValidW64UnwindInfo();
  MCWin64EHUnwindInfo *CurFrame = CurrentW64UnwindInfo;
  CurFrame->PrologEnd = getContext().CreateTempSymbol();
  EmitLabel(CurFrame->PrologEnd);
}

void MCStreamer::EmitFnStart() {
  errs() << "Not implemented yet\n";
  abort();
}

void MCStreamer::EmitFnEnd() {
  errs() << "Not implemented yet\n";
  abort();
}

void MCStreamer::EmitCantUnwind() {
  errs() << "Not implemented yet\n";
  abort();
}

void MCStreamer::EmitHandlerData() {
  errs() << "Not implemented yet\n";
  abort();
}

void MCStreamer::EmitPersonality(const MCSymbol *Personality) {
  errs() << "Not implemented yet\n";
  abort();
}

void MCStreamer::EmitSetFP(unsigned FpReg, unsigned SpReg, int64_t Offset) {
  errs() << "Not implemented yet\n";
  abort();
}

void MCStreamer::EmitPad(int64_t Offset) {
  errs() << "Not implemented yet\n";
  abort();
}

void MCStreamer::EmitRegSave(const SmallVectorImpl<unsigned> &RegList, bool) {
  errs() << "Not implemented yet\n";
  abort();
}

/// EmitRawText - If this file is backed by an assembly streamer, this dumps
/// the specified string in the output .s file.  This capability is
/// indicated by the hasRawTextSupport() predicate.
void MCStreamer::EmitRawText(StringRef String) {
  errs() << "EmitRawText called on an MCStreamer that doesn't support it, "
  " something must not be fully mc'ized\n";
  abort();
}

void MCStreamer::EmitRawText(const Twine &T) {
  SmallString<128> Str;
  T.toVector(Str);
  EmitRawText(Str.str());
}

void MCStreamer::EmitFrames(bool usingCFI) {
  if (!getNumFrameInfos())
    return;

  if (EmitEHFrame)
    MCDwarfFrameEmitter::Emit(*this, usingCFI, true);

  if (EmitDebugFrame)
    MCDwarfFrameEmitter::Emit(*this, usingCFI, false);
}

void MCStreamer::EmitW64Tables() {
  if (!getNumW64UnwindInfos())
    return;

  MCWin64EHUnwindEmitter::Emit(*this);
}
