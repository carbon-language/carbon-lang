//===-- ARM64AsmPrinter.cpp - ARM64 LLVM assembly writer ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a printer that converts from our internal representation
// of machine-dependent LLVM code to the ARM64 assembly language.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "asm-printer"
#include "ARM64.h"
#include "ARM64MachineFunctionInfo.h"
#include "ARM64MCInstLower.h"
#include "ARM64RegisterInfo.h"
#include "ARM64Subtarget.h"
#include "InstPrinter/ARM64InstPrinter.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/StackMaps.h"
#include "llvm/CodeGen/MachineModuleInfoImpls.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/MC/MCLinkerOptimizationHint.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/TargetRegistry.h"
using namespace llvm;

namespace {

class ARM64AsmPrinter : public AsmPrinter {
  /// Subtarget - Keep a pointer to the ARM64Subtarget around so that we can
  /// make the right decision when printing asm code for different targets.
  const ARM64Subtarget *Subtarget;

  ARM64MCInstLower MCInstLowering;
  StackMaps SM;

public:
  ARM64AsmPrinter(TargetMachine &TM, MCStreamer &Streamer)
      : AsmPrinter(TM, Streamer), Subtarget(&TM.getSubtarget<ARM64Subtarget>()),
        MCInstLowering(OutContext, *Mang, *this), SM(*this), ARM64FI(NULL),
        LOHLabelCounter(0) {}

  virtual const char *getPassName() const { return "ARM64 Assembly Printer"; }

  /// \brief Wrapper for MCInstLowering.lowerOperand() for the
  /// tblgen'erated pseudo lowering.
  bool lowerOperand(const MachineOperand &MO, MCOperand &MCOp) const {
    return MCInstLowering.lowerOperand(MO, MCOp);
  }

  void LowerSTACKMAP(MCStreamer &OutStreamer, StackMaps &SM,
                     const MachineInstr &MI);
  void LowerPATCHPOINT(MCStreamer &OutStreamer, StackMaps &SM,
                       const MachineInstr &MI);
  /// \brief tblgen'erated driver function for lowering simple MI->MC
  /// pseudo instructions.
  bool emitPseudoExpansionLowering(MCStreamer &OutStreamer,
                                   const MachineInstr *MI);

  void EmitInstruction(const MachineInstr *MI);

  void getAnalysisUsage(AnalysisUsage &AU) const {
    AsmPrinter::getAnalysisUsage(AU);
    AU.setPreservesAll();
  }

  bool runOnMachineFunction(MachineFunction &F) {
    ARM64FI = F.getInfo<ARM64FunctionInfo>();
    return AsmPrinter::runOnMachineFunction(F);
  }

private:
  MachineLocation getDebugValueLocation(const MachineInstr *MI) const;
  void printOperand(const MachineInstr *MI, unsigned OpNum, raw_ostream &O);
  bool printAsmMRegister(const MachineOperand &MO, char Mode, raw_ostream &O);
  bool printAsmRegInClass(const MachineOperand &MO,
                          const TargetRegisterClass *RC, bool isVector,
                          raw_ostream &O);

  bool PrintAsmOperand(const MachineInstr *MI, unsigned OpNum,
                       unsigned AsmVariant, const char *ExtraCode,
                       raw_ostream &O);
  bool PrintAsmMemoryOperand(const MachineInstr *MI, unsigned OpNum,
                             unsigned AsmVariant, const char *ExtraCode,
                             raw_ostream &O);

  void PrintDebugValueComment(const MachineInstr *MI, raw_ostream &OS);

  void EmitFunctionBodyEnd();

  MCSymbol *GetCPISymbol(unsigned CPID) const;
  void EmitEndOfAsmFile(Module &M);
  ARM64FunctionInfo *ARM64FI;

  /// \brief Emit the LOHs contained in ARM64FI.
  void EmitLOHs();

  typedef std::map<const MachineInstr *, MCSymbol *> MInstToMCSymbol;
  MInstToMCSymbol LOHInstToLabel;
  unsigned LOHLabelCounter;
};

} // end of anonymous namespace

//===----------------------------------------------------------------------===//

void ARM64AsmPrinter::EmitEndOfAsmFile(Module &M) {
  // Funny Darwin hack: This flag tells the linker that no global symbols
  // contain code that falls through to other global symbols (e.g. the obvious
  // implementation of multiple entry points).  If this doesn't occur, the
  // linker can safely perform dead code stripping.  Since LLVM never
  // generates code that does this, it is always safe to set.
  OutStreamer.EmitAssemblerFlag(MCAF_SubsectionsViaSymbols);
  SM.serializeToStackMapSection();

  // Emit a .data.rel section containing any stubs that were created.
  if (Subtarget->isTargetELF()) {
    const TargetLoweringObjectFileELF &TLOFELF =
      static_cast<const TargetLoweringObjectFileELF &>(getObjFileLowering());

    MachineModuleInfoELF &MMIELF = MMI->getObjFileInfo<MachineModuleInfoELF>();

    // Output stubs for external and common global variables.
    MachineModuleInfoELF::SymbolListTy Stubs = MMIELF.GetGVStubList();
    if (!Stubs.empty()) {
      OutStreamer.SwitchSection(TLOFELF.getDataRelSection());
      const DataLayout *TD = TM.getDataLayout();

      for (unsigned i = 0, e = Stubs.size(); i != e; ++i) {
        OutStreamer.EmitLabel(Stubs[i].first);
        OutStreamer.EmitSymbolValue(Stubs[i].second.getPointer(),
                                    TD->getPointerSize(0));
      }
      Stubs.clear();
    }
  }

}

MachineLocation
ARM64AsmPrinter::getDebugValueLocation(const MachineInstr *MI) const {
  MachineLocation Location;
  assert(MI->getNumOperands() == 4 && "Invalid no. of machine operands!");
  // Frame address.  Currently handles register +- offset only.
  if (MI->getOperand(0).isReg() && MI->getOperand(1).isImm())
    Location.set(MI->getOperand(0).getReg(), MI->getOperand(1).getImm());
  else {
    DEBUG(dbgs() << "DBG_VALUE instruction ignored! " << *MI << "\n");
  }
  return Location;
}

void ARM64AsmPrinter::EmitLOHs() {
  SmallVector<MCSymbol *, 3> MCArgs;

  for (const auto &D : ARM64FI->getLOHContainer()) {
    for (const MachineInstr *MI : D.getArgs()) {
      MInstToMCSymbol::iterator LabelIt = LOHInstToLabel.find(MI);
      assert(LabelIt != LOHInstToLabel.end() &&
             "Label hasn't been inserted for LOH related instruction");
      MCArgs.push_back(LabelIt->second);
    }
    OutStreamer.EmitLOHDirective(D.getKind(), MCArgs);
    MCArgs.clear();
  }
}

void ARM64AsmPrinter::EmitFunctionBodyEnd() {
  if (!ARM64FI->getLOHRelated().empty())
    EmitLOHs();
}

/// GetCPISymbol - Return the symbol for the specified constant pool entry.
MCSymbol *ARM64AsmPrinter::GetCPISymbol(unsigned CPID) const {
  // Darwin uses a linker-private symbol name for constant-pools (to
  // avoid addends on the relocation?), ELF has no such concept and
  // uses a normal private symbol.
  if (getDataLayout().getLinkerPrivateGlobalPrefix()[0])
    return OutContext.GetOrCreateSymbol(
        Twine(getDataLayout().getLinkerPrivateGlobalPrefix()) + "CPI" +
        Twine(getFunctionNumber()) + "_" + Twine(CPID));

  return OutContext.GetOrCreateSymbol(
      Twine(getDataLayout().getPrivateGlobalPrefix()) + "CPI" +
      Twine(getFunctionNumber()) + "_" + Twine(CPID));
}

void ARM64AsmPrinter::printOperand(const MachineInstr *MI, unsigned OpNum,
                                   raw_ostream &O) {
  const MachineOperand &MO = MI->getOperand(OpNum);
  switch (MO.getType()) {
  default:
    assert(0 && "<unknown operand type>");
  case MachineOperand::MO_Register: {
    unsigned Reg = MO.getReg();
    assert(TargetRegisterInfo::isPhysicalRegister(Reg));
    assert(!MO.getSubReg() && "Subregs should be eliminated!");
    O << ARM64InstPrinter::getRegisterName(Reg);
    break;
  }
  case MachineOperand::MO_Immediate: {
    int64_t Imm = MO.getImm();
    O << '#' << Imm;
    break;
  }
  }
}

bool ARM64AsmPrinter::printAsmMRegister(const MachineOperand &MO, char Mode,
                                        raw_ostream &O) {
  unsigned Reg = MO.getReg();
  switch (Mode) {
  default:
    return true; // Unknown mode.
  case 'w':
    Reg = getWRegFromXReg(Reg);
    break;
  case 'x':
    Reg = getXRegFromWReg(Reg);
    break;
  }

  O << ARM64InstPrinter::getRegisterName(Reg);
  return false;
}

// Prints the register in MO using class RC using the offset in the
// new register class. This should not be used for cross class
// printing.
bool ARM64AsmPrinter::printAsmRegInClass(const MachineOperand &MO,
                                         const TargetRegisterClass *RC,
                                         bool isVector, raw_ostream &O) {
  assert(MO.isReg() && "Should only get here with a register!");
  const ARM64RegisterInfo *RI =
      static_cast<const ARM64RegisterInfo *>(TM.getRegisterInfo());
  unsigned Reg = MO.getReg();
  unsigned RegToPrint = RC->getRegister(RI->getEncodingValue(Reg));
  assert(RI->regsOverlap(RegToPrint, Reg));
  O << ARM64InstPrinter::getRegisterName(
           RegToPrint, isVector ? ARM64::vreg : ARM64::NoRegAltName);
  return false;
}

bool ARM64AsmPrinter::PrintAsmOperand(const MachineInstr *MI, unsigned OpNum,
                                      unsigned AsmVariant,
                                      const char *ExtraCode, raw_ostream &O) {
  const MachineOperand &MO = MI->getOperand(OpNum);
  // Does this asm operand have a single letter operand modifier?
  if (ExtraCode && ExtraCode[0]) {
    if (ExtraCode[1] != 0)
      return true; // Unknown modifier.

    switch (ExtraCode[0]) {
    default:
      return true; // Unknown modifier.
    case 'w':      // Print W register
    case 'x':      // Print X register
      if (MO.isReg())
        return printAsmMRegister(MO, ExtraCode[0], O);
      if (MO.isImm() && MO.getImm() == 0) {
        unsigned Reg = ExtraCode[0] == 'w' ? ARM64::WZR : ARM64::XZR;
        O << ARM64InstPrinter::getRegisterName(Reg);
        return false;
      }
      printOperand(MI, OpNum, O);
      return false;
    case 'b': // Print B register.
    case 'h': // Print H register.
    case 's': // Print S register.
    case 'd': // Print D register.
    case 'q': // Print Q register.
      if (MO.isReg()) {
        const TargetRegisterClass *RC;
        switch (ExtraCode[0]) {
        case 'b':
          RC = &ARM64::FPR8RegClass;
          break;
        case 'h':
          RC = &ARM64::FPR16RegClass;
          break;
        case 's':
          RC = &ARM64::FPR32RegClass;
          break;
        case 'd':
          RC = &ARM64::FPR64RegClass;
          break;
        case 'q':
          RC = &ARM64::FPR128RegClass;
          break;
        default:
          return true;
        }
        return printAsmRegInClass(MO, RC, false /* vector */, O);
      }
      printOperand(MI, OpNum, O);
      return false;
    }
  }

  // According to ARM, we should emit x and v registers unless we have a
  // modifier.
  if (MO.isReg()) {
    unsigned Reg = MO.getReg();

    // If this is a w or x register, print an x register.
    if (ARM64::GPR32allRegClass.contains(Reg) ||
        ARM64::GPR64allRegClass.contains(Reg))
      return printAsmMRegister(MO, 'x', O);

    // If this is a b, h, s, d, or q register, print it as a v register.
    return printAsmRegInClass(MO, &ARM64::FPR128RegClass, true /* vector */, O);
  }

  printOperand(MI, OpNum, O);
  return false;
}

bool ARM64AsmPrinter::PrintAsmMemoryOperand(const MachineInstr *MI,
                                            unsigned OpNum, unsigned AsmVariant,
                                            const char *ExtraCode,
                                            raw_ostream &O) {
  if (ExtraCode && ExtraCode[0])
    return true; // Unknown modifier.

  const MachineOperand &MO = MI->getOperand(OpNum);
  assert(MO.isReg() && "unexpected inline asm memory operand");
  O << "[" << ARM64InstPrinter::getRegisterName(MO.getReg()) << "]";
  return false;
}

void ARM64AsmPrinter::PrintDebugValueComment(const MachineInstr *MI,
                                             raw_ostream &OS) {
  unsigned NOps = MI->getNumOperands();
  assert(NOps == 4);
  OS << '\t' << MAI->getCommentString() << "DEBUG_VALUE: ";
  // cast away const; DIetc do not take const operands for some reason.
  DIVariable V(const_cast<MDNode *>(MI->getOperand(NOps - 1).getMetadata()));
  OS << V.getName();
  OS << " <- ";
  // Frame address.  Currently handles register +- offset only.
  assert(MI->getOperand(0).isReg() && MI->getOperand(1).isImm());
  OS << '[';
  printOperand(MI, 0, OS);
  OS << '+';
  printOperand(MI, 1, OS);
  OS << ']';
  OS << "+";
  printOperand(MI, NOps - 2, OS);
}

void ARM64AsmPrinter::LowerSTACKMAP(MCStreamer &OutStreamer, StackMaps &SM,
                                    const MachineInstr &MI) {
  unsigned NumNOPBytes = MI.getOperand(1).getImm();

  SM.recordStackMap(MI);
  // Emit padding.
  assert(NumNOPBytes % 4 == 0 && "Invalid number of NOP bytes requested!");
  for (unsigned i = 0; i < NumNOPBytes; i += 4)
    EmitToStreamer(OutStreamer, MCInstBuilder(ARM64::HINT).addImm(0));
}

// Lower a patchpoint of the form:
// [<def>], <id>, <numBytes>, <target>, <numArgs>
void ARM64AsmPrinter::LowerPATCHPOINT(MCStreamer &OutStreamer, StackMaps &SM,
                                      const MachineInstr &MI) {
  SM.recordPatchPoint(MI);

  PatchPointOpers Opers(&MI);

  int64_t CallTarget = Opers.getMetaOper(PatchPointOpers::TargetPos).getImm();
  unsigned EncodedBytes = 0;
  if (CallTarget) {
    assert((CallTarget & 0xFFFFFFFFFFFF) == CallTarget &&
           "High 16 bits of call target should be zero.");
    unsigned ScratchReg = MI.getOperand(Opers.getNextScratchIdx()).getReg();
    EncodedBytes = 16;
    // Materialize the jump address:
    EmitToStreamer(OutStreamer, MCInstBuilder(ARM64::MOVZWi)
                                    .addReg(ScratchReg)
                                    .addImm((CallTarget >> 32) & 0xFFFF)
                                    .addImm(32));
    EmitToStreamer(OutStreamer, MCInstBuilder(ARM64::MOVKWi)
                                    .addReg(ScratchReg)
                                    .addReg(ScratchReg)
                                    .addImm((CallTarget >> 16) & 0xFFFF)
                                    .addImm(16));
    EmitToStreamer(OutStreamer, MCInstBuilder(ARM64::MOVKWi)
                                    .addReg(ScratchReg)
                                    .addReg(ScratchReg)
                                    .addImm(CallTarget & 0xFFFF)
                                    .addImm(0));
    EmitToStreamer(OutStreamer, MCInstBuilder(ARM64::BLR).addReg(ScratchReg));
  }
  // Emit padding.
  unsigned NumBytes = Opers.getMetaOper(PatchPointOpers::NBytesPos).getImm();
  assert(NumBytes >= EncodedBytes &&
         "Patchpoint can't request size less than the length of a call.");
  assert((NumBytes - EncodedBytes) % 4 == 0 &&
         "Invalid number of NOP bytes requested!");
  for (unsigned i = EncodedBytes; i < NumBytes; i += 4)
    EmitToStreamer(OutStreamer, MCInstBuilder(ARM64::HINT).addImm(0));
}

// Simple pseudo-instructions have their lowering (with expansion to real
// instructions) auto-generated.
#include "ARM64GenMCPseudoLowering.inc"

static unsigned getRealIndexedOpcode(unsigned Opc) {
  switch (Opc) {
  case ARM64::LDRXpre_isel:    return ARM64::LDRXpre;
  case ARM64::LDRWpre_isel:    return ARM64::LDRWpre;
  case ARM64::LDRDpre_isel:    return ARM64::LDRDpre;
  case ARM64::LDRSpre_isel:    return ARM64::LDRSpre;
  case ARM64::LDRBBpre_isel:   return ARM64::LDRBBpre;
  case ARM64::LDRHHpre_isel:   return ARM64::LDRHHpre;
  case ARM64::LDRSBWpre_isel:  return ARM64::LDRSBWpre;
  case ARM64::LDRSBXpre_isel:  return ARM64::LDRSBXpre;
  case ARM64::LDRSHWpre_isel:  return ARM64::LDRSHWpre;
  case ARM64::LDRSHXpre_isel:  return ARM64::LDRSHXpre;
  case ARM64::LDRSWpre_isel:   return ARM64::LDRSWpre;

  case ARM64::LDRDpost_isel:   return ARM64::LDRDpost;
  case ARM64::LDRSpost_isel:   return ARM64::LDRSpost;
  case ARM64::LDRXpost_isel:   return ARM64::LDRXpost;
  case ARM64::LDRWpost_isel:   return ARM64::LDRWpost;
  case ARM64::LDRHHpost_isel:  return ARM64::LDRHHpost;
  case ARM64::LDRBBpost_isel:  return ARM64::LDRBBpost;
  case ARM64::LDRSWpost_isel:  return ARM64::LDRSWpost;
  case ARM64::LDRSHWpost_isel: return ARM64::LDRSHWpost;
  case ARM64::LDRSHXpost_isel: return ARM64::LDRSHXpost;
  case ARM64::LDRSBWpost_isel: return ARM64::LDRSBWpost;
  case ARM64::LDRSBXpost_isel: return ARM64::LDRSBXpost;

  case ARM64::STRXpre_isel:    return ARM64::STRXpre;
  case ARM64::STRWpre_isel:    return ARM64::STRWpre;
  case ARM64::STRHHpre_isel:   return ARM64::STRHHpre;
  case ARM64::STRBBpre_isel:   return ARM64::STRBBpre;
  case ARM64::STRDpre_isel:    return ARM64::STRDpre;
  case ARM64::STRSpre_isel:    return ARM64::STRSpre;
  }
  llvm_unreachable("Unexpected pre-indexed opcode!");
}

void ARM64AsmPrinter::EmitInstruction(const MachineInstr *MI) {
  // Do any auto-generated pseudo lowerings.
  if (emitPseudoExpansionLowering(OutStreamer, MI))
    return;

  if (ARM64FI->getLOHRelated().count(MI)) {
    // Generate a label for LOH related instruction
    MCSymbol *LOHLabel = GetTempSymbol("loh", LOHLabelCounter++);
    // Associate the instruction with the label
    LOHInstToLabel[MI] = LOHLabel;
    OutStreamer.EmitLabel(LOHLabel);
  }

  // Do any manual lowerings.
  switch (MI->getOpcode()) {
  default:
    break;
  case ARM64::DBG_VALUE: {
    if (isVerbose() && OutStreamer.hasRawTextSupport()) {
      SmallString<128> TmpStr;
      raw_svector_ostream OS(TmpStr);
      PrintDebugValueComment(MI, OS);
      OutStreamer.EmitRawText(StringRef(OS.str()));
    }
    return;
  }
  // Indexed loads and stores use a pseudo to handle complex operand
  // tricks and writeback to the base register. We strip off the writeback
  // operand and switch the opcode here. Post-indexed stores were handled by the
  // tablegen'erated pseudos above. (The complex operand <--> simple
  // operand isel is beyond tablegen's ability, so we do these manually).
  case ARM64::LDRHHpre_isel:
  case ARM64::LDRBBpre_isel:
  case ARM64::LDRXpre_isel:
  case ARM64::LDRWpre_isel:
  case ARM64::LDRDpre_isel:
  case ARM64::LDRSpre_isel:
  case ARM64::LDRSBWpre_isel:
  case ARM64::LDRSBXpre_isel:
  case ARM64::LDRSHWpre_isel:
  case ARM64::LDRSHXpre_isel:
  case ARM64::LDRSWpre_isel:
  case ARM64::LDRDpost_isel:
  case ARM64::LDRSpost_isel:
  case ARM64::LDRXpost_isel:
  case ARM64::LDRWpost_isel:
  case ARM64::LDRHHpost_isel:
  case ARM64::LDRBBpost_isel:
  case ARM64::LDRSWpost_isel:
  case ARM64::LDRSHWpost_isel:
  case ARM64::LDRSHXpost_isel:
  case ARM64::LDRSBWpost_isel:
  case ARM64::LDRSBXpost_isel: {
    MCInst TmpInst;
    // For loads, the writeback operand to be skipped is the second.
    TmpInst.setOpcode(getRealIndexedOpcode(MI->getOpcode()));
    TmpInst.addOperand(MCOperand::CreateReg(MI->getOperand(0).getReg()));
    TmpInst.addOperand(MCOperand::CreateReg(MI->getOperand(2).getReg()));
    TmpInst.addOperand(MCOperand::CreateImm(MI->getOperand(3).getImm()));
    EmitToStreamer(OutStreamer, TmpInst);
    return;
  }
  case ARM64::STRXpre_isel:
  case ARM64::STRWpre_isel:
  case ARM64::STRHHpre_isel:
  case ARM64::STRBBpre_isel:
  case ARM64::STRDpre_isel:
  case ARM64::STRSpre_isel: {
    MCInst TmpInst;
    // For loads, the writeback operand to be skipped is the first.
    TmpInst.setOpcode(getRealIndexedOpcode(MI->getOpcode()));
    TmpInst.addOperand(MCOperand::CreateReg(MI->getOperand(1).getReg()));
    TmpInst.addOperand(MCOperand::CreateReg(MI->getOperand(2).getReg()));
    TmpInst.addOperand(MCOperand::CreateImm(MI->getOperand(3).getImm()));
    EmitToStreamer(OutStreamer, TmpInst);
    return;
  }

  // Tail calls use pseudo instructions so they have the proper code-gen
  // attributes (isCall, isReturn, etc.). We lower them to the real
  // instruction here.
  case ARM64::TCRETURNri: {
    MCInst TmpInst;
    TmpInst.setOpcode(ARM64::BR);
    TmpInst.addOperand(MCOperand::CreateReg(MI->getOperand(0).getReg()));
    EmitToStreamer(OutStreamer, TmpInst);
    return;
  }
  case ARM64::TCRETURNdi: {
    MCOperand Dest;
    MCInstLowering.lowerOperand(MI->getOperand(0), Dest);
    MCInst TmpInst;
    TmpInst.setOpcode(ARM64::B);
    TmpInst.addOperand(Dest);
    EmitToStreamer(OutStreamer, TmpInst);
    return;
  }
  case ARM64::TLSDESC_BLR: {
    MCOperand Callee, Sym;
    MCInstLowering.lowerOperand(MI->getOperand(0), Callee);
    MCInstLowering.lowerOperand(MI->getOperand(1), Sym);

    // First emit a relocation-annotation. This expands to no code, but requests
    // the following instruction gets an R_AARCH64_TLSDESC_CALL.
    MCInst TLSDescCall;
    TLSDescCall.setOpcode(ARM64::TLSDESCCALL);
    TLSDescCall.addOperand(Sym);
    EmitToStreamer(OutStreamer, TLSDescCall);

    // Other than that it's just a normal indirect call to the function loaded
    // from the descriptor.
    MCInst BLR;
    BLR.setOpcode(ARM64::BLR);
    BLR.addOperand(Callee);
    EmitToStreamer(OutStreamer, BLR);

    return;
  }

  case TargetOpcode::STACKMAP:
    return LowerSTACKMAP(OutStreamer, SM, *MI);

  case TargetOpcode::PATCHPOINT:
    return LowerPATCHPOINT(OutStreamer, SM, *MI);
  }

  // Finally, do the automated lowerings for everything else.
  MCInst TmpInst;
  MCInstLowering.Lower(MI, TmpInst);
  EmitToStreamer(OutStreamer, TmpInst);
}

// Force static initialization.
extern "C" void LLVMInitializeARM64AsmPrinter() {
  RegisterAsmPrinter<ARM64AsmPrinter> X(TheARM64Target);
}
