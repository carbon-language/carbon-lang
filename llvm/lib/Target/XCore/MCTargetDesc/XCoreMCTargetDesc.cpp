//===-- XCoreMCTargetDesc.cpp - XCore Target Descriptions -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides XCore specific target descriptions.
//
//===----------------------------------------------------------------------===//

#include "XCoreMCTargetDesc.h"
#include "InstPrinter/XCoreInstPrinter.h"
#include "XCoreMCAsmInfo.h"
#include "XCoreTargetStreamer.h"
#include "llvm/MC/MCCodeGenInfo.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Support/TargetRegistry.h"

#define GET_INSTRINFO_MC_DESC
#include "XCoreGenInstrInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "XCoreGenSubtargetInfo.inc"

#define GET_REGINFO_MC_DESC
#include "XCoreGenRegisterInfo.inc"

using namespace llvm;

static MCInstrInfo *createXCoreMCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitXCoreMCInstrInfo(X);
  return X;
}

static MCRegisterInfo *createXCoreMCRegisterInfo(StringRef TT) {
  MCRegisterInfo *X = new MCRegisterInfo();
  InitXCoreMCRegisterInfo(X, XCore::LR);
  return X;
}

static MCSubtargetInfo *createXCoreMCSubtargetInfo(StringRef TT, StringRef CPU,
                                                   StringRef FS) {
  MCSubtargetInfo *X = new MCSubtargetInfo();
  InitXCoreMCSubtargetInfo(X, TT, CPU, FS);
  return X;
}

static MCAsmInfo *createXCoreMCAsmInfo(const MCRegisterInfo &MRI,
                                       StringRef TT) {
  MCAsmInfo *MAI = new XCoreMCAsmInfo(TT);

  // Initial state of the frame pointer is SP.
  MCCFIInstruction Inst = MCCFIInstruction::createDefCfa(0, XCore::SP, 0);
  MAI->addInitialFrameState(Inst);

  return MAI;
}

static MCCodeGenInfo *createXCoreMCCodeGenInfo(StringRef TT, Reloc::Model RM,
                                               CodeModel::Model CM,
                                               CodeGenOpt::Level OL) {
  MCCodeGenInfo *X = new MCCodeGenInfo();
  if (RM == Reloc::Default) {
    RM = Reloc::Static;
  }
  if (CM == CodeModel::Default) {
    CM = CodeModel::Small;
  }
  if (CM != CodeModel::Small && CM != CodeModel::Large)
    report_fatal_error("Target only supports CodeModel Small or Large");

  X->InitMCCodeGenInfo(RM, CM, OL);
  return X;
}

static MCInstPrinter *createXCoreMCInstPrinter(const Target &T,
                                               unsigned SyntaxVariant,
                                               const MCAsmInfo &MAI,
                                               const MCInstrInfo &MII,
                                               const MCRegisterInfo &MRI,
                                               const MCSubtargetInfo &STI) {
  return new XCoreInstPrinter(MAI, MII, MRI);
}

XCoreTargetStreamer::XCoreTargetStreamer(MCStreamer &S) : MCTargetStreamer(S) {}
XCoreTargetStreamer::~XCoreTargetStreamer() {}

namespace {

class XCoreTargetAsmStreamer : public XCoreTargetStreamer {
  formatted_raw_ostream &OS;
public:
  XCoreTargetAsmStreamer(MCStreamer &S, formatted_raw_ostream &OS);
  virtual void emitCCTopData(StringRef Name) LLVM_OVERRIDE;
  virtual void emitCCTopFunction(StringRef Name) LLVM_OVERRIDE;
  virtual void emitCCBottomData(StringRef Name) LLVM_OVERRIDE;
  virtual void emitCCBottomFunction(StringRef Name) LLVM_OVERRIDE;
};

XCoreTargetAsmStreamer::XCoreTargetAsmStreamer(MCStreamer &S,
                                               formatted_raw_ostream &OS)
    : XCoreTargetStreamer(S), OS(OS) {}

void XCoreTargetAsmStreamer::emitCCTopData(StringRef Name) {
  OS << "\t.cc_top " << Name << ".data," << Name << '\n';
}

void XCoreTargetAsmStreamer::emitCCTopFunction(StringRef Name) {
  OS << "\t.cc_top " << Name << ".function," << Name << '\n';
}

void XCoreTargetAsmStreamer::emitCCBottomData(StringRef Name) {
  OS << "\t.cc_bottom " << Name << ".data\n";
}

void XCoreTargetAsmStreamer::emitCCBottomFunction(StringRef Name) {
  OS << "\t.cc_bottom " << Name << ".function\n";
}
}

static MCStreamer *
createXCoreMCAsmStreamer(MCContext &Ctx, formatted_raw_ostream &OS,
                         bool isVerboseAsm, bool useCFI, bool useDwarfDirectory,
                         MCInstPrinter *InstPrint, MCCodeEmitter *CE,
                         MCAsmBackend *TAB, bool ShowInst) {
  MCStreamer *S =
      llvm::createAsmStreamer(Ctx, OS, isVerboseAsm, useCFI, useDwarfDirectory,
                              InstPrint, CE, TAB, ShowInst);
  new XCoreTargetAsmStreamer(*S, OS);
  return S;
}

// Force static initialization.
extern "C" void LLVMInitializeXCoreTargetMC() {
  // Register the MC asm info.
  RegisterMCAsmInfoFn X(TheXCoreTarget, createXCoreMCAsmInfo);

  // Register the MC codegen info.
  TargetRegistry::RegisterMCCodeGenInfo(TheXCoreTarget,
                                        createXCoreMCCodeGenInfo);

  // Register the MC instruction info.
  TargetRegistry::RegisterMCInstrInfo(TheXCoreTarget, createXCoreMCInstrInfo);

  // Register the MC register info.
  TargetRegistry::RegisterMCRegInfo(TheXCoreTarget, createXCoreMCRegisterInfo);

  // Register the MC subtarget info.
  TargetRegistry::RegisterMCSubtargetInfo(TheXCoreTarget,
                                          createXCoreMCSubtargetInfo);

  // Register the MCInstPrinter
  TargetRegistry::RegisterMCInstPrinter(TheXCoreTarget,
                                        createXCoreMCInstPrinter);

  TargetRegistry::RegisterAsmStreamer(TheXCoreTarget, createXCoreMCAsmStreamer);
}
