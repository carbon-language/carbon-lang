//===-- WebAssemblyAsmPrinter.cpp - WebAssembly LLVM assembly writer ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file contains a printer that converts from our internal
/// representation of machine-dependent LLVM code to the WebAssembly assembly
/// language.
///
//===----------------------------------------------------------------------===//

#include "WebAssembly.h"
#include "WebAssemblyMachineFunctionInfo.h"
#include "WebAssemblyRegisterInfo.h"
#include "WebAssemblySubtarget.h"
#include "InstPrinter/WebAssemblyInstPrinter.h"
#include "MCTargetDesc/WebAssemblyMCTargetDesc.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "asm-printer"

namespace {

class WebAssemblyAsmPrinter final : public AsmPrinter {
  bool hasAddr64;
  const WebAssemblyInstrInfo *TII;

public:
  WebAssemblyAsmPrinter(TargetMachine &TM, std::unique_ptr<MCStreamer> Streamer)
      : AsmPrinter(TM, std::move(Streamer)), TII(nullptr) {}

private:
  const char *getPassName() const override {
    return "WebAssembly Assembly Printer";
  }

  //===------------------------------------------------------------------===//
  // MachineFunctionPass Implementation.
  //===------------------------------------------------------------------===//

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AsmPrinter::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    const auto &Subtarget = MF.getSubtarget<WebAssemblySubtarget>();
    hasAddr64 = Subtarget.hasAddr64();
    TII = Subtarget.getInstrInfo();
    return AsmPrinter::runOnMachineFunction(MF);
  }

  //===------------------------------------------------------------------===//
  // AsmPrinter Implementation.
  //===------------------------------------------------------------------===//

  void EmitGlobalVariable(const GlobalVariable *GV) override;

  void EmitConstantPool() override;
  void EmitFunctionEntryLabel() override;
  void EmitFunctionBodyStart() override;
  void EmitFunctionBodyEnd() override;

  void EmitInstruction(const MachineInstr *MI) override;
};

} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Helpers.
//===----------------------------------------------------------------------===//

// Untyped, lower-case version of the opcode's name matching the names
// WebAssembly opcodes are expected to have. The tablegen names are uppercase
// and suffixed with their type (after an underscore).
static SmallString<32> OpcodeName(const WebAssemblyInstrInfo *TII,
                                  const MachineInstr *MI) {
  std::string N(StringRef(TII->getName(MI->getOpcode())).lower());
  std::string::size_type End = N.rfind('_');
  End = std::string::npos == End ? N.length() : End;
  return SmallString<32>(&N[0], &N[End]);
}

static std::string toSymbol(StringRef S) { return ("$" + S).str(); }

static std::string toString(const APFloat &FP) {
  static const size_t BufBytes = 128;
  char buf[BufBytes];
  if (FP.isNaN())
    assert((FP.bitwiseIsEqual(APFloat::getQNaN(FP.getSemantics())) ||
            FP.bitwiseIsEqual(
                APFloat::getQNaN(FP.getSemantics(), /*Negative=*/true))) &&
           "convertToHexString handles neither SNaN nor NaN payloads");
  // Use C99's hexadecimal floating-point representation.
  auto Written = FP.convertToHexString(
      buf, /*hexDigits=*/0, /*upperCase=*/false, APFloat::rmNearestTiesToEven);
  (void)Written;
  assert(Written != 0);
  assert(Written < BufBytes);
  return buf;
}

static const char *toString(const Type *Ty, bool hasAddr64) {
  switch (Ty->getTypeID()) {
  default: break;
  // Treat all pointers as the underlying integer into linear memory.
  case Type::PointerTyID: return hasAddr64 ? "i64" : "i32";
  case Type::FloatTyID:  return "f32";
  case Type::DoubleTyID: return "f64";
  case Type::IntegerTyID:
    switch (Ty->getIntegerBitWidth()) {
    case 8: return "i8";
    case 16: return "i16";
    case 32: return "i32";
    case 64: return "i64";
    default: break;
    }
  }
  DEBUG(dbgs() << "Invalid type "; Ty->print(dbgs()); dbgs() << '\n');
  llvm_unreachable("invalid type");
  return "<invalid>";
}


//===----------------------------------------------------------------------===//
// WebAssemblyAsmPrinter Implementation.
//===----------------------------------------------------------------------===//

void WebAssemblyAsmPrinter::EmitGlobalVariable(const GlobalVariable *GV) {
  SmallString<128> Str;
  raw_svector_ostream OS(Str);
  StringRef Name = GV->getName();
  DEBUG(dbgs() << "Global " << Name << '\n');

  if (!GV->hasInitializer()) {
    DEBUG(dbgs() << "  Skipping declaration.\n");
    return;
  }

  // Check to see if this is a special global used by LLVM.
  static const char *Ignored[] = {"llvm.used", "llvm.metadata"};
  for (const char *I : Ignored)
    if (Name == I)
      return;
  // FIXME: Handle the following globals.
  static const char *Unhandled[] = {"llvm.global_ctors", "llvm.global_dtors"};
  for (const char *U : Unhandled)
    if (Name == U)
      report_fatal_error("Unhandled global");
  if (Name.startswith("llvm."))
    report_fatal_error("Unknown LLVM-internal global");

  if (GV->isThreadLocal())
    report_fatal_error("TLS isn't yet supported by WebAssembly");

  const DataLayout &DL = getDataLayout();
  const Constant *Init = GV->getInitializer();
  if (isa<UndefValue>(Init))
    Init = Constant::getNullValue(Init->getType());
  unsigned Align = DL.getPrefTypeAlignment(Init->getType());

  switch (GV->getLinkage()) {
  case GlobalValue::InternalLinkage:
  case GlobalValue::PrivateLinkage:
    break;
  case GlobalValue::AppendingLinkage:
  case GlobalValue::LinkOnceAnyLinkage:
  case GlobalValue::LinkOnceODRLinkage:
  case GlobalValue::WeakAnyLinkage:
  case GlobalValue::WeakODRLinkage:
  case GlobalValue::ExternalLinkage:
  case GlobalValue::CommonLinkage:
    report_fatal_error("Linkage types other than internal and private aren't "
                       "supported by WebAssembly yet");
  default:
    llvm_unreachable("Unknown linkage type");
    return;
  }

  OS << "(global " << toSymbol(Name) << ' '
     << toString(Init->getType(), hasAddr64) << ' ';
  if (const auto *C = dyn_cast<ConstantInt>(Init)) {
    assert(C->getBitWidth() <= 64 && "Printing wider types unimplemented");
    OS << C->getZExtValue();
  } else if (const auto *C = dyn_cast<ConstantFP>(Init)) {
    OS << toString(C->getValueAPF());
  } else {
    assert(false && "Only integer and floating-point constants are supported");
  }
  OS << ") ;; align " << Align;
  OutStreamer->EmitRawText(OS.str());
}

void WebAssemblyAsmPrinter::EmitConstantPool() {
  assert(MF->getConstantPool()->getConstants().empty() &&
         "WebAssembly disables constant pools");
}

void WebAssemblyAsmPrinter::EmitFunctionEntryLabel() {
  SmallString<128> Str;
  raw_svector_ostream OS(Str);

  CurrentFnSym->redefineIfPossible();

  // The function label could have already been emitted if two symbols end up
  // conflicting due to asm renaming.  Detect this and emit an error.
  if (CurrentFnSym->isVariable())
    report_fatal_error("'" + Twine(CurrentFnSym->getName()) +
                       "' is a protected alias");
  if (CurrentFnSym->isDefined())
    report_fatal_error("'" + Twine(CurrentFnSym->getName()) +
                       "' label emitted multiple times to assembly file");

  OS << "(func " << toSymbol(CurrentFnSym->getName());
  OutStreamer->EmitRawText(OS.str());
}

void WebAssemblyAsmPrinter::EmitFunctionBodyStart() {
  SmallString<128> Str;
  raw_svector_ostream OS(Str);
  const Function *F = MF->getFunction();
  const Type *Rt = F->getReturnType();
  if (!Rt->isVoidTy() || !F->arg_empty()) {
    for (const Argument &A : F->args())
      OS << " (param " << toString(A.getType(), hasAddr64) << ')';
    if (!Rt->isVoidTy())
      OS << " (result " << toString(Rt, hasAddr64) << ')';
    OutStreamer->EmitRawText(OS.str());
  }
}

void WebAssemblyAsmPrinter::EmitFunctionBodyEnd() {
  SmallString<128> Str;
  raw_svector_ostream OS(Str);
  OS << ") ;; end func " << toSymbol(CurrentFnSym->getName());
  OutStreamer->EmitRawText(OS.str());
}

void WebAssemblyAsmPrinter::EmitInstruction(const MachineInstr *MI) {
  DEBUG(dbgs() << "EmitInstruction: " << *MI << '\n');
  SmallString<128> Str;
  raw_svector_ostream OS(Str);

  unsigned NumDefs = MI->getDesc().getNumDefs();
  assert(NumDefs <= 1 &&
         "Instructions with multiple result values not implemented");

  OS << '\t';

  if (NumDefs != 0) {
    const MachineOperand &MO = MI->getOperand(0);
    unsigned Reg = MO.getReg();
    OS << "(setlocal @" << TargetRegisterInfo::virtReg2Index(Reg) << ' ';
  }

  if (MI->getOpcode() == WebAssembly::COPY) {
    OS << '@' << TargetRegisterInfo::virtReg2Index(MI->getOperand(1).getReg());
  } else {
    OS << '(' << OpcodeName(TII, MI);
    for (const MachineOperand &MO : MI->uses())
      switch (MO.getType()) {
      default:
        llvm_unreachable("unexpected machine operand type");
      case MachineOperand::MO_Register: {
        if (MO.isImplicit())
          continue;
        unsigned Reg = MO.getReg();
        OS << " @" << TargetRegisterInfo::virtReg2Index(Reg);
      } break;
      case MachineOperand::MO_Immediate: {
        OS << ' ' << MO.getImm();
      } break;
      case MachineOperand::MO_FPImmediate: {
        OS << ' ' << toString(MO.getFPImm()->getValueAPF());
      } break;
      case MachineOperand::MO_GlobalAddress: {
        OS << ' ' << toSymbol(MO.getGlobal()->getName());
      } break;
      }
    OS << ')';
  }

  if (NumDefs != 0)
    OS << ')';

  OutStreamer->EmitRawText(OS.str());
}

// Force static initialization.
extern "C" void LLVMInitializeWebAssemblyAsmPrinter() {
  RegisterAsmPrinter<WebAssemblyAsmPrinter> X(TheWebAssemblyTarget32);
  RegisterAsmPrinter<WebAssemblyAsmPrinter> Y(TheWebAssemblyTarget64);
}
