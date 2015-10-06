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
#include "llvm/ADT/StringExtras.h"
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
  const WebAssemblyInstrInfo *TII;
  unsigned NumArgs;

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
    TII = Subtarget.getInstrInfo();
    NumArgs = MF.getInfo<WebAssemblyFunctionInfo>()->getNumArguments();
    return AsmPrinter::runOnMachineFunction(MF);
  }

  //===------------------------------------------------------------------===//
  // AsmPrinter Implementation.
  //===------------------------------------------------------------------===//

  void EmitJumpTableInfo() override;
  void EmitConstantPool() override;
  void EmitFunctionBodyStart() override;

  void EmitInstruction(const MachineInstr *MI) override;

  static std::string toString(const APFloat &APF);
  const char *toString(Type *Ty) const;
  std::string regToString(unsigned RegNo);
  std::string argToString(unsigned ArgNo);
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

std::string WebAssemblyAsmPrinter::toString(const APFloat &FP) {
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

std::string WebAssemblyAsmPrinter::regToString(unsigned RegNo) {
  if (TargetRegisterInfo::isPhysicalRegister(RegNo))
    return WebAssemblyInstPrinter::getRegisterName(RegNo);

  // WebAssembly arguments and local variables are in the same index space, and
  // there are no explicit varargs, so we just add the number of arguments to
  // the virtual register number to get the local variable number.
  return '@' + utostr(TargetRegisterInfo::virtReg2Index(RegNo) + NumArgs);
}

std::string WebAssemblyAsmPrinter::argToString(unsigned ArgNo) {
  // Same as above, but we don't need to add NumArgs here.
  return '@' + utostr(ArgNo);
}

const char *WebAssemblyAsmPrinter::toString(Type *Ty) const {
  switch (Ty->getTypeID()) {
  default:
    break;
  // Treat all pointers as the underlying integer into linear memory.
  case Type::PointerTyID:
    switch (getPointerSize()) {
    case 4:
      return "i32";
    case 8:
      return "i64";
    default:
      llvm_unreachable("unsupported pointer size");
    }
    break;
  case Type::FloatTyID:
    return "f32";
  case Type::DoubleTyID:
    return "f64";
  case Type::IntegerTyID:
    switch (Ty->getIntegerBitWidth()) {
    case 8:
      return "i8";
    case 16:
      return "i16";
    case 32:
      return "i32";
    case 64:
      return "i64";
    default:
      break;
    }
  }
  DEBUG(dbgs() << "Invalid type "; Ty->print(dbgs()); dbgs() << '\n');
  llvm_unreachable("invalid type");
  return "<invalid>";
}

//===----------------------------------------------------------------------===//
// WebAssemblyAsmPrinter Implementation.
//===----------------------------------------------------------------------===//

void WebAssemblyAsmPrinter::EmitConstantPool() {
  assert(MF->getConstantPool()->getConstants().empty() &&
         "WebAssembly disables constant pools");
}

void WebAssemblyAsmPrinter::EmitJumpTableInfo() {
  // Nothing to do; jump tables are incorporated into the instruction stream.
}

void WebAssemblyAsmPrinter::EmitFunctionBodyStart() {
  const Function *F = MF->getFunction();
  Type *Rt = F->getReturnType();

  if (!Rt->isVoidTy() || !F->arg_empty()) {
    SmallString<128> Str;
    raw_svector_ostream OS(Str);
    bool First = true;
    for (const Argument &A : F->args()) {
      OS << (First ? "" : "\n") << "\t"
                                   ".param "
         << toString(A.getType());
      First = false;
    }
    if (!Rt->isVoidTy()) {
      OS << (First ? "" : "\n") << "\t"
                                   ".result "
         << toString(Rt);
      First = false;
    }
    OutStreamer->EmitRawText(OS.str());
  }

  AsmPrinter::EmitFunctionBodyStart();
}

void WebAssemblyAsmPrinter::EmitInstruction(const MachineInstr *MI) {
  DEBUG(dbgs() << "EmitInstruction: " << *MI << '\n');
  SmallString<128> Str;
  raw_svector_ostream OS(Str);

  unsigned NumDefs = MI->getDesc().getNumDefs();
  assert(NumDefs <= 1 &&
         "Instructions with multiple result values not implemented");

  OS << '\t';

  switch (MI->getOpcode()) {
  case TargetOpcode::COPY:
    OS << regToString(MI->getOperand(1).getReg());
    break;
  case WebAssembly::GLOBAL:
    // TODO: wasm64
    OS << "i32.const " << toSymbol(MI->getOperand(1).getGlobal()->getName());
    break;
  case WebAssembly::ARGUMENT_I32:
  case WebAssembly::ARGUMENT_I64:
  case WebAssembly::ARGUMENT_F32:
  case WebAssembly::ARGUMENT_F64:
    OS << argToString(MI->getOperand(1).getImm());
    break;
  case WebAssembly::Immediate_I32:
    OS << "i32.const " << MI->getOperand(1).getImm();
    break;
  case WebAssembly::Immediate_I64:
    OS << "i64.const " << MI->getOperand(1).getImm();
    break;
  case WebAssembly::Immediate_F32:
    OS << "f32.const " << toString(MI->getOperand(1).getFPImm()->getValueAPF());
    break;
  case WebAssembly::Immediate_F64:
    OS << "f64.const " << toString(MI->getOperand(1).getFPImm()->getValueAPF());
    break;
  default: {
    OS << OpcodeName(TII, MI);
    bool NeedComma = false;
    for (const MachineOperand &MO : MI->uses()) {
      if (MO.isReg() && MO.isImplicit())
        continue;
      if (NeedComma)
        OS << ',';
      NeedComma = true;
      OS << ' ';
      switch (MO.getType()) {
      default:
        llvm_unreachable("unexpected machine operand type");
      case MachineOperand::MO_Register:
        OS << regToString(MO.getReg());
        break;
      case MachineOperand::MO_Immediate:
        OS << MO.getImm();
        break;
      case MachineOperand::MO_FPImmediate:
        OS << toString(MO.getFPImm()->getValueAPF());
        break;
      case MachineOperand::MO_GlobalAddress:
        OS << toSymbol(MO.getGlobal()->getName());
        break;
      case MachineOperand::MO_MachineBasicBlock:
        OS << toSymbol(MO.getMBB()->getSymbol()->getName());
        break;
      }
    }
    break;
  }
  }

  OutStreamer->EmitRawText(OS.str());

  if (NumDefs != 0) {
    SmallString<128> Str;
    raw_svector_ostream OS(Str);
    OS << "\t" "set_local "
       << regToString(MI->getOperand(0).getReg()) << ", "
          "pop";
    OutStreamer->EmitRawText(OS.str());
  }
}

// Force static initialization.
extern "C" void LLVMInitializeWebAssemblyAsmPrinter() {
  RegisterAsmPrinter<WebAssemblyAsmPrinter> X(TheWebAssemblyTarget32);
  RegisterAsmPrinter<WebAssemblyAsmPrinter> Y(TheWebAssemblyTarget64);
}
