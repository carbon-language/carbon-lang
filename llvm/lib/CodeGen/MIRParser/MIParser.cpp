//===- MIParser.cpp - Machine instructions parser implementation ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the parsing of machine instructions.
//
//===----------------------------------------------------------------------===//

#include "MIParser.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include "llvm/Target/TargetInstrInfo.h"

using namespace llvm;

namespace {

class MIParser {
  SourceMgr &SM;
  MachineFunction &MF;
  SMDiagnostic &Error;
  StringRef Source;
  /// Maps from instruction names to op codes.
  StringMap<unsigned> Names2InstrOpCodes;

public:
  MIParser(SourceMgr &SM, MachineFunction &MF, SMDiagnostic &Error,
           StringRef Source);

  /// Report an error at the current location with the given message.
  ///
  /// This function always return true.
  bool error(const Twine &Msg);

  MachineInstr *parse();

private:
  void initNames2InstrOpCodes();

  /// Try to convert an instruction name to an opcode. Return true if the
  /// instruction name is invalid.
  bool parseInstrName(StringRef InstrName, unsigned &OpCode);
};

} // end anonymous namespace

MIParser::MIParser(SourceMgr &SM, MachineFunction &MF, SMDiagnostic &Error,
                   StringRef Source)
    : SM(SM), MF(MF), Error(Error), Source(Source) {}

bool MIParser::error(const Twine &Msg) {
  // TODO: Get the proper location in the MIR file, not just a location inside
  // the string.
  Error =
      SMDiagnostic(SM, SMLoc(), SM.getMemoryBuffer(SM.getMainFileID())
                                    ->getBufferIdentifier(),
                   1, 0, SourceMgr::DK_Error, Msg.str(), Source, None, None);
  return true;
}

MachineInstr *MIParser::parse() {
  StringRef InstrName = Source;
  unsigned OpCode;
  if (parseInstrName(InstrName, OpCode)) {
    error(Twine("unknown machine instruction name '") + InstrName + "'");
    return nullptr;
  }

  // TODO: Parse the rest of instruction - machine operands, etc.
  const auto &MCID = MF.getSubtarget().getInstrInfo()->get(OpCode);
  auto *MI = MF.CreateMachineInstr(MCID, DebugLoc());
  return MI;
}

void MIParser::initNames2InstrOpCodes() {
  if (!Names2InstrOpCodes.empty())
    return;
  const auto *TII = MF.getSubtarget().getInstrInfo();
  assert(TII && "Expected target instruction info");
  for (unsigned I = 0, E = TII->getNumOpcodes(); I < E; ++I)
    Names2InstrOpCodes.insert(std::make_pair(StringRef(TII->getName(I)), I));
}

bool MIParser::parseInstrName(StringRef InstrName, unsigned &OpCode) {
  initNames2InstrOpCodes();
  auto InstrInfo = Names2InstrOpCodes.find(InstrName);
  if (InstrInfo == Names2InstrOpCodes.end())
    return true;
  OpCode = InstrInfo->getValue();
  return false;
}

MachineInstr *llvm::parseMachineInstr(SourceMgr &SM, MachineFunction &MF,
                                      StringRef Src, SMDiagnostic &Error) {
  return MIParser(SM, MF, Error, Src).parse();
}
