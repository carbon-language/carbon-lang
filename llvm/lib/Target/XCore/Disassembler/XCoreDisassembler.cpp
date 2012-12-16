//===- XCoreDisassembler.cpp - Disassembler for XCore -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is part of the XCore Disassembler.
//
//===----------------------------------------------------------------------===//

#include "XCore.h"
#include "XCoreRegisterInfo.h"
#include "llvm/MC/MCDisassembler.h"
#include "llvm/MC/MCFixedLenDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/MemoryObject.h"
#include "llvm/Support/TargetRegistry.h"

using namespace llvm;

typedef MCDisassembler::DecodeStatus DecodeStatus;

namespace {

/// XCoreDisassembler - a disasembler class for XCore.
class XCoreDisassembler : public MCDisassembler {
  const MCRegisterInfo *RegInfo;
public:
  /// Constructor     - Initializes the disassembler.
  ///
  XCoreDisassembler(const MCSubtargetInfo &STI, const MCRegisterInfo *Info) :
    MCDisassembler(STI), RegInfo(Info) {}

  /// getInstruction - See MCDisassembler.
  virtual DecodeStatus getInstruction(MCInst &instr,
                                      uint64_t &size,
                                      const MemoryObject &region,
                                      uint64_t address,
                                      raw_ostream &vStream,
                                      raw_ostream &cStream) const;

  const MCRegisterInfo *getRegInfo() const { return RegInfo; }
};
}

static bool readInstruction16(const MemoryObject &region,
                              uint64_t address,
                              uint64_t &size,
                              uint16_t &insn) {
  uint8_t Bytes[4];

  // We want to read exactly 2 Bytes of data.
  if (region.readBytes(address, 2, Bytes, NULL) == -1) {
    size = 0;
    return false;
  }
  // Encoded as a little-endian 16-bit word in the stream.
  insn = (Bytes[0] <<  0) | (Bytes[1] <<  8);
  return true;
}

static unsigned getReg(const void *D, unsigned RC, unsigned RegNo) {
  const XCoreDisassembler *Dis = static_cast<const XCoreDisassembler*>(D);
  return *(Dis->getRegInfo()->getRegClass(RC).begin() + RegNo);
}


static DecodeStatus DecodeGRRegsRegisterClass(MCInst &Inst,
                                              unsigned RegNo,
                                              uint64_t Address,
                                              const void *Decoder);

#include "XCoreGenDisassemblerTables.inc"

static DecodeStatus DecodeGRRegsRegisterClass(MCInst &Inst,
                                              unsigned RegNo,
                                              uint64_t Address,
                                              const void *Decoder)
{
  if (RegNo > 11)
    return MCDisassembler::Fail;
  unsigned Reg = getReg(Decoder, XCore::GRRegsRegClassID, RegNo);
  Inst.addOperand(MCOperand::CreateReg(Reg));
  return MCDisassembler::Success;
}

MCDisassembler::DecodeStatus
XCoreDisassembler::getInstruction(MCInst &instr,
                                  uint64_t &Size,
                                  const MemoryObject &Region,
                                  uint64_t Address,
                                  raw_ostream &vStream,
                                  raw_ostream &cStream) const {
  uint16_t low;

  if (!readInstruction16(Region, Address, Size, low)) {
    return Fail;
  }

  // Calling the auto-generated decoder function.
  DecodeStatus Result = decodeInstruction(DecoderTable16, instr, low, Address,
                             this, STI);
  if (Result != Fail) {
    Size = 2;
    return Result;
  }

  return Fail;
}

namespace llvm {
  extern Target TheXCoreTarget;
}

static MCDisassembler *createXCoreDisassembler(const Target &T,
                                               const MCSubtargetInfo &STI) {
  return new XCoreDisassembler(STI, T.createMCRegInfo(""));
}

extern "C" void LLVMInitializeXCoreDisassembler() {
  // Register the disassembler.
  TargetRegistry::RegisterMCDisassembler(TheXCoreTarget,
                                         createXCoreDisassembler);
}
