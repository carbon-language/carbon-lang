//===-- SparcV9InstrInfo.h - Define TargetInstrInfo for SparcV9 -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class contains information about individual instructions.
// Also see the SparcV9MachineInstrDesc array, which can be found in
// SparcV9TargetMachine.cpp.
// Other information is computed on demand, and most such functions
// default to member functions in base class TargetInstrInfo.
//
//===----------------------------------------------------------------------===//

#ifndef SPARCV9INSTRINFO_H
#define SPARCV9INSTRINFO_H

#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "SparcV9Internals.h"
#include "SparcV9RegisterInfo.h"

namespace llvm {

/// SparcV9InstrInfo - TargetInstrInfo specialized for the SparcV9 target.
///
struct SparcV9InstrInfo : public TargetInstrInfo {
  const SparcV9RegisterInfo RI;
public:
  SparcV9InstrInfo()
    : TargetInstrInfo(SparcV9MachineInstrDesc, V9::NUM_TOTAL_OPCODES) { }

  /// getRegisterInfo - TargetInstrInfo is a superset of MRegister info.  As
  /// such, whenever a client has an instance of instruction info, it should
  /// always be able to get register info as well (through this method).
  ///
  virtual const MRegisterInfo &getRegisterInfo() const { return RI; }

  // All immediate constants are in position 1 except the
  // store instructions and SETxx.
  //
  virtual int getImmedConstantPos(MachineOpCode opCode) const {
    bool ignore;
    if (this->maxImmedConstant(opCode, ignore) != 0) {
      // 1st store opcode
      assert(! this->isStore((MachineOpCode) V9::STBr - 1));
      // last store opcode
      assert(! this->isStore((MachineOpCode) V9::STXFSRi + 1));

      if (opCode == V9::SETHI)
        return 0;
      if (opCode >= V9::STBr && opCode <= V9::STXFSRi)
        return 2;
      return 1;
    }
    else
      return -1;
  }

  virtual bool hasResultInterlock(MachineOpCode opCode) const
  {
    // All UltraSPARC instructions have interlocks (note that delay slots
    // are not considered here).
    // However, instructions that use the result of an FCMP produce a
    // 9-cycle stall if they are issued less than 3 cycles after the FCMP.
    // Force the compiler to insert a software interlock (i.e., gap of
    // 2 other groups, including NOPs if necessary).
    return (opCode == V9::FCMPS || opCode == V9::FCMPD || opCode == V9::FCMPQ);
  }
};

} // End llvm namespace

#endif
