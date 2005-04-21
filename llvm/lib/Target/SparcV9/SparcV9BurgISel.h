//===-- SparcV9BurgISel.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Global functions exposed by the BURG-based instruction selector
// for the SparcV9 target.
//
//===----------------------------------------------------------------------===//

#ifndef SPARCV9BURGISEL_H
#define SPARCV9BURGISEL_H

//#include "llvm/DerivedTypes.h"
//#include "llvm/Instruction.h"
//#include "SparcV9Internals.h"

namespace llvm {

class Constant;
class Instruction;
class TargetMachine;
class Function;
class Value;
class MachineInstr;
class MachineCodeForInstruction;
class FunctionPass;

/// ConstantMayNotFitInImmedField - Test if this constant may not fit in the
/// immediate field of the machine instructions (probably) generated for this
/// instruction.
///
bool ConstantMayNotFitInImmedField (const Constant *CV, const Instruction *I);

/// CreateCodeToLoadConst - Create an instruction sequence to put the
/// constant `val' into the virtual register `dest'.  `val' may be a Constant
/// or a GlobalValue, viz., the constant address of a global variable or
/// function.  The generated instructions are returned in `mvec'.  Any temp.
/// registers (TmpInstruction) created are recorded in mcfi.
///
void CreateCodeToLoadConst (const TargetMachine &target, Function *F,
  Value *val, Instruction *dest, std::vector<MachineInstr*> &mvec,
  MachineCodeForInstruction &mcfi);

/// createSparcV9BurgInstSelector - Creates and returns a new SparcV9
/// BURG-based instruction selection pass.
///
FunctionPass *createSparcV9BurgInstSelector(TargetMachine &TM);

} // End llvm namespace

#endif
