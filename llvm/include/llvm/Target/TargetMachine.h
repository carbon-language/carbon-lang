//===-- llvm/Target/TargetMachine.h - Target Information --------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file describes the general parts of a Target machine.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETMACHINE_H
#define LLVM_TARGET_TARGETMACHINE_H

#include "llvm/Target/TargetData.h"
#include <cassert>

namespace llvm {

class TargetInstrInfo;
class TargetInstrDescriptor;
class TargetJITInfo;
class TargetSchedInfo;
class SparcV9RegInfo;
class TargetFrameInfo;
class MachineCodeEmitter;
class MRegisterInfo;
class FunctionPassManager;
class PassManager;
class Pass;
class IntrinsicLowering;

//===----------------------------------------------------------------------===//
///
/// TargetMachine - Primary interface to the complete machine description for
/// the target machine.  All target-specific information should be accessible
/// through this interface.
/// 
class TargetMachine {
  const std::string Name;
  const TargetData DataLayout;       // Calculates type size & alignment
  IntrinsicLowering *IL;             // Specifies how to lower intrinsic calls
  
  TargetMachine(const TargetMachine&);   // DO NOT IMPLEMENT
  void operator=(const TargetMachine&);  // DO NOT IMPLEMENT
protected: // Can only create subclasses...
  TargetMachine(const std::string &name, IntrinsicLowering *IL,                
		bool LittleEndian = false,
		unsigned char PtrSize = 8, unsigned char PtrAl = 8,
		unsigned char DoubleAl = 8, unsigned char FloatAl = 4,
		unsigned char LongAl = 8, unsigned char IntAl = 4,
		unsigned char ShortAl = 2, unsigned char ByteAl = 1);

  // This constructor is used for targets that support arbitrary TargetData
  // layouts, like the C backend.  It initializes the TargetData to match that
  // of the specified module.
  TargetMachine(const std::string &name, IntrinsicLowering *IL,
                const Module &M);
public:
  virtual ~TargetMachine();

  const std::string &getName() const { return Name; }

  // getIntrinsicLowering - This method returns a reference to an
  // IntrinsicLowering instance which should be used by the code generator to
  // lower unknown intrinsic functions to the equivalent LLVM expansion.
  IntrinsicLowering &getIntrinsicLowering() const { return *IL; }
  
  // Interfaces to the major aspects of target machine information:
  // -- Instruction opcode and operand information
  // -- Pipelines and scheduling information
  // -- Stack frame information
  // 
  virtual const TargetInstrInfo        *getInstrInfo() const { return 0; }
  virtual const TargetFrameInfo        *getFrameInfo() const { return 0; }
  const TargetData &getTargetData() const { return DataLayout; }

  /// getRegisterInfo - If register information is available, return it.  If
  /// not, return null.  This is kept separate from RegInfo until RegInfo has
  /// details of graph coloring register allocation removed from it.
  ///
  virtual const MRegisterInfo*          getRegisterInfo() const { return 0; }

  /// getJITInfo - If this target supports a JIT, return information for it,
  /// otherwise return null.
  ///
  virtual TargetJITInfo *getJITInfo() { return 0; }

  // These are deprecated interfaces.
  virtual const TargetSchedInfo        *getSchedInfo() const { return 0; }
  virtual const SparcV9RegInfo         *getRegInfo()   const { return 0; }

  /// addPassesToEmitAssembly - Add passes to the specified pass manager to get
  /// assembly langage code emitted.  Typically this will involve several steps
  /// of code generation.  This method should return true if assembly emission
  /// is not supported.
  ///
  virtual bool addPassesToEmitAssembly(PassManager &PM, std::ostream &Out) {
    return true;
  }

  /// addPassesToEmitMachineCode - Add passes to the specified pass manager to
  /// get machine code emitted.  This uses a MachineCodeEmitter object to handle
  /// actually outputting the machine code and resolving things like the address
  /// of functions.  This method should returns true if machine code emission is
  /// not supported.
  ///
  virtual bool addPassesToEmitMachineCode(FunctionPassManager &PM,
                                          MachineCodeEmitter &MCE) {
    return true;
  }
};

} // End llvm namespace

#endif
