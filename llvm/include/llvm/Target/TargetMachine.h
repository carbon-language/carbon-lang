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

class TargetInstrInfo;
class TargetInstrDescriptor;
class TargetSchedInfo;
class TargetRegInfo;
class TargetFrameInfo;
class TargetCacheInfo;
class MachineCodeEmitter;
class MRegisterInfo;
class FunctionPassManager;
class PassManager;
class Pass;

//===----------------------------------------------------------------------===//
///
/// TargetMachine - Primary interface to the complete machine description for
/// the target machine.  All target-specific information should be accessible
/// through this interface.
/// 
class TargetMachine {
  const std::string Name;
  const TargetData DataLayout;		 // Calculates type size & alignment
  
  TargetMachine(const TargetMachine&);   // DO NOT IMPLEMENT
  void operator=(const TargetMachine&);  // DO NOT IMPLEMENT
protected:
  TargetMachine(const std::string &name, // Can only create subclasses...
		bool LittleEndian = false,
		unsigned char PtrSize = 8, unsigned char PtrAl = 8,
		unsigned char DoubleAl = 8, unsigned char FloatAl = 4,
		unsigned char LongAl = 8, unsigned char IntAl = 4,
		unsigned char ShortAl = 2, unsigned char ByteAl = 1)
    : Name(name), DataLayout(name, LittleEndian,
			     PtrSize, PtrAl, DoubleAl, FloatAl, LongAl,
                             IntAl, ShortAl, ByteAl) {}
public:
  virtual ~TargetMachine() {}

  const std::string &getName() const { return Name; }
  
  // Interfaces to the major aspects of target machine information:
  // -- Instruction opcode and operand information
  // -- Pipelines and scheduling information
  // -- Register information
  // -- Stack frame information
  // -- Cache hierarchy information
  // -- Machine-level optimization information (peephole only)
  // 
  virtual const TargetInstrInfo&        getInstrInfo() const = 0;
  virtual const TargetSchedInfo&        getSchedInfo() const = 0;
  virtual const TargetRegInfo&          getRegInfo()   const = 0;
  virtual const TargetFrameInfo&        getFrameInfo() const = 0;
  virtual const TargetCacheInfo&        getCacheInfo() const = 0;
  const TargetData &getTargetData() const { return DataLayout; }

  /// getRegisterInfo - If register information is available, return it.  If
  /// not, return null.  This is kept separate from RegInfo until RegInfo has
  /// details of graph coloring register allocation removed from it.
  ///
  virtual const MRegisterInfo*          getRegisterInfo() const { return 0; }

  // Data storage information
  // 
  virtual unsigned findOptimalStorageSize(const Type* ty) const;
  
  /// addPassesToJITCompile - Add passes to the specified pass manager to
  /// implement a fast dynamic compiler for this target.  Return true if this is
  /// not supported for this target.
  ///
  virtual bool addPassesToJITCompile(FunctionPassManager &PM) { return true; }

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

  /// replaceMachineCodeForFunction - Make it so that calling the
  /// function whose machine code is at OLD turns into a call to NEW,
  /// perhaps by overwriting OLD with a branch to NEW. FIXME: this is
  /// JIT-specific.
  ///
  virtual void replaceMachineCodeForFunction (void *Old, void *New) {
    assert (0 && "Current target cannot replace machine code for functions");
  }
};

#endif
