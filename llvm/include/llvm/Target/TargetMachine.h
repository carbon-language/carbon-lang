//===-- llvm/Target/Machine.h - General Target Information -------*- C++ -*-==//
//
// This file describes the general parts of a Target machine.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETMACHINE_H
#define LLVM_TARGET_TARGETMACHINE_H

#include "llvm/Target/TargetData.h"
#include "Support/NonCopyable.h"

class MachineInstrInfo;
class MachineInstrDescriptor;
class MachineSchedInfo;
class MachineRegInfo;
class MachineFrameInfo;
class MachineCacheInfo;
class MachineOptInfo;
class MRegisterInfo;
class PassManager;
class Pass;

//---------------------------------------------------------------------------
// class TargetMachine
// 
// Purpose:
//   Primary interface to the complete machine description for the
//   target machine.  All target-specific information should be
//   accessible through this interface.
// 
//---------------------------------------------------------------------------

class TargetMachine : public NonCopyableV {
  const std::string Name;
public:
  const TargetData DataLayout;		// Calculates type size & alignment
  
protected:
  TargetMachine(const std::string &name, // Can only create subclasses...
                unsigned char SubWordSize = 1, unsigned char IntRegSize = 8,
		unsigned char PtrSize = 8, unsigned char PtrAl = 8,
		unsigned char DoubleAl = 8, unsigned char FloatAl = 4,
		unsigned char LongAl = 8, unsigned char IntAl = 4,
		unsigned char ShortAl = 2, unsigned char ByteAl = 1)
    : Name(name), DataLayout(name, SubWordSize, IntRegSize, PtrSize, PtrAl,
                             DoubleAl, FloatAl, LongAl,
                             IntAl, ShortAl, ByteAl) {}
public:
  virtual ~TargetMachine() {}

  const std::string &getName() const { return Name; }
  
  // 
  // Interfaces to the major aspects of target machine information:
  // -- Instruction opcode and operand information
  // -- Pipelines and scheduling information
  // -- Register information
  // -- Stack frame information
  // -- Cache hierarchy information
  // -- Machine-level optimization information (peephole only)
  // 
  virtual const MachineInstrInfo&       getInstrInfo() const = 0;
  virtual const MachineSchedInfo&       getSchedInfo() const = 0;
  virtual const MachineRegInfo&	        getRegInfo()   const = 0;
  virtual const MachineFrameInfo&       getFrameInfo() const = 0;
  virtual const MachineCacheInfo&       getCacheInfo() const = 0;
  virtual const MachineOptInfo&         getOptInfo()   const = 0;

  /// getRegisterInfo - If register information is available, return it.  If
  /// not, return null.  This is kept seperate from RegInfo until RegInfo gets
  /// straightened out.
  ///
  virtual const MRegisterInfo*          getRegisterInfo() const { return 0; }

  // Data storage information
  // 
  virtual unsigned findOptimalStorageSize(const Type* ty) const;
  
  /// addPassesToEmitAssembly - Add passes to the specified pass manager to get
  /// assembly langage code emited.  Typically this will involve several steps
  /// of code generation.  This method should return true if code generation is
  /// not supported.
  ///
  virtual bool addPassesToEmitAssembly(PassManager &PM, std::ostream &Out) {
    return true;
  }

  /// addPassesToJITCompile - Add passes to the specified pass manager to
  /// implement a fast dynamic compiler for this target.  Return true if this is
  /// not supported for this target.
  ///
  virtual bool addPassesToJITCompile(PassManager &PM) { return true; }
};

#endif
