//===-- llvm/Target/Machine.h - General Target Information -------*- C++ -*-==//
//
// This file describes the general parts of a Target machine.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETMACHINE_H
#define LLVM_TARGET_TARGETMACHINE_H

#include "llvm/Target/TargetData.h"
#include "llvm/Support/NonCopyable.h"
#include <string>

class TargetMachine;
class MachineInstrInfo;
class MachineInstrDescriptor;
class MachineSchedInfo;
class MachineRegInfo;

//---------------------------------------------------------------------------
// Data types used to define information about a single machine instruction
//---------------------------------------------------------------------------

typedef int MachineOpCode;
typedef int OpCodeMask;


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
public:
  const string     TargetName;
  const TargetData DataLayout;		// Calculates type size & alignment
  int              optSizeForSubWordData;
  int	           minMemOpWordSize;
  int	           maxAtomicMemOpWordSize;
  
protected:
  TargetMachine(const string &targetname, // Can only create subclasses...
		unsigned char PtrSize = 8, unsigned char PtrAl = 8,
		unsigned char DoubleAl = 8, unsigned char FloatAl = 4,
		unsigned char LongAl = 8, unsigned char IntAl = 4,
		unsigned char ShortAl = 2, unsigned char ByteAl = 1)
    : TargetName(targetname), DataLayout(targetname, PtrSize, PtrAl,
					 DoubleAl, FloatAl, LongAl, IntAl, 
					 ShortAl, ByteAl) { }
public:
  virtual ~TargetMachine() {}
  
  // 
  // Interfaces to the major aspects of target machine information:
  // -- Instruction opcode and operand information
  // -- Pipelines and scheduling information
  // -- Register information
  // 
  virtual const MachineInstrInfo&	getInstrInfo() const = 0;

  virtual const MachineSchedInfo&	getSchedInfo() const = 0;
  
  virtual const MachineRegInfo&		getRegInfo()   const = 0;
  
  //
  // Data storage information
  // 
  virtual unsigned int	findOptimalStorageSize	(const Type* ty) const;
  
  //
  // compileMethod - Everything neccesary to compile a method into the
  // built in representation.  This allows the target to have complete control
  // over how it does compilation.  This does not emit assembly or output
  // machine code, however; those are done later.
  //
  virtual bool compileMethod(Method *M) = 0;

  //
  // emitAssembly - Output assembly language code (a .s file) for the specified
  // method. The specified method must have been compiled before this may be
  // used.
  //
  virtual void emitAssembly(Method *M, ostream &OutStr) {  /* todo */ }
};

#endif
