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
public:
  const std::string TargetName;
  const TargetData DataLayout;		// Calculates type size & alignment
  int              optSizeForSubWordData;
  int	           minMemOpWordSize;
  int	           maxAtomicMemOpWordSize;
  
protected:
  TargetMachine(const std::string &targetname, // Can only create subclasses...
                unsigned char IntRegSize = 8,
		unsigned char PtrSize = 8, unsigned char PtrAl = 8,
		unsigned char DoubleAl = 8, unsigned char FloatAl = 4,
		unsigned char LongAl = 8, unsigned char IntAl = 4,
		unsigned char ShortAl = 2, unsigned char ByteAl = 1)
    : TargetName(targetname), DataLayout(targetname, IntRegSize,
                                         PtrSize, PtrAl,
					 DoubleAl, FloatAl, LongAl, IntAl, 
					 ShortAl, ByteAl) { }
public:
  virtual ~TargetMachine() {}
  
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

  // Data storage information
  // 
  virtual unsigned int	findOptimalStorageSize	(const Type* ty) const;
  
  /// addPassesToEmitAssembly - Add passes to the specified pass manager to get
  /// assembly langage code emited.  Typically this will involve several steps
  /// of code generation.
  ///
  virtual void addPassesToEmitAssembly(PassManager &PM, std::ostream &Out) = 0;

  /// getPrologEpilogCodeInserter - Create pass to insert prolog/epilog code.
  /// 
  virtual Pass* getPrologEpilogInsertionPass() = 0;

  /// getFunctionAsmPrinterPass - Create a pass to write out the generated
  /// machine code for a single function to the generated assembly file.
  /// 
  virtual Pass* getFunctionAsmPrinterPass(std::ostream &Out) = 0;

  /// getModuleAsmPrinterPass - Create a pass to write out module-level
  /// information to the generated assembly file.
  /// 
  virtual Pass* getModuleAsmPrinterPass(std::ostream &Out) = 0;

  /// getEmitBytecodeToAsmPass - Create a pass to emit the final LLVM bytecode
  /// to the generated assembly file.
  /// 
  virtual Pass* getEmitBytecodeToAsmPass(std::ostream &Out) = 0;
};

#endif
