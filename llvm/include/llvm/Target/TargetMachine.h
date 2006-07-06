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

#include "llvm/Target/TargetInstrItineraries.h"
#include <cassert>
#include <string>

namespace llvm {

class TargetData;
class TargetSubtarget;
class TargetInstrInfo;
class TargetInstrDescriptor;
class TargetJITInfo;
class TargetLowering;
class TargetFrameInfo;
class MachineCodeEmitter;
class MRegisterInfo;
class Module;
class FunctionPassManager;
class PassManager;
class Pass;

// Relocation model types.
namespace Reloc {
  enum Model {
    Default,
    Static,
    PIC,
    DynamicNoPIC
  };
}

// Code model types.
namespace CodeModel {
  enum Model {
    Default,
    Small,
    Kernel,
    Medium,
    Large
  };
}

//===----------------------------------------------------------------------===//
///
/// TargetMachine - Primary interface to the complete machine description for
/// the target machine.  All target-specific information should be accessible
/// through this interface.
///
class TargetMachine {
  const std::string Name;

  TargetMachine(const TargetMachine&);   // DO NOT IMPLEMENT
  void operator=(const TargetMachine&);  // DO NOT IMPLEMENT
protected: // Can only create subclasses...
  TargetMachine(const std::string &name) : Name(name) { };

  /// This constructor is used for targets that support arbitrary TargetData
  /// layouts, like the C backend.  It initializes the TargetData to match that
  /// of the specified module.
  ///
  TargetMachine(const std::string &name, const Module &M);

  /// getSubtargetImpl - virtual method implemented by subclasses that returns
  /// a reference to that target's TargetSubtarget-derived member variable.
  virtual const TargetSubtarget *getSubtargetImpl() const { return 0; }
public:
  virtual ~TargetMachine();

  /// getModuleMatchQuality - This static method should be implemented by
  /// targets to indicate how closely they match the specified module.  This is
  /// used by the LLC tool to determine which target to use when an explicit
  /// -march option is not specified.  If a target returns zero, it will never
  /// be chosen without an explicit -march option.
  static unsigned getModuleMatchQuality(const Module &M) { return 0; }

  /// getJITMatchQuality - This static method should be implemented by targets
  /// that provide JIT capabilities to indicate how suitable they are for
  /// execution on the current host.  If a value of 0 is returned, the target
  /// will not be used unless an explicit -march option is used.
  static unsigned getJITMatchQuality() { return 0; }


  const std::string &getName() const { return Name; }

  // Interfaces to the major aspects of target machine information:
  // -- Instruction opcode and operand information
  // -- Pipelines and scheduling information
  // -- Stack frame information
  // -- Selection DAG lowering information
  //
  virtual const TargetInstrInfo        *getInstrInfo() const { return 0; }
  virtual const TargetFrameInfo        *getFrameInfo() const { return 0; }
  virtual       TargetLowering    *getTargetLowering() const { return 0; }
  virtual const TargetData            *getTargetData() const { return 0; }

  /// getSubtarget - This method returns a pointer to the specified type of
  /// TargetSubtarget.  In debug builds, it verifies that the object being
  /// returned is of the correct type.
  template<typename STC> const STC &getSubtarget() const {
    const TargetSubtarget *TST = getSubtargetImpl();
    assert(TST && dynamic_cast<const STC*>(TST) &&
           "Not the right kind of subtarget!");
    return *static_cast<const STC*>(TST);
  }

  /// getRegisterInfo - If register information is available, return it.  If
  /// not, return null.  This is kept separate from RegInfo until RegInfo has
  /// details of graph coloring register allocation removed from it.
  ///
  virtual const MRegisterInfo*          getRegisterInfo() const { return 0; }

  /// getJITInfo - If this target supports a JIT, return information for it,
  /// otherwise return null.
  ///
  virtual TargetJITInfo *getJITInfo() { return 0; }
  
  /// getInstrItineraryData - Returns instruction itinerary data for the target
  /// or specific subtarget.
  ///
  virtual const InstrItineraryData getInstrItineraryData() const {  
    return InstrItineraryData();
  }

  /// getRelocationModel - Returns the code generation relocation model. The
  /// choices are static, PIC, and dynamic-no-pic, and target default.
  static Reloc::Model getRelocationModel();

  /// setRelocationModel - Sets the code generation relocation model.
  static void setRelocationModel(Reloc::Model Model);

  /// getCodeModel - Returns the code model. The choices are small, kernel,
  /// medium, large, and target default.
  static CodeModel::Model getCodeModel();

  /// setCodeModel - Sets the code model.
  static void setCodeModel(CodeModel::Model Model);

  /// CodeGenFileType - These enums are meant to be passed into
  /// addPassesToEmitFile to indicate what type of file to emit.
  enum CodeGenFileType {
    AssemblyFile, ObjectFile, DynamicLibrary
  };

  /// addPassesToEmitFile - Add passes to the specified pass manager to get
  /// the specified file emitted.  Typically this will involve several steps of
  /// code generation.  If Fast is set to true, the code generator should emit
  /// code as fast as possible, without regard for compile time.  This method
  /// should return true if emission of this file type is not supported.
  ///
  virtual bool addPassesToEmitFile(PassManager &PM, std::ostream &Out,
                                   CodeGenFileType FileType, bool Fast) {
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
