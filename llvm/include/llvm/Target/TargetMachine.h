//===-- llvm/Target/TargetMachine.h - Target Information --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the TargetMachine and LLVMTargetMachine classes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETMACHINE_H
#define LLVM_TARGET_TARGETMACHINE_H

#include "llvm/Target/TargetInstrItineraries.h"
#include <cassert>
#include <string>

namespace llvm {

class Target;
class MCAsmInfo;
class TargetData;
class TargetSubtarget;
class TargetInstrInfo;
class TargetIntrinsicInfo;
class TargetJITInfo;
class TargetLowering;
class TargetSelectionDAGInfo;
class TargetFrameLowering;
class JITCodeEmitter;
class MCContext;
class TargetRegisterInfo;
class PassManagerBase;
class PassManager;
class Pass;
class TargetELFWriterInfo;
class formatted_raw_ostream;

// Relocation model types.
namespace Reloc {
  enum Model {
    Default,
    Static,
    PIC_,         // Cannot be named PIC due to collision with -DPIC
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

// Code generation optimization level.
namespace CodeGenOpt {
  enum Level {
    None,        // -O0
    Less,        // -O1
    Default,     // -O2, -Os
    Aggressive   // -O3
  };
}

namespace Sched {
  enum Preference {
    None,             // No preference
    Latency,          // Scheduling for shortest total latency.
    RegPressure,      // Scheduling for lowest register pressure.
    Hybrid,           // Scheduling for both latency and register pressure.
    ILP               // Scheduling for ILP in low register pressure mode.
  };
}

//===----------------------------------------------------------------------===//
///
/// TargetMachine - Primary interface to the complete machine description for
/// the target machine.  All target-specific information should be accessible
/// through this interface.
///
class TargetMachine {
  TargetMachine(const TargetMachine &);   // DO NOT IMPLEMENT
  void operator=(const TargetMachine &);  // DO NOT IMPLEMENT
protected: // Can only create subclasses.
  TargetMachine(const Target &);

  /// getSubtargetImpl - virtual method implemented by subclasses that returns
  /// a reference to that target's TargetSubtarget-derived member variable.
  virtual const TargetSubtarget *getSubtargetImpl() const { return 0; }

  /// TheTarget - The Target that this machine was created for.
  const Target &TheTarget;

  /// AsmInfo - Contains target specific asm information.
  ///
  const MCAsmInfo *AsmInfo;

  unsigned MCRelaxAll : 1;
  unsigned MCNoExecStack : 1;
  unsigned MCUseLoc : 1;

public:
  virtual ~TargetMachine();

  const Target &getTarget() const { return TheTarget; }

  // Interfaces to the major aspects of target machine information:
  // -- Instruction opcode and operand information
  // -- Pipelines and scheduling information
  // -- Stack frame information
  // -- Selection DAG lowering information
  //
  virtual const TargetInstrInfo         *getInstrInfo() const { return 0; }
  virtual const TargetFrameLowering *getFrameLowering() const { return 0; }
  virtual const TargetLowering    *getTargetLowering() const { return 0; }
  virtual const TargetSelectionDAGInfo *getSelectionDAGInfo() const{ return 0; }
  virtual const TargetData             *getTargetData() const { return 0; }

  /// getMCAsmInfo - Return target specific asm information.
  ///
  const MCAsmInfo *getMCAsmInfo() const { return AsmInfo; }

  /// getSubtarget - This method returns a pointer to the specified type of
  /// TargetSubtarget.  In debug builds, it verifies that the object being
  /// returned is of the correct type.
  template<typename STC> const STC &getSubtarget() const {
    return *static_cast<const STC*>(getSubtargetImpl());
  }

  /// getRegisterInfo - If register information is available, return it.  If
  /// not, return null.  This is kept separate from RegInfo until RegInfo has
  /// details of graph coloring register allocation removed from it.
  ///
  virtual const TargetRegisterInfo *getRegisterInfo() const { return 0; }

  /// getIntrinsicInfo - If intrinsic information is available, return it.  If
  /// not, return null.
  ///
  virtual const TargetIntrinsicInfo *getIntrinsicInfo() const { return 0; }

  /// getJITInfo - If this target supports a JIT, return information for it,
  /// otherwise return null.
  ///
  virtual TargetJITInfo *getJITInfo() { return 0; }

  /// getInstrItineraryData - Returns instruction itinerary data for the target
  /// or specific subtarget.
  ///
  virtual const InstrItineraryData *getInstrItineraryData() const {
    return 0;
  }

  /// getELFWriterInfo - If this target supports an ELF writer, return
  /// information for it, otherwise return null.
  ///
  virtual const TargetELFWriterInfo *getELFWriterInfo() const { return 0; }

  /// hasMCRelaxAll - Check whether all machine code instructions should be
  /// relaxed.
  bool hasMCRelaxAll() const { return MCRelaxAll; }

  /// setMCRelaxAll - Set whether all machine code instructions should be
  /// relaxed.
  void setMCRelaxAll(bool Value) { MCRelaxAll = Value; }

  /// hasMCNoExecStack - Check whether an executable stack is not needed.
  bool hasMCNoExecStack() const { return MCNoExecStack; }

  /// setMCNoExecStack - Set whether an executabel stack is not needed.
  void setMCNoExecStack(bool Value) { MCNoExecStack = Value; }

  /// hasMCUseLoc - Check whether we should use dwarf's .loc directive.
  bool hasMCUseLoc() const { return MCUseLoc; }

  /// setMCUseLoc - Set whether all we should use dwarf's .loc directive.
  void setMCUseLoc(bool Value) { MCUseLoc = Value; }

  /// getRelocationModel - Returns the code generation relocation model. The
  /// choices are static, PIC, and dynamic-no-pic, and target default.
  static Reloc::Model getRelocationModel();

  /// setRelocationModel - Sets the code generation relocation model.
  ///
  static void setRelocationModel(Reloc::Model Model);

  /// getCodeModel - Returns the code model. The choices are small, kernel,
  /// medium, large, and target default.
  static CodeModel::Model getCodeModel();

  /// setCodeModel - Sets the code model.
  ///
  static void setCodeModel(CodeModel::Model Model);

  /// getAsmVerbosityDefault - Returns the default value of asm verbosity.
  ///
  static bool getAsmVerbosityDefault();

  /// setAsmVerbosityDefault - Set the default value of asm verbosity. Default
  /// is false.
  static void setAsmVerbosityDefault(bool);

  /// getDataSections - Return true if data objects should be emitted into their
  /// own section, corresponds to -fdata-sections.
  static bool getDataSections();

  /// getFunctionSections - Return true if functions should be emitted into
  /// their own section, corresponding to -ffunction-sections.
  static bool getFunctionSections();

  /// setDataSections - Set if the data are emit into separate sections.
  static void setDataSections(bool);

  /// setFunctionSections - Set if the functions are emit into separate
  /// sections.
  static void setFunctionSections(bool);

  /// CodeGenFileType - These enums are meant to be passed into
  /// addPassesToEmitFile to indicate what type of file to emit, and returned by
  /// it to indicate what type of file could actually be made.
  enum CodeGenFileType {
    CGFT_AssemblyFile,
    CGFT_ObjectFile,
    CGFT_Null         // Do not emit any output.
  };

  /// getEnableTailMergeDefault - the default setting for -enable-tail-merge
  /// on this target.  User flag overrides.
  virtual bool getEnableTailMergeDefault() const { return true; }

  /// addPassesToEmitFile - Add passes to the specified pass manager to get the
  /// specified file emitted.  Typically this will involve several steps of code
  /// generation.  This method should return true if emission of this file type
  /// is not supported, or false on success.
  virtual bool addPassesToEmitFile(PassManagerBase &,
                                   formatted_raw_ostream &,
                                   CodeGenFileType,
                                   CodeGenOpt::Level,
                                   bool = true) {
    return true;
  }

  /// addPassesToEmitMachineCode - Add passes to the specified pass manager to
  /// get machine code emitted.  This uses a JITCodeEmitter object to handle
  /// actually outputting the machine code and resolving things like the address
  /// of functions.  This method returns true if machine code emission is
  /// not supported.
  ///
  virtual bool addPassesToEmitMachineCode(PassManagerBase &,
                                          JITCodeEmitter &,
                                          CodeGenOpt::Level,
                                          bool = true) {
    return true;
  }

  /// addPassesToEmitMC - Add passes to the specified pass manager to get
  /// machine code emitted with the MCJIT. This method returns true if machine
  /// code is not supported. It fills the MCContext Ctx pointer which can be
  /// used to build custom MCStreamer.
  ///
  virtual bool addPassesToEmitMC(PassManagerBase &,
                                 MCContext *&,
                                 CodeGenOpt::Level,
                                 bool = true) {
    return true;
  }
};

/// LLVMTargetMachine - This class describes a target machine that is
/// implemented with the LLVM target-independent code generator.
///
class LLVMTargetMachine : public TargetMachine {
  std::string TargetTriple;

protected: // Can only create subclasses.
  LLVMTargetMachine(const Target &T, const std::string &TargetTriple);

private:
  /// addCommonCodeGenPasses - Add standard LLVM codegen passes used for
  /// both emitting to assembly files or machine code output.
  ///
  bool addCommonCodeGenPasses(PassManagerBase &, CodeGenOpt::Level,
                              bool DisableVerify, MCContext *&OutCtx);

  virtual void setCodeModelForJIT();
  virtual void setCodeModelForStatic();

public:

  const std::string &getTargetTriple() const { return TargetTriple; }

  /// addPassesToEmitFile - Add passes to the specified pass manager to get the
  /// specified file emitted.  Typically this will involve several steps of code
  /// generation.  If OptLevel is None, the code generator should emit code as
  /// fast as possible, though the generated code may be less efficient.
  virtual bool addPassesToEmitFile(PassManagerBase &PM,
                                   formatted_raw_ostream &Out,
                                   CodeGenFileType FileType,
                                   CodeGenOpt::Level,
                                   bool DisableVerify = true);

  /// addPassesToEmitMachineCode - Add passes to the specified pass manager to
  /// get machine code emitted.  This uses a JITCodeEmitter object to handle
  /// actually outputting the machine code and resolving things like the address
  /// of functions.  This method returns true if machine code emission is
  /// not supported.
  ///
  virtual bool addPassesToEmitMachineCode(PassManagerBase &PM,
                                          JITCodeEmitter &MCE,
                                          CodeGenOpt::Level,
                                          bool DisableVerify = true);

  /// addPassesToEmitMC - Add passes to the specified pass manager to get
  /// machine code emitted with the MCJIT. This method returns true if machine
  /// code is not supported. It fills the MCContext Ctx pointer which can be
  /// used to build custom MCStreamer.
  ///
  virtual bool addPassesToEmitMC(PassManagerBase &PM,
                                 MCContext *&Ctx,
                                 CodeGenOpt::Level OptLevel,
                                 bool DisableVerify = true);

  /// Target-Independent Code Generator Pass Configuration Options.

  /// addPreISelPasses - This method should add any "last minute" LLVM->LLVM
  /// passes (which are run just before instruction selector).
  virtual bool addPreISel(PassManagerBase &, CodeGenOpt::Level) {
    return true;
  }

  /// addInstSelector - This method should install an instruction selector pass,
  /// which converts from LLVM code to machine instructions.
  virtual bool addInstSelector(PassManagerBase &, CodeGenOpt::Level) {
    return true;
  }

  /// addPreRegAlloc - This method may be implemented by targets that want to
  /// run passes immediately before register allocation. This should return
  /// true if -print-machineinstrs should print after these passes.
  virtual bool addPreRegAlloc(PassManagerBase &, CodeGenOpt::Level) {
    return false;
  }

  /// addPostRegAlloc - This method may be implemented by targets that want
  /// to run passes after register allocation but before prolog-epilog
  /// insertion.  This should return true if -print-machineinstrs should print
  /// after these passes.
  virtual bool addPostRegAlloc(PassManagerBase &, CodeGenOpt::Level) {
    return false;
  }

  /// addPreSched2 - This method may be implemented by targets that want to
  /// run passes after prolog-epilog insertion and before the second instruction
  /// scheduling pass.  This should return true if -print-machineinstrs should
  /// print after these passes.
  virtual bool addPreSched2(PassManagerBase &, CodeGenOpt::Level) {
    return false;
  }

  /// addPreEmitPass - This pass may be implemented by targets that want to run
  /// passes immediately before machine code is emitted.  This should return
  /// true if -print-machineinstrs should print out the code after the passes.
  virtual bool addPreEmitPass(PassManagerBase &, CodeGenOpt::Level) {
    return false;
  }


  /// addCodeEmitter - This pass should be overridden by the target to add a
  /// code emitter, if supported.  If this is not supported, 'true' should be
  /// returned.
  virtual bool addCodeEmitter(PassManagerBase &, CodeGenOpt::Level,
                              JITCodeEmitter &) {
    return true;
  }

  /// getEnableTailMergeDefault - the default setting for -enable-tail-merge
  /// on this target.  User flag overrides.
  virtual bool getEnableTailMergeDefault() const { return true; }
};

} // End llvm namespace

#endif
