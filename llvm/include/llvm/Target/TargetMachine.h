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
class TargetFrameInfo;
class MachineCodeEmitter;
class JITCodeEmitter;
class ObjectCodeEmitter;
class TargetRegisterInfo;
class PassManagerBase;
class PassManager;
class Pass;
class TargetMachOWriterInfo;
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

namespace FileModel {
  enum Model {
    Error,
    None,
    AsmFile,
    MachOFile,
    ElfFile
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

// Specify if we should encode the LSDA pointer in the FDE as 4- or 8-bytes.
namespace DwarfLSDAEncoding {
  enum Encoding {
    Default,
    FourByte,
    EightByte
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
  
public:
  virtual ~TargetMachine();

  const Target &getTarget() const { return TheTarget; }

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
  virtual const InstrItineraryData getInstrItineraryData() const {  
    return InstrItineraryData();
  }

  /// getMachOWriterInfo - If this target supports a Mach-O writer, return
  /// information for it, otherwise return null.
  /// 
  virtual const TargetMachOWriterInfo *getMachOWriterInfo() const { return 0; }

  /// getELFWriterInfo - If this target supports an ELF writer, return
  /// information for it, otherwise return null.
  /// 
  virtual const TargetELFWriterInfo *getELFWriterInfo() const { return 0; }

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

  /// getLSDAEncoding - Returns the LSDA pointer encoding. The choices are
  /// 4-byte, 8-byte, and target default. The CIE is hard-coded to indicate that
  /// the LSDA pointer in the FDE section is an "sdata4", and should be encoded
  /// as a 4-byte pointer by default. However, some systems may require a
  /// different size due to bugs or other conditions. We will default to a
  /// 4-byte encoding unless the system tells us otherwise.
  ///
  /// FIXME: This call-back isn't good! We should be using the correct encoding
  /// regardless of the system. However, there are some systems which have bugs
  /// that prevent this from occuring.
  virtual DwarfLSDAEncoding::Encoding getLSDAEncoding() const {
    return DwarfLSDAEncoding::Default;
  }

  /// CodeGenFileType - These enums are meant to be passed into
  /// addPassesToEmitFile to indicate what type of file to emit.
  enum CodeGenFileType {
    AssemblyFile, ObjectFile, DynamicLibrary
  };

  /// getEnableTailMergeDefault - the default setting for -enable-tail-merge
  /// on this target.  User flag overrides.
  virtual bool getEnableTailMergeDefault() const { return true; }

  /// addPassesToEmitFile - Add passes to the specified pass manager to get the
  /// specified file emitted.  Typically this will involve several steps of code
  /// generation.
  /// This method should return FileModel::Error if emission of this file type
  /// is not supported.
  ///
  virtual FileModel::Model addPassesToEmitFile(PassManagerBase &,
                                               formatted_raw_ostream &,
                                               CodeGenFileType,
                                               CodeGenOpt::Level) {
    return FileModel::None;
  }

  /// addPassesToEmitFileFinish - If the passes to emit the specified file had
  /// to be split up (e.g., to add an object writer pass), this method can be
  /// used to finish up adding passes to emit the file, if necessary.
  ///
  virtual bool addPassesToEmitFileFinish(PassManagerBase &,
                                         MachineCodeEmitter *,
                                         CodeGenOpt::Level) {
    return true;
  }
 
  /// addPassesToEmitFileFinish - If the passes to emit the specified file had
  /// to be split up (e.g., to add an object writer pass), this method can be
  /// used to finish up adding passes to emit the file, if necessary.
  ///
  virtual bool addPassesToEmitFileFinish(PassManagerBase &,
                                         JITCodeEmitter *,
                                         CodeGenOpt::Level) {
    return true;
  }
 
  /// addPassesToEmitFileFinish - If the passes to emit the specified file had
  /// to be split up (e.g., to add an object writer pass), this method can be
  /// used to finish up adding passes to emit the file, if necessary.
  ///
  virtual bool addPassesToEmitFileFinish(PassManagerBase &,
                                         ObjectCodeEmitter *,
                                         CodeGenOpt::Level) {
    return true;
  }
 
  /// addPassesToEmitMachineCode - Add passes to the specified pass manager to
  /// get machine code emitted.  This uses a MachineCodeEmitter object to handle
  /// actually outputting the machine code and resolving things like the address
  /// of functions.  This method returns true if machine code emission is
  /// not supported.
  ///
  virtual bool addPassesToEmitMachineCode(PassManagerBase &,
                                          MachineCodeEmitter &,
                                          CodeGenOpt::Level) {
    return true;
  }

  /// addPassesToEmitMachineCode - Add passes to the specified pass manager to
  /// get machine code emitted.  This uses a MachineCodeEmitter object to handle
  /// actually outputting the machine code and resolving things like the address
  /// of functions.  This method returns true if machine code emission is
  /// not supported.
  ///
  virtual bool addPassesToEmitMachineCode(PassManagerBase &,
                                          JITCodeEmitter &,
                                          CodeGenOpt::Level) {
    return true;
  }

  /// addPassesToEmitWholeFile - This method can be implemented by targets that 
  /// require having the entire module at once.  This is not recommended, do not
  /// use this.
  virtual bool WantsWholeFile() const { return false; }
  virtual bool addPassesToEmitWholeFile(PassManager &, formatted_raw_ostream &,
                                        CodeGenFileType,
                                        CodeGenOpt::Level) {
    return true;
  }
};

/// LLVMTargetMachine - This class describes a target machine that is
/// implemented with the LLVM target-independent code generator.
///
class LLVMTargetMachine : public TargetMachine {
protected: // Can only create subclasses.
  LLVMTargetMachine(const Target &T, const std::string &TargetTriple);
  
  /// addCommonCodeGenPasses - Add standard LLVM codegen passes used for
  /// both emitting to assembly files or machine code output.
  ///
  bool addCommonCodeGenPasses(PassManagerBase &, CodeGenOpt::Level);

private:
  // These routines are used by addPassesToEmitFileFinish and
  // addPassesToEmitMachineCode to set the CodeModel if it's still marked
  // as default.
  virtual void setCodeModelForJIT();
  virtual void setCodeModelForStatic();
  
public:
  
  /// addPassesToEmitFile - Add passes to the specified pass manager to get the
  /// specified file emitted.  Typically this will involve several steps of code
  /// generation.  If OptLevel is None, the code generator should emit code as fast
  /// as possible, though the generated code may be less efficient.  This method
  /// should return FileModel::Error if emission of this file type is not
  /// supported.
  ///
  /// The default implementation of this method adds components from the
  /// LLVM retargetable code generator, invoking the methods below to get
  /// target-specific passes in standard locations.
  ///
  virtual FileModel::Model addPassesToEmitFile(PassManagerBase &PM,
                                               formatted_raw_ostream &Out,
                                               CodeGenFileType FileType,
                                               CodeGenOpt::Level);
  
  /// addPassesToEmitFileFinish - If the passes to emit the specified file had
  /// to be split up (e.g., to add an object writer pass), this method can be
  /// used to finish up adding passes to emit the file, if necessary.
  ///
  virtual bool addPassesToEmitFileFinish(PassManagerBase &PM,
                                         MachineCodeEmitter *MCE,
                                         CodeGenOpt::Level);
 
  /// addPassesToEmitFileFinish - If the passes to emit the specified file had
  /// to be split up (e.g., to add an object writer pass), this method can be
  /// used to finish up adding passes to emit the file, if necessary.
  ///
  virtual bool addPassesToEmitFileFinish(PassManagerBase &PM,
                                         JITCodeEmitter *JCE,
                                         CodeGenOpt::Level);
 
  /// addPassesToEmitFileFinish - If the passes to emit the specified file had
  /// to be split up (e.g., to add an object writer pass), this method can be
  /// used to finish up adding passes to emit the file, if necessary.
  ///
  virtual bool addPassesToEmitFileFinish(PassManagerBase &PM,
                                         ObjectCodeEmitter *OCE,
                                         CodeGenOpt::Level);
 
  /// addPassesToEmitMachineCode - Add passes to the specified pass manager to
  /// get machine code emitted.  This uses a MachineCodeEmitter object to handle
  /// actually outputting the machine code and resolving things like the address
  /// of functions.  This method returns true if machine code emission is
  /// not supported.
  ///
  virtual bool addPassesToEmitMachineCode(PassManagerBase &PM,
                                          MachineCodeEmitter &MCE,
                                          CodeGenOpt::Level);
  
  /// addPassesToEmitMachineCode - Add passes to the specified pass manager to
  /// get machine code emitted.  This uses a MachineCodeEmitter object to handle
  /// actually outputting the machine code and resolving things like the address
  /// of functions.  This method returns true if machine code emission is
  /// not supported.
  ///
  virtual bool addPassesToEmitMachineCode(PassManagerBase &PM,
                                          JITCodeEmitter &MCE,
                                          CodeGenOpt::Level);
  
  /// Target-Independent Code Generator Pass Configuration Options.
  
  /// addInstSelector - This method should add any "last minute" LLVM->LLVM
  /// passes, then install an instruction selector pass, which converts from
  /// LLVM code to machine instructions.
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
                              MachineCodeEmitter &) {
    return true;
  }

  /// addCodeEmitter - This pass should be overridden by the target to add a
  /// code emitter, if supported.  If this is not supported, 'true' should be
  /// returned.
  virtual bool addCodeEmitter(PassManagerBase &, CodeGenOpt::Level,
                              JITCodeEmitter &) {
    return true;
  }

  /// addSimpleCodeEmitter - This pass should be overridden by the target to add
  /// a code emitter (without setting flags), if supported.  If this is not
  /// supported, 'true' should be returned.
  virtual bool addSimpleCodeEmitter(PassManagerBase &, CodeGenOpt::Level,
                                    MachineCodeEmitter &) {
    return true;
  }

  /// addSimpleCodeEmitter - This pass should be overridden by the target to add
  /// a code emitter (without setting flags), if supported.  If this is not
  /// supported, 'true' should be returned.
  virtual bool addSimpleCodeEmitter(PassManagerBase &, CodeGenOpt::Level,
                                    JITCodeEmitter &) {
    return true;
  }

  /// addSimpleCodeEmitter - This pass should be overridden by the target to add
  /// a code emitter (without setting flags), if supported.  If this is not
  /// supported, 'true' should be returned.
  virtual bool addSimpleCodeEmitter(PassManagerBase &, CodeGenOpt::Level,
                                    ObjectCodeEmitter &) {
    return true;
  }

  /// getEnableTailMergeDefault - the default setting for -enable-tail-merge
  /// on this target.  User flag overrides.
  virtual bool getEnableTailMergeDefault() const { return true; }

  /// addAssemblyEmitter - Helper function which creates a target specific
  /// assembly printer, if available.
  ///
  /// \return Returns 'false' on success.
  bool addAssemblyEmitter(PassManagerBase &, CodeGenOpt::Level,
                          bool /* VerboseAsmDefault */,
                          formatted_raw_ostream &);
};

} // End llvm namespace

#endif
