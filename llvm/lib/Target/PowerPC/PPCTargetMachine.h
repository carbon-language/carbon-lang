//===-- PPCTargetMachine.h - Define TargetMachine for PowerPC -----*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the PowerPC specific subclass of TargetMachine.
//
//===----------------------------------------------------------------------===//

#ifndef PPC_TARGETMACHINE_H
#define PPC_TARGETMACHINE_H

#include "PPCFrameInfo.h"
#include "PPCSubtarget.h"
#include "PPCJITInfo.h"
#include "PPCInstrInfo.h"
#include "PPCISelLowering.h"
#include "PPCMachOWriterInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetData.h"

namespace llvm {
class PassManager;
class GlobalValue;

/// PPCTargetMachine - Common code between 32-bit and 64-bit PowerPC targets.
///
class PPCTargetMachine : public LLVMTargetMachine {
  PPCSubtarget        Subtarget;
  const TargetData    DataLayout;       // Calculates type size & alignment
  PPCInstrInfo        InstrInfo;
  PPCFrameInfo        FrameInfo;
  PPCJITInfo          JITInfo;
  PPCTargetLowering   TLInfo;
  InstrItineraryData  InstrItins;
  PPCMachOWriterInfo  MachOWriterInfo;

public:
  PPCTargetMachine(const Target &T, const std::string &TT,
                   const std::string &FS, bool is64Bit);

  virtual const PPCInstrInfo     *getInstrInfo() const { return &InstrInfo; }
  virtual const PPCFrameInfo     *getFrameInfo() const { return &FrameInfo; }
  virtual       PPCJITInfo       *getJITInfo()         { return &JITInfo; }
  virtual       PPCTargetLowering *getTargetLowering() const { 
   return const_cast<PPCTargetLowering*>(&TLInfo); 
  }
  virtual const PPCRegisterInfo  *getRegisterInfo() const {
    return &InstrInfo.getRegisterInfo();
  }
  
  virtual const TargetData    *getTargetData() const    { return &DataLayout; }
  virtual const PPCSubtarget  *getSubtargetImpl() const { return &Subtarget; }
  virtual const InstrItineraryData getInstrItineraryData() const {  
    return InstrItins;
  }
  virtual const PPCMachOWriterInfo *getMachOWriterInfo() const {
    return &MachOWriterInfo;
  }

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
  virtual DwarfLSDAEncoding::Encoding getLSDAEncoding() const;

  // Pass Pipeline Configuration
  virtual bool addInstSelector(PassManagerBase &PM, CodeGenOpt::Level OptLevel);
  virtual bool addPreEmitPass(PassManagerBase &PM, CodeGenOpt::Level OptLevel);
  virtual bool addCodeEmitter(PassManagerBase &PM, CodeGenOpt::Level OptLevel,
                              MachineCodeEmitter &MCE);
  virtual bool addCodeEmitter(PassManagerBase &PM, CodeGenOpt::Level OptLevel,
                              JITCodeEmitter &JCE);
  virtual bool addCodeEmitter(PassManagerBase &PM, CodeGenOpt::Level OptLevel,
                              ObjectCodeEmitter &OCE);
  virtual bool addSimpleCodeEmitter(PassManagerBase &PM,
                                    CodeGenOpt::Level OptLevel,
                                    MachineCodeEmitter &MCE);
  virtual bool addSimpleCodeEmitter(PassManagerBase &PM,
                                    CodeGenOpt::Level OptLevel,
                                    JITCodeEmitter &JCE);
  virtual bool addSimpleCodeEmitter(PassManagerBase &PM,
                                    CodeGenOpt::Level OptLevel,
                                    ObjectCodeEmitter &OCE);
  virtual bool getEnableTailMergeDefault() const;
};

/// PPC32TargetMachine - PowerPC 32-bit target machine.
///
class PPC32TargetMachine : public PPCTargetMachine {
public:
  PPC32TargetMachine(const Target &T, const std::string &TT,
                     const std::string &FS);
};

/// PPC64TargetMachine - PowerPC 64-bit target machine.
///
class PPC64TargetMachine : public PPCTargetMachine {
public:
  PPC64TargetMachine(const Target &T, const std::string &TT,
                     const std::string &FS);
};

} // end namespace llvm

#endif
