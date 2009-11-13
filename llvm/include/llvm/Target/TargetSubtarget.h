//==-- llvm/Target/TargetSubtarget.h - Target Information --------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the subtarget options of a Target machine.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETSUBTARGET_H
#define LLVM_TARGET_TARGETSUBTARGET_H

#include "llvm/Target/TargetMachine.h"

namespace llvm {

class SDep;
class SUnit;
class TargetRegisterClass;
template <typename T> class SmallVectorImpl;

//===----------------------------------------------------------------------===//
///
/// TargetSubtarget - Generic base class for all target subtargets.  All
/// Target-specific options that control code generation and printing should
/// be exposed through a TargetSubtarget-derived class.
///
class TargetSubtarget {
  TargetSubtarget(const TargetSubtarget&);   // DO NOT IMPLEMENT
  void operator=(const TargetSubtarget&);  // DO NOT IMPLEMENT
protected: // Can only create subclasses...
  TargetSubtarget();
public:
  // AntiDepBreakMode - Type of anti-dependence breaking that should
  // be performed before post-RA scheduling.
  typedef enum { ANTIDEP_NONE, ANTIDEP_CRITICAL, ANTIDEP_ALL } AntiDepBreakMode;
  typedef SmallVectorImpl<TargetRegisterClass*> RegClassVector;

  virtual ~TargetSubtarget();

  /// getSpecialAddressLatency - For targets where it is beneficial to
  /// backschedule instructions that compute addresses, return a value
  /// indicating the number of scheduling cycles of backscheduling that
  /// should be attempted.
  virtual unsigned getSpecialAddressLatency() const { return 0; }

  // enablePostRAScheduler - If the target can benefit from post-regalloc
  // scheduling and the specified optimization level meets the requirement
  // return true to enable post-register-allocation scheduling. In
  // CriticalPathRCs return any register classes that should only be broken
  // if on the critical path. 
  virtual bool enablePostRAScheduler(CodeGenOpt::Level OptLevel,
                                     AntiDepBreakMode& Mode,
                                     RegClassVector& CriticalPathRCs) const;
  // adjustSchedDependency - Perform target specific adjustments to
  // the latency of a schedule dependency.
  virtual void adjustSchedDependency(SUnit *def, SUnit *use, 
                                     SDep& dep) const { }
};

} // End llvm namespace

#endif
