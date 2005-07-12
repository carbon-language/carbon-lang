//==-- llvm/Target/TargetSubtarget.h - Target Information --------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Nate Begeman and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the subtarget options of a Target machine.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETSUBTARGET_H
#define LLVM_TARGET_TARGETSUBTARGET_H

namespace llvm {

class Module;

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
  /// This constructor initializes the data members to match that 
  /// of the specified module.
  ///
  TargetSubtarget(const Module &M);
public:
  virtual ~TargetSubtarget();
};

} // End llvm namespace

#endif
