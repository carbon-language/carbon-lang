//===-- llvm/Target/TargetIntrinsicInfo.h - Instruction Info ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the target intrinsic instructions to the code generator.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETINTRINSICINFO_H
#define LLVM_TARGET_TARGETINTRINSICINFO_H

namespace llvm {

class Function;
class Module;
class Type;

//---------------------------------------------------------------------------
///
/// TargetIntrinsicInfo - Interface to description of machine instruction set
///
class TargetIntrinsicInfo {
  TargetIntrinsicInfo(const TargetIntrinsicInfo &); // DO NOT IMPLEMENT
  void operator=(const TargetIntrinsicInfo &);      // DO NOT IMPLEMENT
public:
  TargetIntrinsicInfo();
  virtual ~TargetIntrinsicInfo();

  /// Return the name of a target intrinsic, e.g. "llvm.bfin.ssync".
  virtual const char *getName(unsigned IntrID) const =0;

  /// Look up target intrinsic by name. Return intrinsic ID or 0 for unknown
  /// names.
  virtual unsigned lookupName(const char *Name, unsigned Len) const =0;

  /// Return the target intrinsic ID of a function, or 0.
  virtual unsigned getIntrinsicID(Function *F) const;
};

} // End llvm namespace

#endif
