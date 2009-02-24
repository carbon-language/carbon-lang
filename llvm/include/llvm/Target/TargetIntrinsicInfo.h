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
  
  const char **Intrinsics;               // Raw array to allow static init'n
  unsigned NumIntrinsics;                // Number of entries in the desc array

  TargetIntrinsicInfo(const TargetIntrinsicInfo &);  // DO NOT IMPLEMENT
  void operator=(const TargetIntrinsicInfo &);   // DO NOT IMPLEMENT
public:
  TargetIntrinsicInfo(const char **desc, unsigned num);
  virtual ~TargetIntrinsicInfo();

  unsigned getNumIntrinsics() const { return NumIntrinsics; }

  virtual Function *getDeclaration(Module *M, const char *BuiltinName) const {
    return 0;
  }

  // Returns the Function declaration for intrinsic BuiltinName.  If the
  // intrinsic can be overloaded, uses Tys to return the correct function.
  virtual Function *getDeclaration(Module *M, const char *BuiltinName,
                                   const Type **Tys, unsigned numTys) const {
    return 0;
  }

  // Returns true if the Builtin can be overloaded.
  virtual bool isOverloaded(Module *M, const char *BuiltinName) const {
    return false;
  }

  virtual unsigned getIntrinsicID(Function *F) const { return 0; }
};

} // End llvm namespace

#endif
