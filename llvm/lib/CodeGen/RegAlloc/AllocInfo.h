//===-- AllocInfo.h - Store info about regalloc decisions -------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This header file contains the data structure used to save the state
// of the global, graph-coloring register allocator.
//
//===----------------------------------------------------------------------===//

#ifndef ALLOCINFO_H
#define ALLOCINFO_H

#include "llvm/Type.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Constants.h"

/// AllocInfo - Structure representing one instruction's operand's-worth of
/// register allocation state. We create tables made out of these data
/// structures to generate mapping information for this register allocator.
///
struct AllocInfo {
  unsigned Instruction;
  unsigned Operand;
  unsigned AllocState;
  int Placement;
  AllocInfo (unsigned Instruction_, unsigned Operand_,
             unsigned AllocState_, int Placement_) :
    Instruction (Instruction_), Operand (Operand_),
       AllocState (AllocState_), Placement (Placement_) { }

  /// getConstantType - Return a StructType representing an AllocInfo object.
  ///
  static StructType *getConstantType () {
    std::vector<const Type *> TV;
    TV.push_back (Type::UIntTy);
    TV.push_back (Type::UIntTy);
    TV.push_back (Type::UIntTy);
    TV.push_back (Type::IntTy);
    return StructType::get (TV);
  }

  /// toConstant - Convert this AllocInfo into an LLVM Constant of type
  /// getConstantType(), and return the Constant.
  ///
  Constant *toConstant () const {
    StructType *ST = getConstantType ();
    std::vector<Constant *> CV;
    CV.push_back (ConstantUInt::get (Type::UIntTy, Instruction));
    CV.push_back (ConstantUInt::get (Type::UIntTy, Operand));
    CV.push_back (ConstantUInt::get (Type::UIntTy, AllocState));
    CV.push_back (ConstantSInt::get (Type::IntTy, Placement));
    return ConstantStruct::get (ST, CV);
  }
};

#endif // ALLOCINFO_H
