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

namespace llvm {

/// AllocInfo - Structure representing one instruction's operand's-worth of
/// register allocation state. We create tables made out of these data
/// structures to generate mapping information for this register allocator.
///
struct AllocInfo {
  int Instruction; // (-1 if Argument, or 0 .. n - 1 for an instruction).
  int Operand; // (-1 if Instruction, or 0 .. n-1 for an operand).
  enum AllocStateTy { NotAllocated = 0, Allocated, Spilled };
  AllocStateTy AllocState;
  int Placement;

  AllocInfo (unsigned Instruction_, unsigned Operand_,
             AllocStateTy AllocState_, int Placement_) :
    Instruction (Instruction_), Operand (Operand_),
       AllocState (AllocState_), Placement (Placement_) { }

  /// getConstantType - Return a StructType representing an AllocInfo object.
  ///
  static StructType *getConstantType () {
    std::vector<const Type *> TV;
    TV.push_back (Type::IntTy);
    TV.push_back (Type::IntTy);
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
    CV.push_back (ConstantSInt::get (Type::IntTy, Instruction));
    CV.push_back (ConstantSInt::get (Type::IntTy, Operand));
    CV.push_back (ConstantUInt::get (Type::UIntTy, AllocState));
    CV.push_back (ConstantSInt::get (Type::IntTy, Placement));
    return ConstantStruct::get (ST, CV);
  }

  /// AllocInfos compare equal if the allocation placements are equal
  /// (i.e., they can be equal even if they refer to operands from two
  /// different instructions.)
  ///
  bool operator== (const AllocInfo &X) const {
    return (X.AllocState == AllocState) && (X.Placement == Placement);
  } 
  bool operator!= (const AllocInfo &X) const { return !(*this == X); } 

  /// Returns a human-readable string representation of the AllocState member.
  ///
  const std::string allocStateToString () const {
    static const char *AllocStateNames[] =
      { "NotAllocated", "Allocated", "Spilled" };
    return std::string (AllocStateNames[AllocState]);
  }
};

static inline std::ostream &operator << (std::ostream &OS, AllocInfo &S) {
  OS << "(Instruction " << S.Instruction << " Operand " << S.Operand
     << " AllocState " << S.allocStateToString () << " Placement "
     << S.Placement << ")";
  return OS;
}

} // End llvm namespace

#endif // ALLOCINFO_H
