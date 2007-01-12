//===-- TargetLowering.cpp - Asm Info --------------------------------------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer  and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements methods of the TargetLowering class.
//
//===----------------------------------------------------------------------===//
//

#include "llvm/Target/TargetLowering.h"
#include "llvm/DerivedTypes.h"
#include "llvm/CodeGen/ValueTypes.h"

using namespace llvm;

MVT::ValueType TargetLowering::getValueType(const Type *Ty) const {
  switch (Ty->getTypeID()) {
  default: assert(0 && "Unknown type!");
  case Type::VoidTyID:    return MVT::isVoid;
  case Type::IntegerTyID:
    switch (cast<IntegerType>(Ty)->getBitWidth()) {
      default: assert(0 && "Invalid width for value type");
      case 1:    return MVT::i1;
      case 8:    return MVT::i8;
      case 16:   return MVT::i16;
      case 32:   return MVT::i32;
      case 64:   return MVT::i64;
    }
    break;
  case Type::FloatTyID:   return MVT::f32;
  case Type::DoubleTyID:  return MVT::f64;
  case Type::PointerTyID: return PointerTy;
  case Type::PackedTyID:  return MVT::Vector;
  }
}
