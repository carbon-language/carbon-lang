//===-- ConstantHandling.h - Stuff for manipulating constants ---*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// WARNING: These operators may return a null object if I don't know how to 
//          perform the specified operation on the specified constant types.
//
//===----------------------------------------------------------------------===//

#ifndef CONSTANTHANDLING_H
#define CONSTANTHANDLING_H

#include "llvm/Constants.h"
#include "llvm/Type.h"

namespace llvm {

class PointerType;

struct ConstRules {
  ConstRules() {}

  // Binary Operators...
  virtual Constant *add(const Constant *V1, const Constant *V2) const = 0;
  virtual Constant *sub(const Constant *V1, const Constant *V2) const = 0;
  virtual Constant *mul(const Constant *V1, const Constant *V2) const = 0;
  virtual Constant *div(const Constant *V1, const Constant *V2) const = 0;
  virtual Constant *rem(const Constant *V1, const Constant *V2) const = 0;
  virtual Constant *op_and(const Constant *V1, const Constant *V2) const = 0;
  virtual Constant *op_or (const Constant *V1, const Constant *V2) const = 0;
  virtual Constant *op_xor(const Constant *V1, const Constant *V2) const = 0;
  virtual Constant *shl(const Constant *V1, const Constant *V2) const = 0;
  virtual Constant *shr(const Constant *V1, const Constant *V2) const = 0;

  virtual ConstantBool *lessthan(const Constant *V1, 
                                 const Constant *V2) const = 0;
  virtual ConstantBool *equalto(const Constant *V1, 
                                const Constant *V2) const = 0;

  // Casting operators.  ick
  virtual ConstantBool *castToBool  (const Constant *V) const = 0;
  virtual ConstantSInt *castToSByte (const Constant *V) const = 0;
  virtual ConstantUInt *castToUByte (const Constant *V) const = 0;
  virtual ConstantSInt *castToShort (const Constant *V) const = 0;
  virtual ConstantUInt *castToUShort(const Constant *V) const = 0;
  virtual ConstantSInt *castToInt   (const Constant *V) const = 0;
  virtual ConstantUInt *castToUInt  (const Constant *V) const = 0;
  virtual ConstantSInt *castToLong  (const Constant *V) const = 0;
  virtual ConstantUInt *castToULong (const Constant *V) const = 0;
  virtual ConstantFP   *castToFloat (const Constant *V) const = 0;
  virtual ConstantFP   *castToDouble(const Constant *V) const = 0;
  virtual Constant     *castToPointer(const Constant *V,
                                      const PointerType *Ty) const = 0;

  inline Constant *castTo(const Constant *V, const Type *Ty) const {
    switch (Ty->getPrimitiveID()) {
    case Type::BoolTyID:   return castToBool(V);
    case Type::UByteTyID:  return castToUByte(V);
    case Type::SByteTyID:  return castToSByte(V);
    case Type::UShortTyID: return castToUShort(V);
    case Type::ShortTyID:  return castToShort(V);
    case Type::UIntTyID:   return castToUInt(V);
    case Type::IntTyID:    return castToInt(V);
    case Type::ULongTyID:  return castToULong(V);
    case Type::LongTyID:   return castToLong(V);
    case Type::FloatTyID:  return castToFloat(V);
    case Type::DoubleTyID: return castToDouble(V);
    case Type::PointerTyID:
      return castToPointer(V, reinterpret_cast<const PointerType*>(Ty));
    default: return 0;
    }
  }

  // ConstRules::get - Return an instance of ConstRules for the specified
  // constant operands.
  //
  static ConstRules &get(const Constant *V1, const Constant *V2);
private:
  ConstRules(const ConstRules &);             // Do not implement
  ConstRules &operator=(const ConstRules &);  // Do not implement
};


//===----------------------------------------------------------------------===//
//  Implement higher level instruction folding type instructions
//===----------------------------------------------------------------------===//

// Constant fold various types of instruction...
Constant *ConstantFoldCastInstruction(const Constant *V, const Type *DestTy);
Constant *ConstantFoldBinaryInstruction(unsigned Opcode, const Constant *V1,
                                        const Constant *V2);
Constant *ConstantFoldGetElementPtr(const Constant *C,
                                    const std::vector<Constant*> &IdxList);

} // End llvm namespace

#endif
