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

#include <vector>

namespace llvm {
  class Constant;
  class Type;
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

  virtual Constant *lessthan(const Constant *V1, const Constant *V2) const = 0;
                             
  virtual Constant *equalto(const Constant *V1, const Constant *V2) const = 0;

  // Casting operators.  ick
  virtual Constant *castToBool  (const Constant *V) const = 0;
  virtual Constant *castToSByte (const Constant *V) const = 0;
  virtual Constant *castToUByte (const Constant *V) const = 0;
  virtual Constant *castToShort (const Constant *V) const = 0;
  virtual Constant *castToUShort(const Constant *V) const = 0;
  virtual Constant *castToInt   (const Constant *V) const = 0;
  virtual Constant *castToUInt  (const Constant *V) const = 0;
  virtual Constant *castToLong  (const Constant *V) const = 0;
  virtual Constant *castToULong (const Constant *V) const = 0;
  virtual Constant *castToFloat (const Constant *V) const = 0;
  virtual Constant *castToDouble(const Constant *V) const = 0;
  virtual Constant *castToPointer(const Constant *V,
                                  const PointerType *Ty) const = 0;

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
