//===-- TransformInternals.h - Shared functions for Transforms ---*- C++ -*--=//
//
//  This header file declares shared functions used by the different components
//  of the Transforms library.
//
//===----------------------------------------------------------------------===//

#ifndef TRANSFORM_INTERNALS_H
#define TRANSFORM_INTERNALS_H

#include "llvm/BasicBlock.h"
#include "llvm/Instruction.h"
#include "llvm/Target/TargetData.h"
#include "llvm/DerivedTypes.h"
#include "llvm/ConstantVals.h"
#include <map>
#include <set>

// TargetData Hack: Eventually we will have annotations given to us by the
// backend so that we know stuff about type size and alignments.  For now
// though, just use this, because it happens to match the model that GCC uses.
//
// FIXME: This should use annotations
//
extern const TargetData TD;

static inline int getConstantValue(const ConstantInt *CPI) {
  if (const ConstantSInt *CSI = dyn_cast<ConstantSInt>(CPI))
    return CSI->getValue();
  return cast<ConstantUInt>(CPI)->getValue();
}


// getPointedToComposite - If the argument is a pointer type, and the pointed to
// value is a composite type, return the composite type, else return null.
//
static inline const CompositeType *getPointedToComposite(const Type *Ty) {
  const PointerType *PT = dyn_cast<PointerType>(Ty);
  return PT ? dyn_cast<CompositeType>(PT->getElementType()) : 0;
}


// ReplaceInstWithValue - Replace all uses of an instruction (specified by BI)
// with a value, then remove and delete the original instruction.
//
void ReplaceInstWithValue(BasicBlock::InstListType &BIL,
                          BasicBlock::iterator &BI, Value *V);

// ReplaceInstWithInst - Replace the instruction specified by BI with the
// instruction specified by I.  The original instruction is deleted and BI is
// updated to point to the new instruction.
//
void ReplaceInstWithInst(BasicBlock::InstListType &BIL,
                         BasicBlock::iterator &BI, Instruction *I);

void ReplaceInstWithInst(Instruction *From, Instruction *To);


// ConvertableToGEP - This function returns true if the specified value V is
// a valid index into a pointer of type Ty.  If it is valid, Idx is filled in
// with the values that would be appropriate to make this a getelementptr
// instruction.  The type returned is the root type that the GEP would point
// to if it were synthesized with this operands.
//
// If BI is nonnull, cast instructions are inserted as appropriate for the
// arguments of the getelementptr.
//
const Type *ConvertableToGEP(const Type *Ty, Value *V,
                             std::vector<Value*> &Indices,
                             BasicBlock::iterator *BI = 0);


// ------------- Expression Conversion ---------------------

typedef std::map<const Value*, const Type*>         ValueTypeCache;

struct ValueMapCache {
  // Operands mapped - Contains an entry if the first value (the user) has had
  // the second value (the operand) mapped already.
  //
  std::set<const User*> OperandsMapped;

  // Expression Map - Contains an entry from the old value to the new value of
  // an expression that has been converted over.
  //
  std::map<const Value *, Value *> ExprMap;
  typedef std::map<const Value *, Value *> ExprMapTy;
};


bool ExpressionConvertableToType(Value *V, const Type *Ty, ValueTypeCache &Map);
Value *ConvertExpressionToType(Value *V, const Type *Ty, ValueMapCache &VMC);

// ValueConvertableToType - Return true if it is possible
bool ValueConvertableToType(Value *V, const Type *Ty,
                            ValueTypeCache &ConvertedTypes);

void ConvertValueToNewType(Value *V, Value *NewVal, ValueMapCache &VMC);


//===----------------------------------------------------------------------===//
//  ValueHandle Class - Smart pointer that occupies a slot on the users USE list
//  that prevents it from being destroyed.  This "looks" like an Instruction
//  with Opcode UserOp1.
// 
class ValueHandle : public Instruction {
  ValueHandle(const ValueHandle &); // DO NOT IMPLEMENT
  ValueMapCache &Cache;
public:
  ValueHandle(ValueMapCache &VMC, Value *V);
  ~ValueHandle();

  virtual Instruction *clone() const { abort(); return 0; }

  virtual const char *getOpcodeName() const {
    return "ValueHandle";
  }

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const ValueHandle *) { return true; }
  static inline bool classof(const Instruction *I) {
    return (I->getOpcode() == Instruction::UserOp1);
  }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) && classof(cast<Instruction>(V));
  }
};

// getStructOffsetType - Return a vector of offsets that are to be used to index
// into the specified struct type to get as close as possible to index as we
// can.  Note that it is possible that we cannot get exactly to Offset, in which
// case we update offset to be the offset we actually obtained.  The resultant
// leaf type is returned.
//
// If StopEarly is set to true (the default), the first object with the
// specified type is returned, even if it is a struct type itself.  In this
// case, this routine will not drill down to the leaf type.  Set StopEarly to
// false if you want a leaf
//
const Type *getStructOffsetType(const Type *Ty, unsigned &Offset,
                                std::vector<Value*> &Offsets,
                                bool StopEarly = true);

#endif
