//===-- TransformInternals.h - Shared functions for Transforms ---*- C++ -*--=//
//
//  This header file declares shared functions used by the different components
//  of the Transforms library.
//
//===----------------------------------------------------------------------===//

#ifndef TRANSFORM_INTERNALS_H
#define TRANSFORM_INTERNALS_H

#include "llvm/BasicBlock.h"
#include "llvm/Target/TargetData.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Constants.h"
#include <map>
#include <set>

static inline int64_t getConstantValue(const ConstantInt *CPI) {
  return (int64_t)cast<ConstantInt>(CPI)->getRawValue();
}


// getPointedToComposite - If the argument is a pointer type, and the pointed to
// value is a composite type, return the composite type, else return null.
//
static inline const CompositeType *getPointedToComposite(const Type *Ty) {
  const PointerType *PT = dyn_cast<PointerType>(Ty);
  return PT ? dyn_cast<CompositeType>(PT->getElementType()) : 0;
}

// ConvertibleToGEP - This function returns true if the specified value V is
// a valid index into a pointer of type Ty.  If it is valid, Idx is filled in
// with the values that would be appropriate to make this a getelementptr
// instruction.  The type returned is the root type that the GEP would point
// to if it were synthesized with this operands.
//
// If BI is nonnull, cast instructions are inserted as appropriate for the
// arguments of the getelementptr.
//
const Type *ConvertibleToGEP(const Type *Ty, Value *V,
                             std::vector<Value*> &Indices,
                             const TargetData &TD,
                             BasicBlock::iterator *BI = 0);


//===----------------------------------------------------------------------===//
//  ValueHandle Class - Smart pointer that occupies a slot on the users USE list
//  that prevents it from being destroyed.  This "looks" like an Instruction
//  with Opcode UserOp1.
// 
class ValueMapCache;
class ValueHandle : public Instruction {
  ValueMapCache &Cache;
public:
  ValueHandle(ValueMapCache &VMC, Value *V);
  ValueHandle(const ValueHandle &);
  ~ValueHandle();

  virtual Instruction *clone() const { abort(); return 0; }

  virtual const char *getOpcodeName() const {
    return "ValueHandle";
  }

  inline bool operator<(const ValueHandle &VH) const {
    return getOperand(0) < VH.getOperand(0);
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


// ------------- Expression Conversion ---------------------

typedef std::map<const Value*, const Type*> ValueTypeCache;

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

  // Cast Map - Cast instructions can have their source and destination values
  // changed independantly for each part.  Because of this, our old naive
  // implementation would create a TWO new cast instructions, which would cause
  // all kinds of problems.  Here we keep track of the newly allocated casts, so
  // that we only create one for a particular instruction.
  //
  std::set<ValueHandle> NewCasts;
};


bool ExpressionConvertibleToType(Value *V, const Type *Ty, ValueTypeCache &Map,
                                 const TargetData &TD);
Value *ConvertExpressionToType(Value *V, const Type *Ty, ValueMapCache &VMC,
                               const TargetData &TD);

// ValueConvertibleToType - Return true if it is possible
bool ValueConvertibleToType(Value *V, const Type *Ty,
                            ValueTypeCache &ConvertedTypes,
                            const TargetData &TD);

void ConvertValueToNewType(Value *V, Value *NewVal, ValueMapCache &VMC,
                           const TargetData &TD);


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
                                const TargetData &TD, bool StopEarly = true);

#endif
