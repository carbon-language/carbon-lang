//===- llvm/Transforms/MutateStructTypes.h - Change struct defns -*- C++ -*--=//
//
// This pass is used to change structure accesses and type definitions in some
// way.  It can be used to arbitrarily permute structure fields, safely, without
// breaking code.  A transformation may only be done on a type if that type has
// been found to be "safe" by the 'FindUnsafePointerTypes' pass.  This pass will
// assert and die if you try to do an illegal transformation.
//
// This is an interprocedural pass that requires the entire program to do a
// transformation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_MUTATESTRUCTTYPES_H
#define LLVM_TRANSFORMS_MUTATESTRUCTTYPES_H

#include "llvm/Pass.h"
#include <map>

class StructType;
class CompositeType;
class GlobalValue;

class MutateStructTypes : public Pass {
  // TransformType - Representation of the destination type for a particular
  // incoming structure.  The first member is the destination type that we are
  // mapping to, and the second member is the destination slot # to put each
  // incoming slot [or negative if the specified incoming slot should be
  // removed].
  //
  typedef std::pair<const StructType*, std::vector<int> > TransformType;

  // Transforms to do for each structure type...
  std::map<const StructType*, TransformType> Transforms;

  // Mapping of old type to new types...
  std::map<const Type *, PATypeHolder<Type> > TypeMap;

  // Mapping from global value of old type, to a global value of the new type...
  std::map<const GlobalValue*, GlobalValue*> GlobalMap;

  // Mapping from intra method value to intra method value
  std::map<const Value*, Value*> LocalValueMap;

public:
  // Ctor - Take a map that specifies what transformation to do for each field
  // of the specified structure types.  There is one element of the vector for
  // each field of the structure.  The value specified indicates which slot of
  // the destination structure the field should end up in.  A negative value 
  // indicates that the field should be deleted entirely.
  //
  typedef std::map<const StructType*, std::vector<int> > TransformsType;

  MutateStructTypes(const TransformsType &Transforms);


  // doPassInitialization - This loops over global constants defined in the
  // module, converting them to their new type.  Also this creates placeholder
  // methods for methods than need to be copied because they have a new
  // signature type.
  //
  bool doPassInitialization(Module *M);

  // doPerMethodWork - This transforms the instructions of the method to use the
  // new types.
  //
  bool doPerMethodWork(Method *M);

  // doPassFinalization - This removes the old versions of methods that are no
  // longer needed.
  //
  virtual bool doPassFinalization(Module *M);

private:
  // ConvertType - Convert from the old type system to the new one...
  const Type *ConvertType(const Type *Ty);

  // ConvertValue - Convert from the old value in the old type system to the new
  // type system.
  //
  Value *ConvertValue(const Value *V);

  // AdjustIndices - Convert the indexes specifed by Idx to the new changed form
  // using the specified OldTy as the base type being indexed into.
  //
  void AdjustIndices(const CompositeType *OldTy, std::vector<Value*> &Idx,
                     unsigned idx = 0);
};

#endif
