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
#include <map>

// TargetData Hack: Eventually we will have annotations given to us by the
// backend so that we know stuff about type size and alignments.  For now
// though, just use this, because it happens to match the model that GCC uses.
//
// FIXME: This should use annotations
//
extern const TargetData TD;

// losslessCastableTypes - Return true if the types are bitwise equivalent.
// This predicate returns true if it is possible to cast from one type to
// another without gaining or losing precision, or altering the bits in any way.
//
bool losslessCastableTypes(const Type *T1, const Type *T2);


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


// ------------- Expression Conversion ---------------------

typedef map<const Value*, const Type*> ValueTypeCache;
typedef map<const Value*, Value*>      ValueMapCache;

// RetValConvertableToType - Return true if it is possible
bool RetValConvertableToType(Value *V, const Type *Ty,
                             ValueTypeCache &ConvertedTypes);

void ConvertUsersType(Value *V, Value *NewVal, ValueMapCache &VMC);


#endif
