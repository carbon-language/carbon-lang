//===-- Bitcode/Writer/ValueEnumerator.h - Number values --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class gives values and types Unique ID's.
//
//===----------------------------------------------------------------------===//

#ifndef VALUE_ENUMERATOR_H
#define VALUE_ENUMERATOR_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/Attributes.h"
#include <vector>

namespace llvm {

class Type;
class Value;
class BasicBlock;
class Function;
class Module;
class PAListPtr;
class TypeSymbolTable;
class ValueSymbolTable;

class ValueEnumerator {
public:
  // For each type, we remember its Type* and occurrence frequency.
  typedef std::vector<std::pair<const Type*, unsigned> > TypeList;

  // For each value, we remember its Value* and occurrence frequency.
  typedef std::vector<std::pair<const Value*, unsigned> > ValueList;
private:
  typedef DenseMap<const Type*, unsigned> TypeMapType;
  TypeMapType TypeMap;
  TypeList Types;

  typedef DenseMap<const Value*, unsigned> ValueMapType;
  ValueMapType ValueMap;
  ValueList Values;
  
  typedef DenseMap<void*, unsigned> ParamAttrMapType;
  ParamAttrMapType ParamAttrMap;
  std::vector<PAListPtr> ParamAttrs;
  
  /// BasicBlocks - This contains all the basic blocks for the currently
  /// incorporated function.  Their reverse mapping is stored in ValueMap.
  std::vector<const BasicBlock*> BasicBlocks;
  
  /// When a function is incorporated, this is the size of the Values list
  /// before incorporation.
  unsigned NumModuleValues;
  unsigned FirstFuncConstantID;
  unsigned FirstInstID;
  
  ValueEnumerator(const ValueEnumerator &);  // DO NOT IMPLEMENT
  void operator=(const ValueEnumerator &);   // DO NOT IMPLEMENT
public:
  ValueEnumerator(const Module *M);

  unsigned getValueID(const Value *V) const {
    ValueMapType::const_iterator I = ValueMap.find(V);
    assert(I != ValueMap.end() && "Value not in slotcalculator!");
    return I->second-1;
  }
  
  unsigned getTypeID(const Type *T) const {
    TypeMapType::const_iterator I = TypeMap.find(T);
    assert(I != TypeMap.end() && "Type not in ValueEnumerator!");
    return I->second-1;
  }
  
  unsigned getParamAttrID(const PAListPtr &PAL) const {
    if (PAL.isEmpty()) return 0;  // Null maps to zero.
    ParamAttrMapType::const_iterator I = ParamAttrMap.find(PAL.getRawPointer());
    assert(I != ParamAttrMap.end() && "ParamAttr not in ValueEnumerator!");
    return I->second;
  }

  /// getFunctionConstantRange - Return the range of values that corresponds to
  /// function-local constants.
  void getFunctionConstantRange(unsigned &Start, unsigned &End) const {
    Start = FirstFuncConstantID;
    End = FirstInstID;
  }
  
  const ValueList &getValues() const { return Values; }
  const TypeList &getTypes() const { return Types; }
  const std::vector<const BasicBlock*> &getBasicBlocks() const {
    return BasicBlocks; 
  }
  const std::vector<PAListPtr> &getParamAttrs() const {
    return ParamAttrs;
  }

  /// PurgeAggregateValues - If there are any aggregate values at the end of the
  /// value list, remove them and return the count of the remaining values.  If
  /// there are none, return -1.
  int PurgeAggregateValues();
  
  /// incorporateFunction/purgeFunction - If you'd like to deal with a function,
  /// use these two methods to get its data into the ValueEnumerator!
  ///
  void incorporateFunction(const Function &F);
  void purgeFunction();

private:
  void OptimizeConstants(unsigned CstStart, unsigned CstEnd);
    
  void EnumerateValue(const Value *V);
  void EnumerateType(const Type *T);
  void EnumerateOperandType(const Value *V);
  void EnumerateParamAttrs(const PAListPtr &PAL);
  
  void EnumerateTypeSymbolTable(const TypeSymbolTable &ST);
  void EnumerateValueSymbolTable(const ValueSymbolTable &ST);
};

} // End llvm namespace

#endif
