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
class Instruction;
class BasicBlock;
class Function;
class Module;
class MetadataBase;
class AttrListPtr;
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
  ValueList MDValues;
  ValueMapType MDValueMap;
  
  typedef DenseMap<void*, unsigned> AttributeMapType;
  AttributeMapType AttributeMap;
  std::vector<AttrListPtr> Attributes;
  
  typedef DenseMap<const Instruction*, unsigned> InstructionMapType;
  InstructionMapType InstructionMap;
  unsigned InstructionCount;

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

  unsigned getValueID(const Value *V) const;

  unsigned getTypeID(const Type *T) const {
    TypeMapType::const_iterator I = TypeMap.find(T);
    assert(I != TypeMap.end() && "Type not in ValueEnumerator!");
    return I->second-1;
  }

  unsigned getInstructionID(const Instruction *I) const;
  void setInstructionID(const Instruction *I);

  unsigned getAttributeID(const AttrListPtr &PAL) const {
    if (PAL.isEmpty()) return 0;  // Null maps to zero.
    AttributeMapType::const_iterator I = AttributeMap.find(PAL.getRawPointer());
    assert(I != AttributeMap.end() && "Attribute not in ValueEnumerator!");
    return I->second;
  }

  /// getFunctionConstantRange - Return the range of values that corresponds to
  /// function-local constants.
  void getFunctionConstantRange(unsigned &Start, unsigned &End) const {
    Start = FirstFuncConstantID;
    End = FirstInstID;
  }
  
  const ValueList &getValues() const { return Values; }
  const ValueList &getMDValues() const { return MDValues; }
  const TypeList &getTypes() const { return Types; }
  const std::vector<const BasicBlock*> &getBasicBlocks() const {
    return BasicBlocks; 
  }
  const std::vector<AttrListPtr> &getAttributes() const {
    return Attributes;
  }

  /// incorporateFunction/purgeFunction - If you'd like to deal with a function,
  /// use these two methods to get its data into the ValueEnumerator!
  ///
  void incorporateFunction(const Function &F);
  void purgeFunction();

private:
  void OptimizeConstants(unsigned CstStart, unsigned CstEnd);
    
  void EnumerateMetadata(const MetadataBase *MD);
  void EnumerateValue(const Value *V);
  void EnumerateType(const Type *T);
  void EnumerateOperandType(const Value *V);
  void EnumerateAttributes(const AttrListPtr &PAL);
  
  void EnumerateTypeSymbolTable(const TypeSymbolTable &ST);
  void EnumerateValueSymbolTable(const ValueSymbolTable &ST);
};

} // End llvm namespace

#endif
