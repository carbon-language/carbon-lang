//===-- SlotTable.cpp - Abstract data type for slot numbers ---------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements an abstract data type for keeping track of slot numbers 
// for bytecode and assembly writing or any other purpose. 
//
//===----------------------------------------------------------------------===//

#include "llvm/Constants.h"
#include "llvm/Type.h"
#include "llvm/GlobalValue.h"
#include "llvm/Internal/SlotTable.h"

using namespace llvm;

//===----------------------------------------------------------------------===//
//                            SlotTable Implementation
//===----------------------------------------------------------------------===//

SlotTable::SlotTable( bool dont_insert_primitives ) {
  if ( ! dont_insert_primitives ) 
    this->insertPrimitives();
}

// empty - determine if the slot table is completely empty.
bool SlotTable::empty() const {
  return vTable.empty() && vMap.empty() && tPlane.empty() && tMap.empty();
}

// getSlot - get the slot number associated with value Val
SlotTable::SlotNum SlotTable::getSlot(const Value* Val) const {
  ValueMap::const_iterator I = vMap.find( Val );
  if ( I != vMap.end() )
    return I->second;

  // Do not number ConstantPointerRef's at all.  They are an abomination.
  if (const ConstantPointerRef *CPR = dyn_cast<ConstantPointerRef>(Val))
    return this->getSlot(CPR->getValue());

  return BAD_SLOT;
}

// getSlot - get the slot number associated with type Typ
SlotTable::SlotNum SlotTable::getSlot(const Type* Typ) const {
  TypeMap::const_iterator I = tMap.find( Typ );
  if ( I != tMap.end() )
    return I->second;

  return BAD_SLOT;
}

// clear - completely clear the slot table of all entries
void SlotTable::clear() {
  vTable.clear();
  vMap.clear();
  tPlane.clear();
  tMap.clear();
}

// resize - make sure there's enough room for specific number of planes
void SlotTable::resize( size_t new_size ) {
  vTable.resize( new_size );
}

// insert - insert a Value into a specific plane
SlotTable::SlotNum SlotTable::insert( const Value* Val, PlaneNum plane ) {
  if ( vTable.size() <= plane ) // Make sure we have the type plane allocated
    vTable.resize(plane+1, ValuePlane());

  // Insert node into table and map
  SlotNum DestSlot = vMap[Val] = vTable[plane].size();
  vTable[plane].push_back(Val);
  return DestSlot;
}

// insert - insert a type 
SlotTable::SlotNum SlotTable::insert( const Type* Typ ) {
  // Insert node into table and map making sure that
  // the same type isn't inserted twice.
  assert(tMap.find(Typ) == tMap.end() && "Can't insert a Type multiple times");
  SlotNum DestSlot = tMap[Typ] = tPlane.size();
  tPlane.push_back(Typ);
  return DestSlot;
}

// remove - remove a value from the slot table
SlotTable::SlotNum SlotTable::remove( const Value* Val, PlaneNum plane ) {
  // FIXME: not implemented - not sure we need it
  return BAD_SLOT;
}

// remove - remove a type from the slot table
SlotTable::SlotNum SlotTable::remove( const Type* Typ ) {
  // FIXME: not implemented - not sure we need it
  return BAD_SLOT;
}

// insertPrimitives - insert the primitive types for initialization
// Make sure that all of the primitive types are in the table
// and that their Primitive ID is equal to their slot #
void SlotTable::insertPrimitives() {
  for (PlaneNum plane = 0; plane < Type::FirstDerivedTyID; ++plane) {
    const Type* Ty = Type::getPrimitiveType((Type::PrimitiveID) plane);
    assert(Ty && "Couldn't get primitive type id");
    SlotNum slot = this->insert(Ty);
    assert(slot == plane && "Type slot didn't match plane number");
  }
}

// vim: sw=2
