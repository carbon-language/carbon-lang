//===-- Internal/SlotTable.h - Type/Value Slot Holder -----------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file declares the SlotTable class for type plane numbering.
// 
//===----------------------------------------------------------------------===//

#ifndef LLVM_INTERNAL_SLOTTABLE_H
#define LLVM_INTERNAL_SLOTTABLE_H

#include <vector>
#include <map>

namespace llvm {

// Forward declarations
class Value;
class Type;
class Module;
class Function;
class SymbolTable;
class ConstantArray;

/// This class is the common abstract data type for both the SlotMachine and
/// the SlotCalculator. It provides the two-way mapping between Values and 
/// Slots as well as the two-way mapping between Types and Slots. For Values,
/// the slot number can be extracted by simply using the getSlot()
/// method and passing in the Value. For Types, it is the same. 
/// @brief Abstract data type for slot numbers.
class SlotTable
{
/// @name Types
/// @{
public:

  /// This type is used throughout the code to make it clear that 
  /// an unsigned value refers to a Slot number and not something else.
  /// @brief Type slot number identification type.
  typedef unsigned SlotNum;

  /// This type is used throughout the code to make it clear that an
  /// unsigned value refers to a type plane number and not something else.
  /// @brief The type of a plane number (corresponds to Type::PrimitiveID).
  typedef unsigned PlaneNum;

  /// @brief Some constants used as flags instead of actual slot numbers
  enum Constants {
      MAX_SLOT = 4294967294U,
      BAD_SLOT = 4294967295U
  };

  /// @brief A single plane of Values. Intended index is slot number.
  typedef std::vector<const Value*> ValuePlane; 

  /// @brief A table of Values. Intended index is Type::PrimitiveID.
  typedef std::vector<ValuePlane> ValueTable; 

  /// @brief A map of values to slot numbers.
  typedef std::map<const Value*,SlotNum> ValueMap; 

  /// @brief A single plane of Types. Intended index is slot number.
  typedef std::vector<const Type*>  TypePlane;

  /// @brief A map of types to slot numbers.
  typedef std::map<const Type*,SlotNum> TypeMap;

/// @}
/// @name Constructors
/// @{
public:
  /// This constructor initializes all the containers in the SlotTable
  /// to empty and then inserts all the primitive types into the type plane
  /// by default. This is done as a convenience since most uses of the
  /// SlotTable will need the primitive types. If you don't need them, pass
  /// in true.
  /// @brief Default Constructor
  SlotTable( 
      bool dont_insert_primitives = false ///< Control insertion of primitives.
  );

/// @}
/// @name Accessors
/// @{
public:
  /// @brief Get the number of planes of values.
  size_t value_size() const { return vTable.size(); }

  /// @brief Get the number of types.
  size_t type_size() const { return tPlane.size(); }

  /// @brief Determine if a specific type plane in the value table exists
  bool plane_exists(PlaneNum plane) const {
    return vTable.size() > plane;
  }

  /// @brief Determine if a specific type plane in the value table is empty
  bool plane_empty(PlaneNum plane) const {
    return (plane_exists(plane) ? vTable[plane].empty() : true);
  }

  /// @brief Get the number of entries in a specific plane of the value table
  size_t plane_size(PlaneNum plane) const {
    return (plane_exists(plane) ? vTable[plane].size() : 0 );
  }

  /// @returns true if the slot table is completely empty.
  /// @brief Determine if the SlotTable is empty.
  bool empty() const;

  /// @returns the slot number or BAD_SLOT if Val is not in table.
  /// @brief Get a slot number for a Value.
  SlotNum getSlot(const Value* Val) const;

  /// @returns the slot number or BAD_SLOT if Type is not in the table.
  /// @brief Get a slot number for a Type.
  SlotNum getSlot(const Type* Typ) const;

  /// @returns true iff the Value is in the table.
  /// @brief Determine if a Value has a slot number.
  bool hasSlot(const Value* Val) { return getSlot(Val) != BAD_SLOT; }

  /// @returns true iff the Type is in the table.
  /// @brief Determine if a Type has a slot number.
  bool hasSlot(const Type* Typ) { return getSlot(Typ) != BAD_SLOT; }

/// @}
/// @name Mutators
/// @{
public:
  /// @brief Completely clear the SlotTable;
  void clear();

  /// @brief Resize the table to incorporate at least \p new_size planes
  void resize( size_t new_size );

  /// @returns the slot number of the newly inserted value in its plane
  /// @brief Add a Value to the SlotTable
  SlotNum insert(const Value* Val, PlaneNum plane );

  /// @returns the slot number of the newly inserted type
  /// @brief Add a Type to the SlotTable
  SlotNum insert( const Type* Typ );

  /// @returns the slot number that \p Val had when it was in the table
  /// @brief Remove a Value from the SlotTable
  SlotNum remove( const Value* Val, PlaneNum plane );

  /// @returns the slot number that \p Typ had when it was in the table
  /// @brief Remove a Type from the SlotTable
  SlotNum remove( const Type* Typ );

/// @}
/// @name Implementation Details
/// @{
private:
  /// Insert the primitive types into the type plane. This is called
  /// by the constructor to initialize the type plane.
  void insertPrimitives();

/// @}
/// @name Data
/// @{
private:
  /// A two dimensional table of Values indexed by type and slot number. This
  /// allows for efficient lookup of a Value by its type and slot number.
  ValueTable vTable;

  /// A map of Values to unsigned integer. This allows for efficient lookup of
  /// A Value's slot number in its type plane. 
  ValueMap   vMap;

  /// A one dimensional vector of Types indexed by slot number. Types are
  /// handled separately because they are not Values. 
  TypePlane  tPlane;

  /// A map of Types to unsigned integer. This allows for efficient lookup of
  /// a Type's slot number in the type plane.
  TypeMap    tMap;

/// @}

};

} // End llvm namespace

// vim: sw=2 

#endif
