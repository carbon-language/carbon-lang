//===-- llvm/ConstantPool.h - Define the constant pool class ------*- C++ -*-=//
//
// This file implements a constant pool that is split into different type 
// planes.  This allows searching for a typed object to go a little faster.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CONSTANTPOOL_H
#define LLVM_CONSTANTPOOL_H

#include <vector>
#include "llvm/ValueHolder.h"
class SymTabValue;
class ConstPoolVal;
class Type;
class Value;

class ConstantPool {
public:
  typedef ValueHolder<ConstPoolVal, SymTabValue, SymTabValue> PlaneType;
private:
  typedef vector<PlaneType*> PlanesType;
  PlanesType Planes;
  SymTabValue *Parent;

  inline void resize(unsigned size);
public:
  inline ConstantPool(SymTabValue *P) { Parent = P; }
  inline ~ConstantPool() { delete_all(); }

  inline       SymTabValue *getParent()       { return Parent; }
  inline const SymTabValue *getParent() const { return Parent; }
  const Value *getParentV() const;
        Value *getParentV()      ;

  void setParent(SymTabValue *STV);

  void dropAllReferences();  // Drop all references to other constants

  // Constant getPlane - Returns true if the type plane does not exist, 
  // otherwise updates the pointer to point to the correct plane.
  //
  bool getPlane(const Type *T, const PlaneType *&Plane) const;
  bool getPlane(const Type *T,       PlaneType *&Plane);

  // Normal getPlane - Resizes constant pool to contain type even if it doesn't
  // already have it.
  //
  PlaneType &getPlane(const Type *T);

  // insert - Add constant into the symbol table...
  void insert(ConstPoolVal *N);
  bool remove(ConstPoolVal *N);   // Returns true on failure 

  // delete_all - Remove all elements from the constant pool
  void delete_all();

  // find - Search to see if a constant of the specified value is already in
  // the constant table.
  //
  const ConstPoolVal *find(const ConstPoolVal *V) const;
        ConstPoolVal *find(const ConstPoolVal *V)      ;
  const ConstPoolVal *find(const Type *Ty) const;
        ConstPoolVal *find(const Type *Ty)      ;

  // Plane iteration support
  //
  typedef PlanesType::iterator       plane_iterator;
  typedef PlanesType::const_iterator plane_const_iterator;

  inline plane_iterator       begin()       { return Planes.begin(); }
  inline plane_const_iterator begin() const { return Planes.begin(); }
  inline plane_iterator       end()         { return Planes.end(); }
  inline plane_const_iterator end()   const { return Planes.end(); }

  // ensureTypeAvailable - This is used to make sure that the specified type is
  // in the constant pool.  If it is not already in the constant pool, it is
  // added.
  //
  const Type *ensureTypeAvailable(const Type *);
};

#endif
