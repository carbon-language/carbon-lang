//===-- llvm/DerivedTypes.h - Classes for handling data types ----*- C++ -*--=//
//
// This file contains the declarations of classes that represent "derived 
// types".  These are things like "arrays of x" or "structure of x, y, z" or
// "method returning x taking (y,z) as parameters", etc...
//
// The implementations of these classes live in the Type.cpp file.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DERIVED_TYPES_H
#define LLVM_DERIVED_TYPES_H

#include "llvm/Type.h"

class DerivedType : public Type {
  // AbstractTypeUsers - Implement a list of the users that need to be notified
  // if I am a type, and I get resolved into a more concrete type.
  //
  ///// FIXME: kill mutable nonsense when Type's are not const
  mutable vector<AbstractTypeUser *> AbstractTypeUsers;

  char isRefining;                                   // Used for recursive types

protected:
  inline DerivedType(const string &Name, PrimitiveID id) : Type(Name, id) {
    isRefining = false;
  }

  // typeIsRefined - Notify AbstractTypeUsers of this type that the current type
  // has been refined a bit.  The pointer is still valid and still should be
  // used, but the subtypes have changed.
  //
  void typeIsRefined();
  
  // setDerivedTypeProperties - Based on the subtypes, set the name of this
  // type so that it is printed nicely by the type printer.  Also calculate
  // whether this type is abstract or not.  Used by the constructor and when
  // the type is refined.
  //
  void setDerivedTypeProperties();

public:

  //===--------------------------------------------------------------------===//
  // Abstract Type handling methods - These types have special lifetimes, which
  // are managed by (add|remove)AbstractTypeUser. See comments in
  // AbstractTypeUser.h for more information.

  // addAbstractTypeUser - Notify an abstract type that there is a new user of
  // it.  This function is called primarily by the PATypeHandle class.
  //
  void addAbstractTypeUser(AbstractTypeUser *U) const {
    assert(isAbstract() && "addAbstractTypeUser: Current type not abstract!");
#if 0
    cerr << "  addAbstractTypeUser[" << (void*)this << ", " << getDescription() 
	 << "][" << AbstractTypeUsers.size() << "] User = " << U << endl;
#endif
    AbstractTypeUsers.push_back(U);
  }

  // removeAbstractTypeUser - Notify an abstract type that a user of the class
  // no longer has a handle to the type.  This function is called primarily by
  // the PATypeHandle class.  When there are no users of the abstract type, it
  // is anihilated, because there is no way to get a reference to it ever again.
  //
  void removeAbstractTypeUser(AbstractTypeUser *U) const;

  // getNumAbstractTypeUsers - Return the number of users registered to the type
  inline unsigned getNumAbstractTypeUsers() const {
    assert(isAbstract() && "getNumAbstractTypeUsers: Type not abstract!");
    return AbstractTypeUsers.size(); 
  }

  // refineAbstractTypeTo - This function is used to when it is discovered that
  // the 'this' abstract type is actually equivalent to the NewType specified.
  // This causes all users of 'this' to switch to reference the more concrete
  // type NewType and for 'this' to be deleted.
  //
  void refineAbstractTypeTo(const Type *NewType);
};




class MethodType : public DerivedType {
public:
  typedef vector<PATypeHandle<Type> > ParamTypes;
private:
  PATypeHandle<Type> ResultType;
  ParamTypes ParamTys;
  bool isVarArgs;

  MethodType(const MethodType &);                   // Do not implement
  const MethodType &operator=(const MethodType &);  // Do not implement
protected:
  // This should really be private, but it squelches a bogus warning
  // from GCC to make them protected:  warning: `class MethodType' only 
  // defines private constructors and has no friends

  // Private ctor - Only can be created by a static member...
  MethodType(const Type *Result, const vector<const Type*> &Params, 
             bool IsVarArgs);

public:

  inline bool isVarArg() const { return isVarArgs; }
  inline const Type *getReturnType() const { return ResultType; }
  inline const ParamTypes &getParamTypes() const { return ParamTys; }


  virtual const Type *getContainedType(unsigned i) const {
    return i == 0 ? ResultType : (i <= ParamTys.size() ? ParamTys[i-1] : 0);
  }
  virtual unsigned getNumContainedTypes() const { return ParamTys.size()+1; }

  // refineAbstractType - Called when a contained type is found to be more
  // concrete - this could potentially change us from an abstract type to a
  // concrete type.
  //
  virtual void refineAbstractType(const DerivedType *OldTy, const Type *NewTy);

  static MethodType *get(const Type *Result, const vector<const Type*> &Params);
};


class ArrayType : public DerivedType {
private:
  PATypeHandle<Type> ElementType;
  int NumElements;       // >= 0 for sized array, -1 for unbounded/unknown array

  ArrayType(const ArrayType &);                   // Do not implement
  const ArrayType &operator=(const ArrayType &);  // Do not implement
protected:
  // This should really be private, but it squelches a bogus warning
  // from GCC to make them protected:  warning: `class ArrayType' only 
  // defines private constructors and has no friends


  // Private ctor - Only can be created by a static member...
  ArrayType(const Type *ElType, int NumEl);

public:

  inline const Type *getElementType() const { return ElementType; }
  inline int         getNumElements() const { return NumElements; }

  inline bool isSized()   const { return NumElements >= 0; }
  inline bool isUnsized() const { return NumElements == -1; }

  virtual const Type *getContainedType(unsigned i) const { 
    return i == 0 ? ElementType : 0;
  }
  virtual unsigned getNumContainedTypes() const { return 1; }

  // refineAbstractType - Called when a contained type is found to be more
  // concrete - this could potentially change us from an abstract type to a
  // concrete type.
  //
  virtual void refineAbstractType(const DerivedType *OldTy, const Type *NewTy);

  static ArrayType *get(const Type *ElementType, int NumElements = -1);
};


class StructType : public DerivedType {
public:
  typedef vector<PATypeHandle<Type> > ElementTypes;

private:
  ElementTypes ETypes;                              // Element types of struct

  StructType(const StructType &);                   // Do not implement
  const StructType &operator=(const StructType &);  // Do not implement

protected:
  // This should really be private, but it squelches a bogus warning
  // from GCC to make them protected:  warning: `class StructType' only 
  // defines private constructors and has no friends

  // Private ctor - Only can be created by a static member...
  StructType(const vector<const Type*> &Types);
  
public:
  inline const ElementTypes &getElementTypes() const { return ETypes; }

  virtual const Type *getContainedType(unsigned i) const { 
    return i < ETypes.size() ? ETypes[i] : 0;
  }
  virtual unsigned getNumContainedTypes() const { return ETypes.size(); }

  // refineAbstractType - Called when a contained type is found to be more
  // concrete - this could potentially change us from an abstract type to a
  // concrete type.
  //
  virtual void refineAbstractType(const DerivedType *OldTy, const Type *NewTy);

  static StructType *get(const vector<const Type*> &Params);
};


class PointerType : public DerivedType {
private:
  PATypeHandle<Type> ValueType;

  PointerType(const PointerType &);                   // Do not implement
  const PointerType &operator=(const PointerType &);  // Do not implement
protected:
  // This should really be private, but it squelches a bogus warning
  // from GCC to make them protected:  warning: `class PointerType' only 
  // defines private constructors and has no friends


  // Private ctor - Only can be created by a static member...
  PointerType(const Type *ElType);

public:

  inline const Type *getValueType() const { return ValueType; }

  virtual const Type *getContainedType(unsigned i) const { 
    return i == 0 ? ValueType : 0;
  }
  virtual unsigned getNumContainedTypes() const { return 1; }

  static PointerType *get(const Type *ElementType);

  // refineAbstractType - Called when a contained type is found to be more
  // concrete - this could potentially change us from an abstract type to a
  // concrete type.
  //
  virtual void refineAbstractType(const DerivedType *OldTy, const Type *NewTy);
};


class OpaqueType : public DerivedType {
private:
  OpaqueType(const OpaqueType &);                   // Do not implement
  const OpaqueType &operator=(const OpaqueType &);  // Do not implement
protected:
  // This should really be private, but it squelches a bogus warning
  // from GCC to make them protected:  warning: `class OpaqueType' only 
  // defines private constructors and has no friends

  // Private ctor - Only can be created by a static member...
  OpaqueType();

public:

  // get - Static factory method for the OpaqueType class...
  static OpaqueType *get() {
    return new OpaqueType();           // All opaque types are distinct
  }
};


// Define some inline methods for the AbstractTypeUser.h:PATypeHandle class.
// These are defined here because they MUST be inlined, yet are dependant on 
// the definition of the Type class.  Of course Type derives from Value, which
// contains an AbstractTypeUser instance, so there is no good way to factor out
// the code.  Hence this bit of uglyness.
//
template <class TypeSubClass> void PATypeHandle<TypeSubClass>::addUser() {
  if (Ty->isAbstract())
    Ty->castDerivedTypeAsserting()->addAbstractTypeUser(User);
}
template <class TypeSubClass> void PATypeHandle<TypeSubClass>::removeUser() {
  if (Ty->isAbstract())
    Ty->castDerivedTypeAsserting()->removeAbstractTypeUser(User);
}

#endif
