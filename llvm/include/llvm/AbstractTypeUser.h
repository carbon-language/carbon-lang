//===-- llvm/AbstractTypeUser.h - AbstractTypeUser Interface -----*- C++ -*--=//
//
// The AbstractTypeUser class is an interface to be implemented by classes who
// could possible use an abstract type.  Abstract types are denoted by the
// isAbstract flag set to true in the Type class.  These are classes that
// contain an Opaque type in their structure somehow.
//
// Classes must implement this interface so that they may be notified when an
// abstract type is resolved.  Abstract types may be resolved into more concrete
// types through: linking, parsing, and bytecode reading.  When this happens,
// all of the users of the type must be updated to reference the new, more
// concrete type.  They are notified through the AbstractTypeUser interface.
//
// In addition to this, AbstractTypeUsers must keep the use list of the
// potentially abstract type that they reference up-to-date.  To do this in a
// nice, transparent way, the PATypeHandle class is used to hold "Potentially
// Abstract Types", and keep the use list of the abstract types up-to-date.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ABSTRACT_TYPE_USER_H
#define LLVM_ABSTRACT_TYPE_USER_H

//
// This is the "master" include for assert.h
// Whether this file needs it or not, it must always include assert.h for the
// files which include llvm/AbstractTypeUser.h
//
// In this way, most every LLVM source file will have access to the assert()
// macro without having to #include <assert.h> directly.
//
#include "Config/assert.h"

class Type;
class DerivedType;

class AbstractTypeUser {
protected:
  virtual ~AbstractTypeUser() {}                        // Derive from me
public:

  // refineAbstractType - The callback method invoked when an abstract type
  // has been found to be more concrete.  A class must override this method to
  // update its internal state to reference NewType instead of OldType.  Soon
  // after this method is invoked, OldType shall be deleted, so referencing it
  // is quite unwise.
  //
  // Another case that is important to consider is when a type is refined, but
  // stays in the same place in memory.  In this case OldTy will equal NewTy.
  // This callback just notifies ATU's that the underlying structure of the type
  // has changed... but any previously used properties are still valid.
  //
  // Note that it is possible to refine a type with parameters OldTy==NewTy, and
  // OldTy is no longer abstract.  In this case, abstract type users should
  // release their hold on a type, because it went from being abstract to
  // concrete.
  //
  virtual void refineAbstractType(const DerivedType *OldTy,
				  const Type *NewTy) = 0;
  // for debugging...
  virtual void dump() const = 0;
};


// PATypeHandle - Handle to a Type subclass.  This class is parameterized so
// that users can have handles to FunctionType's that are still specialized, for
// example.  This class is a simple class used to keep the use list of abstract
// types up-to-date.
//
class PATypeHandle {
  const Type *Ty;
  AbstractTypeUser * const User;

  // These functions are defined at the bottom of Type.h.  See the comment there
  // for justification.
  inline void addUser();
  inline void removeUser();
public:
  // ctor - Add use to type if abstract.  Note that Ty must not be null
  inline PATypeHandle(const Type *ty, AbstractTypeUser *user) 
    : Ty(ty), User(user) {
    addUser();
  }

  // ctor - Add use to type if abstract.
  inline PATypeHandle(const PATypeHandle &T) : Ty(T.Ty), User(T.User) {
    addUser();
  }

  // dtor - Remove reference to type...
  inline ~PATypeHandle() { removeUser(); }

  // Automatic casting operator so that the handle may be used naturally
  inline operator const Type *() const { return Ty; }
  inline const Type *get() const { return Ty; }

  // operator= - Allow assignment to handle
  inline const Type *operator=(const Type *ty) {
    if (Ty != ty) {   // Ensure we don't accidentally drop last ref to Ty
      removeUser();
      Ty = ty;
      addUser();
    }
    return Ty;
  }

  // operator= - Allow assignment to handle
  inline const Type *operator=(const PATypeHandle &T) {
    return operator=(T.Ty);
  }

  inline bool operator==(const Type *ty) {
    return Ty == ty;
  }

  // operator-> - Allow user to dereference handle naturally...
  inline const Type *operator->() const { return Ty; }

  // removeUserFromConcrete - This function should be called when the User is
  // notified that our type is refined... and the type is being refined to
  // itself, which is now a concrete type.  When a type becomes concrete like
  // this, we MUST remove ourself from the AbstractTypeUser list, even though
  // the type is apparently concrete.
  //
  inline void removeUserFromConcrete();
};


// PATypeHolder - Holder class for a potentially abstract type.  This functions
// as both a handle (as above) and an AbstractTypeUser.  It uses the callback to
// keep its pointer member updated to the current version of the type.
//
struct PATypeHolder : public AbstractTypeUser, public PATypeHandle {
  inline PATypeHolder(const Type *ty) : PATypeHandle(ty, this) {}
  inline PATypeHolder(const PATypeHolder &T)
    : AbstractTypeUser(T), PATypeHandle(T, this) {}

  // refineAbstractType - All we do is update our PATypeHandle member to point
  // to the new type.
  //
  virtual void refineAbstractType(const DerivedType *OldTy, const Type *NewTy) {
    assert(get() == (const Type*)OldTy && "Can't refine to unknown value!");

    // Check to see if the type just became concrete.  If so, we have to
    // removeUser to get off its AbstractTypeUser list
    removeUserFromConcrete();

    if ((const Type*)OldTy != NewTy)
      PATypeHandle::operator=(NewTy);
  }

  // operator= - Allow assignment to handle
  inline const Type *operator=(const Type *ty) {
    return PATypeHandle::operator=(ty);
  }

  // operator= - Allow assignment to handle
  inline const Type *operator=(const PATypeHandle &T) {
    return PATypeHandle::operator=(T);
  }
  inline const Type *operator=(const PATypeHolder &H) {
    return PATypeHandle::operator=(H);
  }

  void dump() const;
};

#endif
