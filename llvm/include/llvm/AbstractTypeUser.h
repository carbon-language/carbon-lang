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
  virtual void refineAbstractType(const DerivedType *OldTy,
				  const Type *NewTy) = 0;
};


// PATypeHandle - Handle to a Type subclass.  This class is parameterized so
// that users can have handles to MethodType's that are still specialized, for
// example.  This class is a simple class used to keep the use list of abstract
// types up-to-date.
//
template <class TypeSubClass>
class PATypeHandle {
  const TypeSubClass *Ty;
  AbstractTypeUser * const User;

  // These functions are defined at the bottom of Type.h.  See the comment there
  // for justification.
  inline void addUser();
  inline void removeUser();
public:
  // ctor - Add use to type if abstract.  Note that Ty must not be null
  inline PATypeHandle(const TypeSubClass *ty, AbstractTypeUser *user) 
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
  inline operator const TypeSubClass *() const { return Ty; }
  inline const TypeSubClass *get() const { return Ty; }

  // operator= - Allow assignment to handle
  inline const TypeSubClass *operator=(const TypeSubClass *ty) {
    if (Ty != ty) {   // Ensure we don't accidentally drop last ref to Ty
      removeUser();
      Ty = ty;
      addUser();
    }
    return Ty;
  }

  // operator= - Allow assignment to handle
  inline const TypeSubClass *operator=(const PATypeHandle &T) {
    return operator=(T.Ty);
  }

  inline bool operator==(const TypeSubClass *ty) {
    return Ty == ty;
  }

  // operator-> - Allow user to dereference handle naturally...
  inline const TypeSubClass *operator->() const { return Ty; }
};


// PATypeHolder - Holder class for a potentially abstract type.  This functions
// as both a handle (as above) and an AbstractTypeUser.  It uses the callback to
// keep its pointer member updated to the current version of the type.
//
template <class TypeSC>
class PATypeHolder : public AbstractTypeUser, public PATypeHandle<TypeSC> {
public:
  inline PATypeHolder(const TypeSC *ty) : PATypeHandle<TypeSC>(ty, this) {}
  inline PATypeHolder(const PATypeHolder &T)
    : AbstractTypeUser(T), PATypeHandle<TypeSC>(T, this) {}

  // refineAbstractType - All we do is update our PATypeHandle member to point
  // to the new type.
  //
  virtual void refineAbstractType(const DerivedType *OldTy, const Type *NewTy) {
    assert(get() == OldTy && "Can't refine to unknown value!");
    PATypeHandle<TypeSC>::operator=((const TypeSC*)NewTy);
  }

  // operator= - Allow assignment to handle
  inline const TypeSC *operator=(const TypeSC *ty) {
    return PATypeHandle<TypeSC>::operator=(ty);
  }

  // operator= - Allow assignment to handle
  inline const TypeSC *operator=(const PATypeHandle<TypeSC> &T) {
    return PATypeHandle<TypeSC>::operator=(T);
  }
  inline const TypeSC *operator=(const PATypeHolder<TypeSC> &H) {
    return PATypeHandle<TypeSC>::operator=(H);
  }
};


#endif
