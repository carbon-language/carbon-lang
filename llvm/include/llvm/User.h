//===-- llvm/User.h - User class definition ---------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This class defines the interface that one who 'use's a Value must implement.
// Each instance of the Value class keeps track of what User's have handles
// to it.
//
//  * Instructions are the largest class of User's.
//  * Constants may be users of other constants (think arrays and stuff)
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_USER_H
#define LLVM_USER_H

#include "llvm/Value.h"

namespace llvm {

/*==============================================================================


   -----------------------------------------------------------------
   --- Interaction and relationship between User and Use objects ---
   -----------------------------------------------------------------


A subclass of User can choose between incorporating its Use objects
or refer to them out-of-line by means of a pointer. A mixed variant
(some Uses inline others hung off) is impractical and breaks the invariant
that the Use objects belonging to the same User form a contiguous array.

We have 2 different layouts in the User (sub)classes:

Layout a)
The Use object(s) are inside (resp. at fixed offset) of the User
object and there are a fixed number of them.

Layout b)
The Use object(s) are referenced by a pointer to an
array from the User object and there may be a variable
number of them.

Initially each layout will possess a direct pointer to the
start of the array of Uses. Though not mandatory for layout a),
we stick to this redundancy for the sake of simplicity.
The User object will also store the number of Use objects it
has. (Theoretically this information can also be calculated
given the scheme presented below.)

Special forms of allocation operators (operator new)
will enforce the following memory layouts:


#  Layout a) will be modelled by prepending the User object
#  by the Use[] array.
#      
#      ...---.---.---.---.-------...
#        | P | P | P | P | User
#      '''---'---'---'---'-------'''


#  Layout b) will be modelled by pointing at the Use[] array.
#      
#      .-------...
#      | User
#      '-------'''
#          |
#          v
#          .---.---.---.---...
#          | P | P | P | P |
#          '---'---'---'---'''

   (In the above figures 'P' stands for the Use** that
    is stored in each Use object in the member Use::Prev)


Since the Use objects will be deprived of the direct pointer to
their User objects, there must be a fast and exact method to
recover it. This is accomplished by the following scheme:

A bit-encoding in the 2 LSBits of the Use::Prev will allow to find the
start of the User object:

00 --> binary digit 0
01 --> binary digit 1
10 --> stop and calc (s)
11 --> full stop (S)

Given a Use*, all we have to do is to walk till we get
a stop and we either have a User immediately behind or
we have to walk to the next stop picking up digits
and calculating the offset:

.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.---.----------------
| 1 | s | 1 | 0 | 1 | 0 | s | 1 | 1 | 0 | s | 1 | 1 | s | 1 | S | User (or User*)
'---'---'---'---'---'---'---'---'---'---'---'---'---'---'---'---'----------------
    |+15                |+10            |+6         |+3     |+1
    |                   |               |           |       |__>
    |                   |               |           |__________>
    |                   |               |______________________>
    |                   |______________________________________>
    |__________________________________________________________>


Only the significant number of bits need to be stored between the
stops, so that the worst case is 20 memory accesses when there are
1000 Use objects.

The following literate Haskell fragment demonstrates the concept:

> import Test.QuickCheck
> 
> digits :: Int -> [Char] -> [Char]
> digits 0 acc = '0' : acc
> digits 1 acc = '1' : acc
> digits n acc = digits (n `div` 2) $ digits (n `mod` 2) acc
> 
> dist :: Int -> [Char] -> [Char]
> dist 0 [] = ['S']
> dist 0 acc = acc
> dist 1 acc = let r = dist 0 acc in 's' : digits (length r) r
> dist n acc = dist (n - 1) $ dist 1 acc
> 
> takeLast n ss = reverse $ take n $ reverse ss
> 
> test = takeLast 40 $ dist 20 []
> 

Printing <test> gives: "1s100000s11010s10100s1111s1010s110s11s1S"

The reverse algorithm computes the
length of the string just by examining
a certain prefix:

> pref :: [Char] -> Int
> pref "S" = 1
> pref ('s':'1':rest) = decode 2 1 rest
> pref (_:rest) = 1 + pref rest
> 
> decode walk acc ('0':rest) = decode (walk + 1) (acc * 2) rest
> decode walk acc ('1':rest) = decode (walk + 1) (acc * 2 + 1) rest
> decode walk acc _ = walk + acc
> 

Now, as expected, printing <pref test> gives 40.

We can quickCheck this with following property:

> testcase = dist 2000 []
> testcaseLength = length testcase
> 
> identityProp n = n > 0 && n <= testcaseLength ==> length arr == pref arr
>     where arr = takeLast n testcase

As expected <quickCheck identityProp> gives:

*Main> quickCheck identityProp
OK, passed 100 tests.

Let's be a bit more exhaustive:

> 
> deepCheck p = check (defaultConfig { configMaxTest = 500 }) p
> 

And here is the result of <deepCheck identityProp>:

*Main> deepCheck identityProp
OK, passed 500 tests.


To maintain the invariant that the 2 LSBits of each Use** in Use
never change after being set up, setters of Use::Prev must re-tag the
new Use** on every modification. Accordingly getters must strip the
tag bits.

For layout b) instead of the User we will find a pointer (User* with LSBit set).
Following this pointer brings us to the User. A portable trick will ensure
that the first bytes of User (if interpreted as a pointer) will never have
the LSBit set.

==============================================================================*/

/// OperandTraits - Compile-time customization of
/// operand-related allocators and accessors
/// for use of the User class
template <class>
struct OperandTraits;

class User;

/// OperandTraits<User> - specialization to User
template <>
struct OperandTraits<User> {
  static inline Use *op_begin(User*);
  static inline Use *op_end(User*);
  static inline unsigned operands(const User*);
  template <class U>
  struct Layout {
    typedef U overlay;
  };
  static inline void *allocate(unsigned);
};

class User : public Value {
  User(const User &);             // Do not implement
  void *operator new(size_t);     // Do not implement
  template <unsigned>
  friend struct HungoffOperandTraits;
protected:
  /// OperandList - This is a pointer to the array of Users for this operand.
  /// For nodes of fixed arity (e.g. a binary operator) this array will live
  /// prefixed to the derived class.  For nodes of resizable variable arity
  /// (e.g. PHINodes, SwitchInst etc.), this memory will be dynamically
  /// allocated and should be destroyed by the classes' 
  /// virtual dtor.
  Use *OperandList;

  /// NumOperands - The number of values used by this User.
  ///
  unsigned NumOperands;

  void *operator new(size_t s, unsigned Us);
  User(const Type *ty, unsigned vty, Use *OpList, unsigned NumOps)
    : Value(ty, vty), OperandList(OpList), NumOperands(NumOps) {}
  Use *allocHungoffUses(unsigned) const;
  void dropHungoffUses(Use *U) {
    if (OperandList == U) {
      OperandList = 0;
      NumOperands = 0;
    }
    Use::zap(U, U->getImpliedUser(), true);
  }
public:
  ~User() {
    Use::zap(OperandList, OperandList + NumOperands);
  }
  /// operator delete - free memory allocated for User and Use objects
  void operator delete(void *Usr);
  /// placement delete - required by std, but never called.
  void operator delete(void*, unsigned) {
    assert(0 && "Constructor throws?");
  }
  template <unsigned Idx> Use &Op() {
    return OperandTraits<User>::op_begin(this)[Idx];
  }
  template <unsigned Idx> const Use &Op() const {
    return OperandTraits<User>::op_begin(const_cast<User*>(this))[Idx];
  }
  Value *getOperand(unsigned i) const {
    assert(i < NumOperands && "getOperand() out of range!");
    return OperandList[i];
  }
  void setOperand(unsigned i, Value *Val) {
    assert(i < NumOperands && "setOperand() out of range!");
    OperandList[i] = Val;
  }
  unsigned getNumOperands() const { return NumOperands; }

  // ---------------------------------------------------------------------------
  // Operand Iterator interface...
  //
  typedef Use*       op_iterator;
  typedef const Use* const_op_iterator;

  inline op_iterator       op_begin()       { return OperandList; }
  inline const_op_iterator op_begin() const { return OperandList; }
  inline op_iterator       op_end()         { return OperandList+NumOperands; }
  inline const_op_iterator op_end()   const { return OperandList+NumOperands; }

  // dropAllReferences() - This function is in charge of "letting go" of all
  // objects that this User refers to.  This allows one to
  // 'delete' a whole class at a time, even though there may be circular
  // references...  First all references are dropped, and all use counts go to
  // zero.  Then everything is deleted for real.  Note that no operations are
  // valid on an object that has "dropped all references", except operator
  // delete.
  //
  void dropAllReferences() {
    for (op_iterator i = op_begin(), e = op_end(); i != e; ++i)
      i->set(0);
  }

  /// replaceUsesOfWith - Replaces all references to the "From" definition with
  /// references to the "To" definition.
  ///
  void replaceUsesOfWith(Value *From, Value *To);

  // Methods for support type inquiry through isa, cast, and dyn_cast:
  static inline bool classof(const User *) { return true; }
  static inline bool classof(const Value *V) {
    return isa<Instruction>(V) || isa<Constant>(V);
  }
};

inline Use *OperandTraits<User>::op_begin(User *U) {
  return U->op_begin();
}

inline Use *OperandTraits<User>::op_end(User *U) {
  return U->op_end();
}

inline unsigned OperandTraits<User>::operands(const User *U) {
  return U->getNumOperands();
}

template<> struct simplify_type<User::op_iterator> {
  typedef Value* SimpleType;

  static SimpleType getSimplifiedValue(const User::op_iterator &Val) {
    return static_cast<SimpleType>(Val->get());
  }
};

template<> struct simplify_type<const User::op_iterator>
  : public simplify_type<User::op_iterator> {};

template<> struct simplify_type<User::const_op_iterator> {
  typedef Value* SimpleType;

  static SimpleType getSimplifiedValue(const User::const_op_iterator &Val) {
    return static_cast<SimpleType>(Val->get());
  }
};

template<> struct simplify_type<const User::const_op_iterator>
  : public simplify_type<User::const_op_iterator> {};


// value_use_iterator::getOperandNo - Requires the definition of the User class.
template<typename UserTy>
unsigned value_use_iterator<UserTy>::getOperandNo() const {
  return U - U->getUser()->op_begin();
}

} // End llvm namespace

#endif
