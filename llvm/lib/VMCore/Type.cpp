//===-- Type.cpp - Implement the Type class -------------------------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements the Type class for the VMCore library.
//
//===----------------------------------------------------------------------===//

#include "llvm/DerivedTypes.h"
#include "llvm/SymbolTable.h"
#include "llvm/Constants.h"
#include "Support/StringExtras.h"
#include "Support/STLExtras.h"
#include <algorithm>

using namespace llvm;

// DEBUG_MERGE_TYPES - Enable this #define to see how and when derived types are
// created and later destroyed, all in an effort to make sure that there is only
// a single canonical version of a type.
//
//#define DEBUG_MERGE_TYPES 1


//===----------------------------------------------------------------------===//
//                         Type Class Implementation
//===----------------------------------------------------------------------===//

static unsigned CurUID = 0;
static std::vector<const Type *> UIDMappings;

// Concrete/Abstract TypeDescriptions - We lazily calculate type descriptions
// for types as they are needed.  Because resolution of types must invalidate
// all of the abstract type descriptions, we keep them in a seperate map to make
// this easy.
static std::map<const Type*, std::string> ConcreteTypeDescriptions;
static std::map<const Type*, std::string> AbstractTypeDescriptions;

Type::Type(const std::string &name, PrimitiveID id)
  : Value(Type::TypeTy, Value::TypeVal), ForwardType(0) {
  if (!name.empty())
    ConcreteTypeDescriptions[this] = name;
  ID = id;
  Abstract = false;
  UID = CurUID++;       // Assign types UID's as they are created
  UIDMappings.push_back(this);
}

void Type::setName(const std::string &Name, SymbolTable *ST) {
  assert(ST && "Type::setName - Must provide symbol table argument!");

  if (Name.size()) ST->insert(Name, this);
}


const Type *Type::getUniqueIDType(unsigned UID) {
  assert(UID < UIDMappings.size() && 
         "Type::getPrimitiveType: UID out of range!");
  return UIDMappings[UID];
}

const Type *Type::getPrimitiveType(PrimitiveID IDNumber) {
  switch (IDNumber) {
  case VoidTyID  : return VoidTy;
  case BoolTyID  : return BoolTy;
  case UByteTyID : return UByteTy;
  case SByteTyID : return SByteTy;
  case UShortTyID: return UShortTy;
  case ShortTyID : return ShortTy;
  case UIntTyID  : return UIntTy;
  case IntTyID   : return IntTy;
  case ULongTyID : return ULongTy;
  case LongTyID  : return LongTy;
  case FloatTyID : return FloatTy;
  case DoubleTyID: return DoubleTy;
  case TypeTyID  : return TypeTy;
  case LabelTyID : return LabelTy;
  default:
    return 0;
  }
}

// isLosslesslyConvertibleTo - Return true if this type can be converted to
// 'Ty' without any reinterpretation of bits.  For example, uint to int.
//
bool Type::isLosslesslyConvertibleTo(const Type *Ty) const {
  if (this == Ty) return true;
  if ((!isPrimitiveType()    && !isa<PointerType>(this)) ||
      (!isa<PointerType>(Ty) && !Ty->isPrimitiveType())) return false;

  if (getPrimitiveID() == Ty->getPrimitiveID())
    return true;  // Handles identity cast, and cast of differing pointer types

  // Now we know that they are two differing primitive or pointer types
  switch (getPrimitiveID()) {
  case Type::UByteTyID:   return Ty == Type::SByteTy;
  case Type::SByteTyID:   return Ty == Type::UByteTy;
  case Type::UShortTyID:  return Ty == Type::ShortTy;
  case Type::ShortTyID:   return Ty == Type::UShortTy;
  case Type::UIntTyID:    return Ty == Type::IntTy;
  case Type::IntTyID:     return Ty == Type::UIntTy;
  case Type::ULongTyID:   return Ty == Type::LongTy;
  case Type::LongTyID:    return Ty == Type::ULongTy;
  case Type::PointerTyID: return isa<PointerType>(Ty);
  default:
    return false;  // Other types have no identity values
  }
}

// getPrimitiveSize - Return the basic size of this type if it is a primitive
// type.  These are fixed by LLVM and are not target dependent.  This will
// return zero if the type does not have a size or is not a primitive type.
//
unsigned Type::getPrimitiveSize() const {
  switch (getPrimitiveID()) {
#define HANDLE_PRIM_TYPE(TY,SIZE)  case TY##TyID: return SIZE;
#include "llvm/Type.def"
  default: return 0;
  }
}


/// getForwardedTypeInternal - This method is used to implement the union-find
/// algorithm for when a type is being forwarded to another type.
const Type *Type::getForwardedTypeInternal() const {
  assert(ForwardType && "This type is not being forwarded to another type!");
  
  // Check to see if the forwarded type has been forwarded on.  If so, collapse
  // the forwarding links.
  const Type *RealForwardedType = ForwardType->getForwardedType();
  if (!RealForwardedType)
    return ForwardType;  // No it's not forwarded again

  // Yes, it is forwarded again.  First thing, add the reference to the new
  // forward type.
  if (RealForwardedType->isAbstract())
    cast<DerivedType>(RealForwardedType)->addRef();

  // Now drop the old reference.  This could cause ForwardType to get deleted.
  cast<DerivedType>(ForwardType)->dropRef();
  
  // Return the updated type.
  ForwardType = RealForwardedType;
  return ForwardType;
}

// getTypeDescription - This is a recursive function that walks a type hierarchy
// calculating the description for a type.
//
static std::string getTypeDescription(const Type *Ty,
                                      std::vector<const Type *> &TypeStack) {
  if (isa<OpaqueType>(Ty)) {                     // Base case for the recursion
    std::map<const Type*, std::string>::iterator I =
      AbstractTypeDescriptions.lower_bound(Ty);
    if (I != AbstractTypeDescriptions.end() && I->first == Ty)
      return I->second;
    std::string Desc = "opaque"+utostr(Ty->getUniqueID());
    AbstractTypeDescriptions.insert(std::make_pair(Ty, Desc));
    return Desc;
  }
  
  if (!Ty->isAbstract()) {                       // Base case for the recursion
    std::map<const Type*, std::string>::iterator I =
      ConcreteTypeDescriptions.find(Ty);
    if (I != ConcreteTypeDescriptions.end()) return I->second;
  }
      
  // Check to see if the Type is already on the stack...
  unsigned Slot = 0, CurSize = TypeStack.size();
  while (Slot < CurSize && TypeStack[Slot] != Ty) ++Slot; // Scan for type
  
  // This is another base case for the recursion.  In this case, we know 
  // that we have looped back to a type that we have previously visited.
  // Generate the appropriate upreference to handle this.
  // 
  if (Slot < CurSize)
    return "\\" + utostr(CurSize-Slot);         // Here's the upreference

  // Recursive case: derived types...
  std::string Result;
  TypeStack.push_back(Ty);    // Add us to the stack..
      
  switch (Ty->getPrimitiveID()) {
  case Type::FunctionTyID: {
    const FunctionType *FTy = cast<FunctionType>(Ty);
    Result = getTypeDescription(FTy->getReturnType(), TypeStack) + " (";
    for (FunctionType::ParamTypes::const_iterator
           I = FTy->getParamTypes().begin(),
           E = FTy->getParamTypes().end(); I != E; ++I) {
      if (I != FTy->getParamTypes().begin())
        Result += ", ";
      Result += getTypeDescription(*I, TypeStack);
    }
    if (FTy->isVarArg()) {
      if (!FTy->getParamTypes().empty()) Result += ", ";
      Result += "...";
    }
    Result += ")";
    break;
  }
  case Type::StructTyID: {
    const StructType *STy = cast<StructType>(Ty);
    Result = "{ ";
    for (StructType::ElementTypes::const_iterator
           I = STy->getElementTypes().begin(),
           E = STy->getElementTypes().end(); I != E; ++I) {
      if (I != STy->getElementTypes().begin())
        Result += ", ";
      Result += getTypeDescription(*I, TypeStack);
    }
    Result += " }";
    break;
  }
  case Type::PointerTyID: {
    const PointerType *PTy = cast<PointerType>(Ty);
    Result = getTypeDescription(PTy->getElementType(), TypeStack) + " *";
    break;
  }
  case Type::ArrayTyID: {
    const ArrayType *ATy = cast<ArrayType>(Ty);
    unsigned NumElements = ATy->getNumElements();
    Result = "[";
    Result += utostr(NumElements) + " x ";
    Result += getTypeDescription(ATy->getElementType(), TypeStack) + "]";
    break;
  }
  default:
    Result = "<error>";
    assert(0 && "Unhandled type in getTypeDescription!");
  }

  TypeStack.pop_back();       // Remove self from stack...

  return Result;
}



static const std::string &getOrCreateDesc(std::map<const Type*,std::string>&Map,
                                          const Type *Ty) {
  std::map<const Type*, std::string>::iterator I = Map.find(Ty);
  if (I != Map.end()) return I->second;
    
  std::vector<const Type *> TypeStack;
  return Map[Ty] = getTypeDescription(Ty, TypeStack);
}


const std::string &Type::getDescription() const {
  if (isAbstract())
    return getOrCreateDesc(AbstractTypeDescriptions, this);
  else
    return getOrCreateDesc(ConcreteTypeDescriptions, this);
}


bool StructType::indexValid(const Value *V) const {
  if (!isa<Constant>(V)) return false;
  if (V->getType() != Type::UByteTy) return false;
  unsigned Idx = cast<ConstantUInt>(V)->getValue();
  return Idx < ETypes.size();
}

// getTypeAtIndex - Given an index value into the type, return the type of the
// element.  For a structure type, this must be a constant value...
//
const Type *StructType::getTypeAtIndex(const Value *V) const {
  assert(isa<Constant>(V) && "Structure index must be a constant!!");
  assert(V->getType() == Type::UByteTy && "Structure index must be ubyte!");
  unsigned Idx = cast<ConstantUInt>(V)->getValue();
  assert(Idx < ETypes.size() && "Structure index out of range!");
  assert(indexValid(V) && "Invalid structure index!"); // Duplicate check

  return ETypes[Idx];
}


//===----------------------------------------------------------------------===//
//                           Auxiliary classes
//===----------------------------------------------------------------------===//
//
// These classes are used to implement specialized behavior for each different
// type.
//
struct SignedIntType : public Type {
  SignedIntType(const std::string &Name, PrimitiveID id) : Type(Name, id) {}

  // isSigned - Return whether a numeric type is signed.
  virtual bool isSigned() const { return 1; }

  // isInteger - Equivalent to isSigned() || isUnsigned, but with only a single
  // virtual function invocation.
  //
  virtual bool isInteger() const { return 1; }
};

struct UnsignedIntType : public Type {
  UnsignedIntType(const std::string &N, PrimitiveID id) : Type(N, id) {}

  // isUnsigned - Return whether a numeric type is signed.
  virtual bool isUnsigned() const { return 1; }

  // isInteger - Equivalent to isSigned() || isUnsigned, but with only a single
  // virtual function invocation.
  //
  virtual bool isInteger() const { return 1; }
};

struct OtherType : public Type {
  OtherType(const std::string &N, PrimitiveID id) : Type(N, id) {}
};

static struct TypeType : public Type {
  TypeType() : Type("type", TypeTyID) {}
} TheTypeTy;   // Implement the type that is global.


//===----------------------------------------------------------------------===//
//                           Static 'Type' data
//===----------------------------------------------------------------------===//

static OtherType       TheVoidTy  ("void"  , Type::VoidTyID);
static OtherType       TheBoolTy  ("bool"  , Type::BoolTyID);
static SignedIntType   TheSByteTy ("sbyte" , Type::SByteTyID);
static UnsignedIntType TheUByteTy ("ubyte" , Type::UByteTyID);
static SignedIntType   TheShortTy ("short" , Type::ShortTyID);
static UnsignedIntType TheUShortTy("ushort", Type::UShortTyID);
static SignedIntType   TheIntTy   ("int"   , Type::IntTyID); 
static UnsignedIntType TheUIntTy  ("uint"  , Type::UIntTyID);
static SignedIntType   TheLongTy  ("long"  , Type::LongTyID);
static UnsignedIntType TheULongTy ("ulong" , Type::ULongTyID);
static OtherType       TheFloatTy ("float" , Type::FloatTyID);
static OtherType       TheDoubleTy("double", Type::DoubleTyID);
static OtherType       TheLabelTy ("label" , Type::LabelTyID);

Type *Type::VoidTy   = &TheVoidTy;
Type *Type::BoolTy   = &TheBoolTy;
Type *Type::SByteTy  = &TheSByteTy;
Type *Type::UByteTy  = &TheUByteTy;
Type *Type::ShortTy  = &TheShortTy;
Type *Type::UShortTy = &TheUShortTy;
Type *Type::IntTy    = &TheIntTy;
Type *Type::UIntTy   = &TheUIntTy;
Type *Type::LongTy   = &TheLongTy;
Type *Type::ULongTy  = &TheULongTy;
Type *Type::FloatTy  = &TheFloatTy;
Type *Type::DoubleTy = &TheDoubleTy;
Type *Type::TypeTy   = &TheTypeTy;
Type *Type::LabelTy  = &TheLabelTy;


//===----------------------------------------------------------------------===//
//                          Derived Type Constructors
//===----------------------------------------------------------------------===//

FunctionType::FunctionType(const Type *Result,
                           const std::vector<const Type*> &Params, 
                           bool IsVarArgs) : DerivedType(FunctionTyID), 
    ResultType(PATypeHandle(Result, this)),
    isVarArgs(IsVarArgs) {
  bool isAbstract = Result->isAbstract();
  ParamTys.reserve(Params.size());
  for (unsigned i = 0; i < Params.size(); ++i) {
    ParamTys.push_back(PATypeHandle(Params[i], this));
    isAbstract |= Params[i]->isAbstract();
  }

  // Calculate whether or not this type is abstract
  setAbstract(isAbstract);
}

StructType::StructType(const std::vector<const Type*> &Types)
  : CompositeType(StructTyID) {
  ETypes.reserve(Types.size());
  bool isAbstract = false;
  for (unsigned i = 0; i < Types.size(); ++i) {
    assert(Types[i] != Type::VoidTy && "Void type in method prototype!!");
    ETypes.push_back(PATypeHandle(Types[i], this));
    isAbstract |= Types[i]->isAbstract();
  }

  // Calculate whether or not this type is abstract
  setAbstract(isAbstract);
}

ArrayType::ArrayType(const Type *ElType, unsigned NumEl)
  : SequentialType(ArrayTyID, ElType) {
  NumElements = NumEl;

  // Calculate whether or not this type is abstract
  setAbstract(ElType->isAbstract());
}

PointerType::PointerType(const Type *E) : SequentialType(PointerTyID, E) {
  // Calculate whether or not this type is abstract
  setAbstract(E->isAbstract());
}

OpaqueType::OpaqueType() : DerivedType(OpaqueTyID) {
  setAbstract(true);
#ifdef DEBUG_MERGE_TYPES
  std::cerr << "Derived new type: " << *this << "\n";
#endif
}


// getAlwaysOpaqueTy - This function returns an opaque type.  It doesn't matter
// _which_ opaque type it is, but the opaque type must never get resolved.
//
static Type *getAlwaysOpaqueTy() {
  static Type *AlwaysOpaqueTy = OpaqueType::get();
  static PATypeHolder Holder(AlwaysOpaqueTy);
  return AlwaysOpaqueTy;
}


//===----------------------------------------------------------------------===//
// dropAllTypeUses methods - These methods eliminate any possibly recursive type
// references from a derived type.  The type must remain abstract, so we make
// sure to use an always opaque type as an argument.
//

void FunctionType::dropAllTypeUses() {
  ResultType = getAlwaysOpaqueTy();
  ParamTys.clear();
}

void ArrayType::dropAllTypeUses() {
  ElementType = getAlwaysOpaqueTy();
}

void StructType::dropAllTypeUses() {
  ETypes.clear();
  ETypes.push_back(PATypeHandle(getAlwaysOpaqueTy(), this));
}

void PointerType::dropAllTypeUses() {
  ElementType = getAlwaysOpaqueTy();
}




// isTypeAbstract - This is a recursive function that walks a type hierarchy
// calculating whether or not a type is abstract.  Worst case it will have to do
// a lot of traversing if you have some whacko opaque types, but in most cases,
// it will do some simple stuff when it hits non-abstract types that aren't
// recursive.
//
bool Type::isTypeAbstract() {
  if (!isAbstract())                           // Base case for the recursion
    return false;                              // Primitive = leaf type
  
  if (isa<OpaqueType>(this))                   // Base case for the recursion
    return true;                               // This whole type is abstract!

  // We have to guard against recursion.  To do this, we temporarily mark this
  // type as concrete, so that if we get back to here recursively we will think
  // it's not abstract, and thus not scan it again.
  setAbstract(false);

  // Scan all of the sub-types.  If any of them are abstract, than so is this
  // one!
  for (Type::subtype_iterator I = subtype_begin(), E = subtype_end();
       I != E; ++I)
    if (const_cast<Type*>(*I)->isTypeAbstract()) {
      setAbstract(true);        // Restore the abstract bit.
      return true;              // This type is abstract if subtype is abstract!
    }
  
  // Restore the abstract bit.
  setAbstract(true);

  // Nothing looks abstract here...
  return false;
}


//===----------------------------------------------------------------------===//
//                      Type Structural Equality Testing
//===----------------------------------------------------------------------===//

// TypesEqual - Two types are considered structurally equal if they have the
// same "shape": Every level and element of the types have identical primitive
// ID's, and the graphs have the same edges/nodes in them.  Nodes do not have to
// be pointer equals to be equivalent though.  This uses an optimistic algorithm
// that assumes that two graphs are the same until proven otherwise.
//
static bool TypesEqual(const Type *Ty, const Type *Ty2,
		       std::map<const Type *, const Type *> &EqTypes) {
  if (Ty == Ty2) return true;
  if (Ty->getPrimitiveID() != Ty2->getPrimitiveID()) return false;
  if (Ty->isPrimitiveType()) return true;
  if (isa<OpaqueType>(Ty))
    return false;  // Two unequal opaque types are never equal

  std::map<const Type*, const Type*>::iterator It = EqTypes.lower_bound(Ty);
  if (It != EqTypes.end() && It->first == Ty)
    return It->second == Ty2;    // Looping back on a type, check for equality

  // Otherwise, add the mapping to the table to make sure we don't get
  // recursion on the types...
  EqTypes.insert(It, std::make_pair(Ty, Ty2));

  // Two really annoying special cases that breaks an otherwise nice simple
  // algorithm is the fact that arraytypes have sizes that differentiates types,
  // and that function types can be varargs or not.  Consider this now.
  //
  if (const PointerType *PTy = dyn_cast<PointerType>(Ty)) {
    return TypesEqual(PTy->getElementType(),
                      cast<PointerType>(Ty2)->getElementType(), EqTypes);
  } else if (const StructType *STy = dyn_cast<StructType>(Ty)) {
    const StructType::ElementTypes &STyE = STy->getElementTypes();
    const StructType::ElementTypes &STyE2 =
      cast<StructType>(Ty2)->getElementTypes();
    if (STyE.size() != STyE2.size()) return false;
    for (unsigned i = 0, e = STyE.size(); i != e; ++i)
      if (!TypesEqual(STyE[i], STyE2[i], EqTypes))
        return false;
    return true;
  } else if (const ArrayType *ATy = dyn_cast<ArrayType>(Ty)) {
    const ArrayType *ATy2 = cast<ArrayType>(Ty2);
    return ATy->getNumElements() == ATy2->getNumElements() &&
           TypesEqual(ATy->getElementType(), ATy2->getElementType(), EqTypes);
  } else if (const FunctionType *FTy = dyn_cast<FunctionType>(Ty)) {
    const FunctionType *FTy2 = cast<FunctionType>(Ty2);
    if (FTy->isVarArg() != FTy2->isVarArg() ||
        FTy->getParamTypes().size() != FTy2->getParamTypes().size() ||
        !TypesEqual(FTy->getReturnType(), FTy2->getReturnType(), EqTypes))
      return false;
    const FunctionType::ParamTypes &FTyP = FTy->getParamTypes();
    const FunctionType::ParamTypes &FTy2P = FTy2->getParamTypes();
    for (unsigned i = 0, e = FTyP.size(); i != e; ++i)
      if (!TypesEqual(FTyP[i], FTy2P[i], EqTypes))
        return false;
    return true;
  } else {
    assert(0 && "Unknown derived type!");
    return false;
  }
}

static bool TypesEqual(const Type *Ty, const Type *Ty2) {
  std::map<const Type *, const Type *> EqTypes;
  return TypesEqual(Ty, Ty2, EqTypes);
}



//===----------------------------------------------------------------------===//
//                       Derived Type Factory Functions
//===----------------------------------------------------------------------===//

// TypeMap - Make sure that only one instance of a particular type may be
// created on any given run of the compiler... note that this involves updating
// our map if an abstract type gets refined somehow...
//
namespace llvm {
template<class ValType, class TypeClass>
class TypeMap {
  typedef std::map<ValType, TypeClass *> MapTy;
  MapTy Map;
public:
  typedef typename MapTy::iterator iterator;
  ~TypeMap() { print("ON EXIT"); }

  inline TypeClass *get(const ValType &V) {
    iterator I = Map.find(V);
    return I != Map.end() ? I->second : 0;
  }

  inline void add(const ValType &V, TypeClass *T) {
    Map.insert(std::make_pair(V, T));
    print("add");
  }

  iterator getEntryForType(TypeClass *Ty) {
    iterator I = Map.find(ValType::get(Ty));
    if (I == Map.end()) print("ERROR!");
    assert(I != Map.end() && "Didn't find type entry!");
    assert(I->second == Ty && "Type entry wrong?");
    return I;
  }


  void finishRefinement(iterator TyIt) {
    TypeClass *Ty = TyIt->second;

    // The old record is now out-of-date, because one of the children has been
    // updated.  Remove the obsolete entry from the map.
    Map.erase(TyIt);

    // Now we check to see if there is an existing entry in the table which is
    // structurally identical to the newly refined type.  If so, this type gets
    // refined to the pre-existing type.
    //
    for (iterator I = Map.begin(), E = Map.end(); I != E; ++I)
      if (TypesEqual(Ty, I->second)) {
        assert(Ty->isAbstract() && "Replacing a non-abstract type?");
        TypeClass *NewTy = I->second;

        // Refined to a different type altogether?
        Ty->refineAbstractTypeTo(NewTy);
        return;
      }

    // If there is no existing type of the same structure, we reinsert an
    // updated record into the map.
    Map.insert(std::make_pair(ValType::get(Ty), Ty));

    // If the type is currently thought to be abstract, rescan all of our
    // subtypes to see if the type has just become concrete!
    if (Ty->isAbstract()) {
      Ty->setAbstract(Ty->isTypeAbstract());

      // If the type just became concrete, notify all users!
      if (!Ty->isAbstract())
        Ty->notifyUsesThatTypeBecameConcrete();
    }
  }
  
  void remove(const ValType &OldVal) {
    iterator I = Map.find(OldVal);
    assert(I != Map.end() && "TypeMap::remove, element not found!");
    Map.erase(I);
  }

  void remove(iterator I) {
    assert(I != Map.end() && "Cannot remove invalid iterator pointer!");
    Map.erase(I);
  }

  void print(const char *Arg) const {
#ifdef DEBUG_MERGE_TYPES
    std::cerr << "TypeMap<>::" << Arg << " table contents:\n";
    unsigned i = 0;
    for (typename MapTy::const_iterator I = Map.begin(), E = Map.end();
         I != E; ++I)
      std::cerr << " " << (++i) << ". " << (void*)I->second << " " 
                << *I->second << "\n";
#endif
  }

  void dump() const { print("dump output"); }
};
}


//===----------------------------------------------------------------------===//
// Function Type Factory and Value Class...
//

// FunctionValType - Define a class to hold the key that goes into the TypeMap
//
namespace llvm {
class FunctionValType {
  const Type *RetTy;
  std::vector<const Type*> ArgTypes;
  bool isVarArg;
public:
  FunctionValType(const Type *ret, const std::vector<const Type*> &args,
                  bool IVA) : RetTy(ret), isVarArg(IVA) {
    for (unsigned i = 0; i < args.size(); ++i)
      ArgTypes.push_back(args[i]);
  }

  static FunctionValType get(const FunctionType *FT);

  // Subclass should override this... to update self as usual
  void doRefinement(const DerivedType *OldType, const Type *NewType) {
    if (RetTy == OldType) RetTy = NewType;
    for (unsigned i = 0, e = ArgTypes.size(); i != e; ++i)
      if (ArgTypes[i] == OldType) ArgTypes[i] = NewType;
  }

  inline bool operator<(const FunctionValType &MTV) const {
    if (RetTy < MTV.RetTy) return true;
    if (RetTy > MTV.RetTy) return false;

    if (ArgTypes < MTV.ArgTypes) return true;
    return ArgTypes == MTV.ArgTypes && isVarArg < MTV.isVarArg;
  }
};
}

// Define the actual map itself now...
static TypeMap<FunctionValType, FunctionType> FunctionTypes;

FunctionValType FunctionValType::get(const FunctionType *FT) {
  // Build up a FunctionValType
  std::vector<const Type *> ParamTypes;
  ParamTypes.reserve(FT->getParamTypes().size());
  for (unsigned i = 0, e = FT->getParamTypes().size(); i != e; ++i)
    ParamTypes.push_back(FT->getParamType(i));
  return FunctionValType(FT->getReturnType(), ParamTypes, FT->isVarArg());
}


// FunctionType::get - The factory function for the FunctionType class...
FunctionType *FunctionType::get(const Type *ReturnType, 
                                const std::vector<const Type*> &Params,
                                bool isVarArg) {
  FunctionValType VT(ReturnType, Params, isVarArg);
  FunctionType *MT = FunctionTypes.get(VT);
  if (MT) return MT;

  FunctionTypes.add(VT, MT = new FunctionType(ReturnType, Params, isVarArg));

#ifdef DEBUG_MERGE_TYPES
  std::cerr << "Derived new type: " << MT << "\n";
#endif
  return MT;
}

//===----------------------------------------------------------------------===//
// Array Type Factory...
//
namespace llvm {
class ArrayValType {
  const Type *ValTy;
  unsigned Size;
public:
  ArrayValType(const Type *val, int sz) : ValTy(val), Size(sz) {}

  static ArrayValType get(const ArrayType *AT) {
    return ArrayValType(AT->getElementType(), AT->getNumElements());
  }

  // Subclass should override this... to update self as usual
  void doRefinement(const DerivedType *OldType, const Type *NewType) {
    assert(ValTy == OldType);
    ValTy = NewType;
  }

  inline bool operator<(const ArrayValType &MTV) const {
    if (Size < MTV.Size) return true;
    return Size == MTV.Size && ValTy < MTV.ValTy;
  }
};
}
static TypeMap<ArrayValType, ArrayType> ArrayTypes;


ArrayType *ArrayType::get(const Type *ElementType, unsigned NumElements) {
  assert(ElementType && "Can't get array of null types!");

  ArrayValType AVT(ElementType, NumElements);
  ArrayType *AT = ArrayTypes.get(AVT);
  if (AT) return AT;           // Found a match, return it!

  // Value not found.  Derive a new type!
  ArrayTypes.add(AVT, AT = new ArrayType(ElementType, NumElements));

#ifdef DEBUG_MERGE_TYPES
  std::cerr << "Derived new type: " << *AT << "\n";
#endif
  return AT;
}

//===----------------------------------------------------------------------===//
// Struct Type Factory...
//

namespace llvm {
// StructValType - Define a class to hold the key that goes into the TypeMap
//
class StructValType {
  std::vector<const Type*> ElTypes;
public:
  StructValType(const std::vector<const Type*> &args) : ElTypes(args) {}

  static StructValType get(const StructType *ST) {
    std::vector<const Type *> ElTypes;
    ElTypes.reserve(ST->getElementTypes().size());
    for (unsigned i = 0, e = ST->getElementTypes().size(); i != e; ++i)
      ElTypes.push_back(ST->getElementTypes()[i]);
    
    return StructValType(ElTypes);
  }

  // Subclass should override this... to update self as usual
  void doRefinement(const DerivedType *OldType, const Type *NewType) {
    for (unsigned i = 0; i < ElTypes.size(); ++i)
      if (ElTypes[i] == OldType) ElTypes[i] = NewType;
  }

  inline bool operator<(const StructValType &STV) const {
    return ElTypes < STV.ElTypes;
  }
};
}

static TypeMap<StructValType, StructType> StructTypes;

StructType *StructType::get(const std::vector<const Type*> &ETypes) {
  StructValType STV(ETypes);
  StructType *ST = StructTypes.get(STV);
  if (ST) return ST;

  // Value not found.  Derive a new type!
  StructTypes.add(STV, ST = new StructType(ETypes));

#ifdef DEBUG_MERGE_TYPES
  std::cerr << "Derived new type: " << *ST << "\n";
#endif
  return ST;
}



//===----------------------------------------------------------------------===//
// Pointer Type Factory...
//

// PointerValType - Define a class to hold the key that goes into the TypeMap
//
namespace llvm {
class PointerValType {
  const Type *ValTy;
public:
  PointerValType(const Type *val) : ValTy(val) {}

  static PointerValType get(const PointerType *PT) {
    return PointerValType(PT->getElementType());
  }

  // Subclass should override this... to update self as usual
  void doRefinement(const DerivedType *OldType, const Type *NewType) {
    assert(ValTy == OldType);
    ValTy = NewType;
  }

  bool operator<(const PointerValType &MTV) const {
    return ValTy < MTV.ValTy;
  }
};
}

static TypeMap<PointerValType, PointerType> PointerTypes;

PointerType *PointerType::get(const Type *ValueType) {
  assert(ValueType && "Can't get a pointer to <null> type!");
  PointerValType PVT(ValueType);

  PointerType *PT = PointerTypes.get(PVT);
  if (PT) return PT;

  // Value not found.  Derive a new type!
  PointerTypes.add(PVT, PT = new PointerType(ValueType));

#ifdef DEBUG_MERGE_TYPES
  std::cerr << "Derived new type: " << *PT << "\n";
#endif
  return PT;
}

namespace llvm {
void debug_type_tables() {
  FunctionTypes.dump();
  ArrayTypes.dump();
  StructTypes.dump();
  PointerTypes.dump();
}
}

//===----------------------------------------------------------------------===//
//                     Derived Type Refinement Functions
//===----------------------------------------------------------------------===//

// removeAbstractTypeUser - Notify an abstract type that a user of the class
// no longer has a handle to the type.  This function is called primarily by
// the PATypeHandle class.  When there are no users of the abstract type, it
// is annihilated, because there is no way to get a reference to it ever again.
//
void DerivedType::removeAbstractTypeUser(AbstractTypeUser *U) const {
  // Search from back to front because we will notify users from back to
  // front.  Also, it is likely that there will be a stack like behavior to
  // users that register and unregister users.
  //
  unsigned i;
  for (i = AbstractTypeUsers.size(); AbstractTypeUsers[i-1] != U; --i)
    assert(i != 0 && "AbstractTypeUser not in user list!");

  --i;  // Convert to be in range 0 <= i < size()
  assert(i < AbstractTypeUsers.size() && "Index out of range!");  // Wraparound?

  AbstractTypeUsers.erase(AbstractTypeUsers.begin()+i);
      
#ifdef DEBUG_MERGE_TYPES
  std::cerr << "  remAbstractTypeUser[" << (void*)this << ", "
            << *this << "][" << i << "] User = " << U << "\n";
#endif
    
  if (AbstractTypeUsers.empty() && RefCount == 0 && isAbstract()) {
#ifdef DEBUG_MERGE_TYPES
    std::cerr << "DELETEing unused abstract type: <" << *this
              << ">[" << (void*)this << "]" << "\n";
#endif
    delete this;                  // No users of this abstract type!
  }
}


// refineAbstractTypeTo - This function is used to when it is discovered that
// the 'this' abstract type is actually equivalent to the NewType specified.
// This causes all users of 'this' to switch to reference the more concrete type
// NewType and for 'this' to be deleted.
//
void DerivedType::refineAbstractTypeTo(const Type *NewType) {
  assert(isAbstract() && "refineAbstractTypeTo: Current type is not abstract!");
  assert(this != NewType && "Can't refine to myself!");
  assert(ForwardType == 0 && "This type has already been refined!");

  // The descriptions may be out of date.  Conservatively clear them all!
  AbstractTypeDescriptions.clear();

#ifdef DEBUG_MERGE_TYPES
  std::cerr << "REFINING abstract type [" << (void*)this << " "
            << *this << "] to [" << (void*)NewType << " "
            << *NewType << "]!\n";
#endif

  // Make sure to put the type to be refined to into a holder so that if IT gets
  // refined, that we will not continue using a dead reference...
  //
  PATypeHolder NewTy(NewType);

  // Any PATypeHolders referring to this type will now automatically forward to
  // the type we are resolved to.
  ForwardType = NewType;
  if (NewType->isAbstract())
    cast<DerivedType>(NewType)->addRef();

  // Add a self use of the current type so that we don't delete ourself until
  // after the function exits.
  //
  PATypeHolder CurrentTy(this);

  // To make the situation simpler, we ask the subclass to remove this type from
  // the type map, and to replace any type uses with uses of non-abstract types.
  // This dramatically limits the amount of recursive type trouble we can find
  // ourselves in.
  dropAllTypeUses();

  // Iterate over all of the uses of this type, invoking callback.  Each user
  // should remove itself from our use list automatically.  We have to check to
  // make sure that NewTy doesn't _become_ 'this'.  If it does, resolving types
  // will not cause users to drop off of the use list.  If we resolve to ourself
  // we succeed!
  //
  while (!AbstractTypeUsers.empty() && NewTy != this) {
    AbstractTypeUser *User = AbstractTypeUsers.back();

    unsigned OldSize = AbstractTypeUsers.size();
#ifdef DEBUG_MERGE_TYPES
    std::cerr << " REFINING user " << OldSize-1 << "[" << (void*)User
              << "] of abstract type [" << (void*)this << " "
              << *this << "] to [" << (void*)NewTy.get() << " "
              << *NewTy << "]!\n";
#endif
    User->refineAbstractType(this, NewTy);

    assert(AbstractTypeUsers.size() != OldSize &&
           "AbsTyUser did not remove self from user list!");
  }

  // If we were successful removing all users from the type, 'this' will be
  // deleted when the last PATypeHolder is destroyed or updated from this type.
  // This may occur on exit of this function, as the CurrentTy object is
  // destroyed.
}

// notifyUsesThatTypeBecameConcrete - Notify AbstractTypeUsers of this type that
// the current type has transitioned from being abstract to being concrete.
//
void DerivedType::notifyUsesThatTypeBecameConcrete() {
#ifdef DEBUG_MERGE_TYPES
  std::cerr << "typeIsREFINED type: " << (void*)this << " " << *this << "\n";
#endif

  unsigned OldSize = AbstractTypeUsers.size();
  while (!AbstractTypeUsers.empty()) {
    AbstractTypeUser *ATU = AbstractTypeUsers.back();
    ATU->typeBecameConcrete(this);

    assert(AbstractTypeUsers.size() < OldSize-- &&
           "AbstractTypeUser did not remove itself from the use list!");
  }
}
  



// refineAbstractType - Called when a contained type is found to be more
// concrete - this could potentially change us from an abstract type to a
// concrete type.
//
void FunctionType::refineAbstractType(const DerivedType *OldType,
                                      const Type *NewType) {
  assert((isAbstract() || !OldType->isAbstract()) &&
         "Refining a non-abstract type!");
#ifdef DEBUG_MERGE_TYPES
  std::cerr << "FunctionTy::refineAbstractTy(" << (void*)OldType << "[" 
            << *OldType << "], " << (void*)NewType << " [" 
            << *NewType << "])\n";
#endif

  // Look up our current type map entry..
  TypeMap<FunctionValType, FunctionType>::iterator TMI =
    FunctionTypes.getEntryForType(this);

  // Find the type element we are refining...
  if (ResultType == OldType) {
    ResultType.removeUserFromConcrete();
    ResultType = NewType;
  }
  for (unsigned i = 0, e = ParamTys.size(); i != e; ++i)
    if (ParamTys[i] == OldType) {
      ParamTys[i].removeUserFromConcrete();
      ParamTys[i] = NewType;
    }

  FunctionTypes.finishRefinement(TMI);
}

void FunctionType::typeBecameConcrete(const DerivedType *AbsTy) {
  refineAbstractType(AbsTy, AbsTy);
}


// refineAbstractType - Called when a contained type is found to be more
// concrete - this could potentially change us from an abstract type to a
// concrete type.
//
void ArrayType::refineAbstractType(const DerivedType *OldType,
				   const Type *NewType) {
  assert((isAbstract() || !OldType->isAbstract()) &&
         "Refining a non-abstract type!");
#ifdef DEBUG_MERGE_TYPES
  std::cerr << "ArrayTy::refineAbstractTy(" << (void*)OldType << "[" 
            << *OldType << "], " << (void*)NewType << " [" 
            << *NewType << "])\n";
#endif

  // Look up our current type map entry..
  TypeMap<ArrayValType, ArrayType>::iterator TMI =
    ArrayTypes.getEntryForType(this);

  assert(getElementType() == OldType);
  ElementType.removeUserFromConcrete();
  ElementType = NewType;

  ArrayTypes.finishRefinement(TMI);
}

void ArrayType::typeBecameConcrete(const DerivedType *AbsTy) {
  refineAbstractType(AbsTy, AbsTy);
}


// refineAbstractType - Called when a contained type is found to be more
// concrete - this could potentially change us from an abstract type to a
// concrete type.
//
void StructType::refineAbstractType(const DerivedType *OldType,
				    const Type *NewType) {
  assert((isAbstract() || !OldType->isAbstract()) &&
         "Refining a non-abstract type!");
#ifdef DEBUG_MERGE_TYPES
  std::cerr << "StructTy::refineAbstractTy(" << (void*)OldType << "[" 
            << *OldType << "], " << (void*)NewType << " [" 
            << *NewType << "])\n";
#endif

  // Look up our current type map entry..
  TypeMap<StructValType, StructType>::iterator TMI =
    StructTypes.getEntryForType(this);

  for (int i = ETypes.size()-1; i >= 0; --i)
    if (ETypes[i] == OldType) {
      ETypes[i].removeUserFromConcrete();

      // Update old type to new type in the array...
      ETypes[i] = NewType;
    }

  StructTypes.finishRefinement(TMI);
}

void StructType::typeBecameConcrete(const DerivedType *AbsTy) {
  refineAbstractType(AbsTy, AbsTy);
}

// refineAbstractType - Called when a contained type is found to be more
// concrete - this could potentially change us from an abstract type to a
// concrete type.
//
void PointerType::refineAbstractType(const DerivedType *OldType,
				     const Type *NewType) {
  assert((isAbstract() || !OldType->isAbstract()) &&
         "Refining a non-abstract type!");
#ifdef DEBUG_MERGE_TYPES
  std::cerr << "PointerTy::refineAbstractTy(" << (void*)OldType << "[" 
            << *OldType << "], " << (void*)NewType << " [" 
            << *NewType << "])\n";
#endif

  // Look up our current type map entry..
  TypeMap<PointerValType, PointerType>::iterator TMI =
    PointerTypes.getEntryForType(this);

  assert(ElementType == OldType);
  ElementType.removeUserFromConcrete();
  ElementType = NewType;

  PointerTypes.finishRefinement(TMI);
}

void PointerType::typeBecameConcrete(const DerivedType *AbsTy) {
  refineAbstractType(AbsTy, AbsTy);
}
