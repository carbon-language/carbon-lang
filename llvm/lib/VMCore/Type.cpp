//===-- Type.cpp - Implement the Type class -------------------------------===//
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
  case Type::ULongTyID:
  case Type::LongTyID:
  case Type::PointerTyID:
    return Ty == Type::ULongTy || Ty == Type::LongTy || isa<PointerType>(Ty);
  default:
    return false;  // Other types have no identity values
  }
}

// getPrimitiveSize - Return the basic size of this type if it is a primative
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
//                           Auxilliary classes
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
    return false;  // Two nonequal opaque types are never equal

  std::map<const Type*, const Type*>::iterator It = EqTypes.find(Ty);
  if (It != EqTypes.end())
    return It->second == Ty2;    // Looping back on a type, check for equality

  // Otherwise, add the mapping to the table to make sure we don't get
  // recursion on the types...
  EqTypes.insert(std::make_pair(Ty, Ty2));

  // Iterate over the types and make sure the the contents are equivalent...
  Type::subtype_iterator I  = Ty ->subtype_begin(), IE  = Ty ->subtype_end();
  Type::subtype_iterator I2 = Ty2->subtype_begin(), IE2 = Ty2->subtype_end();
  for (; I != IE && I2 != IE2; ++I, ++I2)
    if (!TypesEqual(*I, *I2, EqTypes)) return false;

  // Two really annoying special cases that breaks an otherwise nice simple
  // algorithm is the fact that arraytypes have sizes that differentiates types,
  // and that function types can be varargs or not.  Consider this now.
  if (const ArrayType *ATy = dyn_cast<ArrayType>(Ty)) {
    if (ATy->getNumElements() != cast<ArrayType>(Ty2)->getNumElements())
      return false;
  } else if (const FunctionType *FTy = dyn_cast<FunctionType>(Ty)) {
    if (FTy->isVarArg() != cast<FunctionType>(Ty2)->isVarArg())
      return false;
  }

  return I == IE && I2 == IE2;    // Types equal if both iterators are done
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
template<class ValType, class TypeClass>
class TypeMap : public AbstractTypeUser {
  typedef std::map<ValType, PATypeHandle> MapTy;
  MapTy Map;
public:
  typedef typename MapTy::iterator iterator;
  ~TypeMap() { print("ON EXIT"); }

  inline TypeClass *get(const ValType &V) {
    iterator I = Map.find(V);
    return I != Map.end() ? (TypeClass*)I->second.get() : 0;
  }

  inline void add(const ValType &V, TypeClass *T) {
    Map.insert(std::make_pair(V, PATypeHandle(T, this)));
    print("add");
  }

  iterator getEntryForType(TypeClass *Ty) {
    iterator I = Map.find(ValType::get(Ty));
    if (I == Map.end()) print("ERROR!");
    assert(I != Map.end() && "Didn't find type entry!");
    assert(T->second == Ty && "Type entry wrong?");
    return I;
  }


  void finishRefinement(TypeClass *Ty) {
    //const TypeClass *Ty = (const TypeClass*)TyIt->second.get();
    for (iterator I = Map.begin(), E = Map.end(); I != E; ++I)
      if (I->second.get() != Ty && TypesEqual(Ty, I->second.get())) {
        assert(Ty->isAbstract() && "Replacing a non-abstract type?");
        TypeClass *NewTy = (TypeClass*)I->second.get();
#if 0
        //Map.erase(TyIt);                // The old entry is now dead!
#endif
        // Refined to a different type altogether?
        Ty->refineAbstractTypeToInternal(NewTy, false);
        return;
      }

    // If the type is currently thought to be abstract, rescan all of our
    // subtypes to see if the type has just become concrete!
    if (Ty->isAbstract())
      Ty->setAbstract(Ty->isTypeAbstract());

    // This method may be called with either an abstract or a concrete type.
    // Concrete types might get refined if a subelement type got refined which
    // was previously marked as abstract, but was realized to be concrete.  This
    // can happen for recursive types.
    Ty->typeIsRefined();                     // Same type, different contents...
  }

  // refineAbstractType - This is called when one of the contained abstract
  // types gets refined... this simply removes the abstract type from our table.
  // We expect that whoever refined the type will add it back to the table,
  // corrected.
  //
  virtual void refineAbstractType(const DerivedType *OldTy, const Type *NewTy) {
#ifdef DEBUG_MERGE_TYPES
    std::cerr << "Removing Old type from Tab: " << (void*)OldTy << ", "
              << *OldTy << "  replacement == " << (void*)NewTy
              << ", " << *NewTy << "\n";
#endif
    for (iterator I = Map.begin(), E = Map.end(); I != E; ++I)
      if (I->second.get() == OldTy) {
        // Check to see if the type just became concrete.  If so, remove self
        // from user list.
        I->second.removeUserFromConcrete();
        I->second = cast<TypeClass>(NewTy);
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
      std::cerr << " " << (++i) << ". " << (void*)I->second.get() << " " 
                << *I->second.get() << "\n";
#endif
  }

  void dump() const { print("dump output"); }
};


// ValTypeBase - This is the base class that is used by the various
// instantiations of TypeMap.  This class is an AbstractType user that notifies
// the underlying TypeMap when it gets modified.
//
template<class ValType, class TypeClass>
class ValTypeBase : public AbstractTypeUser {
  TypeMap<ValType, TypeClass> &MyTable;
protected:
  inline ValTypeBase(TypeMap<ValType, TypeClass> &tab) : MyTable(tab) {}

  // Subclass should override this... to update self as usual
  virtual void doRefinement(const DerivedType *OldTy, const Type *NewTy) = 0;

  // typeBecameConcrete - This callback occurs when a contained type refines
  // to itself, but becomes concrete in the process.  Our subclass should remove
  // itself from the ATU list of the specified type.
  //
  virtual void typeBecameConcrete(const DerivedType *Ty) = 0;
  
  virtual void refineAbstractType(const DerivedType *OldTy, const Type *NewTy) {
    assert(OldTy == NewTy || OldTy->isAbstract());

    if (!OldTy->isAbstract())
      typeBecameConcrete(OldTy);

    TypeMap<ValType, TypeClass> &Table = MyTable;     // Copy MyTable reference
    ValType Tmp(*(ValType*)this);                     // Copy this.
    PATypeHandle OldType(Table.get(*(ValType*)this), this);
    Table.remove(*(ValType*)this);                    // Destroy's this!

    // Refine temporary to new state...
    if (OldTy != NewTy)
      Tmp.doRefinement(OldTy, NewTy); 

    // FIXME: when types are not const!
    Table.add((ValType&)Tmp, (TypeClass*)OldType.get());
  }

  void dump() const {
    std::cerr << "ValTypeBase instance!\n";
  }
};



//===----------------------------------------------------------------------===//
// Function Type Factory and Value Class...
//

// FunctionValType - Define a class to hold the key that goes into the TypeMap
//
class FunctionValType : public ValTypeBase<FunctionValType, FunctionType> {
  PATypeHandle RetTy;
  std::vector<PATypeHandle> ArgTypes;
  bool isVarArg;
public:
  FunctionValType(const Type *ret, const std::vector<const Type*> &args,
		bool IVA, TypeMap<FunctionValType, FunctionType> &Tab)
    : ValTypeBase<FunctionValType, FunctionType>(Tab), RetTy(ret, this),
      isVarArg(IVA) {
    for (unsigned i = 0; i < args.size(); ++i)
      ArgTypes.push_back(PATypeHandle(args[i], this));
  }

  // We *MUST* have an explicit copy ctor so that the TypeHandles think that
  // this FunctionValType owns them, not the old one!
  //
  FunctionValType(const FunctionValType &MVT) 
    : ValTypeBase<FunctionValType, FunctionType>(MVT), RetTy(MVT.RetTy, this),
      isVarArg(MVT.isVarArg) {
    ArgTypes.reserve(MVT.ArgTypes.size());
    for (unsigned i = 0; i < MVT.ArgTypes.size(); ++i)
      ArgTypes.push_back(PATypeHandle(MVT.ArgTypes[i], this));
  }

  static FunctionValType get(const FunctionType *FT);

  // Subclass should override this... to update self as usual
  virtual void doRefinement(const DerivedType *OldType, const Type *NewType) {
    if (RetTy == OldType) RetTy = NewType;
    for (unsigned i = 0, e = ArgTypes.size(); i != e; ++i)
      if (ArgTypes[i] == OldType) ArgTypes[i] = NewType;
  }

  virtual void typeBecameConcrete(const DerivedType *Ty) {
    if (RetTy == Ty) RetTy.removeUserFromConcrete();

    for (unsigned i = 0; i < ArgTypes.size(); ++i)
      if (ArgTypes[i] == Ty) ArgTypes[i].removeUserFromConcrete();
  }

  inline bool operator<(const FunctionValType &MTV) const {
    if (RetTy.get() < MTV.RetTy.get()) return true;
    if (RetTy.get() > MTV.RetTy.get()) return false;

    if (ArgTypes < MTV.ArgTypes) return true;
    return (ArgTypes == MTV.ArgTypes) && isVarArg < MTV.isVarArg;
  }
};

// Define the actual map itself now...
static TypeMap<FunctionValType, FunctionType> FunctionTypes;

FunctionValType FunctionValType::get(const FunctionType *FT) {
  // Build up a FunctionValType
  std::vector<const Type *> ParamTypes;
  ParamTypes.reserve(FT->getParamTypes().size());
  for (unsigned i = 0, e = FT->getParamTypes().size(); i != e; ++i)
    ParamTypes.push_back(FT->getParamType(i));
  return FunctionValType(FT->getReturnType(), ParamTypes, FT->isVarArg(),
                         FunctionTypes);
}


// FunctionType::get - The factory function for the FunctionType class...
FunctionType *FunctionType::get(const Type *ReturnType, 
                                const std::vector<const Type*> &Params,
                                bool isVarArg) {
  FunctionValType VT(ReturnType, Params, isVarArg, FunctionTypes);
  FunctionType *MT = FunctionTypes.get(VT);
  if (MT) return MT;

  FunctionTypes.add(VT, MT = new FunctionType(ReturnType, Params, isVarArg));

#ifdef DEBUG_MERGE_TYPES
  std::cerr << "Derived new type: " << MT << "\n";
#endif
  return MT;
}

void FunctionType::dropAllTypeUses(bool inMap) {
#if 0
  if (inMap) FunctionTypes.remove(FunctionTypes.getEntryForType(this));
  // Drop all uses of other types, which might be recursive.
#endif
  ResultType = OpaqueType::get();
  ParamTys.clear();
}


//===----------------------------------------------------------------------===//
// Array Type Factory...
//
class ArrayValType : public ValTypeBase<ArrayValType, ArrayType> {
  PATypeHandle ValTy;
  unsigned Size;
public:
  ArrayValType(const Type *val, int sz, TypeMap<ArrayValType, ArrayType> &Tab)
    : ValTypeBase<ArrayValType, ArrayType>(Tab), ValTy(val, this), Size(sz) {}

  // We *MUST* have an explicit copy ctor so that the ValTy thinks that this
  // ArrayValType owns it, not the old one!
  //
  ArrayValType(const ArrayValType &AVT) 
    : ValTypeBase<ArrayValType, ArrayType>(AVT), ValTy(AVT.ValTy, this),
      Size(AVT.Size) {}

  static ArrayValType get(const ArrayType *AT);


  // Subclass should override this... to update self as usual
  virtual void doRefinement(const DerivedType *OldType, const Type *NewType) {
    assert(ValTy == OldType);
    ValTy = NewType;
  }

  virtual void typeBecameConcrete(const DerivedType *Ty) {
    assert(ValTy == Ty &&
           "Contained type became concrete but we're not using it!");
    ValTy.removeUserFromConcrete();
  }

  inline bool operator<(const ArrayValType &MTV) const {
    if (Size < MTV.Size) return true;
    return Size == MTV.Size && ValTy.get() < MTV.ValTy.get();
  }
};

static TypeMap<ArrayValType, ArrayType> ArrayTypes;

ArrayValType ArrayValType::get(const ArrayType *AT) {
  return ArrayValType(AT->getElementType(), AT->getNumElements(), ArrayTypes);
}


ArrayType *ArrayType::get(const Type *ElementType, unsigned NumElements) {
  assert(ElementType && "Can't get array of null types!");

  ArrayValType AVT(ElementType, NumElements, ArrayTypes);
  ArrayType *AT = ArrayTypes.get(AVT);
  if (AT) return AT;           // Found a match, return it!

  // Value not found.  Derive a new type!
  ArrayTypes.add(AVT, AT = new ArrayType(ElementType, NumElements));

#ifdef DEBUG_MERGE_TYPES
  std::cerr << "Derived new type: " << *AT << "\n";
#endif
  return AT;
}

void ArrayType::dropAllTypeUses(bool inMap) {
#if 0
  if (inMap) ArrayTypes.remove(ArrayTypes.getEntryForType(this));
#endif
  ElementType = OpaqueType::get();
}




//===----------------------------------------------------------------------===//
// Struct Type Factory...
//

// StructValType - Define a class to hold the key that goes into the TypeMap
//
class StructValType : public ValTypeBase<StructValType, StructType> {
  std::vector<PATypeHandle> ElTypes;
public:
  StructValType(const std::vector<const Type*> &args,
		TypeMap<StructValType, StructType> &Tab)
    : ValTypeBase<StructValType, StructType>(Tab) {
    ElTypes.reserve(args.size());
    for (unsigned i = 0, e = args.size(); i != e; ++i)
      ElTypes.push_back(PATypeHandle(args[i], this));
  }

  // We *MUST* have an explicit copy ctor so that the TypeHandles think that
  // this StructValType owns them, not the old one!
  //
  StructValType(const StructValType &SVT) 
    : ValTypeBase<StructValType, StructType>(SVT){
    ElTypes.reserve(SVT.ElTypes.size());
    for (unsigned i = 0, e = SVT.ElTypes.size(); i != e; ++i)
      ElTypes.push_back(PATypeHandle(SVT.ElTypes[i], this));
  }

  static StructValType get(const StructType *ST);

  // Subclass should override this... to update self as usual
  virtual void doRefinement(const DerivedType *OldType, const Type *NewType) {
    for (unsigned i = 0; i < ElTypes.size(); ++i)
      if (ElTypes[i] == OldType) ElTypes[i] = NewType;
  }

  virtual void typeBecameConcrete(const DerivedType *Ty) {
    for (unsigned i = 0, e = ElTypes.size(); i != e; ++i)
      if (ElTypes[i] == Ty)
        ElTypes[i].removeUserFromConcrete();
  }

  inline bool operator<(const StructValType &STV) const {
    return ElTypes < STV.ElTypes;
  }
};

static TypeMap<StructValType, StructType> StructTypes;

StructValType StructValType::get(const StructType *ST) {
  std::vector<const Type *> ElTypes;
  ElTypes.reserve(ST->getElementTypes().size());
  for (unsigned i = 0, e = ST->getElementTypes().size(); i != e; ++i)
    ElTypes.push_back(ST->getElementTypes()[i]);
  
  return StructValType(ElTypes, StructTypes);
}



StructType *StructType::get(const std::vector<const Type*> &ETypes) {
  StructValType STV(ETypes, StructTypes);
  StructType *ST = StructTypes.get(STV);
  if (ST) return ST;

  // Value not found.  Derive a new type!
  StructTypes.add(STV, ST = new StructType(ETypes));

#ifdef DEBUG_MERGE_TYPES
  std::cerr << "Derived new type: " << *ST << "\n";
#endif
  return ST;
}

void StructType::dropAllTypeUses(bool inMap) {
#if 0
  if (inMap) StructTypes.remove(StructTypes.getEntryForType(this));
#endif
  ETypes.clear();
  ETypes.push_back(PATypeHandle(OpaqueType::get(), this));
}



//===----------------------------------------------------------------------===//
// Pointer Type Factory...
//

// PointerValType - Define a class to hold the key that goes into the TypeMap
//
class PointerValType : public ValTypeBase<PointerValType, PointerType> {
  PATypeHandle ValTy;
public:
  PointerValType(const Type *val, TypeMap<PointerValType, PointerType> &Tab)
    : ValTypeBase<PointerValType, PointerType>(Tab), ValTy(val, this) {}

  // We *MUST* have an explicit copy ctor so that the ValTy thinks that this
  // PointerValType owns it, not the old one!
  //
  PointerValType(const PointerValType &PVT) 
    : ValTypeBase<PointerValType, PointerType>(PVT), ValTy(PVT.ValTy, this) {}

  static PointerValType get(const PointerType *PT);

  // Subclass should override this... to update self as usual
  virtual void doRefinement(const DerivedType *OldType, const Type *NewType) {
    assert(ValTy == OldType);
    ValTy = NewType;
  }

  virtual void typeBecameConcrete(const DerivedType *Ty) {
    assert(ValTy == Ty &&
           "Contained type became concrete but we're not using it!");
    ValTy.removeUserFromConcrete();
  }

  inline bool operator<(const PointerValType &MTV) const {
    return ValTy.get() < MTV.ValTy.get();
  }
};

static TypeMap<PointerValType, PointerType> PointerTypes;

PointerValType PointerValType::get(const PointerType *PT) {
  return PointerValType(PT->getElementType(), PointerTypes);
}


PointerType *PointerType::get(const Type *ValueType) {
  assert(ValueType && "Can't get a pointer to <null> type!");
  PointerValType PVT(ValueType, PointerTypes);

  PointerType *PT = PointerTypes.get(PVT);
  if (PT) return PT;

  // Value not found.  Derive a new type!
  PointerTypes.add(PVT, PT = new PointerType(ValueType));

#ifdef DEBUG_MERGE_TYPES
  std::cerr << "Derived new type: " << *PT << "\n";
#endif
  return PT;
}

void PointerType::dropAllTypeUses(bool inMap) {
#if 0
  if (inMap) PointerTypes.remove(PointerTypes.getEntryForType(this));
#endif
  ElementType = OpaqueType::get();
}

void debug_type_tables() {
  FunctionTypes.dump();
  ArrayTypes.dump();
  StructTypes.dump();
  PointerTypes.dump();
}


//===----------------------------------------------------------------------===//
//                     Derived Type Refinement Functions
//===----------------------------------------------------------------------===//

// removeAbstractTypeUser - Notify an abstract type that a user of the class
// no longer has a handle to the type.  This function is called primarily by
// the PATypeHandle class.  When there are no users of the abstract type, it
// is anihilated, because there is no way to get a reference to it ever again.
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


// refineAbstractTypeToInternal - This function is used to when it is discovered
// that the 'this' abstract type is actually equivalent to the NewType
// specified.  This causes all users of 'this' to switch to reference the more
// concrete type NewType and for 'this' to be deleted.
//
void DerivedType::refineAbstractTypeToInternal(const Type *NewType, bool inMap){
  assert(isAbstract() && "refineAbstractTypeTo: Current type is not abstract!");
  assert(this != NewType && "Can't refine to myself!");
  
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
  dropAllTypeUses(inMap);

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

#ifdef DEBUG_MERGE_TYPES
    if (AbstractTypeUsers.size() == OldSize) {
      User->refineAbstractType(this, NewTy);
      if (AbstractTypeUsers.back() != User)
        std::cerr << "User changed!\n";
      std::cerr << "Top of user list is:\n";
      AbstractTypeUsers.back()->dump();
      
      std::cerr <<"\nOld User=\n";
      User->dump();
    }
#endif
    assert(AbstractTypeUsers.size() != OldSize &&
           "AbsTyUser did not remove self from user list!");
  }

  // If we were successful removing all users from the type, 'this' will be
  // deleted when the last PATypeHolder is destroyed or updated from this type.
  // This may occur on exit of this function, as the CurrentTy object is
  // destroyed.
}

// typeIsRefined - Notify AbstractTypeUsers of this type that the current type
// has been refined a bit.  The pointer is still valid and still should be
// used, but the subtypes have changed.
//
void DerivedType::typeIsRefined() {
  assert(isRefining >= 0 && isRefining <= 2 && "isRefining out of bounds!");
  if (isRefining == 1) return;  // Kill recursion here...
  ++isRefining;

#ifdef DEBUG_MERGE_TYPES
  std::cerr << "typeIsREFINED type: " << (void*)this << " " << *this << "\n";
#endif

  // In this loop we have to be very careful not to get into infinite loops and
  // other problem cases.  Specifically, we loop through all of the abstract
  // type users in the user list, notifying them that the type has been refined.
  // At their choice, they may or may not choose to remove themselves from the
  // list of users.  Regardless of whether they do or not, we have to be sure
  // that we only notify each user exactly once.  Because the refineAbstractType
  // method can cause an arbitrary permutation to the user list, we cannot loop
  // through it in any particular order and be guaranteed that we will be
  // successful at this aim.  Because of this, we keep track of all the users we
  // have visited and only visit users we have not seen.  Because this user list
  // should be small, we use a vector instead of a full featured set to keep
  // track of what users we have notified so far.
  //
  std::vector<AbstractTypeUser*> Refined;
  while (1) {
    unsigned i;
    for (i = AbstractTypeUsers.size(); i != 0; --i)
      if (find(Refined.begin(), Refined.end(), AbstractTypeUsers[i-1]) ==
          Refined.end())
        break;    // Found an unrefined user?
    
    if (i == 0) break;  // Noone to refine left, break out of here!

    AbstractTypeUser *ATU = AbstractTypeUsers[--i];
    Refined.push_back(ATU);  // Keep track of which users we have refined!

#ifdef DEBUG_MERGE_TYPES
    std::cerr << " typeIsREFINED user " << i << "[" << ATU
              << "] of abstract type [" << (void*)this << " "
              << *this << "]\n";
#endif
    ATU->refineAbstractType(this, this);
  }

  --isRefining;

#ifndef _NDEBUG
  if (!(isAbstract() || AbstractTypeUsers.empty()))
    for (unsigned i = 0; i < AbstractTypeUsers.size(); ++i) {
      if (AbstractTypeUsers[i] != this) {
        // Debugging hook
        std::cerr << "FOUND FAILURE\nUser: ";
        AbstractTypeUsers[i]->dump();
        std::cerr << "\nCatch:\n";
        AbstractTypeUsers[i]->refineAbstractType(this, this);
        assert(0 && "Type became concrete,"
               " but it still has abstract type users hanging around!");
      }
  }
#endif
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
#if 0
  TypeMap<FunctionValType, FunctionType>::iterator TMI =
    FunctionTypes.getEntryForType(this);
#endif

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

  FunctionTypes.finishRefinement(this);
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

#if 0
  // Look up our current type map entry..
  TypeMap<ArrayValType, ArrayType>::iterator TMI =
    ArrayTypes.getEntryForType(this);
#endif

  assert(getElementType() == OldType);
  ElementType.removeUserFromConcrete();
  ElementType = NewType;

  ArrayTypes.finishRefinement(this);
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

#if 0
  // Look up our current type map entry..
  TypeMap<StructValType, StructType>::iterator TMI =
    StructTypes.getEntryForType(this);
#endif

  for (int i = ETypes.size()-1; i >= 0; --i)
    if (ETypes[i] == OldType) {
      ETypes[i].removeUserFromConcrete();

      // Update old type to new type in the array...
      ETypes[i] = NewType;
    }

  StructTypes.finishRefinement(this);
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

#if 0
  // Look up our current type map entry..
  TypeMap<PointerValType, PointerType>::iterator TMI =
    PointerTypes.getEntryForType(this);
#endif

  assert(ElementType == OldType);
  ElementType.removeUserFromConcrete();
  ElementType = NewType;

  PointerTypes.finishRefinement(this);
}

