//===-- Type.cpp - Implement the Type class ----------------------*- C++ -*--=//
//
// This file implements the Type class for the VMCore library.
//
//===----------------------------------------------------------------------===//

#include "llvm/DerivedTypes.h"
#include "llvm/SymbolTable.h"
#include "Support/StringExtras.h"
#include "Support/STLExtras.h"

// DEBUG_MERGE_TYPES - Enable this #define to see how and when derived types are
// created and later destroyed, all in an effort to make sure that there is only
// a single cannonical version of a type.
//
//#define DEBUG_MERGE_TYPES 1



//===----------------------------------------------------------------------===//
//                         Type Class Implementation
//===----------------------------------------------------------------------===//

static unsigned CurUID = 0;
static vector<const Type *> UIDMappings;

Type::Type(const string &name, PrimitiveID id)
  : Value(Type::TypeTy, Value::TypeVal) {
  setDescription(name);
  ID = id;
  Abstract = Recursive = false;
  UID = CurUID++;       // Assign types UID's as they are created
  UIDMappings.push_back(this);
}

void Type::setName(const string &Name, SymbolTable *ST) {
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

// isLosslesslyConvertableTo - Return true if this type can be converted to
// 'Ty' without any reinterpretation of bits.  For example, uint to int.
//
bool Type::isLosslesslyConvertableTo(const Type *Ty) const {
  if (this == Ty) return true;
  if ((!isPrimitiveType()   && !isPointerType()) ||
      (!Ty->isPointerType() && !Ty->isPrimitiveType())) return false;

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
    return Ty == Type::ULongTy || Ty == Type::LongTy ||
           Ty->getPrimitiveID() == Type::PointerTyID;
  default:
    return false;  // Other types have no identity values
  }
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
class SignedIntType : public Type {
  int Size;
public:
  SignedIntType(const string &Name, PrimitiveID id, int size) : Type(Name, id) {
    Size = size;
  }

  // isSigned - Return whether a numeric type is signed.
  virtual bool isSigned() const { return 1; }

  // isIntegral - Equivalent to isSigned() || isUnsigned, but with only a single
  // virtual function invocation.
  //
  virtual bool isIntegral() const { return 1; }
};

class UnsignedIntType : public Type {
  uint64_t Size;
public:
  UnsignedIntType(const string &N, PrimitiveID id, int size) : Type(N, id) {
    Size = size;
  }

  // isUnsigned - Return whether a numeric type is signed.
  virtual bool isUnsigned() const { return 1; }

  // isIntegral - Equivalent to isSigned() || isUnsigned, but with only a single
  // virtual function invocation.
  //
  virtual bool isIntegral() const { return 1; }
};

static struct TypeType : public Type {
  TypeType() : Type("type", TypeTyID) {}
} TheTypeType;   // Implement the type that is global.


//===----------------------------------------------------------------------===//
//                           Static 'Type' data
//===----------------------------------------------------------------------===//

Type *Type::VoidTy   = new            Type("void"  , VoidTyID),
     *Type::BoolTy   = new            Type("bool"  , BoolTyID),
     *Type::SByteTy  = new   SignedIntType("sbyte" , SByteTyID, 1),
     *Type::UByteTy  = new UnsignedIntType("ubyte" , UByteTyID, 1),
     *Type::ShortTy  = new   SignedIntType("short" ,  ShortTyID, 2),
     *Type::UShortTy = new UnsignedIntType("ushort", UShortTyID, 2),
     *Type::IntTy    = new   SignedIntType("int"   ,  IntTyID, 4), 
     *Type::UIntTy   = new UnsignedIntType("uint"  , UIntTyID, 4),
     *Type::LongTy   = new   SignedIntType("long"  ,  LongTyID, 8),
     *Type::ULongTy  = new UnsignedIntType("ulong" , ULongTyID, 8),
     *Type::FloatTy  = new            Type("float" , FloatTyID),
     *Type::DoubleTy = new            Type("double", DoubleTyID),
     *Type::TypeTy   =        &TheTypeType,
     *Type::LabelTy  = new            Type("label" , LabelTyID);


//===----------------------------------------------------------------------===//
//                          Derived Type Constructors
//===----------------------------------------------------------------------===//

MethodType::MethodType(const Type *Result, const vector<const Type*> &Params, 
                       bool IsVarArgs) : DerivedType("", MethodTyID), 
    ResultType(PATypeHandle<Type>(Result, this)),
    isVarArgs(IsVarArgs) {
  ParamTys.reserve(Params.size());
  for (unsigned i = 0; i < Params.size(); ++i)
    ParamTys.push_back(PATypeHandle<Type>(Params[i], this));

  setDerivedTypeProperties();
}

ArrayType::ArrayType(const Type *ElType, int NumEl)
  : CompositeType("", ArrayTyID), ElementType(PATypeHandle<Type>(ElType, this)){
  NumElements = NumEl;
  setDerivedTypeProperties();
}

StructType::StructType(const vector<const Type*> &Types)
  : CompositeType("", StructTyID) {
  ETypes.reserve(Types.size());
  for (unsigned i = 0; i < Types.size(); ++i) {
    assert(Types[i] != Type::VoidTy && "Void type in method prototype!!");
    ETypes.push_back(PATypeHandle<Type>(Types[i], this));
  }
  setDerivedTypeProperties();
}

PointerType::PointerType(const Type *E) : DerivedType("", PointerTyID),
			  ValueType(PATypeHandle<Type>(E, this)) {
  setDerivedTypeProperties();
}

OpaqueType::OpaqueType() : DerivedType("", OpaqueTyID) {
  setAbstract(true);
  setDescription("opaque"+utostr(getUniqueID()));
#ifdef DEBUG_MERGE_TYPES
  cerr << "Derived new type: " << getDescription() << endl;
#endif
}




//===----------------------------------------------------------------------===//
//               Derived Type setDerivedTypeProperties Function
//===----------------------------------------------------------------------===//

// getTypeProps - This is a recursive function that walks a type hierarchy
// calculating the description for a type and whether or not it is abstract or
// recursive.  Worst case it will have to do a lot of traversing if you have
// some whacko opaque types, but in most cases, it will do some simple stuff
// when it hits non-abstract types that aren't recursive.
//
static string getTypeProps(const Type *Ty, vector<const Type *> &TypeStack,
			   bool &isAbstract, bool &isRecursive) {
  string Result;
  if (!Ty->isAbstract() && !Ty->isRecursive() && // Base case for the recursion
      Ty->getDescription().size()) {
    Result = Ty->getDescription();               // Primitive = leaf type
  } else if (isa<OpaqueType>(Ty)) {              // Base case for the recursion
    Result = Ty->getDescription();               // Opaque = leaf type
    isAbstract = true;                           // This whole type is abstract!
  } else {
    // Check to see if the Type is already on the stack...
    unsigned Slot = 0, CurSize = TypeStack.size();
    while (Slot < CurSize && TypeStack[Slot] != Ty) ++Slot; // Scan for type
    
    // This is another base case for the recursion.  In this case, we know 
    // that we have looped back to a type that we have previously visited.
    // Generate the appropriate upreference to handle this.
    // 
    if (Slot < CurSize) {
      Result = "\\" + utostr(CurSize-Slot);       // Here's the upreference
      isRecursive = true;                         // We know we are recursive
    } else {                      // Recursive case: abstract derived type...
      TypeStack.push_back(Ty);    // Add us to the stack..
      
      switch (Ty->getPrimitiveID()) {
      case Type::MethodTyID: {
	const MethodType *MTy = cast<const MethodType>(Ty);
	Result = getTypeProps(MTy->getReturnType(), TypeStack,
			      isAbstract, isRecursive)+" (";
	for (MethodType::ParamTypes::const_iterator
	       I = MTy->getParamTypes().begin(),
	       E = MTy->getParamTypes().end(); I != E; ++I) {
	  if (I != MTy->getParamTypes().begin())
	    Result += ", ";
	  Result += getTypeProps(*I, TypeStack, isAbstract, isRecursive);
	}
	if (MTy->isVarArg()) {
	  if (!MTy->getParamTypes().empty()) Result += ", ";
	  Result += "...";
	}
	Result += ")";
	break;
      }
      case Type::StructTyID: {
	const StructType *STy = cast<const StructType>(Ty);
	Result = "{ ";
	for (StructType::ElementTypes::const_iterator
	       I = STy->getElementTypes().begin(),
	       E = STy->getElementTypes().end(); I != E; ++I) {
	  if (I != STy->getElementTypes().begin())
	    Result += ", ";
	  Result += getTypeProps(*I, TypeStack, isAbstract, isRecursive);
	}
	Result += " }";
	break;
      }
      case Type::PointerTyID: {
	const PointerType *PTy = cast<const PointerType>(Ty);
	Result = getTypeProps(PTy->getElementType(), TypeStack,
			      isAbstract, isRecursive) + " *";
	break;
      }
      case Type::ArrayTyID: {
	const ArrayType *ATy = cast<const ArrayType>(Ty);
	int NumElements = ATy->getNumElements();
	Result = "[";
	if (NumElements != -1) Result += itostr(NumElements) + " x ";
	Result += getTypeProps(ATy->getElementType(), TypeStack,
			       isAbstract, isRecursive) + "]";
	break;
      }
      default:
	assert(0 && "Unhandled case in getTypeProps!");
	Result = "<error>";
      }

      TypeStack.pop_back();       // Remove self from stack...
    }
  }
  return Result;
}


// setDerivedTypeProperties - This function is used to calculate the
// isAbstract, isRecursive, and the Description settings for a type.  The
// getTypeProps function does all the dirty work.
//
void DerivedType::setDerivedTypeProperties() {
  vector<const Type *> TypeStack;
  bool isAbstract = false, isRecursive = false;
  
  setDescription(getTypeProps(this, TypeStack, isAbstract, isRecursive));
  setAbstract(isAbstract);
  setRecursive(isRecursive);
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
		       map<const Type *, const Type *> &EqTypes) {
  if (Ty == Ty2) return true;
  if (Ty->getPrimitiveID() != Ty2->getPrimitiveID()) return false;
  if (Ty->isPrimitiveType()) return true;
  if (isa<OpaqueType>(Ty))
    return false;  // Two nonequal opaque types are never equal

  map<const Type*, const Type*>::iterator It = EqTypes.find(Ty);
  if (It != EqTypes.end())
    return It->second == Ty2;    // Looping back on a type, check for equality

  // Otherwise, add the mapping to the table to make sure we don't get
  // recursion on the types...
  EqTypes.insert(make_pair(Ty, Ty2));

  // Iterate over the types and make sure the the contents are equivalent...
  Type::subtype_iterator I  = Ty ->subtype_begin(), IE  = Ty ->subtype_end();
  Type::subtype_iterator I2 = Ty2->subtype_begin(), IE2 = Ty2->subtype_end();
  for (; I != IE && I2 != IE2; ++I, ++I2)
    if (!TypesEqual(*I, *I2, EqTypes)) return false;

  // Two really annoying special cases that breaks an otherwise nice simple
  // algorithm is the fact that arraytypes have sizes that differentiates types,
  // and that method types can be varargs or not.  Consider this now.
  if (const ArrayType *ATy = dyn_cast<ArrayType>(Ty)) {
    if (ATy->getNumElements() != cast<const ArrayType>(Ty2)->getNumElements())
      return false;
  } else if (const MethodType *MTy = dyn_cast<MethodType>(Ty)) {
    if (MTy->isVarArg() != cast<const MethodType>(Ty2)->isVarArg())
      return false;
  }

  return I == IE && I2 == IE2;    // Types equal if both iterators are done
}

static bool TypesEqual(const Type *Ty, const Type *Ty2) {
  map<const Type *, const Type *> EqTypes;
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
  typedef map<ValType, PATypeHandle<TypeClass> > MapTy;
  MapTy Map;
public:

  ~TypeMap() { print("ON EXIT"); }

  inline TypeClass *get(const ValType &V) {
    map<ValType, PATypeHandle<TypeClass> >::iterator I = Map.find(V);
    // TODO: FIXME: When Types are not CONST.
    return (I != Map.end()) ? (TypeClass*)I->second.get() : 0;
  }

  inline void add(const ValType &V, TypeClass *T) {
    Map.insert(make_pair(V, PATypeHandle<TypeClass>(T, this)));
    print("add");
  }

  // containsEquivalent - Return true if the typemap contains a type that is
  // structurally equivalent to the specified type.
  //
  inline const TypeClass *containsEquivalent(const TypeClass *Ty) {
    for (MapTy::iterator I = Map.begin(), E = Map.end(); I != E; ++I)
      if (I->second.get() != Ty && TypesEqual(Ty, I->second.get()))
	return (TypeClass*)I->second.get();  // FIXME TODO when types not const
    return 0;
  }

  // refineAbstractType - This is called when one of the contained abstract
  // types gets refined... this simply removes the abstract type from our table.
  // We expect that whoever refined the type will add it back to the table,
  // corrected.
  //
  virtual void refineAbstractType(const DerivedType *OldTy, const Type *NewTy) {
    if (OldTy == NewTy) {
      if (!OldTy->isAbstract()) {
        // Check to see if the type just became concrete.
        // If so, remove self from user list.
        for (MapTy::iterator I = Map.begin(), E = Map.end(); I != E; ++I)
          if (I->second == OldTy)
            I->second.removeUserFromConcrete();
      }
      return;
    }
#ifdef DEBUG_MERGE_TYPES
    cerr << "Removing Old type from Tab: " << (void*)OldTy << ", "
	 << OldTy->getDescription() << "  replacement == " << (void*)NewTy
	 << ", " << NewTy->getDescription() << endl;
#endif
    for (MapTy::iterator I = Map.begin(), E = Map.end(); I != E; ++I)
      if (I->second == OldTy) {
	Map.erase(I);
	print("refineAbstractType after");
	return;
      }
    assert(0 && "Abstract type not found in table!");
  }

  void remove(const ValType &OldVal) {
    MapTy::iterator I = Map.find(OldVal);
    assert(I != Map.end() && "TypeMap::remove, element not found!");
    Map.erase(I);
  }

  void print(const char *Arg) {
#ifdef DEBUG_MERGE_TYPES
    cerr << "TypeMap<>::" << Arg << " table contents:\n";
    unsigned i = 0;
    for (MapTy::iterator I = Map.begin(), E = Map.end(); I != E; ++I)
      cerr << " " << (++i) << ". " << I->second << " " 
	   << I->second->getDescription() << endl;
#endif
  }
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
    if (OldTy == NewTy) {
      if (!OldTy->isAbstract())
        typeBecameConcrete(OldTy);
      return;
    }
    TypeMap<ValType, TypeClass> &Table = MyTable;     // Copy MyTable reference
    ValType Tmp(*(ValType*)this);                     // Copy this.
    PATypeHandle<TypeClass> OldType(Table.get(*(ValType*)this), this);
    Table.remove(*(ValType*)this);                    // Destroy's this!
#if 1
    // Refine temporary to new state...
    Tmp.doRefinement(OldTy, NewTy); 

    Table.add((ValType&)Tmp, (TypeClass*)OldType.get());
#endif
  }
};



//===----------------------------------------------------------------------===//
// Method Type Factory and Value Class...
//

// MethodValType - Define a class to hold the key that goes into the TypeMap
//
class MethodValType : public ValTypeBase<MethodValType, MethodType> {
  PATypeHandle<Type> RetTy;
  vector<PATypeHandle<Type> > ArgTypes;
  bool isVarArg;
public:
  MethodValType(const Type *ret, const vector<const Type*> &args,
		bool IVA, TypeMap<MethodValType, MethodType> &Tab)
    : ValTypeBase<MethodValType, MethodType>(Tab), RetTy(ret, this),
      isVarArg(IVA) {
    for (unsigned i = 0; i < args.size(); ++i)
      ArgTypes.push_back(PATypeHandle<Type>(args[i], this));
  }

  // We *MUST* have an explicit copy ctor so that the TypeHandles think that
  // this MethodValType owns them, not the old one!
  //
  MethodValType(const MethodValType &MVT) 
    : ValTypeBase<MethodValType, MethodType>(MVT), RetTy(MVT.RetTy, this),
      isVarArg(MVT.isVarArg) {
    ArgTypes.reserve(MVT.ArgTypes.size());
    for (unsigned i = 0; i < MVT.ArgTypes.size(); ++i)
      ArgTypes.push_back(PATypeHandle<Type>(MVT.ArgTypes[i], this));
  }

  // Subclass should override this... to update self as usual
  virtual void doRefinement(const DerivedType *OldType, const Type *NewType) {
    if (RetTy == OldType) RetTy = NewType;
    for (unsigned i = 0; i < ArgTypes.size(); ++i)
      if (ArgTypes[i] == OldType) ArgTypes[i] = NewType;
  }

  virtual void typeBecameConcrete(const DerivedType *Ty) {
    if (RetTy == Ty) RetTy.removeUserFromConcrete();

    for (unsigned i = 0; i < ArgTypes.size(); ++i)
      if (ArgTypes[i] == Ty) ArgTypes[i].removeUserFromConcrete();
  }

  inline bool operator<(const MethodValType &MTV) const {
    if (RetTy.get() < MTV.RetTy.get()) return true;
    if (RetTy.get() > MTV.RetTy.get()) return false;

    if (ArgTypes < MTV.ArgTypes) return true;
    return (ArgTypes == MTV.ArgTypes) && isVarArg < MTV.isVarArg;
  }
};

// Define the actual map itself now...
static TypeMap<MethodValType, MethodType> MethodTypes;

// MethodType::get - The factory function for the MethodType class...
MethodType *MethodType::get(const Type *ReturnType, 
			    const vector<const Type*> &Params,
			    bool isVarArg) {
  MethodValType VT(ReturnType, Params, isVarArg, MethodTypes);
  MethodType *MT = MethodTypes.get(VT);
  if (MT) return MT;

  MethodTypes.add(VT, MT = new MethodType(ReturnType, Params, isVarArg));

#ifdef DEBUG_MERGE_TYPES
  cerr << "Derived new type: " << MT << endl;
#endif
  return MT;
}

//===----------------------------------------------------------------------===//
// Array Type Factory...
//
class ArrayValType : public ValTypeBase<ArrayValType, ArrayType> {
  PATypeHandle<Type> ValTy;
  int Size;
public:
  ArrayValType(const Type *val, int sz, TypeMap<ArrayValType, ArrayType> &Tab)
    : ValTypeBase<ArrayValType, ArrayType>(Tab), ValTy(val, this), Size(sz) {}

  // We *MUST* have an explicit copy ctor so that the ValTy thinks that this
  // ArrayValType owns it, not the old one!
  //
  ArrayValType(const ArrayValType &AVT) 
    : ValTypeBase<ArrayValType, ArrayType>(AVT), ValTy(AVT.ValTy, this),
      Size(AVT.Size) {}

  // Subclass should override this... to update self as usual
  virtual void doRefinement(const DerivedType *OldType, const Type *NewType) {
    if (ValTy == OldType) ValTy = NewType;
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

ArrayType *ArrayType::get(const Type *ElementType, int NumElements = -1) {
  assert(ElementType && "Can't get array of null types!");

  ArrayValType AVT(ElementType, NumElements, ArrayTypes);
  ArrayType *AT = ArrayTypes.get(AVT);
  if (AT) return AT;           // Found a match, return it!

  // Value not found.  Derive a new type!
  ArrayTypes.add(AVT, AT = new ArrayType(ElementType, NumElements));

#ifdef DEBUG_MERGE_TYPES
  cerr << "Derived new type: " << AT->getDescription() << endl;
#endif
  return AT;
}

//===----------------------------------------------------------------------===//
// Struct Type Factory...
//

// StructValType - Define a class to hold the key that goes into the TypeMap
//
class StructValType : public ValTypeBase<StructValType, StructType> {
  vector<PATypeHandle<Type> > ElTypes;
public:
  StructValType(const vector<const Type*> &args,
		TypeMap<StructValType, StructType> &Tab)
    : ValTypeBase<StructValType, StructType>(Tab) {
    for (unsigned i = 0; i < args.size(); ++i)
      ElTypes.push_back(PATypeHandle<Type>(args[i], this));
  }

  // We *MUST* have an explicit copy ctor so that the TypeHandles think that
  // this StructValType owns them, not the old one!
  //
  StructValType(const StructValType &SVT) 
    : ValTypeBase<StructValType, StructType>(SVT){
    ElTypes.reserve(SVT.ElTypes.size());
    for (unsigned i = 0; i < SVT.ElTypes.size(); ++i)
      ElTypes.push_back(PATypeHandle<Type>(SVT.ElTypes[i], this));
  }

  // Subclass should override this... to update self as usual
  virtual void doRefinement(const DerivedType *OldType, const Type *NewType) {
    for (unsigned i = 0; i < ElTypes.size(); ++i)
      if (ElTypes[i] == OldType) ElTypes[i] = NewType;
  }

  virtual void typeBecameConcrete(const DerivedType *Ty) {
    for (unsigned i = 0; i < ElTypes.size(); ++i)
      if (ElTypes[i] == Ty) ElTypes[i].removeUserFromConcrete();
  }

  inline bool operator<(const StructValType &STV) const {
    return ElTypes < STV.ElTypes;
  }
};

static TypeMap<StructValType, StructType> StructTypes;

StructType *StructType::get(const vector<const Type*> &ETypes) {
  StructValType STV(ETypes, StructTypes);
  StructType *ST = StructTypes.get(STV);
  if (ST) return ST;

  // Value not found.  Derive a new type!
  StructTypes.add(STV, ST = new StructType(ETypes));

#ifdef DEBUG_MERGE_TYPES
  cerr << "Derived new type: " << ST->getDescription() << endl;
#endif
  return ST;
}

//===----------------------------------------------------------------------===//
// Pointer Type Factory...
//

// PointerValType - Define a class to hold the key that goes into the TypeMap
//
class PointerValType : public ValTypeBase<PointerValType, PointerType> {
  PATypeHandle<Type> ValTy;
public:
  PointerValType(const Type *val, TypeMap<PointerValType, PointerType> &Tab)
    : ValTypeBase<PointerValType, PointerType>(Tab), ValTy(val, this) {}

  // We *MUST* have an explicit copy ctor so that the ValTy thinks that this
  // PointerValType owns it, not the old one!
  //
  PointerValType(const PointerValType &PVT) 
    : ValTypeBase<PointerValType, PointerType>(PVT), ValTy(PVT.ValTy, this) {}

  // Subclass should override this... to update self as usual
  virtual void doRefinement(const DerivedType *OldType, const Type *NewType) {
    if (ValTy == OldType) ValTy = NewType;
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

PointerType *PointerType::get(const Type *ValueType) {
  assert(ValueType && "Can't get a pointer to <null> type!");
  PointerValType PVT(ValueType, PointerTypes);

  PointerType *PT = PointerTypes.get(PVT);
  if (PT) return PT;

  // Value not found.  Derive a new type!
  PointerTypes.add(PVT, PT = new PointerType(ValueType));

#ifdef DEBUG_MERGE_TYPES
  cerr << "Derived new type: " << PT->getDescription() << endl;
#endif
  return PT;
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
  for (unsigned i = AbstractTypeUsers.size(); i > 0; --i) {
    if (AbstractTypeUsers[i-1] == U) {
      AbstractTypeUsers.erase(AbstractTypeUsers.begin()+i-1);
      
#ifdef DEBUG_MERGE_TYPES
      cerr << "  removeAbstractTypeUser<" << (void*)this << ", "
	   << getDescription() << ">[" << i << "] User = " << U << endl;
#endif

      if (AbstractTypeUsers.empty() && isAbstract()) {
#ifdef DEBUG_MERGE_TYPES
	cerr << "DELETEing unused abstract type: <" << getDescription()
	     << ">[" << (void*)this << "]" << endl;
#endif
	delete this;                  // No users of this abstract type!
      }
      return;
    }
  }
  assert(0 && "AbstractTypeUser not in user list!");
}


// refineAbstractTypeTo - This function is used to when it is discovered that
// the 'this' abstract type is actually equivalent to the NewType specified.
// This causes all users of 'this' to switch to reference the more concrete
// type NewType and for 'this' to be deleted.
//
void DerivedType::refineAbstractTypeTo(const Type *NewType) {
  assert(isAbstract() && "refineAbstractTypeTo: Current type is not abstract!");
  assert(this != NewType && "Can't refine to myself!");

#ifdef DEBUG_MERGE_TYPES
  cerr << "REFINING abstract type [" << (void*)this << " " << getDescription()
       << "] to [" << (void*)NewType << " " << NewType->getDescription()
       << "]!\n";
#endif


  // Make sure to put the type to be refined to into a holder so that if IT gets
  // refined, that we will not continue using a dead reference...
  //
  PATypeHolder<Type> NewTy(NewType);

  // Add a self use of the current type so that we don't delete ourself until
  // after this while loop.  We are careful to never invoke refine on ourself,
  // so this extra reference shouldn't be a problem.  Note that we must only
  // remove a single reference at the end, but we must tolerate multiple self
  // references because we could be refineAbstractTypeTo'ing recursively on the
  // same type.
  //
  addAbstractTypeUser(this);

  // Count the number of self uses.  Stop looping when sizeof(list) == NSU.
  unsigned NumSelfUses = 0;

  // Iterate over all of the uses of this type, invoking callback.  Each user
  // should remove itself from our use list automatically.
  //
  while (AbstractTypeUsers.size() > NumSelfUses) {
    AbstractTypeUser *User = AbstractTypeUsers.back();

    if (User == this) {
      // Move self use to the start of the list.  Increment NSU.
      swap(AbstractTypeUsers.back(), AbstractTypeUsers[NumSelfUses++]);
    } else {
      unsigned OldSize = AbstractTypeUsers.size();
#ifdef DEBUG_MERGE_TYPES
      cerr << " REFINING user " << OldSize-1 << " of abstract type ["
	   << (void*)this << " " << getDescription() << "] to [" 
	   << (void*)NewTy.get() << " " << NewTy->getDescription() << "]!\n";
#endif
      User->refineAbstractType(this, NewTy);

      if (AbstractTypeUsers.size() == OldSize) {
        User->refineAbstractType(this, NewTy);
      }
      assert(AbstractTypeUsers.size() != OldSize &&
	     "AbsTyUser did not remove self from user list!");
    }
  }

  // Remove a single self use, even though there may be several here. This will
  // probably 'delete this', so no instance variables may be used after this
  // occurs...
  assert(AbstractTypeUsers.back() == this && "Only self uses should be left!");
  removeAbstractTypeUser(this);
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
  cerr << "typeIsREFINED type: " << (void*)this <<" "<<getDescription() << endl;
#endif
  for (unsigned i = 0; i < AbstractTypeUsers.size(); ) {
    AbstractTypeUser *ATU = AbstractTypeUsers[i];
#ifdef DEBUG_MERGE_TYPES
    cerr << " typeIsREFINED user " << i << " of abstract type ["
	 << (void*)this << " " << getDescription() << "]\n";
#endif
    ATU->refineAbstractType(this, this);
    
    // If the user didn't remove itself from the list, continue...
    if (AbstractTypeUsers.size() > i && AbstractTypeUsers[i] == ATU) {
      ++i;
    }
  }

  --isRefining;

#ifndef _NDEBUG
  if (!(isAbstract() || AbstractTypeUsers.empty()))
    for (unsigned i = 0; i < AbstractTypeUsers.size(); ++i) {
      if (AbstractTypeUsers[i] != this) {
        // Debugging hook
        cerr << "FOUND FAILURE\n";
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
void MethodType::refineAbstractType(const DerivedType *OldType,
				    const Type *NewType) {
#ifdef DEBUG_MERGE_TYPES
  cerr << "MethodTy::refineAbstractTy(" << (void*)OldType << "[" 
       << OldType->getDescription() << "], " << (void*)NewType << " [" 
       << NewType->getDescription() << "])\n";
#endif

  if (!OldType->isAbstract()) {
    if (ResultType == OldType) ResultType.removeUserFromConcrete();
    for (unsigned i = 0; i < ParamTys.size(); ++i)
      if (ParamTys[i] == OldType) ParamTys[i].removeUserFromConcrete();
  }

  if (OldType != NewType) {
    if (ResultType == OldType) ResultType = NewType;

    for (unsigned i = 0; i < ParamTys.size(); ++i)
      if (ParamTys[i] == OldType) ParamTys[i] = NewType;
  }

  const MethodType *MT = MethodTypes.containsEquivalent(this);
  if (MT && MT != this) {
    refineAbstractTypeTo(MT);            // Different type altogether...
  } else {
    setDerivedTypeProperties();          // Update the name and isAbstract
    typeIsRefined();                     // Same type, different contents...
  }
}


// refineAbstractType - Called when a contained type is found to be more
// concrete - this could potentially change us from an abstract type to a
// concrete type.
//
void ArrayType::refineAbstractType(const DerivedType *OldType,
				   const Type *NewType) {
#ifdef DEBUG_MERGE_TYPES
  cerr << "ArrayTy::refineAbstractTy(" << (void*)OldType << "[" 
       << OldType->getDescription() << "], " << (void*)NewType << " [" 
       << NewType->getDescription() << "])\n";
#endif

  if (!OldType->isAbstract()) {
    assert(ElementType == OldType);
    ElementType.removeUserFromConcrete();
  }

  ElementType = NewType;
  const ArrayType *AT = ArrayTypes.containsEquivalent(this);
  if (AT && AT != this) {
    refineAbstractTypeTo(AT);          // Different type altogether...
  } else {
    setDerivedTypeProperties();        // Update the name and isAbstract
    typeIsRefined();                   // Same type, different contents...
  }
}


// refineAbstractType - Called when a contained type is found to be more
// concrete - this could potentially change us from an abstract type to a
// concrete type.
//
void StructType::refineAbstractType(const DerivedType *OldType,
				    const Type *NewType) {
#ifdef DEBUG_MERGE_TYPES
  cerr << "StructTy::refineAbstractTy(" << (void*)OldType << "[" 
       << OldType->getDescription() << "], " << (void*)NewType << " [" 
       << NewType->getDescription() << "])\n";
#endif
  if (!OldType->isAbstract()) {
    for (unsigned i = 0; i < ETypes.size(); ++i)
      if (ETypes[i] == OldType)
        ETypes[i].removeUserFromConcrete();
  }

  if (OldType != NewType) {
    // Update old type to new type in the array...
    for (unsigned i = 0; i < ETypes.size(); ++i)
      if (ETypes[i] == OldType)
        ETypes[i] = NewType;
  } 

  const StructType *ST = StructTypes.containsEquivalent(this);
  if (ST && ST != this) {
    refineAbstractTypeTo(ST);          // Different type altogether...
  } else {
    setDerivedTypeProperties();        // Update the name and isAbstract
    typeIsRefined();                   // Same type, different contents...
  }
}

// refineAbstractType - Called when a contained type is found to be more
// concrete - this could potentially change us from an abstract type to a
// concrete type.
//
void PointerType::refineAbstractType(const DerivedType *OldType,
				     const Type *NewType) {
#ifdef DEBUG_MERGE_TYPES
  cerr << "PointerTy::refineAbstractTy(" << (void*)OldType << "[" 
       << OldType->getDescription() << "], " << (void*)NewType << " [" 
       << NewType->getDescription() << "])\n";
#endif

  if (!OldType->isAbstract()) {
    assert(ValueType == OldType);
    ValueType.removeUserFromConcrete();
  }

  ValueType = NewType;
  const PointerType *PT = PointerTypes.containsEquivalent(this);

  if (PT && PT != this) {
    refineAbstractTypeTo(PT);          // Different type altogether...
  } else {
    setDerivedTypeProperties();        // Update the name and isAbstract
    typeIsRefined();                   // Same type, different contents...
  }
}

