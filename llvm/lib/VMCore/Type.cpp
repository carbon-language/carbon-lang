//===-- Type.cpp - Implement the Type class ----------------------*- C++ -*--=//
//
// This file implements the Type class for the VMCore library.
//
//===----------------------------------------------------------------------===//

#include "llvm/DerivedTypes.h"
#include "llvm/Support/StringExtras.h"
#include "llvm/CodeGen/TargetMachine.h"

//===----------------------------------------------------------------------===//
//                         Type Class Implementation
//===----------------------------------------------------------------------===//

static unsigned CurUID = 0;
static vector<const Type *> UIDMappings;

Type::Type(const string &name, PrimitiveID id) 
  : Value(Type::TypeTy, Value::TypeVal, name) {
  ID = id;
  ConstRulesImpl = 0;

  UID = CurUID++;       // Assign types UID's as they are created
  UIDMappings.push_back(this);
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
  case LockTyID  : return LockTy;
  case FillerTyID: return new Type("XXX FILLER XXX", FillerTyID); // TODO:KILLME
  default:
    return 0;
  }
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

const Type *Type::VoidTy   = new            Type("void"  , VoidTyID),
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
           *Type::LabelTy  = new            Type("label" , LabelTyID),
           *Type::LockTy   = new            Type("lock"  , LockTyID);


//===----------------------------------------------------------------------===//
//                      Derived Type Implementations
//===----------------------------------------------------------------------===//

// Make sure that only one instance of a particular type may be created on any
// given run of the compiler...
//
// TODO: This list should be kept in sorted order so that we can do a binary
// TODO: search instead of linear search!
//
// TODO: This should be templatized so that every derived type can use the same
// TODO: code!
//
#define TEST_MERGE_TYPES 0

#if TEST_MERGE_TYPES
#include "llvm/Assembly/Writer.h"
#endif

#undef RESURRECT_OLD_LAYOUT_CODE
#ifdef RESURRECT_OLD_LAYOUT_CODE

unsigned int
StructType::getStorageSize(const TargetMachine& tmi) const
{
  if (layoutCache->targetInfo && layoutCache->targetInfo != &tmi)
    {// target machine has changed (hey it could happen). discard cached info.
      ResetCachedInfo();
      layoutCache->targetInfo = &tmi;
    }
  
  if (layoutCache->storageSize < 0) {
    layoutCache->storageSize = tmi.findOptimalStorageSize(this);
    assert(layoutCache->storageSize >= 0);
  }
  
  return layoutCache->storageSize;
}

unsigned int
StructType::getElementOffset(int i, const TargetMachine& tmi) const
{
  // target machine has changed (hey it could happen). discard cached info.
  if (layoutCache->targetInfo && layoutCache->targetInfo != &tmi)
    ResetCachedInfo();
  
  if (layoutCache->memberOffsets[i] < 0) {
    layoutCache->targetInfo = &tmi;  // remember which target was used
    
    unsigned int *offsetVec = tmi.findOptimalMemberOffsets(this);
    for (unsigned i=0, N=layoutCache->memberOffsets.size(); i < N; ++i) {
      layoutCache->memberOffsets[i] = offsetVec[i];
      assert(layoutCache->memberOffsets[i] >= 0);
    }
    delete[] offsetVec; 
  }
  
  return layoutCache->memberOffsets[i];
}

#endif

//===----------------------------------------------------------------------===//
//                          Derived Type Constructors
//===----------------------------------------------------------------------===//

MethodType::MethodType(const Type *Result, const vector<const Type*> &Params, 
                       bool IsVarArgs, const string &Name) 
  : Type(Name, MethodTyID), ResultType(Result), 
    ParamTys(Params.begin(), Params.end()-IsVarArgs), 
    isVarArgs(IsVarArgs) {
}

ArrayType::ArrayType(const Type *ElType, int NumEl, const string &Name) 
  : Type(Name, ArrayTyID), ElementType(ElType) {
  NumElements = NumEl;
}

StructType::StructType(const vector<const Type*> &Types, const string &Name) 
  : Type(Name, StructTyID), ETypes(Types) {
}

PointerType::PointerType(const Type *E) 
  : Type(E->getName() + " *", PointerTyID), ValueType(E) {
}

//===----------------------------------------------------------------------===//
//                         Derived Type Creator Functions
//===----------------------------------------------------------------------===//

const MethodType *MethodType::getMethodType(const Type *ReturnType, 
                                            const vector<const Type*> &Params) {
  static vector<const MethodType*> ExistingMethodTypesCache;

  bool IsVarArg = Params.size() && (Params[Params.size()-1] == Type::VoidTy);

  for (unsigned i = 0; i < ExistingMethodTypesCache.size(); ++i) {
    const MethodType *T = ExistingMethodTypesCache[i];
    if (T->getReturnType() == ReturnType && T->isVarArg() == IsVarArg) {
      const ParamTypes &EParams = T->getParamTypes();
      ParamTypes::const_iterator I = Params.begin(), EI = Params.end()-IsVarArg;
      ParamTypes::const_iterator J = EParams.begin(); 
      for (; I != EI && J != EParams.end(); ++I, ++J)
        if (*I != *J) break;  // These types aren't equal!

      if (I == EI && J == EParams.end()) {
#if TEST_MERGE_TYPES == 2
        ostream_iterator<const Type*> out(cerr, ", ");
        cerr << "Type: \"";
        copy(Params.begin(), EI, out);
        cerr << "\"\nEquals: \"";
        copy(EParams.begin(), EParams.end(), out);
        cerr << "\"" << endl;
#endif
        return T;
      }
    }
  }  
#if TEST_MERGE_TYPES == 2
  ostream_iterator<const Type*> out(cerr, ", ");
  cerr << "Input Types: ";
  copy(Params.begin(), Params.end()-IsVarArg, out);
  cerr << endl;
#endif

  // Calculate the string name for the new type...
  string Name = ReturnType->getName() + " (";
  for (ParamTypes::const_iterator I = Params.begin();  
       I != (Params.end()-IsVarArg); ++I) {
    if (I != Params.begin())
      Name += ", ";
    Name += (*I)->getName();
  }
  if (IsVarArg) {
    if (Params.size() > 1) Name += ", ";
    Name += "...";
  }
  Name += ")";

#if TEST_MERGE_TYPES
  cerr << "Derived new type: " << Name << endl;
#endif

  MethodType *Result = new MethodType(ReturnType, Params, IsVarArg, Name);
  ExistingMethodTypesCache.push_back(Result);
  return Result;
}


const ArrayType *ArrayType::getArrayType(const Type *ElementType, 
					 int NumElements = -1) {
  assert(ElementType && "Can't get array of null types!");
  static vector<const ArrayType*> ExistingTypesCache;

  // Search cache for value...
  for (unsigned i = 0; i < ExistingTypesCache.size(); ++i) {
    const ArrayType *T = ExistingTypesCache[i];

    if (T->getElementType() == ElementType && 
	T->getNumElements() == NumElements)
      return T;
  }

  // Value not found.  Derive a new type!
  string Name = "[";
  if (NumElements != -1) Name += itostr(NumElements) + " x ";

  Name += ElementType->getName();
  
  ArrayType *Result = new ArrayType(ElementType, NumElements, Name + "]");
  ExistingTypesCache.push_back(Result);

#if TEST_MERGE_TYPES
  cerr << "Derived new type: " << Result->getName() << endl;
#endif
  return Result;
}

const StructType *StructType::getStructType(const ElementTypes &ETypes) {
  static vector<const StructType*> ExistingStructTypesCache;

  for (unsigned i = 0; i < ExistingStructTypesCache.size(); ++i) {
    const StructType *T = ExistingStructTypesCache[i];

    const ElementTypes &Elements = T->getElementTypes();
    ElementTypes::const_iterator I = ETypes.begin(); 
    ElementTypes::const_iterator J = Elements.begin(); 
    for (; I != ETypes.end() && J != Elements.end(); ++I, ++J)
      if (*I != *J) break;  // These types aren't equal!
    
    if (I == ETypes.end() && J == Elements.end()) {
#if TEST_MERGE_TYPES == 2
      ostream_iterator<const Type*> out(cerr, ", ");
      cerr << "Type: \"";
      copy(ETypes.begin(), ETypes.end(), out);
      cerr << "\"\nEquals: \"";
      copy(Elements.begin(), Elements.end(), out);
      cerr << "\"" << endl;
#endif
      return T;
    }
  }

#if TEST_MERGE_TYPES == 2
  ostream_iterator<const Type*> out(cerr, ", ");
  cerr << "Input Types: ";
  copy(ETypes.begin(), ETypes.end(), out);
  cerr << endl;
#endif

  // Calculate the string name for the new type...
  string Name = "{ ";
  for (ElementTypes::const_iterator I = ETypes.begin();  
       I != ETypes.end(); ++I) {
    if (I != ETypes.begin())
      Name += ", ";
    Name += (*I)->getName();
  }
  Name += " }";

#if TEST_MERGE_TYPES
  cerr << "Derived new type: " << Name << endl;
#endif

  StructType *Result = new StructType(ETypes, Name);
  ExistingStructTypesCache.push_back(Result);
  return Result;
}


const PointerType *PointerType::getPointerType(const Type *ValueType) {
  assert(ValueType && "Can't get a pointer to <null> type!");
  static vector<const PointerType*> ExistingTypesCache;

  // Search cache for value...
  for (unsigned i = 0; i < ExistingTypesCache.size(); ++i) {
    const PointerType *T = ExistingTypesCache[i];

    if (T->getValueType() == ValueType)
      return T;
  }

  PointerType *Result = new PointerType(ValueType);
  ExistingTypesCache.push_back(Result);

#if TEST_MERGE_TYPES
  cerr << "Derived new type: " << Result->getName() << endl;
#endif
  return Result;
}

