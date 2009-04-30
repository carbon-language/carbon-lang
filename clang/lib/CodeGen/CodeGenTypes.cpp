//===--- CodeGenTypes.cpp - Type translation for LLVM CodeGen -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is the code that handles AST -> LLVM type lowering. 
//
//===----------------------------------------------------------------------===//

#include "CodeGenTypes.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/Expr.h"
#include "clang/AST/RecordLayout.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/Target/TargetData.h"

#include "CGCall.h"

using namespace clang;
using namespace CodeGen;

namespace {
  /// RecordOrganizer - This helper class, used by CGRecordLayout, layouts 
  /// structs and unions. It manages transient information used during layout.
  /// FIXME : Handle field aligments. Handle packed structs.
  class RecordOrganizer {
  public:
    explicit RecordOrganizer(CodeGenTypes &Types, const RecordDecl& Record) : 
      CGT(Types), RD(Record), STy(NULL) {}

    /// layoutStructFields - Do the actual work and lay out all fields. Create
    /// corresponding llvm struct type.  This should be invoked only after
    /// all fields are added.
    void layoutStructFields(const ASTRecordLayout &RL);

    /// layoutUnionFields - Do the actual work and lay out all fields. Create
    /// corresponding llvm struct type.  This should be invoked only after
    /// all fields are added.
    void layoutUnionFields(const ASTRecordLayout &RL);

    /// getLLVMType - Return associated llvm struct type. This may be NULL
    /// if fields are not laid out.
    llvm::Type *getLLVMType() const {
      return STy;
    }

    llvm::SmallSet<unsigned, 8> &getPaddingFields() {
      return PaddingFields;
    }

  private:
    CodeGenTypes &CGT;
    const RecordDecl& RD;
    llvm::Type *STy;
    llvm::SmallSet<unsigned, 8> PaddingFields;
  };
}

CodeGenTypes::CodeGenTypes(ASTContext &Ctx, llvm::Module& M,
                           const llvm::TargetData &TD)
  : Context(Ctx), Target(Ctx.Target), TheModule(M), TheTargetData(TD),
    TheABIInfo(0) {
}

CodeGenTypes::~CodeGenTypes() {
  for(llvm::DenseMap<const Type *, CGRecordLayout *>::iterator
        I = CGRecordLayouts.begin(), E = CGRecordLayouts.end();
      I != E; ++I)
    delete I->second;
  CGRecordLayouts.clear();
}

/// ConvertType - Convert the specified type to its LLVM form.
const llvm::Type *CodeGenTypes::ConvertType(QualType T) {
  llvm::PATypeHolder Result = ConvertTypeRecursive(T);

  // Any pointers that were converted defered evaluation of their pointee type,
  // creating an opaque type instead.  This is in order to avoid problems with
  // circular types.  Loop through all these defered pointees, if any, and
  // resolve them now.
  while (!PointersToResolve.empty()) {
    std::pair<QualType, llvm::OpaqueType*> P =
      PointersToResolve.back();
    PointersToResolve.pop_back();
    // We can handle bare pointers here because we know that the only pointers
    // to the Opaque type are P.second and from other types.  Refining the
    // opqaue type away will invalidate P.second, but we don't mind :).
    const llvm::Type *NT = ConvertTypeForMemRecursive(P.first);
    P.second->refineAbstractTypeTo(NT);
  }

  return Result;
}

const llvm::Type *CodeGenTypes::ConvertTypeRecursive(QualType T) {
  T = Context.getCanonicalType(T);
  
  // See if type is already cached.
  llvm::DenseMap<Type *, llvm::PATypeHolder>::iterator
    I = TypeCache.find(T.getTypePtr());
  // If type is found in map and this is not a definition for a opaque
  // place holder type then use it. Otherwise, convert type T.
  if (I != TypeCache.end())
    return I->second.get();

  const llvm::Type *ResultType = ConvertNewType(T);
  TypeCache.insert(std::make_pair(T.getTypePtr(), 
                                  llvm::PATypeHolder(ResultType)));
  return ResultType;
}

const llvm::Type *CodeGenTypes::ConvertTypeForMemRecursive(QualType T) {
  const llvm::Type *ResultType = ConvertTypeRecursive(T);
  if (ResultType == llvm::Type::Int1Ty)
    return llvm::IntegerType::get((unsigned)Context.getTypeSize(T));
  return ResultType;
}

/// ConvertTypeForMem - Convert type T into a llvm::Type.  This differs from
/// ConvertType in that it is used to convert to the memory representation for
/// a type.  For example, the scalar representation for _Bool is i1, but the
/// memory representation is usually i8 or i32, depending on the target.
const llvm::Type *CodeGenTypes::ConvertTypeForMem(QualType T) {
  const llvm::Type *R = ConvertType(T);
  
  // If this is a non-bool type, don't map it.
  if (R != llvm::Type::Int1Ty)
    return R;
    
  // Otherwise, return an integer of the target-specified size.
  return llvm::IntegerType::get((unsigned)Context.getTypeSize(T));
  
}

// Code to verify a given function type is complete, i.e. the return type
// and all of the argument types are complete.
static const TagType *VerifyFuncTypeComplete(const Type* T) {
  const FunctionType *FT = cast<FunctionType>(T);
  if (const TagType* TT = FT->getResultType()->getAsTagType())
    if (!TT->getDecl()->isDefinition())
      return TT;
  if (const FunctionProtoType *FPT = dyn_cast<FunctionProtoType>(T))
    for (unsigned i = 0; i < FPT->getNumArgs(); i++)
      if (const TagType* TT = FPT->getArgType(i)->getAsTagType())
        if (!TT->getDecl()->isDefinition())
          return TT;
  return 0;
}

/// UpdateCompletedType - When we find the full definition for a TagDecl,
/// replace the 'opaque' type we previously made for it if applicable.
void CodeGenTypes::UpdateCompletedType(const TagDecl *TD) {
  const Type *Key = 
    Context.getTagDeclType(const_cast<TagDecl*>(TD)).getTypePtr();
  llvm::DenseMap<const Type*, llvm::PATypeHolder>::iterator TDTI = 
    TagDeclTypes.find(Key);
  if (TDTI == TagDeclTypes.end()) return;
  
  // Remember the opaque LLVM type for this tagdecl.
  llvm::PATypeHolder OpaqueHolder = TDTI->second;
  assert(isa<llvm::OpaqueType>(OpaqueHolder.get()) &&
         "Updating compilation of an already non-opaque type?");
  
  // Remove it from TagDeclTypes so that it will be regenerated.
  TagDeclTypes.erase(TDTI);

  // Generate the new type.
  const llvm::Type *NT = ConvertTagDeclType(TD);

  // Refine the old opaque type to its new definition.
  cast<llvm::OpaqueType>(OpaqueHolder.get())->refineAbstractTypeTo(NT);

  // Since we just completed a tag type, check to see if any function types
  // were completed along with the tag type.
  // FIXME: This is very inefficient; if we track which function types depend
  // on which tag types, though, it should be reasonably efficient.
  llvm::DenseMap<const Type*, llvm::PATypeHolder>::iterator i;
  for (i = FunctionTypes.begin(); i != FunctionTypes.end(); ++i) {
    if (const TagType* TT = VerifyFuncTypeComplete(i->first)) {
      // This function type still depends on an incomplete tag type; make sure
      // that tag type has an associated opaque type.
      ConvertTagDeclType(TT->getDecl());
    } else {
      // This function no longer depends on an incomplete tag type; create the
      // function type, and refine the opaque type to the new function type.
      llvm::PATypeHolder OpaqueHolder = i->second;
      const llvm::Type *NFT = ConvertNewType(QualType(i->first, 0));
      cast<llvm::OpaqueType>(OpaqueHolder.get())->refineAbstractTypeTo(NFT);
      FunctionTypes.erase(i);
    }
  }
}

static const llvm::Type* getTypeForFormat(const llvm::fltSemantics &format) {
  if (&format == &llvm::APFloat::IEEEsingle)
    return llvm::Type::FloatTy;
  if (&format == &llvm::APFloat::IEEEdouble)
    return llvm::Type::DoubleTy;
  if (&format == &llvm::APFloat::IEEEquad)
    return llvm::Type::FP128Ty;
  if (&format == &llvm::APFloat::PPCDoubleDouble)
    return llvm::Type::PPC_FP128Ty;
  if (&format == &llvm::APFloat::x87DoubleExtended)
    return llvm::Type::X86_FP80Ty;
  assert(0 && "Unknown float format!");
  return 0;
}

const llvm::Type *CodeGenTypes::ConvertNewType(QualType T) {
  const clang::Type &Ty = *Context.getCanonicalType(T);
  
  switch (Ty.getTypeClass()) {
#define TYPE(Class, Base)
#define ABSTRACT_TYPE(Class, Base)
#define NON_CANONICAL_TYPE(Class, Base) case Type::Class:
#define DEPENDENT_TYPE(Class, Base) case Type::Class:
#include "clang/AST/TypeNodes.def"
    assert(false && "Non-canonical or dependent types aren't possible.");
    break;

  case Type::Builtin: {
    switch (cast<BuiltinType>(Ty).getKind()) {
    default: assert(0 && "Unknown builtin type!");
    case BuiltinType::Void:
      // LLVM void type can only be used as the result of a function call.  Just
      // map to the same as char.
      return llvm::IntegerType::get(8);

    case BuiltinType::Bool:
      // Note that we always return bool as i1 for use as a scalar type.
      return llvm::Type::Int1Ty;
      
    case BuiltinType::Char_S:
    case BuiltinType::Char_U:
    case BuiltinType::SChar:
    case BuiltinType::UChar:
    case BuiltinType::Short:
    case BuiltinType::UShort:
    case BuiltinType::Int:
    case BuiltinType::UInt:
    case BuiltinType::Long:
    case BuiltinType::ULong:
    case BuiltinType::LongLong:
    case BuiltinType::ULongLong:
    case BuiltinType::WChar:
      return llvm::IntegerType::get(
        static_cast<unsigned>(Context.getTypeSize(T)));
      
    case BuiltinType::Float:
    case BuiltinType::Double:
    case BuiltinType::LongDouble:
      return getTypeForFormat(Context.getFloatTypeSemantics(T));
          
    case BuiltinType::UInt128:
    case BuiltinType::Int128:
      return llvm::IntegerType::get(128);
    }
    break;
  }
  case Type::FixedWidthInt:
    return llvm::IntegerType::get(cast<FixedWidthIntType>(T)->getWidth());
  case Type::Complex: {
    const llvm::Type *EltTy = 
      ConvertTypeRecursive(cast<ComplexType>(Ty).getElementType());
    return llvm::StructType::get(EltTy, EltTy, NULL);
  }
  case Type::LValueReference:
  case Type::RValueReference: {
    const ReferenceType &RTy = cast<ReferenceType>(Ty);
    QualType ETy = RTy.getPointeeType();
    llvm::OpaqueType *PointeeType = llvm::OpaqueType::get();
    PointersToResolve.push_back(std::make_pair(ETy, PointeeType));
    return llvm::PointerType::get(PointeeType, ETy.getAddressSpace());
  }
  case Type::Pointer: {
    const PointerType &PTy = cast<PointerType>(Ty);
    QualType ETy = PTy.getPointeeType();
    llvm::OpaqueType *PointeeType = llvm::OpaqueType::get();
    PointersToResolve.push_back(std::make_pair(ETy, PointeeType));
    return llvm::PointerType::get(PointeeType, ETy.getAddressSpace());
  }
    
  case Type::VariableArray: {
    const VariableArrayType &A = cast<VariableArrayType>(Ty);
    assert(A.getIndexTypeQualifier() == 0 &&
           "FIXME: We only handle trivial array types so far!");
    // VLAs resolve to the innermost element type; this matches
    // the return of alloca, and there isn't any obviously better choice.
    return ConvertTypeForMemRecursive(A.getElementType());
  }
  case Type::IncompleteArray: {
    const IncompleteArrayType &A = cast<IncompleteArrayType>(Ty);
    assert(A.getIndexTypeQualifier() == 0 &&
           "FIXME: We only handle trivial array types so far!");
    // int X[] -> [0 x int]
    return llvm::ArrayType::get(ConvertTypeForMemRecursive(A.getElementType()), 0);
  }
  case Type::ConstantArray: {
    const ConstantArrayType &A = cast<ConstantArrayType>(Ty);
    const llvm::Type *EltTy = ConvertTypeForMemRecursive(A.getElementType());
    return llvm::ArrayType::get(EltTy, A.getSize().getZExtValue());
  }
  case Type::ExtVector:
  case Type::Vector: {
    const VectorType &VT = cast<VectorType>(Ty);
    return llvm::VectorType::get(ConvertTypeRecursive(VT.getElementType()),
                                 VT.getNumElements());
  }
  case Type::FunctionNoProto:
  case Type::FunctionProto: {
    // First, check whether we can build the full function type.
    if (const TagType* TT = VerifyFuncTypeComplete(&Ty)) {
      // This function's type depends on an incomplete tag type; make sure
      // we have an opaque type corresponding to the tag type.
      ConvertTagDeclType(TT->getDecl());
      // Create an opaque type for this function type, save it, and return it.
      llvm::Type *ResultType = llvm::OpaqueType::get();
      FunctionTypes.insert(std::make_pair(&Ty, ResultType));
      return ResultType;
    }
    // The function type can be built; call the appropriate routines to
    // build it.
    if (const FunctionProtoType *FPT = dyn_cast<FunctionProtoType>(&Ty))
      return GetFunctionType(getFunctionInfo(FPT), FPT->isVariadic());

    const FunctionNoProtoType *FNPT = cast<FunctionNoProtoType>(&Ty);
    return GetFunctionType(getFunctionInfo(FNPT), true);
  }
  
  case Type::ExtQual:
    return
      ConvertTypeRecursive(QualType(cast<ExtQualType>(Ty).getBaseType(), 0));

  case Type::ObjCQualifiedInterface: {
    // Lower foo<P1,P2> just like foo.
    ObjCInterfaceDecl *ID = cast<ObjCQualifiedInterfaceType>(Ty).getDecl();
    return ConvertTypeRecursive(Context.getObjCInterfaceType(ID));
  }
      
  case Type::ObjCInterface: {
    // Objective-C interfaces are always opaque (outside of the
    // runtime, which can do whatever it likes); we never refine
    // these.
    const llvm::Type *&T = InterfaceTypes[cast<ObjCInterfaceType>(&Ty)];
    if (!T)
        T = llvm::OpaqueType::get();
    return T;
  }
      
  case Type::ObjCQualifiedId:
    // Protocols don't influence the LLVM type.
    return ConvertTypeRecursive(Context.getObjCIdType());

  case Type::Record:
  case Type::Enum: {
    const TagDecl *TD = cast<TagType>(Ty).getDecl();
    const llvm::Type *Res = ConvertTagDeclType(TD);
    
    std::string TypeName(TD->getKindName());
    TypeName += '.';
    
    // Name the codegen type after the typedef name
    // if there is no tag type name available
    if (TD->getIdentifier())
      TypeName += TD->getNameAsString();
    else if (const TypedefType *TdT = dyn_cast<TypedefType>(T))
      TypeName += TdT->getDecl()->getNameAsString();
    else
      TypeName += "anon";
    
    TheModule.addTypeName(TypeName, Res);  
    return Res;
  }

  case Type::BlockPointer: {
    const QualType FTy = cast<BlockPointerType>(Ty).getPointeeType();
    llvm::OpaqueType *PointeeType = llvm::OpaqueType::get();
    PointersToResolve.push_back(std::make_pair(FTy, PointeeType));
    return llvm::PointerType::get(PointeeType, FTy.getAddressSpace());
  }

  case Type::MemberPointer:
    // FIXME: Implement C++ pointer-to-member. The GCC representation is
    // documented here:
    // http://gcc.gnu.org/onlinedocs/gccint/Type-Layout.html#Type-Layout
    assert(0 && "FIXME: We can't handle member pointers yet.");
    return llvm::OpaqueType::get();

  case Type::TemplateSpecialization:
    assert(false && "Dependent types can't get here");
  }
  
  // FIXME: implement.
  return llvm::OpaqueType::get();
}

/// ConvertTagDeclType - Lay out a tagged decl type like struct or union or
/// enum.
const llvm::Type *CodeGenTypes::ConvertTagDeclType(const TagDecl *TD) {
  // TagDecl's are not necessarily unique, instead use the (clang)
  // type connected to the decl.
  const Type *Key = 
    Context.getTagDeclType(const_cast<TagDecl*>(TD)).getTypePtr();
  llvm::DenseMap<const Type*, llvm::PATypeHolder>::iterator TDTI = 
    TagDeclTypes.find(Key);
  
  // If we've already compiled this tag type, use the previous definition.
  if (TDTI != TagDeclTypes.end())
    return TDTI->second;
  
  // If this is still a forward definition, just define an opaque type to use
  // for this tagged decl.
  if (!TD->isDefinition()) {
    llvm::Type *ResultType = llvm::OpaqueType::get();  
    TagDeclTypes.insert(std::make_pair(Key, ResultType));
    return ResultType;
  }
  
  // Okay, this is a definition of a type.  Compile the implementation now.
  
  if (TD->isEnum()) {
    // Don't bother storing enums in TagDeclTypes.
    return ConvertTypeRecursive(cast<EnumDecl>(TD)->getIntegerType());
  }
  
  // This decl could well be recursive.  In this case, insert an opaque
  // definition of this type, which the recursive uses will get.  We will then
  // refine this opaque version later.

  // Create new OpaqueType now for later use in case this is a recursive
  // type.  This will later be refined to the actual type.
  llvm::PATypeHolder ResultHolder = llvm::OpaqueType::get();
  TagDeclTypes.insert(std::make_pair(Key, ResultHolder));
  
  const llvm::Type *ResultType;
  const RecordDecl *RD = cast<const RecordDecl>(TD);

  // There isn't any extra information for empty structures/unions.
  if (RD->field_empty(getContext())) {
    ResultType = llvm::StructType::get(std::vector<const llvm::Type*>());
  } else {
    // Layout fields.
    RecordOrganizer RO(*this, *RD);
    
    if (TD->isStruct() || TD->isClass())
      RO.layoutStructFields(Context.getASTRecordLayout(RD));
    else {
      assert(TD->isUnion() && "unknown tag decl kind!");
      RO.layoutUnionFields(Context.getASTRecordLayout(RD));
    }
    
    // Get llvm::StructType.
    const Type *Key = 
      Context.getTagDeclType(const_cast<TagDecl*>(TD)).getTypePtr();
    CGRecordLayouts[Key] = new CGRecordLayout(RO.getLLVMType(), 
                                              RO.getPaddingFields());
    ResultType = RO.getLLVMType();
  }
  
  // Refine our Opaque type to ResultType.  This can invalidate ResultType, so
  // make sure to read the result out of the holder.
  cast<llvm::OpaqueType>(ResultHolder.get())
    ->refineAbstractTypeTo(ResultType);
  
  return ResultHolder.get();
}  

/// getLLVMFieldNo - Return llvm::StructType element number
/// that corresponds to the field FD.
unsigned CodeGenTypes::getLLVMFieldNo(const FieldDecl *FD) {
  llvm::DenseMap<const FieldDecl*, unsigned>::iterator I = FieldInfo.find(FD);
  assert (I != FieldInfo.end()  && "Unable to find field info");
  return I->second;
}

/// addFieldInfo - Assign field number to field FD.
void CodeGenTypes::addFieldInfo(const FieldDecl *FD, unsigned No) {
  FieldInfo[FD] = No;
}

/// getBitFieldInfo - Return the BitFieldInfo  that corresponds to the field FD.
CodeGenTypes::BitFieldInfo CodeGenTypes::getBitFieldInfo(const FieldDecl *FD) {
  llvm::DenseMap<const FieldDecl *, BitFieldInfo>::iterator
    I = BitFields.find(FD);
  assert (I != BitFields.end()  && "Unable to find bitfield info");
  return I->second;
}

/// addBitFieldInfo - Assign a start bit and a size to field FD.
void CodeGenTypes::addBitFieldInfo(const FieldDecl *FD, unsigned Begin,
                                   unsigned Size) {
  BitFields.insert(std::make_pair(FD, BitFieldInfo(Begin, Size)));
}

/// getCGRecordLayout - Return record layout info for the given llvm::Type.
const CGRecordLayout *
CodeGenTypes::getCGRecordLayout(const TagDecl *TD) const {
  const Type *Key = 
    Context.getTagDeclType(const_cast<TagDecl*>(TD)).getTypePtr();
  llvm::DenseMap<const Type*, CGRecordLayout *>::iterator I
    = CGRecordLayouts.find(Key);
  assert (I != CGRecordLayouts.end() 
          && "Unable to find record layout information for type");
  return I->second;
}

/// layoutStructFields - Do the actual work and lay out all fields. Create
/// corresponding llvm struct type.
/// Note that this doesn't actually try to do struct layout; it depends on
/// the layout built by the AST.  (We have to do struct layout to do Sema,
/// and there's no point to duplicating the work.)
void RecordOrganizer::layoutStructFields(const ASTRecordLayout &RL) {
  // FIXME: This code currently always generates packed structures.
  // Unpacked structures are more readable, and sometimes more efficient!
  // (But note that any changes here are likely to impact CGExprConstant,
  // which makes some messy assumptions.)
  uint64_t llvmSize = 0;
  // FIXME: Make this a SmallVector
  std::vector<const llvm::Type*> LLVMFields;

  unsigned curField = 0;
  for (RecordDecl::field_iterator Field = RD.field_begin(CGT.getContext()),
                               FieldEnd = RD.field_end(CGT.getContext());
       Field != FieldEnd; ++Field) {
    uint64_t offset = RL.getFieldOffset(curField);
    const llvm::Type *Ty = CGT.ConvertTypeForMemRecursive(Field->getType());
    uint64_t size = CGT.getTargetData().getTypePaddedSizeInBits(Ty);

    if (Field->isBitField()) {
      uint64_t BitFieldSize =
          Field->getBitWidth()->EvaluateAsInt(CGT.getContext()).getZExtValue();

      // Bitfield field info is different from other field info;
      // it actually ignores the underlying LLVM struct because
      // there isn't any convenient mapping.
      CGT.addFieldInfo(*Field, offset / size);
      CGT.addBitFieldInfo(*Field, offset % size, BitFieldSize);
    } else {
      // Put the element into the struct. This would be simpler
      // if we didn't bother, but it seems a bit too strange to
      // allocate all structs as i8 arrays.
      while (llvmSize < offset) {
        LLVMFields.push_back(llvm::Type::Int8Ty);
        llvmSize += 8;
      }

      llvmSize += size;
      CGT.addFieldInfo(*Field, LLVMFields.size());
      LLVMFields.push_back(Ty);
    }
    ++curField;
  }

  while (llvmSize < RL.getSize()) {
    LLVMFields.push_back(llvm::Type::Int8Ty);
    llvmSize += 8;
  }

  STy = llvm::StructType::get(LLVMFields, true);
  assert(CGT.getTargetData().getTypePaddedSizeInBits(STy) == RL.getSize());
}

/// layoutUnionFields - Do the actual work and lay out all fields. Create
/// corresponding llvm struct type.  This should be invoked only after
/// all fields are added.
void RecordOrganizer::layoutUnionFields(const ASTRecordLayout &RL) {
  unsigned curField = 0;
  for (RecordDecl::field_iterator Field = RD.field_begin(CGT.getContext()),
                               FieldEnd = RD.field_end(CGT.getContext());
       Field != FieldEnd; ++Field) {
    // The offset should usually be zero, but bitfields could be strange
    uint64_t offset = RL.getFieldOffset(curField);
    CGT.ConvertTypeRecursive(Field->getType());

    if (Field->isBitField()) {
      Expr *BitWidth = Field->getBitWidth();
      uint64_t BitFieldSize =  
        BitWidth->EvaluateAsInt(CGT.getContext()).getZExtValue();

      CGT.addFieldInfo(*Field, 0);
      CGT.addBitFieldInfo(*Field, offset, BitFieldSize);
    } else {
      CGT.addFieldInfo(*Field, 0);
    }
    ++curField;
  }

  // This looks stupid, but it is correct in the sense that
  // it works no matter how complicated the sizes and alignments
  // of the union elements are. The natural alignment
  // of the result doesn't matter because anyone allocating
  // structures should be aligning them appropriately anyway.
  // FIXME: We can be a bit more intuitive in a lot of cases.
  // FIXME: Make this a struct type to work around PR2399; the
  // C backend doesn't like structs using array types.
  std::vector<const llvm::Type*> LLVMFields;
  LLVMFields.push_back(llvm::ArrayType::get(llvm::Type::Int8Ty,
                                            RL.getSize() / 8));
  STy = llvm::StructType::get(LLVMFields, true);
  assert(CGT.getTargetData().getTypePaddedSizeInBits(STy) == RL.getSize());
}
