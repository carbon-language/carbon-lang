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
#include "clang/Basic/TargetInfo.h"
#include "clang/AST/AST.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/Target/TargetData.h"

using namespace clang;
using namespace CodeGen;

namespace {
  /// RecordOrganizer - This helper class, used by CGRecordLayout, layouts 
  /// structs and unions. It manages transient information used during layout.
  /// FIXME : At the moment assume 
  ///    - one to one mapping between AST FieldDecls and 
  ///      llvm::StructType elements.
  ///    - Ignore bit fields
  ///    - Ignore field aligments
  ///    - Ignore packed structs
  class RecordOrganizer {
  public:
    explicit RecordOrganizer(CodeGenTypes &Types) : 
      CGT(Types), STy(NULL), llvmFieldNo(0), Cursor(0),
      llvmSize(0) {}
    
    /// addField - Add new field.
    void addField(const FieldDecl *FD);

    /// addLLVMField - Add llvm struct field that corresponds to llvm type Ty. 
    /// Increment field count.
    void addLLVMField(const llvm::Type *Ty);

    /// addPaddingFields - Current cursor is not suitable place to add next 
    /// field. Add required padding fields.
    void addPaddingFields(unsigned WaterMark);

    /// layoutStructFields - Do the actual work and lay out all fields. Create
    /// corresponding llvm struct type.  This should be invoked only after
    /// all fields are added.
    void layoutStructFields(const ASTRecordLayout &RL);

    /// layoutUnionFields - Do the actual work and lay out all fields. Create
    /// corresponding llvm struct type.  This should be invoked only after
    /// all fields are added.
    void layoutUnionFields();

    /// getLLVMType - Return associated llvm struct type. This may be NULL
    /// if fields are not laid out.
    llvm::Type *getLLVMType() const {
      return STy;
    }

    /// placeBitField - Find a place for FD, which is a bit-field. 
    void placeBitField(const FieldDecl *FD);

  private:
    CodeGenTypes &CGT;
    llvm::Type *STy;
    unsigned llvmFieldNo;
    uint64_t Cursor; 
    uint64_t llvmSize;
    llvm::SmallVector<const FieldDecl *, 8> FieldDecls;
    std::vector<const llvm::Type*> LLVMFields;
    llvm::SmallVector<uint64_t, 8> Offsets;
  };
}

CodeGenTypes::CodeGenTypes(ASTContext &Ctx, llvm::Module& M,
                           const llvm::TargetData &TD)
  : Context(Ctx), Target(Ctx.Target), TheModule(M), TheTargetData(TD) {
}

CodeGenTypes::~CodeGenTypes() {
  for(llvm::DenseMap<const llvm::Type *, CGRecordLayout *>::iterator
        I = CGRecordLayouts.begin(), E = CGRecordLayouts.end();
      I != E; ++I)
    delete I->second;
  CGRecordLayouts.clear();
}

/// isOpaqueTypeDefinition - Return true if LT is a llvm::OpaqueType
/// and T is tag definition. This helper routine does not check
/// relationship between T and LT.
static bool isOpaqueTypeDefinition(QualType T, llvm::Type *LT) {
  
  if (!isa<llvm::OpaqueType>(LT))
    return false;

  const clang::Type &Ty = *T.getCanonicalType();
  if (Ty.getTypeClass() == Type::Tagged) {
    const TagType &TT = cast<TagType>(Ty);
    const TagDecl *TD = TT.getDecl();
    if (TD->isDefinition())
      return true;
  }

  return false;
}

/// ConvertType - Convert the specified type to its LLVM form.
const llvm::Type *CodeGenTypes::ConvertType(QualType T) {
  // See if type is already cached.
  llvm::DenseMap<Type *, llvm::PATypeHolder>::iterator
    I = TypeHolderMap.find(T.getTypePtr());
  // If type is found in map and this is not a definition for a opaque
  // place holder type then use it. Otherwise convert type T.
  if (I != TypeHolderMap.end() && !isOpaqueTypeDefinition(T, I->second.get()))
    return I->second.get();

  const llvm::Type *ResultType = ConvertNewType(T);
  TypeHolderMap.insert(std::make_pair(T.getTypePtr(), 
                                      llvm::PATypeHolder(ResultType)));
  return ResultType;
}

/// ConvertTypeForMem - Convert type T into a llvm::Type. Maintain and use
/// type cache through TypeHolderMap.  This differs from ConvertType in that
/// it is used to convert to the memory representation for a type.  For
/// example, the scalar representation for _Bool is i1, but the memory
/// representation is usually i8 or i32, depending on the target.
const llvm::Type *CodeGenTypes::ConvertTypeForMem(QualType T) {
  const llvm::Type *R = ConvertType(T);
  
  // If this is a non-bool type, don't map it.
  if (R != llvm::Type::Int1Ty)
    return R;
    
  // Otherwise, return an integer of the target-specified size.
  unsigned BoolWidth = (unsigned)Context.getTypeSize(T, SourceLocation());
  return llvm::IntegerType::get(BoolWidth);
  
}


const llvm::Type *CodeGenTypes::ConvertNewType(QualType T) {
  const clang::Type &Ty = *T.getCanonicalType();
  
  switch (Ty.getTypeClass()) {
  case Type::TypeName:        // typedef isn't canonical.
  case Type::TypeOfExp:       // typeof isn't canonical.
  case Type::TypeOfTyp:       // typeof isn't canonical.
    assert(0 && "Non-canonical type, shouldn't happen");
  case Type::Builtin: {
    switch (cast<BuiltinType>(Ty).getKind()) {
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
      return llvm::IntegerType::get(
        static_cast<unsigned>(Context.getTypeSize(T, SourceLocation())));
      
    case BuiltinType::Float:      return llvm::Type::FloatTy;
    case BuiltinType::Double:     return llvm::Type::DoubleTy;
    case BuiltinType::LongDouble:
      // FIXME: mapping long double onto double.
      return llvm::Type::DoubleTy;
    }
    break;
  }
  case Type::Complex: {
    std::vector<const llvm::Type*> Elts;
    Elts.push_back(ConvertType(cast<ComplexType>(Ty).getElementType()));
    Elts.push_back(Elts[0]);
    return llvm::StructType::get(Elts);
  }
  case Type::Pointer: {
    const PointerType &P = cast<PointerType>(Ty);
    return llvm::PointerType::getUnqual(ConvertType(P.getPointeeType())); 
  }
  case Type::Reference: {
    const ReferenceType &R = cast<ReferenceType>(Ty);
    return llvm::PointerType::getUnqual(ConvertType(R.getReferenceeType()));
  }
    
  case Type::VariableArray: {
    const VariableArrayType &A = cast<VariableArrayType>(Ty);
    assert(A.getSizeModifier() == ArrayType::Normal &&
           A.getIndexTypeQualifier() == 0 &&
           "FIXME: We only handle trivial array types so far!");
    if (A.getSizeExpr() == 0) {
      // int X[] -> [0 x int]
      return llvm::ArrayType::get(ConvertType(A.getElementType()), 0);
    } else {
      assert(0 && "FIXME: VLAs not implemented yet!");
    }
  }
  case Type::ConstantArray: {
    const ConstantArrayType &A = cast<ConstantArrayType>(Ty);
    const llvm::Type *EltTy = ConvertType(A.getElementType());
    return llvm::ArrayType::get(EltTy, A.getSize().getZExtValue());
  }
  case Type::OCUVector:
  case Type::Vector: {
    const VectorType &VT = cast<VectorType>(Ty);
    return llvm::VectorType::get(ConvertType(VT.getElementType()),
                                 VT.getNumElements());
  }
  case Type::FunctionNoProto:
  case Type::FunctionProto: {
    const FunctionType &FP = cast<FunctionType>(Ty);
    const llvm::Type *ResultType;
    
    if (FP.getResultType()->isVoidType())
      ResultType = llvm::Type::VoidTy;    // Result of function uses llvm void.
    else
      ResultType = ConvertType(FP.getResultType());
    
    // FIXME: Convert argument types.
    bool isVarArg;
    std::vector<const llvm::Type*> ArgTys;
    
    // Struct return passes the struct byref.
    if (!ResultType->isFirstClassType() && ResultType != llvm::Type::VoidTy) {
      const llvm::Type *RType = llvm::PointerType::getUnqual(ResultType);
      QualType RTy = Context.getPointerType(FP.getResultType());
      TypeHolderMap.insert(std::make_pair(RTy.getTypePtr(), 
                                          llvm::PATypeHolder(RType)));
  
      ArgTys.push_back(RType);
      ResultType = llvm::Type::VoidTy;
    }
    
    if (const FunctionTypeProto *FTP = dyn_cast<FunctionTypeProto>(&FP)) {
      DecodeArgumentTypes(*FTP, ArgTys);
      isVarArg = FTP->isVariadic();
    } else {
      isVarArg = true;
    }
    
    return llvm::FunctionType::get(ResultType, ArgTys, isVarArg);
  }

  case Type::ObjCInterface:
    assert(0 && "FIXME: add missing functionality here");
    break;
      
  case Type::ObjCQualifiedInterface:
    assert(0 && "FIXME: add missing functionality here");
    break;

  case Type::ObjCQualifiedId:
    assert(0 && "FIXME: add missing functionality here");
    break;

  case Type::Tagged:
    const TagType &TT = cast<TagType>(Ty);
    const TagDecl *TD = TT.getDecl();
    llvm::Type *&ResultType = TagDeclTypes[TD];
      
    // If corresponding llvm type is not a opaque struct type
    // then use it.
    if (ResultType && !isOpaqueTypeDefinition(T, ResultType))
      return ResultType;
    
    if (!TD->isDefinition()) {
      ResultType = llvm::OpaqueType::get();  
    } else if (TD->getKind() == Decl::Enum) {
      return ConvertType(cast<EnumDecl>(TD)->getIntegerType());
    } else if (TD->getKind() == Decl::Struct) {
      const RecordDecl *RD = cast<const RecordDecl>(TD);
      
      // If this is nested record and this RecordDecl is already under
      // process then return associated OpaqueType for now.
      llvm::DenseMap<const RecordDecl *, llvm::Type *>::iterator 
        OpaqueI = RecordTypesToResolve.find(RD);
      if (OpaqueI != RecordTypesToResolve.end())
        return OpaqueI->second;

      llvm::OpaqueType *OpaqueTy = NULL;
      if (ResultType)
        OpaqueTy = dyn_cast<llvm::OpaqueType>(ResultType);
      if (!OpaqueTy) {
        // Create new OpaqueType now for later use.
        // FIXME: This creates a lot of opaque types, most of them are not 
        // needed. Reevaluate this when performance analyis finds tons of 
        // opaque types.
        OpaqueTy = llvm::OpaqueType::get();
        TypeHolderMap.insert(std::make_pair(T.getTypePtr(), 
                                            llvm::PATypeHolder(OpaqueTy)));
      }
      RecordTypesToResolve[RD] = OpaqueTy;

      // Layout fields.
      RecordOrganizer RO(*this);
      for (unsigned i = 0, e = RD->getNumMembers(); i != e; ++i)
        RO.addField(RD->getMember(i));
      const ASTRecordLayout &RL = Context.getASTRecordLayout(RD, 
                                                             SourceLocation());
      RO.layoutStructFields(RL);

      // Get llvm::StructType.
      CGRecordLayout *RLI = new CGRecordLayout(RO.getLLVMType());
      ResultType = RLI->getLLVMType();
      CGRecordLayouts[ResultType] = RLI;

      // Refine any OpaqueType associated with this RecordDecl.
      OpaqueTy->refineAbstractTypeTo(ResultType);
      OpaqueI = RecordTypesToResolve.find(RD);
      assert (OpaqueI != RecordTypesToResolve.end() 
              && "Expected RecordDecl in RecordTypesToResolve");
      RecordTypesToResolve.erase(OpaqueI);

    } else if (TD->getKind() == Decl::Union) {
      const RecordDecl *RD = cast<const RecordDecl>(TD);
      // Just use the largest element of the union, breaking ties with the
      // highest aligned member.

      if (RD->getNumMembers() != 0) {
        RecordOrganizer RO(*this);
        for (unsigned i = 0, e = RD->getNumMembers(); i != e; ++i)
          RO.addField(RD->getMember(i));
        RO.layoutUnionFields();

        // Get llvm::StructType.
        CGRecordLayout *RLI = new CGRecordLayout(RO.getLLVMType());
        ResultType = RLI->getLLVMType();
        CGRecordLayouts[ResultType] = RLI;
      } else {       
        std::vector<const llvm::Type*> Fields;
        ResultType = llvm::StructType::get(Fields);
      }
    } else {
      assert(0 && "FIXME: Implement tag decl kind!");
    }
          
    std::string TypeName(TD->getKindName());
    TypeName += '.';
    
    // Name the codegen type after the typedef name
    // if there is no tag type name available
    if (TD->getIdentifier() == 0) {
      if (T->getTypeClass() == Type::TypeName) {
        const TypedefType *TdT = cast<TypedefType>(T);
        TypeName += TdT->getDecl()->getName();
      } else
        TypeName += "anon";
    } else 
      TypeName += TD->getName();
          
    TheModule.addTypeName(TypeName, ResultType);  
    return ResultType;
  }
  
  // FIXME: implement.
  return llvm::OpaqueType::get();
}

void CodeGenTypes::DecodeArgumentTypes(const FunctionTypeProto &FTP, 
                                       std::vector<const llvm::Type*> &ArgTys) {
  for (unsigned i = 0, e = FTP.getNumArgs(); i != e; ++i) {
    const llvm::Type *Ty = ConvertType(FTP.getArgType(i));
    if (Ty->isFirstClassType())
      ArgTys.push_back(Ty);
    else {
      QualType PTy = Context.getPointerType(FTP.getArgType(i));
      const llvm::Type *PtrTy = llvm::PointerType::getUnqual(Ty);
      TypeHolderMap.insert(std::make_pair(PTy.getTypePtr(), 
                                          llvm::PATypeHolder(PtrTy)));

      ArgTys.push_back(PtrTy);
    }
  }
}

/// getLLVMFieldNo - Return llvm::StructType element number
/// that corresponds to the field FD.
unsigned CodeGenTypes::getLLVMFieldNo(const FieldDecl *FD) {
  llvm::DenseMap<const FieldDecl *, unsigned>::iterator
    I = FieldInfo.find(FD);
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
CodeGenTypes::getCGRecordLayout(const llvm::Type* Ty) const {
  llvm::DenseMap<const llvm::Type*, CGRecordLayout *>::iterator I
    = CGRecordLayouts.find(Ty);
  assert (I != CGRecordLayouts.end() 
          && "Unable to find record layout information for type");
  return I->second;
}

/// addField - Add new field.
void RecordOrganizer::addField(const FieldDecl *FD) {
  assert (!STy && "Record fields are already laid out");
  FieldDecls.push_back(FD);
}

/// layoutStructFields - Do the actual work and lay out all fields. Create
/// corresponding llvm struct type.  This should be invoked only after
/// all fields are added.
/// FIXME : At the moment assume 
///    - one to one mapping between AST FieldDecls and 
///      llvm::StructType elements.
///    - Ignore bit fields
///    - Ignore field aligments
///    - Ignore packed structs
void RecordOrganizer::layoutStructFields(const ASTRecordLayout &RL) {
  // FIXME : Use SmallVector
  llvmSize = 0;
  llvmFieldNo = 0;
  Cursor = 0;
  LLVMFields.clear();
  Offsets.clear();

  for (llvm::SmallVector<const FieldDecl *, 8>::iterator I = FieldDecls.begin(),
         E = FieldDecls.end(); I != E; ++I) {
    const FieldDecl *FD = *I;

    if (FD->isBitField()) 
      placeBitField(FD);
    else {
      const llvm::Type *Ty = CGT.ConvertType(FD->getType());
      addLLVMField(Ty);
      CGT.addFieldInfo(FD, llvmFieldNo - 1);
      Cursor = llvmSize;
    }
  }

  unsigned StructAlign = RL.getAlignment();
  if (llvmSize % StructAlign) {
    unsigned StructPadding = StructAlign - (llvmSize % StructAlign);
    addPaddingFields(llvmSize + StructPadding);
  }

  STy = llvm::StructType::get(LLVMFields);
}

/// addPaddingFields - Current cursor is not suitable place to add next field.
/// Add required padding fields.
void RecordOrganizer::addPaddingFields(unsigned WaterMark) {
  unsigned RequiredBits = WaterMark - llvmSize;
  unsigned RequiredBytes = (RequiredBits + 7) / 8;
  for (unsigned i = 0; i != RequiredBytes; ++i)
    addLLVMField(llvm::Type::Int8Ty);
}

/// addLLVMField - Add llvm struct field that corresponds to llvm type Ty.
/// Increment field count.
void RecordOrganizer::addLLVMField(const llvm::Type *Ty) {

  unsigned AlignmentInBits = CGT.getTargetData().getABITypeAlignment(Ty) * 8;
  if (llvmSize % AlignmentInBits) {
    // At the moment, insert padding fields even if target specific llvm 
    // type alignment enforces implict padding fields for FD. Later on, 
    // optimize llvm fields by removing implicit padding fields and 
    // combining consequetive padding fields.
    unsigned Padding = AlignmentInBits - (llvmSize % AlignmentInBits);
    addPaddingFields(llvmSize + Padding);
  }

  unsigned TySize = CGT.getTargetData().getABITypeSizeInBits(Ty);
  Offsets.push_back(llvmSize);
  llvmSize += TySize;
  LLVMFields.push_back(Ty);
  ++llvmFieldNo;
}

/// layoutUnionFields - Do the actual work and lay out all fields. Create
/// corresponding llvm struct type.  This should be invoked only after
/// all fields are added.
void RecordOrganizer::layoutUnionFields() {
 
  unsigned PrimaryEltNo = 0;
  std::pair<uint64_t, unsigned> PrimaryElt =
    CGT.getContext().getTypeInfo(FieldDecls[0]->getType(), SourceLocation());
  CGT.addFieldInfo(FieldDecls[0], 0);

  unsigned Size = FieldDecls.size();
  for(unsigned i = 1; i != Size; ++i) {
    const FieldDecl *FD = FieldDecls[i];
    assert (!FD->isBitField() && "Bit fields are not yet supported");
    std::pair<uint64_t, unsigned> EltInfo = 
      CGT.getContext().getTypeInfo(FD->getType(), SourceLocation());

    // Use largest element, breaking ties with the hightest aligned member.
    if (EltInfo.first > PrimaryElt.first ||
        (EltInfo.first == PrimaryElt.first &&
         EltInfo.second > PrimaryElt.second)) {
      PrimaryElt = EltInfo;
      PrimaryEltNo = i;
    }

    // In union, each field gets first slot.
    CGT.addFieldInfo(FD, 0);
  }

  std::vector<const llvm::Type*> Fields;
  const llvm::Type *Ty = CGT.ConvertType(FieldDecls[PrimaryEltNo]->getType());
  Fields.push_back(Ty);
  STy = llvm::StructType::get(Fields);
}

/// placeBitField - Find a place for FD, which is a bit-field.
/// This function searches for the last aligned field. If the  bit-field fits in
/// it, it is reused. Otherwise, the bit-field is placed in a new field.
void RecordOrganizer::placeBitField(const FieldDecl *FD) {

  assert (FD->isBitField() && "FD is not a bit-field");
  Expr *BitWidth = FD->getBitWidth();
  llvm::APSInt FieldSize(32);
  bool isBitField = 
    BitWidth->isIntegerConstantExpr(FieldSize, CGT.getContext());
  assert (isBitField  && "Invalid BitField size expression");
  uint64_t BitFieldSize =  FieldSize.getZExtValue();

  bool FoundPrevField = false;
  unsigned TotalOffsets = Offsets.size();
  const llvm::Type *Ty = CGT.ConvertType(FD->getType());
  uint64_t TySize = CGT.getTargetData().getABITypeSizeInBits(Ty);
  
  if (!TotalOffsets) {
    // Special case: the first field. 
    CGT.addFieldInfo(FD, llvmFieldNo);
    CGT.addBitFieldInfo(FD, 0, BitFieldSize);
    addPaddingFields(BitFieldSize);
    Cursor = BitFieldSize;
    return;
  }

  // Search for the last aligned field.
  for (unsigned i = TotalOffsets; i != 0; --i) {
    uint64_t O = Offsets[i - 1];
    if (O % TySize == 0) {
      FoundPrevField = true;
      if (TySize - (Cursor - O) >= BitFieldSize) {
	// The bitfield fits in the last aligned field.
	// This is : struct { char a; int CurrentField:10;};
	// where 'CurrentField' shares first field with 'a'.
	addPaddingFields(Cursor + BitFieldSize);
	CGT.addFieldInfo(FD, i - 1);
	CGT.addBitFieldInfo(FD, Cursor - O, BitFieldSize);
	Cursor += BitFieldSize;
      } else {
	// Place the bitfield in a new LLVM field.
	// This is : struct { char a; short CurrentField:10;};
	// where 'CurrentField' needs a new llvm field.
	addPaddingFields(O + TySize);
	CGT.addFieldInfo(FD, llvmFieldNo);
	CGT.addBitFieldInfo(FD, 0, BitFieldSize);
	addPaddingFields(O + TySize +  BitFieldSize);
	Cursor = O + TySize +  BitFieldSize;
      }
      break;
    }
  }

  assert(FoundPrevField && 
	 "Unable to find a place for bitfield in struct layout");
}
