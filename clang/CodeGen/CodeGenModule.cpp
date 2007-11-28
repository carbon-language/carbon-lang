//===--- CodeGenModule.cpp - Emit LLVM Code from ASTs for a Module --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This coordinates the per-module state used while generating code.
//
//===----------------------------------------------------------------------===//

#include "CodeGenModule.h"
#include "CodeGenFunction.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/Intrinsics.h"
using namespace clang;
using namespace CodeGen;


CodeGenModule::CodeGenModule(ASTContext &C, const LangOptions &LO,
                             llvm::Module &M, const llvm::TargetData &TD)
  : Context(C), Features(LO), TheModule(M), TheTargetData(TD),
    Types(C, M, TD), MemCpyFn(0), CFConstantStringClassRef(0) {}

llvm::Constant *CodeGenModule::GetAddrOfGlobalDecl(const ValueDecl *D) {
  // See if it is already in the map.
  llvm::Constant *&Entry = GlobalDeclMap[D];
  if (Entry) return Entry;
  
  QualType ASTTy = cast<ValueDecl>(D)->getType();
  const llvm::Type *Ty = getTypes().ConvertType(ASTTy);
  if (isa<FunctionDecl>(D)) {
    const llvm::FunctionType *FTy = cast<llvm::FunctionType>(Ty);
    // FIXME: param attributes for sext/zext etc.
    return Entry = new llvm::Function(FTy, llvm::Function::ExternalLinkage,
                                      D->getName(), &getModule());
  }
  
  assert(isa<FileVarDecl>(D) && "Unknown global decl!");
  
  return Entry = new llvm::GlobalVariable(Ty, false, 
                                          llvm::GlobalValue::ExternalLinkage,
                                          0, D->getName(), &getModule());
}

void CodeGenModule::EmitFunction(const FunctionDecl *FD) {
  // If this is not a prototype, emit the body.
  if (FD->getBody())
    CodeGenFunction(*this).GenerateCode(FD);
}

static llvm::Constant *GenerateConstantExpr(const Expr *Expression, 
                                            CodeGenModule& CGModule);

/// GenerateConversionToBool - Generate comparison to zero for conversion to 
/// bool
static llvm::Constant *GenerateConversionToBool(llvm::Constant *Expression, 
                                            QualType Source) {
  if (Source->isRealFloatingType()) {
    // Compare against 0.0 for fp scalars.
    llvm::Constant *Zero = llvm::Constant::getNullValue(Expression->getType());
    return llvm::ConstantExpr::getFCmp(llvm::FCmpInst::FCMP_UNE, Expression, 
                                       Zero);
  }

  assert((Source->isIntegerType() || Source->isPointerType()) &&
         "Unknown scalar type to convert");

  // Compare against an integer or pointer null.
  llvm::Constant *Zero = llvm::Constant::getNullValue(Expression->getType());
  return llvm::ConstantExpr::getICmp(llvm::ICmpInst::ICMP_NE, Expression, Zero);
}

/// GenerateConstantCast - Generates a constant cast to convert the Expression
/// into the Target type.
static llvm::Constant *GenerateConstantCast(const Expr *Expression, 
                                                QualType Target, 
                                                CodeGenModule& CGModule) {
  CodeGenTypes& Types = CGModule.getTypes(); 
  QualType Source = Expression->getType().getCanonicalType();
  Target = Target.getCanonicalType();

  assert (!Target->isVoidType());

  llvm::Constant *SubExpr = GenerateConstantExpr(Expression, CGModule);

  if (Source == Target)
      return SubExpr;

  // Handle conversions to bool first, they are special: comparisons against 0.
  if (Target->isBooleanType())
    return GenerateConversionToBool(SubExpr, Source);
    
  const llvm::Type *SourceType = Types.ConvertType(Source);
  const llvm::Type *TargetType = Types.ConvertType(Target);

  // Ignore conversions like int -> uint.
  if (SubExpr->getType() == TargetType)
    return SubExpr;

  // Handle pointer conversions next: pointers can only be converted to/from
  // other pointers and integers.
  if (isa<llvm::PointerType>(TargetType)) {
    // The source value may be an integer, or a pointer.
    if (isa<llvm::PointerType>(SubExpr->getType()))
      return llvm::ConstantExpr::getBitCast(SubExpr, TargetType);
    assert(Source->isIntegerType() && "Not ptr->ptr or int->ptr conversion?");
    return llvm::ConstantExpr::getIntToPtr(SubExpr, TargetType);
  }

  if (isa<llvm::PointerType>(SourceType)) {
    // Must be an ptr to int cast.
    assert(isa<llvm::IntegerType>(TargetType) && "not ptr->int?");
    return llvm::ConstantExpr::getPtrToInt(SubExpr, TargetType);
  }

  if (Source->isRealFloatingType() && Target->isRealFloatingType()) {
    return llvm::ConstantExpr::getFPCast(SubExpr, TargetType);
  }

  // Finally, we have the arithmetic types: real int/float.
  if (isa<llvm::IntegerType>(SourceType)) {
    bool InputSigned = Source->isSignedIntegerType();
    if (isa<llvm::IntegerType>(TargetType))
      return llvm::ConstantExpr::getIntegerCast(SubExpr, TargetType, 
                                                InputSigned);
    else if (InputSigned)
      return llvm::ConstantExpr::getSIToFP(SubExpr, TargetType);
    else
      return llvm::ConstantExpr::getUIToFP(SubExpr, TargetType);
  }

  assert(SubExpr->getType()->isFloatingPoint() && "Unknown real conversion");
  if (isa<llvm::IntegerType>(TargetType)) {
    if (Target->isSignedIntegerType())
      return llvm::ConstantExpr::getFPToSI(SubExpr, TargetType);
    else
      return llvm::ConstantExpr::getFPToUI(SubExpr, TargetType);
  }

  assert(TargetType->isFloatingPoint() && "Unknown real conversion");
  if (TargetType->getTypeID() < SubExpr->getType()->getTypeID())
    return llvm::ConstantExpr::getFPTrunc(SubExpr, TargetType);
  else
    return llvm::ConstantExpr::getFPExtend(SubExpr, TargetType);

  assert (!"Unsupported cast type in global intialiser.");
  return 0;
}

/// GenerateAggregateInit - Generate a Constant initaliser for global array or
/// struct typed variables.
static llvm::Constant *GenerateAggregateInit(const InitListExpr *ILE, 
                                             CodeGenModule& CGModule) {
  assert (ILE->getType()->isArrayType() || ILE->getType()->isStructureType());
  CodeGenTypes& Types = CGModule.getTypes();

  unsigned NumInitElements = ILE->getNumInits();

  const llvm::CompositeType *CType = 
    cast<llvm::CompositeType>(Types.ConvertType(ILE->getType()));
  assert(CType);
  std::vector<llvm::Constant*> Elts;    
    
  // Copy initializer elements.
  unsigned i = 0;
  for (i = 0; i < NumInitElements; ++i) {
    llvm::Constant *C = GenerateConstantExpr(ILE->getInit(i), CGModule);
    assert (C && "Failed to create initialiser expression");
    Elts.push_back(C);
  }

  if (ILE->getType()->isStructureType())
    return llvm::ConstantStruct::get(cast<llvm::StructType>(CType), Elts);
    
  // Initialising an array requires us to automatically initialise any 
  // elements that have not been initialised explicitly
  const llvm::ArrayType *AType = cast<llvm::ArrayType>(CType);
  assert(AType);
  const llvm::Type *AElemTy = AType->getElementType();
  unsigned NumArrayElements = AType->getNumElements();
  // Initialize remaining array elements.
  for (; i < NumArrayElements; ++i)
    Elts.push_back(llvm::Constant::getNullValue(AElemTy));

  return llvm::ConstantArray::get(AType, Elts);    
}

/// GenerateConstantExpr - Recursively builds a constant initialiser for the
/// given expression.
static llvm::Constant *GenerateConstantExpr(const Expr* Expression, 
                                            CodeGenModule& CGModule) {
  CodeGenTypes& Types = CGModule.getTypes(); 
  ASTContext& Context = CGModule.getContext();
  assert ((Expression->isConstantExpr(Context, 0) ||
           Expression->getStmtClass() == Stmt::InitListExprClass) &&
          "Only constant global initialisers are supported.");

  QualType type = Expression->getType().getCanonicalType();

  if (type->isIntegerType()) {
    llvm::APSInt
      Value(static_cast<uint32_t>(Context.getTypeSize(type, SourceLocation())));
    if (Expression->isIntegerConstantExpr(Value, Context)) {
      return llvm::ConstantInt::get(Value);
    } 
  }

  switch (Expression->getStmtClass()) {
  // Generate constant for floating point literal values.
  case Stmt::FloatingLiteralClass: {
    const FloatingLiteral *FLiteral = cast<FloatingLiteral>(Expression);
    return llvm::ConstantFP::get(Types.ConvertType(type), FLiteral->getValue());
  }

  // Generate constant for string literal values.
  case Stmt::StringLiteralClass: {
    const StringLiteral *SLiteral = cast<StringLiteral>(Expression);
    const char *StrData = SLiteral->getStrData();
    unsigned Len = SLiteral->getByteLength();
    return CGModule.GetAddrOfConstantString(std::string(StrData, 
                                                        StrData + Len));
  }

  // Elide parenthesis.
  case Stmt::ParenExprClass:
    return GenerateConstantExpr(cast<ParenExpr>(Expression)->getSubExpr(),
                                CGModule);
        
  // Generate constant for sizeof operator.
  // FIXME: Need to support AlignOf
  case Stmt::SizeOfAlignOfTypeExprClass: {
    const SizeOfAlignOfTypeExpr *SOExpr = 
      cast<SizeOfAlignOfTypeExpr>(Expression);
    assert (SOExpr->isSizeOf());
    return llvm::ConstantExpr::getSizeOf(Types.ConvertType(type));
  }

  // Generate constant cast expressions.
  case Stmt::CastExprClass:
    return GenerateConstantCast(cast<CastExpr>(Expression)->getSubExpr(), type,
                                CGModule);

  case Stmt::ImplicitCastExprClass: {
    const ImplicitCastExpr *ICExpr = cast<ImplicitCastExpr>(Expression);
    return GenerateConstantCast(ICExpr->getSubExpr(), type, CGModule);
  }

  // Generate a constant array access expression
  // FIXME: Clang's semantic analysis incorrectly prevents array access in 
  // global initialisers, preventing us from testing this.
  case Stmt::ArraySubscriptExprClass: {
    const ArraySubscriptExpr* ASExpr = cast<ArraySubscriptExpr>(Expression);
    llvm::Constant *Base = GenerateConstantExpr(ASExpr->getBase(), CGModule);
    llvm::Constant *Index = GenerateConstantExpr(ASExpr->getIdx(), CGModule);
    return llvm::ConstantExpr::getExtractElement(Base, Index);
  }

  // Generate a constant expression to initialise an aggregate type, such as 
  // an array or struct.
  case Stmt::InitListExprClass: 
    return GenerateAggregateInit(cast<InitListExpr>(Expression), CGModule);

  default:
    assert (!"Unsupported expression in global initialiser.");
  }
  return 0;
}

llvm::Constant *CodeGenModule::EmitGlobalInit(const FileVarDecl *D,
                                              llvm::GlobalVariable *GV) {
  return GenerateConstantExpr(D->getInit(), *this);
}

void CodeGenModule::EmitGlobalVar(const FileVarDecl *D) {
  llvm::GlobalVariable *GV = cast<llvm::GlobalVariable>(GetAddrOfGlobalDecl(D));
  
  // If the storage class is external and there is no initializer, just leave it
  // as a declaration.
  if (D->getStorageClass() == VarDecl::Extern && D->getInit() == 0)
    return;

  // Otherwise, convert the initializer, or use zero if appropriate.
  llvm::Constant *Init = 0;
  if (D->getInit() == 0) {
    Init = llvm::Constant::getNullValue(GV->getType()->getElementType());
  } else if (D->getType()->isIntegerType()) {
    llvm::APSInt Value(static_cast<uint32_t>(
      getContext().getTypeSize(D->getInit()->getType(), SourceLocation())));
    if (D->getInit()->isIntegerConstantExpr(Value, Context))
      Init = llvm::ConstantInt::get(Value);
  }

  if (!Init)
    Init = EmitGlobalInit(D, GV);

  assert(Init && "FIXME: Global variable initializers unimp!");

  GV->setInitializer(Init);
  
  // Set the llvm linkage type as appropriate.
  // FIXME: This isn't right.  This should handle common linkage and other
  // stuff.
  switch (D->getStorageClass()) {
  case VarDecl::Auto:
  case VarDecl::Register:
    assert(0 && "Can't have auto or register globals");
  case VarDecl::None:
  case VarDecl::Extern:
    // todo: common
    break;
  case VarDecl::Static:
    GV->setLinkage(llvm::GlobalVariable::InternalLinkage);
    break;
  }
}

/// EmitGlobalVarDeclarator - Emit all the global vars attached to the specified
/// declarator chain.
void CodeGenModule::EmitGlobalVarDeclarator(const FileVarDecl *D) {
  for (; D; D = cast_or_null<FileVarDecl>(D->getNextDeclarator()))
    EmitGlobalVar(D);
}

/// getBuiltinLibFunction
llvm::Function *CodeGenModule::getBuiltinLibFunction(unsigned BuiltinID) {
  if (BuiltinFunctions.size() <= BuiltinID)
    BuiltinFunctions.resize(BuiltinID);
  
  // Already available?
  llvm::Function *&FunctionSlot = BuiltinFunctions[BuiltinID];
  if (FunctionSlot)
    return FunctionSlot;
  
  assert(Context.BuiltinInfo.isLibFunction(BuiltinID) && "isn't a lib fn");
  
  // Get the name, skip over the __builtin_ prefix.
  const char *Name = Context.BuiltinInfo.GetName(BuiltinID)+10;
  
  // Get the type for the builtin.
  QualType Type = Context.BuiltinInfo.GetBuiltinType(BuiltinID, Context);
  const llvm::FunctionType *Ty = 
    cast<llvm::FunctionType>(getTypes().ConvertType(Type));

  // FIXME: This has a serious problem with code like this:
  //  void abs() {}
  //    ... __builtin_abs(x);
  // The two versions of abs will collide.  The fix is for the builtin to win,
  // and for the existing one to be turned into a constantexpr cast of the
  // builtin.  In the case where the existing one is a static function, it
  // should just be renamed.
  if (llvm::Function *Existing = getModule().getFunction(Name)) {
    if (Existing->getFunctionType() == Ty && Existing->hasExternalLinkage())
      return FunctionSlot = Existing;
    assert(Existing == 0 && "FIXME: Name collision");
  }

  // FIXME: param attributes for sext/zext etc.
  return FunctionSlot = new llvm::Function(Ty, llvm::Function::ExternalLinkage,
                                           Name, &getModule());
}


llvm::Function *CodeGenModule::getMemCpyFn() {
  if (MemCpyFn) return MemCpyFn;
  llvm::Intrinsic::ID IID;
  uint64_t Size; unsigned Align;
  Context.Target.getPointerInfo(Size, Align, SourceLocation());
  switch (Size) {
  default: assert(0 && "Unknown ptr width");
  case 32: IID = llvm::Intrinsic::memcpy_i32; break;
  case 64: IID = llvm::Intrinsic::memcpy_i64; break;
  }
  return MemCpyFn = llvm::Intrinsic::getDeclaration(&TheModule, IID);
}

llvm::Constant *CodeGenModule::
GetAddrOfConstantCFString(const std::string &str) {
  llvm::StringMapEntry<llvm::Constant *> &Entry = 
    CFConstantStringMap.GetOrCreateValue(&str[0], &str[str.length()]);
  
  if (Entry.getValue())
    return Entry.getValue();
  
  std::vector<llvm::Constant*> Fields;
  
  if (!CFConstantStringClassRef) {
    const llvm::Type *Ty = getTypes().ConvertType(getContext().IntTy);
    Ty = llvm::ArrayType::get(Ty, 0);
  
    CFConstantStringClassRef = 
      new llvm::GlobalVariable(Ty, false,
                               llvm::GlobalVariable::ExternalLinkage, 0, 
                               "__CFConstantStringClassReference", 
                               &getModule());
  }
  
  // Class pointer.
  llvm::Constant *Zero = llvm::Constant::getNullValue(llvm::Type::Int32Ty);
  llvm::Constant *Zeros[] = { Zero, Zero };
  llvm::Constant *C = 
    llvm::ConstantExpr::getGetElementPtr(CFConstantStringClassRef, Zeros, 2);
  Fields.push_back(C);
  
  // Flags.
  const llvm::Type *Ty = getTypes().ConvertType(getContext().IntTy);
  Fields.push_back(llvm::ConstantInt::get(Ty, 1992));
    
  // String pointer.
  C = llvm::ConstantArray::get(str);
  C = new llvm::GlobalVariable(C->getType(), true, 
                               llvm::GlobalValue::InternalLinkage,
                               C, ".str", &getModule());
  
  C = llvm::ConstantExpr::getGetElementPtr(C, Zeros, 2);
  Fields.push_back(C);
  
  // String length.
  Ty = getTypes().ConvertType(getContext().LongTy);
  Fields.push_back(llvm::ConstantInt::get(Ty, str.length()));
  
  // The struct.
  Ty = getTypes().ConvertType(getContext().getCFConstantStringType());
  C = llvm::ConstantStruct::get(cast<llvm::StructType>(Ty), Fields);
  llvm::GlobalVariable *GV = 
    new llvm::GlobalVariable(C->getType(), true, 
                             llvm::GlobalVariable::InternalLinkage, 
                             C, "", &getModule());
  GV->setSection("__DATA,__cfstring");
  Entry.setValue(GV);
  return GV;
}

/// GenerateWritableString -- Creates storage for a string literal
static llvm::Constant *GenerateStringLiteral(const std::string &str, 
                                             bool constant,
                                             CodeGenModule& CGModule) {
  // Create Constant for this string literal
  llvm::Constant *C=llvm::ConstantArray::get(str);
  
  // Create a global variable for this string
  C = new llvm::GlobalVariable(C->getType(), constant, 
                               llvm::GlobalValue::InternalLinkage,
                               C, ".str", &CGModule.getModule());
  llvm::Constant *Zero = llvm::Constant::getNullValue(llvm::Type::Int32Ty);
  llvm::Constant *Zeros[] = { Zero, Zero };
  C = llvm::ConstantExpr::getGetElementPtr(C, Zeros, 2);
  return C;
}

/// CodeGenModule::GetAddrOfConstantString -- returns a pointer to the first 
/// element of a character array containing the literal.
llvm::Constant *CodeGenModule::GetAddrOfConstantString(const std::string &str) {
  // Don't share any string literals if writable-strings is turned on.
  if (Features.WritableStrings)
    return GenerateStringLiteral(str, false, *this);
  
  llvm::StringMapEntry<llvm::Constant *> &Entry = 
  ConstantStringMap.GetOrCreateValue(&str[0], &str[str.length()]);

  if (Entry.getValue())
      return Entry.getValue();

  // Create a global variable for this.
  llvm::Constant *C = GenerateStringLiteral(str, true, *this);
  Entry.setValue(C);
  return C;
}
