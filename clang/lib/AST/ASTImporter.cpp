//===--- ASTImporter.cpp - Importing ASTs from other Contexts ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the ASTImporter class which imports AST nodes from one
//  context into another context.
//
//===----------------------------------------------------------------------===//
#include "clang/AST/ASTImporter.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTDiagnostic.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/AST/TypeVisitor.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/Support/MemoryBuffer.h"
#include <deque>

using namespace clang;

namespace {
  class ASTNodeImporter : public TypeVisitor<ASTNodeImporter, QualType>,
                          public DeclVisitor<ASTNodeImporter, Decl *>,
                          public StmtVisitor<ASTNodeImporter, Stmt *> {
    ASTImporter &Importer;
    
  public:
    explicit ASTNodeImporter(ASTImporter &Importer) : Importer(Importer) { }
    
    using TypeVisitor<ASTNodeImporter, QualType>::Visit;
    using DeclVisitor<ASTNodeImporter, Decl *>::Visit;
    using StmtVisitor<ASTNodeImporter, Stmt *>::Visit;

    // Importing types
    QualType VisitType(const Type *T);
    QualType VisitBuiltinType(const BuiltinType *T);
    QualType VisitComplexType(const ComplexType *T);
    QualType VisitPointerType(const PointerType *T);
    QualType VisitBlockPointerType(const BlockPointerType *T);
    QualType VisitLValueReferenceType(const LValueReferenceType *T);
    QualType VisitRValueReferenceType(const RValueReferenceType *T);
    QualType VisitMemberPointerType(const MemberPointerType *T);
    QualType VisitConstantArrayType(const ConstantArrayType *T);
    QualType VisitIncompleteArrayType(const IncompleteArrayType *T);
    QualType VisitVariableArrayType(const VariableArrayType *T);
    // FIXME: DependentSizedArrayType
    // FIXME: DependentSizedExtVectorType
    QualType VisitVectorType(const VectorType *T);
    QualType VisitExtVectorType(const ExtVectorType *T);
    QualType VisitFunctionNoProtoType(const FunctionNoProtoType *T);
    QualType VisitFunctionProtoType(const FunctionProtoType *T);
    // FIXME: UnresolvedUsingType
    QualType VisitTypedefType(const TypedefType *T);
    QualType VisitTypeOfExprType(const TypeOfExprType *T);
    // FIXME: DependentTypeOfExprType
    QualType VisitTypeOfType(const TypeOfType *T);
    QualType VisitDecltypeType(const DecltypeType *T);
    QualType VisitUnaryTransformType(const UnaryTransformType *T);
    QualType VisitAutoType(const AutoType *T);
    // FIXME: DependentDecltypeType
    QualType VisitRecordType(const RecordType *T);
    QualType VisitEnumType(const EnumType *T);
    // FIXME: TemplateTypeParmType
    // FIXME: SubstTemplateTypeParmType
    QualType VisitTemplateSpecializationType(const TemplateSpecializationType *T);
    QualType VisitElaboratedType(const ElaboratedType *T);
    // FIXME: DependentNameType
    // FIXME: DependentTemplateSpecializationType
    QualType VisitObjCInterfaceType(const ObjCInterfaceType *T);
    QualType VisitObjCObjectType(const ObjCObjectType *T);
    QualType VisitObjCObjectPointerType(const ObjCObjectPointerType *T);
                            
    // Importing declarations
    bool ImportDeclParts(NamedDecl *D, DeclContext *&DC, 
                         DeclContext *&LexicalDC, DeclarationName &Name, 
                         SourceLocation &Loc);
    void ImportDeclarationNameLoc(const DeclarationNameInfo &From,
                                  DeclarationNameInfo& To);
    void ImportDeclContext(DeclContext *FromDC, bool ForceImport = false);
    bool ImportDefinition(RecordDecl *From, RecordDecl *To);
    TemplateParameterList *ImportTemplateParameterList(
                                                 TemplateParameterList *Params);
    TemplateArgument ImportTemplateArgument(const TemplateArgument &From);
    bool ImportTemplateArguments(const TemplateArgument *FromArgs,
                                 unsigned NumFromArgs,
                               llvm::SmallVectorImpl<TemplateArgument> &ToArgs);
    bool IsStructuralMatch(RecordDecl *FromRecord, RecordDecl *ToRecord);
    bool IsStructuralMatch(EnumDecl *FromEnum, EnumDecl *ToRecord);
    bool IsStructuralMatch(ClassTemplateDecl *From, ClassTemplateDecl *To);
    Decl *VisitDecl(Decl *D);
    Decl *VisitNamespaceDecl(NamespaceDecl *D);
    Decl *VisitTypedefNameDecl(TypedefNameDecl *D, bool IsAlias);
    Decl *VisitTypedefDecl(TypedefDecl *D);
    Decl *VisitTypeAliasDecl(TypeAliasDecl *D);
    Decl *VisitEnumDecl(EnumDecl *D);
    Decl *VisitRecordDecl(RecordDecl *D);
    Decl *VisitEnumConstantDecl(EnumConstantDecl *D);
    Decl *VisitFunctionDecl(FunctionDecl *D);
    Decl *VisitCXXMethodDecl(CXXMethodDecl *D);
    Decl *VisitCXXConstructorDecl(CXXConstructorDecl *D);
    Decl *VisitCXXDestructorDecl(CXXDestructorDecl *D);
    Decl *VisitCXXConversionDecl(CXXConversionDecl *D);
    Decl *VisitFieldDecl(FieldDecl *D);
    Decl *VisitIndirectFieldDecl(IndirectFieldDecl *D);
    Decl *VisitObjCIvarDecl(ObjCIvarDecl *D);
    Decl *VisitVarDecl(VarDecl *D);
    Decl *VisitImplicitParamDecl(ImplicitParamDecl *D);
    Decl *VisitParmVarDecl(ParmVarDecl *D);
    Decl *VisitObjCMethodDecl(ObjCMethodDecl *D);
    Decl *VisitObjCCategoryDecl(ObjCCategoryDecl *D);
    Decl *VisitObjCProtocolDecl(ObjCProtocolDecl *D);
    Decl *VisitObjCInterfaceDecl(ObjCInterfaceDecl *D);
    Decl *VisitObjCCategoryImplDecl(ObjCCategoryImplDecl *D);
    Decl *VisitObjCImplementationDecl(ObjCImplementationDecl *D);
    Decl *VisitObjCPropertyDecl(ObjCPropertyDecl *D);
    Decl *VisitObjCPropertyImplDecl(ObjCPropertyImplDecl *D);
    Decl *VisitObjCForwardProtocolDecl(ObjCForwardProtocolDecl *D);
    Decl *VisitObjCClassDecl(ObjCClassDecl *D);
    Decl *VisitTemplateTypeParmDecl(TemplateTypeParmDecl *D);
    Decl *VisitNonTypeTemplateParmDecl(NonTypeTemplateParmDecl *D);
    Decl *VisitTemplateTemplateParmDecl(TemplateTemplateParmDecl *D);
    Decl *VisitClassTemplateDecl(ClassTemplateDecl *D);
    Decl *VisitClassTemplateSpecializationDecl(
                                            ClassTemplateSpecializationDecl *D);
                            
    // Importing statements
    Stmt *VisitStmt(Stmt *S);

    // Importing expressions
    Expr *VisitExpr(Expr *E);
    Expr *VisitDeclRefExpr(DeclRefExpr *E);
    Expr *VisitIntegerLiteral(IntegerLiteral *E);
    Expr *VisitCharacterLiteral(CharacterLiteral *E);
    Expr *VisitParenExpr(ParenExpr *E);
    Expr *VisitUnaryOperator(UnaryOperator *E);
    Expr *VisitUnaryExprOrTypeTraitExpr(UnaryExprOrTypeTraitExpr *E);
    Expr *VisitBinaryOperator(BinaryOperator *E);
    Expr *VisitCompoundAssignOperator(CompoundAssignOperator *E);
    Expr *VisitImplicitCastExpr(ImplicitCastExpr *E);
    Expr *VisitCStyleCastExpr(CStyleCastExpr *E);
  };
}

//----------------------------------------------------------------------------
// Structural Equivalence
//----------------------------------------------------------------------------

namespace {
  struct StructuralEquivalenceContext {
    /// \brief AST contexts for which we are checking structural equivalence.
    ASTContext &C1, &C2;
    
    /// \brief The set of "tentative" equivalences between two canonical 
    /// declarations, mapping from a declaration in the first context to the
    /// declaration in the second context that we believe to be equivalent.
    llvm::DenseMap<Decl *, Decl *> TentativeEquivalences;
    
    /// \brief Queue of declarations in the first context whose equivalence
    /// with a declaration in the second context still needs to be verified.
    std::deque<Decl *> DeclsToCheck;
    
    /// \brief Declaration (from, to) pairs that are known not to be equivalent
    /// (which we have already complained about).
    llvm::DenseSet<std::pair<Decl *, Decl *> > &NonEquivalentDecls;
    
    /// \brief Whether we're being strict about the spelling of types when 
    /// unifying two types.
    bool StrictTypeSpelling;
    
    StructuralEquivalenceContext(ASTContext &C1, ASTContext &C2,
               llvm::DenseSet<std::pair<Decl *, Decl *> > &NonEquivalentDecls,
                                 bool StrictTypeSpelling = false)
      : C1(C1), C2(C2), NonEquivalentDecls(NonEquivalentDecls),
        StrictTypeSpelling(StrictTypeSpelling) { }

    /// \brief Determine whether the two declarations are structurally
    /// equivalent.
    bool IsStructurallyEquivalent(Decl *D1, Decl *D2);
    
    /// \brief Determine whether the two types are structurally equivalent.
    bool IsStructurallyEquivalent(QualType T1, QualType T2);

  private:
    /// \brief Finish checking all of the structural equivalences.
    ///
    /// \returns true if an error occurred, false otherwise.
    bool Finish();
    
  public:
    DiagnosticBuilder Diag1(SourceLocation Loc, unsigned DiagID) {
      return C1.getDiagnostics().Report(Loc, DiagID);
    }

    DiagnosticBuilder Diag2(SourceLocation Loc, unsigned DiagID) {
      return C2.getDiagnostics().Report(Loc, DiagID);
    }
  };
}

static bool IsStructurallyEquivalent(StructuralEquivalenceContext &Context,
                                     QualType T1, QualType T2);
static bool IsStructurallyEquivalent(StructuralEquivalenceContext &Context,
                                     Decl *D1, Decl *D2);

/// \brief Determine if two APInts have the same value, after zero-extending
/// one of them (if needed!) to ensure that the bit-widths match.
static bool IsSameValue(const llvm::APInt &I1, const llvm::APInt &I2) {
  if (I1.getBitWidth() == I2.getBitWidth())
    return I1 == I2;
  
  if (I1.getBitWidth() > I2.getBitWidth())
    return I1 == I2.zext(I1.getBitWidth());
  
  return I1.zext(I2.getBitWidth()) == I2;
}

/// \brief Determine if two APSInts have the same value, zero- or sign-extending
/// as needed.
static bool IsSameValue(const llvm::APSInt &I1, const llvm::APSInt &I2) {
  if (I1.getBitWidth() == I2.getBitWidth() && I1.isSigned() == I2.isSigned())
    return I1 == I2;
  
  // Check for a bit-width mismatch.
  if (I1.getBitWidth() > I2.getBitWidth())
    return IsSameValue(I1, I2.extend(I1.getBitWidth()));
  else if (I2.getBitWidth() > I1.getBitWidth())
    return IsSameValue(I1.extend(I2.getBitWidth()), I2);
  
  // We have a signedness mismatch. Turn the signed value into an unsigned 
  // value.
  if (I1.isSigned()) {
    if (I1.isNegative())
      return false;
    
    return llvm::APSInt(I1, true) == I2;
  }
 
  if (I2.isNegative())
    return false;
  
  return I1 == llvm::APSInt(I2, true);
}

/// \brief Determine structural equivalence of two expressions.
static bool IsStructurallyEquivalent(StructuralEquivalenceContext &Context,
                                     Expr *E1, Expr *E2) {
  if (!E1 || !E2)
    return E1 == E2;
  
  // FIXME: Actually perform a structural comparison!
  return true;
}

/// \brief Determine whether two identifiers are equivalent.
static bool IsStructurallyEquivalent(const IdentifierInfo *Name1,
                                     const IdentifierInfo *Name2) {
  if (!Name1 || !Name2)
    return Name1 == Name2;
  
  return Name1->getName() == Name2->getName();
}

/// \brief Determine whether two nested-name-specifiers are equivalent.
static bool IsStructurallyEquivalent(StructuralEquivalenceContext &Context,
                                     NestedNameSpecifier *NNS1,
                                     NestedNameSpecifier *NNS2) {
  // FIXME: Implement!
  return true;
}

/// \brief Determine whether two template arguments are equivalent.
static bool IsStructurallyEquivalent(StructuralEquivalenceContext &Context,
                                     const TemplateArgument &Arg1,
                                     const TemplateArgument &Arg2) {
  if (Arg1.getKind() != Arg2.getKind())
    return false;

  switch (Arg1.getKind()) {
  case TemplateArgument::Null:
    return true;
      
  case TemplateArgument::Type:
    return Context.IsStructurallyEquivalent(Arg1.getAsType(), Arg2.getAsType());
      
  case TemplateArgument::Integral:
    if (!Context.IsStructurallyEquivalent(Arg1.getIntegralType(), 
                                          Arg2.getIntegralType()))
      return false;
    
    return IsSameValue(*Arg1.getAsIntegral(), *Arg2.getAsIntegral());
      
  case TemplateArgument::Declaration:
    return Context.IsStructurallyEquivalent(Arg1.getAsDecl(), Arg2.getAsDecl());
      
  case TemplateArgument::Template:
    return IsStructurallyEquivalent(Context, 
                                    Arg1.getAsTemplate(), 
                                    Arg2.getAsTemplate());

  case TemplateArgument::TemplateExpansion:
    return IsStructurallyEquivalent(Context, 
                                    Arg1.getAsTemplateOrTemplatePattern(), 
                                    Arg2.getAsTemplateOrTemplatePattern());

  case TemplateArgument::Expression:
    return IsStructurallyEquivalent(Context, 
                                    Arg1.getAsExpr(), Arg2.getAsExpr());
      
  case TemplateArgument::Pack:
    if (Arg1.pack_size() != Arg2.pack_size())
      return false;
      
    for (unsigned I = 0, N = Arg1.pack_size(); I != N; ++I)
      if (!IsStructurallyEquivalent(Context, 
                                    Arg1.pack_begin()[I],
                                    Arg2.pack_begin()[I]))
        return false;
      
    return true;
  }
  
  llvm_unreachable("Invalid template argument kind");
  return true;
}

/// \brief Determine structural equivalence for the common part of array 
/// types.
static bool IsArrayStructurallyEquivalent(StructuralEquivalenceContext &Context,
                                          const ArrayType *Array1, 
                                          const ArrayType *Array2) {
  if (!IsStructurallyEquivalent(Context, 
                                Array1->getElementType(), 
                                Array2->getElementType()))
    return false;
  if (Array1->getSizeModifier() != Array2->getSizeModifier())
    return false;
  if (Array1->getIndexTypeQualifiers() != Array2->getIndexTypeQualifiers())
    return false;
  
  return true;
}

/// \brief Determine structural equivalence of two types.
static bool IsStructurallyEquivalent(StructuralEquivalenceContext &Context,
                                     QualType T1, QualType T2) {
  if (T1.isNull() || T2.isNull())
    return T1.isNull() && T2.isNull();
  
  if (!Context.StrictTypeSpelling) {
    // We aren't being strict about token-to-token equivalence of types,
    // so map down to the canonical type.
    T1 = Context.C1.getCanonicalType(T1);
    T2 = Context.C2.getCanonicalType(T2);
  }
  
  if (T1.getQualifiers() != T2.getQualifiers())
    return false;
  
  Type::TypeClass TC = T1->getTypeClass();
  
  if (T1->getTypeClass() != T2->getTypeClass()) {
    // Compare function types with prototypes vs. without prototypes as if
    // both did not have prototypes.
    if (T1->getTypeClass() == Type::FunctionProto &&
        T2->getTypeClass() == Type::FunctionNoProto)
      TC = Type::FunctionNoProto;
    else if (T1->getTypeClass() == Type::FunctionNoProto &&
             T2->getTypeClass() == Type::FunctionProto)
      TC = Type::FunctionNoProto;
    else
      return false;
  }
  
  switch (TC) {
  case Type::Builtin:
    // FIXME: Deal with Char_S/Char_U. 
    if (cast<BuiltinType>(T1)->getKind() != cast<BuiltinType>(T2)->getKind())
      return false;
    break;
  
  case Type::Complex:
    if (!IsStructurallyEquivalent(Context,
                                  cast<ComplexType>(T1)->getElementType(),
                                  cast<ComplexType>(T2)->getElementType()))
      return false;
    break;
  
  case Type::Pointer:
    if (!IsStructurallyEquivalent(Context,
                                  cast<PointerType>(T1)->getPointeeType(),
                                  cast<PointerType>(T2)->getPointeeType()))
      return false;
    break;

  case Type::BlockPointer:
    if (!IsStructurallyEquivalent(Context,
                                  cast<BlockPointerType>(T1)->getPointeeType(),
                                  cast<BlockPointerType>(T2)->getPointeeType()))
      return false;
    break;

  case Type::LValueReference:
  case Type::RValueReference: {
    const ReferenceType *Ref1 = cast<ReferenceType>(T1);
    const ReferenceType *Ref2 = cast<ReferenceType>(T2);
    if (Ref1->isSpelledAsLValue() != Ref2->isSpelledAsLValue())
      return false;
    if (Ref1->isInnerRef() != Ref2->isInnerRef())
      return false;
    if (!IsStructurallyEquivalent(Context,
                                  Ref1->getPointeeTypeAsWritten(),
                                  Ref2->getPointeeTypeAsWritten()))
      return false;
    break;
  }
      
  case Type::MemberPointer: {
    const MemberPointerType *MemPtr1 = cast<MemberPointerType>(T1);
    const MemberPointerType *MemPtr2 = cast<MemberPointerType>(T2);
    if (!IsStructurallyEquivalent(Context,
                                  MemPtr1->getPointeeType(),
                                  MemPtr2->getPointeeType()))
      return false;
    if (!IsStructurallyEquivalent(Context,
                                  QualType(MemPtr1->getClass(), 0),
                                  QualType(MemPtr2->getClass(), 0)))
      return false;
    break;
  }
      
  case Type::ConstantArray: {
    const ConstantArrayType *Array1 = cast<ConstantArrayType>(T1);
    const ConstantArrayType *Array2 = cast<ConstantArrayType>(T2);
    if (!IsSameValue(Array1->getSize(), Array2->getSize()))
      return false;
    
    if (!IsArrayStructurallyEquivalent(Context, Array1, Array2))
      return false;
    break;
  }

  case Type::IncompleteArray:
    if (!IsArrayStructurallyEquivalent(Context, 
                                       cast<ArrayType>(T1), 
                                       cast<ArrayType>(T2)))
      return false;
    break;
      
  case Type::VariableArray: {
    const VariableArrayType *Array1 = cast<VariableArrayType>(T1);
    const VariableArrayType *Array2 = cast<VariableArrayType>(T2);
    if (!IsStructurallyEquivalent(Context, 
                                  Array1->getSizeExpr(), Array2->getSizeExpr()))
      return false;
    
    if (!IsArrayStructurallyEquivalent(Context, Array1, Array2))
      return false;
    
    break;
  }
  
  case Type::DependentSizedArray: {
    const DependentSizedArrayType *Array1 = cast<DependentSizedArrayType>(T1);
    const DependentSizedArrayType *Array2 = cast<DependentSizedArrayType>(T2);
    if (!IsStructurallyEquivalent(Context, 
                                  Array1->getSizeExpr(), Array2->getSizeExpr()))
      return false;
    
    if (!IsArrayStructurallyEquivalent(Context, Array1, Array2))
      return false;
    
    break;
  }
      
  case Type::DependentSizedExtVector: {
    const DependentSizedExtVectorType *Vec1
      = cast<DependentSizedExtVectorType>(T1);
    const DependentSizedExtVectorType *Vec2
      = cast<DependentSizedExtVectorType>(T2);
    if (!IsStructurallyEquivalent(Context, 
                                  Vec1->getSizeExpr(), Vec2->getSizeExpr()))
      return false;
    if (!IsStructurallyEquivalent(Context, 
                                  Vec1->getElementType(), 
                                  Vec2->getElementType()))
      return false;
    break;
  }
   
  case Type::Vector: 
  case Type::ExtVector: {
    const VectorType *Vec1 = cast<VectorType>(T1);
    const VectorType *Vec2 = cast<VectorType>(T2);
    if (!IsStructurallyEquivalent(Context, 
                                  Vec1->getElementType(),
                                  Vec2->getElementType()))
      return false;
    if (Vec1->getNumElements() != Vec2->getNumElements())
      return false;
    if (Vec1->getVectorKind() != Vec2->getVectorKind())
      return false;
    break;
  }

  case Type::FunctionProto: {
    const FunctionProtoType *Proto1 = cast<FunctionProtoType>(T1);
    const FunctionProtoType *Proto2 = cast<FunctionProtoType>(T2);
    if (Proto1->getNumArgs() != Proto2->getNumArgs())
      return false;
    for (unsigned I = 0, N = Proto1->getNumArgs(); I != N; ++I) {
      if (!IsStructurallyEquivalent(Context, 
                                    Proto1->getArgType(I),
                                    Proto2->getArgType(I)))
        return false;
    }
    if (Proto1->isVariadic() != Proto2->isVariadic())
      return false;
    if (Proto1->getExceptionSpecType() != Proto2->getExceptionSpecType())
      return false;
    if (Proto1->getExceptionSpecType() == EST_Dynamic) {
      if (Proto1->getNumExceptions() != Proto2->getNumExceptions())
        return false;
      for (unsigned I = 0, N = Proto1->getNumExceptions(); I != N; ++I) {
        if (!IsStructurallyEquivalent(Context,
                                      Proto1->getExceptionType(I),
                                      Proto2->getExceptionType(I)))
          return false;
      }
    } else if (Proto1->getExceptionSpecType() == EST_ComputedNoexcept) {
      if (!IsStructurallyEquivalent(Context,
                                    Proto1->getNoexceptExpr(),
                                    Proto2->getNoexceptExpr()))
        return false;
    }
    if (Proto1->getTypeQuals() != Proto2->getTypeQuals())
      return false;
    
    // Fall through to check the bits common with FunctionNoProtoType.
  }
      
  case Type::FunctionNoProto: {
    const FunctionType *Function1 = cast<FunctionType>(T1);
    const FunctionType *Function2 = cast<FunctionType>(T2);
    if (!IsStructurallyEquivalent(Context, 
                                  Function1->getResultType(),
                                  Function2->getResultType()))
      return false;
      if (Function1->getExtInfo() != Function2->getExtInfo())
        return false;
    break;
  }
   
  case Type::UnresolvedUsing:
    if (!IsStructurallyEquivalent(Context,
                                  cast<UnresolvedUsingType>(T1)->getDecl(),
                                  cast<UnresolvedUsingType>(T2)->getDecl()))
      return false;
      
    break;

  case Type::Attributed:
    if (!IsStructurallyEquivalent(Context,
                                  cast<AttributedType>(T1)->getModifiedType(),
                                  cast<AttributedType>(T2)->getModifiedType()))
      return false;
    if (!IsStructurallyEquivalent(Context,
                                cast<AttributedType>(T1)->getEquivalentType(),
                                cast<AttributedType>(T2)->getEquivalentType()))
      return false;
    break;
      
  case Type::Paren:
    if (!IsStructurallyEquivalent(Context,
                                  cast<ParenType>(T1)->getInnerType(),
                                  cast<ParenType>(T2)->getInnerType()))
      return false;
    break;

  case Type::Typedef:
    if (!IsStructurallyEquivalent(Context,
                                  cast<TypedefType>(T1)->getDecl(),
                                  cast<TypedefType>(T2)->getDecl()))
      return false;
    break;
      
  case Type::TypeOfExpr:
    if (!IsStructurallyEquivalent(Context,
                                cast<TypeOfExprType>(T1)->getUnderlyingExpr(),
                                cast<TypeOfExprType>(T2)->getUnderlyingExpr()))
      return false;
    break;
      
  case Type::TypeOf:
    if (!IsStructurallyEquivalent(Context,
                                  cast<TypeOfType>(T1)->getUnderlyingType(),
                                  cast<TypeOfType>(T2)->getUnderlyingType()))
      return false;
    break;

  case Type::UnaryTransform:
    if (!IsStructurallyEquivalent(Context,
                             cast<UnaryTransformType>(T1)->getUnderlyingType(),
                             cast<UnaryTransformType>(T1)->getUnderlyingType()))
      return false;
    break;

  case Type::Decltype:
    if (!IsStructurallyEquivalent(Context,
                                  cast<DecltypeType>(T1)->getUnderlyingExpr(),
                                  cast<DecltypeType>(T2)->getUnderlyingExpr()))
      return false;
    break;

  case Type::Auto:
    if (!IsStructurallyEquivalent(Context,
                                  cast<AutoType>(T1)->getDeducedType(),
                                  cast<AutoType>(T2)->getDeducedType()))
      return false;
    break;

  case Type::Record:
  case Type::Enum:
    if (!IsStructurallyEquivalent(Context,
                                  cast<TagType>(T1)->getDecl(),
                                  cast<TagType>(T2)->getDecl()))
      return false;
    break;

  case Type::TemplateTypeParm: {
    const TemplateTypeParmType *Parm1 = cast<TemplateTypeParmType>(T1);
    const TemplateTypeParmType *Parm2 = cast<TemplateTypeParmType>(T2);
    if (Parm1->getDepth() != Parm2->getDepth())
      return false;
    if (Parm1->getIndex() != Parm2->getIndex())
      return false;
    if (Parm1->isParameterPack() != Parm2->isParameterPack())
      return false;
    
    // Names of template type parameters are never significant.
    break;
  }
      
  case Type::SubstTemplateTypeParm: {
    const SubstTemplateTypeParmType *Subst1
      = cast<SubstTemplateTypeParmType>(T1);
    const SubstTemplateTypeParmType *Subst2
      = cast<SubstTemplateTypeParmType>(T2);
    if (!IsStructurallyEquivalent(Context,
                                  QualType(Subst1->getReplacedParameter(), 0),
                                  QualType(Subst2->getReplacedParameter(), 0)))
      return false;
    if (!IsStructurallyEquivalent(Context, 
                                  Subst1->getReplacementType(),
                                  Subst2->getReplacementType()))
      return false;
    break;
  }

  case Type::SubstTemplateTypeParmPack: {
    const SubstTemplateTypeParmPackType *Subst1
      = cast<SubstTemplateTypeParmPackType>(T1);
    const SubstTemplateTypeParmPackType *Subst2
      = cast<SubstTemplateTypeParmPackType>(T2);
    if (!IsStructurallyEquivalent(Context,
                                  QualType(Subst1->getReplacedParameter(), 0),
                                  QualType(Subst2->getReplacedParameter(), 0)))
      return false;
    if (!IsStructurallyEquivalent(Context, 
                                  Subst1->getArgumentPack(),
                                  Subst2->getArgumentPack()))
      return false;
    break;
  }
  case Type::TemplateSpecialization: {
    const TemplateSpecializationType *Spec1
      = cast<TemplateSpecializationType>(T1);
    const TemplateSpecializationType *Spec2
      = cast<TemplateSpecializationType>(T2);
    if (!IsStructurallyEquivalent(Context,
                                  Spec1->getTemplateName(),
                                  Spec2->getTemplateName()))
      return false;
    if (Spec1->getNumArgs() != Spec2->getNumArgs())
      return false;
    for (unsigned I = 0, N = Spec1->getNumArgs(); I != N; ++I) {
      if (!IsStructurallyEquivalent(Context, 
                                    Spec1->getArg(I), Spec2->getArg(I)))
        return false;
    }
    break;
  }
      
  case Type::Elaborated: {
    const ElaboratedType *Elab1 = cast<ElaboratedType>(T1);
    const ElaboratedType *Elab2 = cast<ElaboratedType>(T2);
    // CHECKME: what if a keyword is ETK_None or ETK_typename ?
    if (Elab1->getKeyword() != Elab2->getKeyword())
      return false;
    if (!IsStructurallyEquivalent(Context, 
                                  Elab1->getQualifier(), 
                                  Elab2->getQualifier()))
      return false;
    if (!IsStructurallyEquivalent(Context,
                                  Elab1->getNamedType(),
                                  Elab2->getNamedType()))
      return false;
    break;
  }

  case Type::InjectedClassName: {
    const InjectedClassNameType *Inj1 = cast<InjectedClassNameType>(T1);
    const InjectedClassNameType *Inj2 = cast<InjectedClassNameType>(T2);
    if (!IsStructurallyEquivalent(Context,
                                  Inj1->getInjectedSpecializationType(),
                                  Inj2->getInjectedSpecializationType()))
      return false;
    break;
  }

  case Type::DependentName: {
    const DependentNameType *Typename1 = cast<DependentNameType>(T1);
    const DependentNameType *Typename2 = cast<DependentNameType>(T2);
    if (!IsStructurallyEquivalent(Context, 
                                  Typename1->getQualifier(),
                                  Typename2->getQualifier()))
      return false;
    if (!IsStructurallyEquivalent(Typename1->getIdentifier(),
                                  Typename2->getIdentifier()))
      return false;
    
    break;
  }
  
  case Type::DependentTemplateSpecialization: {
    const DependentTemplateSpecializationType *Spec1 =
      cast<DependentTemplateSpecializationType>(T1);
    const DependentTemplateSpecializationType *Spec2 =
      cast<DependentTemplateSpecializationType>(T2);
    if (!IsStructurallyEquivalent(Context, 
                                  Spec1->getQualifier(),
                                  Spec2->getQualifier()))
      return false;
    if (!IsStructurallyEquivalent(Spec1->getIdentifier(),
                                  Spec2->getIdentifier()))
      return false;
    if (Spec1->getNumArgs() != Spec2->getNumArgs())
      return false;
    for (unsigned I = 0, N = Spec1->getNumArgs(); I != N; ++I) {
      if (!IsStructurallyEquivalent(Context,
                                    Spec1->getArg(I), Spec2->getArg(I)))
        return false;
    }
    break;
  }

  case Type::PackExpansion:
    if (!IsStructurallyEquivalent(Context,
                                  cast<PackExpansionType>(T1)->getPattern(),
                                  cast<PackExpansionType>(T2)->getPattern()))
      return false;
    break;

  case Type::ObjCInterface: {
    const ObjCInterfaceType *Iface1 = cast<ObjCInterfaceType>(T1);
    const ObjCInterfaceType *Iface2 = cast<ObjCInterfaceType>(T2);
    if (!IsStructurallyEquivalent(Context, 
                                  Iface1->getDecl(), Iface2->getDecl()))
      return false;
    break;
  }

  case Type::ObjCObject: {
    const ObjCObjectType *Obj1 = cast<ObjCObjectType>(T1);
    const ObjCObjectType *Obj2 = cast<ObjCObjectType>(T2);
    if (!IsStructurallyEquivalent(Context,
                                  Obj1->getBaseType(),
                                  Obj2->getBaseType()))
      return false;
    if (Obj1->getNumProtocols() != Obj2->getNumProtocols())
      return false;
    for (unsigned I = 0, N = Obj1->getNumProtocols(); I != N; ++I) {
      if (!IsStructurallyEquivalent(Context,
                                    Obj1->getProtocol(I),
                                    Obj2->getProtocol(I)))
        return false;
    }
    break;
  }

  case Type::ObjCObjectPointer: {
    const ObjCObjectPointerType *Ptr1 = cast<ObjCObjectPointerType>(T1);
    const ObjCObjectPointerType *Ptr2 = cast<ObjCObjectPointerType>(T2);
    if (!IsStructurallyEquivalent(Context, 
                                  Ptr1->getPointeeType(),
                                  Ptr2->getPointeeType()))
      return false;
    break;
  }
      
  } // end switch

  return true;
}

/// \brief Determine structural equivalence of two records.
static bool IsStructurallyEquivalent(StructuralEquivalenceContext &Context,
                                     RecordDecl *D1, RecordDecl *D2) {
  if (D1->isUnion() != D2->isUnion()) {
    Context.Diag2(D2->getLocation(), diag::warn_odr_tag_type_inconsistent)
      << Context.C2.getTypeDeclType(D2);
    Context.Diag1(D1->getLocation(), diag::note_odr_tag_kind_here)
      << D1->getDeclName() << (unsigned)D1->getTagKind();
    return false;
  }
  
  // If both declarations are class template specializations, we know
  // the ODR applies, so check the template and template arguments.
  ClassTemplateSpecializationDecl *Spec1
    = dyn_cast<ClassTemplateSpecializationDecl>(D1);
  ClassTemplateSpecializationDecl *Spec2
    = dyn_cast<ClassTemplateSpecializationDecl>(D2);
  if (Spec1 && Spec2) {
    // Check that the specialized templates are the same.
    if (!IsStructurallyEquivalent(Context, Spec1->getSpecializedTemplate(),
                                  Spec2->getSpecializedTemplate()))
      return false;
    
    // Check that the template arguments are the same.
    if (Spec1->getTemplateArgs().size() != Spec2->getTemplateArgs().size())
      return false;
    
    for (unsigned I = 0, N = Spec1->getTemplateArgs().size(); I != N; ++I)
      if (!IsStructurallyEquivalent(Context, 
                                    Spec1->getTemplateArgs().get(I),
                                    Spec2->getTemplateArgs().get(I)))
        return false;
  }  
  // If one is a class template specialization and the other is not, these
  // structures are different.
  else if (Spec1 || Spec2)
    return false;

  // Compare the definitions of these two records. If either or both are
  // incomplete, we assume that they are equivalent.
  D1 = D1->getDefinition();
  D2 = D2->getDefinition();
  if (!D1 || !D2)
    return true;
  
  if (CXXRecordDecl *D1CXX = dyn_cast<CXXRecordDecl>(D1)) {
    if (CXXRecordDecl *D2CXX = dyn_cast<CXXRecordDecl>(D2)) {
      if (D1CXX->getNumBases() != D2CXX->getNumBases()) {
        Context.Diag2(D2->getLocation(), diag::warn_odr_tag_type_inconsistent)
          << Context.C2.getTypeDeclType(D2);
        Context.Diag2(D2->getLocation(), diag::note_odr_number_of_bases)
          << D2CXX->getNumBases();
        Context.Diag1(D1->getLocation(), diag::note_odr_number_of_bases)
          << D1CXX->getNumBases();
        return false;
      }
      
      // Check the base classes. 
      for (CXXRecordDecl::base_class_iterator Base1 = D1CXX->bases_begin(), 
                                           BaseEnd1 = D1CXX->bases_end(),
                                                Base2 = D2CXX->bases_begin();
           Base1 != BaseEnd1;
           ++Base1, ++Base2) {        
        if (!IsStructurallyEquivalent(Context, 
                                      Base1->getType(), Base2->getType())) {
          Context.Diag2(D2->getLocation(), diag::warn_odr_tag_type_inconsistent)
            << Context.C2.getTypeDeclType(D2);
          Context.Diag2(Base2->getSourceRange().getBegin(), diag::note_odr_base)
            << Base2->getType()
            << Base2->getSourceRange();
          Context.Diag1(Base1->getSourceRange().getBegin(), diag::note_odr_base)
            << Base1->getType()
            << Base1->getSourceRange();
          return false;
        }
        
        // Check virtual vs. non-virtual inheritance mismatch.
        if (Base1->isVirtual() != Base2->isVirtual()) {
          Context.Diag2(D2->getLocation(), diag::warn_odr_tag_type_inconsistent)
            << Context.C2.getTypeDeclType(D2);
          Context.Diag2(Base2->getSourceRange().getBegin(),
                        diag::note_odr_virtual_base)
            << Base2->isVirtual() << Base2->getSourceRange();
          Context.Diag1(Base1->getSourceRange().getBegin(), diag::note_odr_base)
            << Base1->isVirtual()
            << Base1->getSourceRange();
          return false;
        }
      }
    } else if (D1CXX->getNumBases() > 0) {
      Context.Diag2(D2->getLocation(), diag::warn_odr_tag_type_inconsistent)
        << Context.C2.getTypeDeclType(D2);
      const CXXBaseSpecifier *Base1 = D1CXX->bases_begin();
      Context.Diag1(Base1->getSourceRange().getBegin(), diag::note_odr_base)
        << Base1->getType()
        << Base1->getSourceRange();
      Context.Diag2(D2->getLocation(), diag::note_odr_missing_base);
      return false;
    }
  }
  
  // Check the fields for consistency.
  CXXRecordDecl::field_iterator Field2 = D2->field_begin(),
                             Field2End = D2->field_end();
  for (CXXRecordDecl::field_iterator Field1 = D1->field_begin(),
                                  Field1End = D1->field_end();
       Field1 != Field1End;
       ++Field1, ++Field2) {
    if (Field2 == Field2End) {
      Context.Diag2(D2->getLocation(), diag::warn_odr_tag_type_inconsistent)
        << Context.C2.getTypeDeclType(D2);
      Context.Diag1(Field1->getLocation(), diag::note_odr_field)
        << Field1->getDeclName() << Field1->getType();
      Context.Diag2(D2->getLocation(), diag::note_odr_missing_field);
      return false;
    }
    
    if (!IsStructurallyEquivalent(Context, 
                                  Field1->getType(), Field2->getType())) {
      Context.Diag2(D2->getLocation(), diag::warn_odr_tag_type_inconsistent)
        << Context.C2.getTypeDeclType(D2);
      Context.Diag2(Field2->getLocation(), diag::note_odr_field)
        << Field2->getDeclName() << Field2->getType();
      Context.Diag1(Field1->getLocation(), diag::note_odr_field)
        << Field1->getDeclName() << Field1->getType();
      return false;
    }
    
    if (Field1->isBitField() != Field2->isBitField()) {
      Context.Diag2(D2->getLocation(), diag::warn_odr_tag_type_inconsistent)
        << Context.C2.getTypeDeclType(D2);
      if (Field1->isBitField()) {
        llvm::APSInt Bits;
        Field1->getBitWidth()->isIntegerConstantExpr(Bits, Context.C1);
        Context.Diag1(Field1->getLocation(), diag::note_odr_bit_field)
          << Field1->getDeclName() << Field1->getType()
          << Bits.toString(10, false);
        Context.Diag2(Field2->getLocation(), diag::note_odr_not_bit_field)
          << Field2->getDeclName();
      } else {
        llvm::APSInt Bits;
        Field2->getBitWidth()->isIntegerConstantExpr(Bits, Context.C2);
        Context.Diag2(Field2->getLocation(), diag::note_odr_bit_field)
          << Field2->getDeclName() << Field2->getType()
          << Bits.toString(10, false);
        Context.Diag1(Field1->getLocation(), 
                          diag::note_odr_not_bit_field)
        << Field1->getDeclName();
      }
      return false;
    }
    
    if (Field1->isBitField()) {
      // Make sure that the bit-fields are the same length.
      llvm::APSInt Bits1, Bits2;
      if (!Field1->getBitWidth()->isIntegerConstantExpr(Bits1, Context.C1))
        return false;
      if (!Field2->getBitWidth()->isIntegerConstantExpr(Bits2, Context.C2))
        return false;
      
      if (!IsSameValue(Bits1, Bits2)) {
        Context.Diag2(D2->getLocation(), diag::warn_odr_tag_type_inconsistent)
          << Context.C2.getTypeDeclType(D2);
        Context.Diag2(Field2->getLocation(), diag::note_odr_bit_field)
          << Field2->getDeclName() << Field2->getType()
          << Bits2.toString(10, false);
        Context.Diag1(Field1->getLocation(), diag::note_odr_bit_field)
          << Field1->getDeclName() << Field1->getType()
          << Bits1.toString(10, false);
        return false;
      }
    }
  }
  
  if (Field2 != Field2End) {
    Context.Diag2(D2->getLocation(), diag::warn_odr_tag_type_inconsistent)
      << Context.C2.getTypeDeclType(D2);
    Context.Diag2(Field2->getLocation(), diag::note_odr_field)
      << Field2->getDeclName() << Field2->getType();
    Context.Diag1(D1->getLocation(), diag::note_odr_missing_field);
    return false;
  }
  
  return true;
}
     
/// \brief Determine structural equivalence of two enums.
static bool IsStructurallyEquivalent(StructuralEquivalenceContext &Context,
                                     EnumDecl *D1, EnumDecl *D2) {
  EnumDecl::enumerator_iterator EC2 = D2->enumerator_begin(),
                             EC2End = D2->enumerator_end();
  for (EnumDecl::enumerator_iterator EC1 = D1->enumerator_begin(),
                                  EC1End = D1->enumerator_end();
       EC1 != EC1End; ++EC1, ++EC2) {
    if (EC2 == EC2End) {
      Context.Diag2(D2->getLocation(), diag::warn_odr_tag_type_inconsistent)
        << Context.C2.getTypeDeclType(D2);
      Context.Diag1(EC1->getLocation(), diag::note_odr_enumerator)
        << EC1->getDeclName() 
        << EC1->getInitVal().toString(10);
      Context.Diag2(D2->getLocation(), diag::note_odr_missing_enumerator);
      return false;
    }
    
    llvm::APSInt Val1 = EC1->getInitVal();
    llvm::APSInt Val2 = EC2->getInitVal();
    if (!IsSameValue(Val1, Val2) || 
        !IsStructurallyEquivalent(EC1->getIdentifier(), EC2->getIdentifier())) {
      Context.Diag2(D2->getLocation(), diag::warn_odr_tag_type_inconsistent)
        << Context.C2.getTypeDeclType(D2);
      Context.Diag2(EC2->getLocation(), diag::note_odr_enumerator)
        << EC2->getDeclName() 
        << EC2->getInitVal().toString(10);
      Context.Diag1(EC1->getLocation(), diag::note_odr_enumerator)
        << EC1->getDeclName() 
        << EC1->getInitVal().toString(10);
      return false;
    }
  }
  
  if (EC2 != EC2End) {
    Context.Diag2(D2->getLocation(), diag::warn_odr_tag_type_inconsistent)
      << Context.C2.getTypeDeclType(D2);
    Context.Diag2(EC2->getLocation(), diag::note_odr_enumerator)
      << EC2->getDeclName() 
      << EC2->getInitVal().toString(10);
    Context.Diag1(D1->getLocation(), diag::note_odr_missing_enumerator);
    return false;
  }
  
  return true;
}

static bool IsStructurallyEquivalent(StructuralEquivalenceContext &Context,
                                     TemplateParameterList *Params1,
                                     TemplateParameterList *Params2) {
  if (Params1->size() != Params2->size()) {
    Context.Diag2(Params2->getTemplateLoc(), 
                  diag::err_odr_different_num_template_parameters)
      << Params1->size() << Params2->size();
    Context.Diag1(Params1->getTemplateLoc(), 
                  diag::note_odr_template_parameter_list);
    return false;
  }
  
  for (unsigned I = 0, N = Params1->size(); I != N; ++I) {
    if (Params1->getParam(I)->getKind() != Params2->getParam(I)->getKind()) {
      Context.Diag2(Params2->getParam(I)->getLocation(), 
                    diag::err_odr_different_template_parameter_kind);
      Context.Diag1(Params1->getParam(I)->getLocation(),
                    diag::note_odr_template_parameter_here);
      return false;
    }
    
    if (!Context.IsStructurallyEquivalent(Params1->getParam(I),
                                          Params2->getParam(I))) {
      
      return false;
    }
  }
  
  return true;
}

static bool IsStructurallyEquivalent(StructuralEquivalenceContext &Context,
                                     TemplateTypeParmDecl *D1,
                                     TemplateTypeParmDecl *D2) {
  if (D1->isParameterPack() != D2->isParameterPack()) {
    Context.Diag2(D2->getLocation(), diag::err_odr_parameter_pack_non_pack)
      << D2->isParameterPack();
    Context.Diag1(D1->getLocation(), diag::note_odr_parameter_pack_non_pack)
      << D1->isParameterPack();
    return false;
  }
  
  return true;
}

static bool IsStructurallyEquivalent(StructuralEquivalenceContext &Context,
                                     NonTypeTemplateParmDecl *D1,
                                     NonTypeTemplateParmDecl *D2) {
  // FIXME: Enable once we have variadic templates.
#if 0
  if (D1->isParameterPack() != D2->isParameterPack()) {
    Context.Diag2(D2->getLocation(), diag::err_odr_parameter_pack_non_pack)
      << D2->isParameterPack();
    Context.Diag1(D1->getLocation(), diag::note_odr_parameter_pack_non_pack)
      << D1->isParameterPack();
    return false;
  }
#endif
  
  // Check types.
  if (!Context.IsStructurallyEquivalent(D1->getType(), D2->getType())) {
    Context.Diag2(D2->getLocation(), 
                  diag::err_odr_non_type_parameter_type_inconsistent)
      << D2->getType() << D1->getType();
    Context.Diag1(D1->getLocation(), diag::note_odr_value_here)
      << D1->getType();
    return false;
  }
  
  return true;
}

static bool IsStructurallyEquivalent(StructuralEquivalenceContext &Context,
                                     TemplateTemplateParmDecl *D1,
                                     TemplateTemplateParmDecl *D2) {
  // FIXME: Enable once we have variadic templates.
#if 0
  if (D1->isParameterPack() != D2->isParameterPack()) {
    Context.Diag2(D2->getLocation(), diag::err_odr_parameter_pack_non_pack)
    << D2->isParameterPack();
    Context.Diag1(D1->getLocation(), diag::note_odr_parameter_pack_non_pack)
    << D1->isParameterPack();
    return false;
  }
#endif
  
  // Check template parameter lists.
  return IsStructurallyEquivalent(Context, D1->getTemplateParameters(),
                                  D2->getTemplateParameters());
}

static bool IsStructurallyEquivalent(StructuralEquivalenceContext &Context,
                                     ClassTemplateDecl *D1, 
                                     ClassTemplateDecl *D2) {
  // Check template parameters.
  if (!IsStructurallyEquivalent(Context,
                                D1->getTemplateParameters(),
                                D2->getTemplateParameters()))
    return false;
  
  // Check the templated declaration.
  return Context.IsStructurallyEquivalent(D1->getTemplatedDecl(), 
                                          D2->getTemplatedDecl());
}

/// \brief Determine structural equivalence of two declarations.
static bool IsStructurallyEquivalent(StructuralEquivalenceContext &Context,
                                     Decl *D1, Decl *D2) {
  // FIXME: Check for known structural equivalences via a callback of some sort.
  
  // Check whether we already know that these two declarations are not
  // structurally equivalent.
  if (Context.NonEquivalentDecls.count(std::make_pair(D1->getCanonicalDecl(),
                                                      D2->getCanonicalDecl())))
    return false;
  
  // Determine whether we've already produced a tentative equivalence for D1.
  Decl *&EquivToD1 = Context.TentativeEquivalences[D1->getCanonicalDecl()];
  if (EquivToD1)
    return EquivToD1 == D2->getCanonicalDecl();
  
  // Produce a tentative equivalence D1 <-> D2, which will be checked later.
  EquivToD1 = D2->getCanonicalDecl();
  Context.DeclsToCheck.push_back(D1->getCanonicalDecl());
  return true;
}

bool StructuralEquivalenceContext::IsStructurallyEquivalent(Decl *D1, 
                                                            Decl *D2) {
  if (!::IsStructurallyEquivalent(*this, D1, D2))
    return false;
  
  return !Finish();
}

bool StructuralEquivalenceContext::IsStructurallyEquivalent(QualType T1, 
                                                            QualType T2) {
  if (!::IsStructurallyEquivalent(*this, T1, T2))
    return false;
  
  return !Finish();
}

bool StructuralEquivalenceContext::Finish() {
  while (!DeclsToCheck.empty()) {
    // Check the next declaration.
    Decl *D1 = DeclsToCheck.front();
    DeclsToCheck.pop_front();
    
    Decl *D2 = TentativeEquivalences[D1];
    assert(D2 && "Unrecorded tentative equivalence?");
    
    bool Equivalent = true;
    
    // FIXME: Switch on all declaration kinds. For now, we're just going to
    // check the obvious ones.
    if (RecordDecl *Record1 = dyn_cast<RecordDecl>(D1)) {
      if (RecordDecl *Record2 = dyn_cast<RecordDecl>(D2)) {
        // Check for equivalent structure names.
        IdentifierInfo *Name1 = Record1->getIdentifier();
        if (!Name1 && Record1->getTypedefNameForAnonDecl())
          Name1 = Record1->getTypedefNameForAnonDecl()->getIdentifier();
        IdentifierInfo *Name2 = Record2->getIdentifier();
        if (!Name2 && Record2->getTypedefNameForAnonDecl())
          Name2 = Record2->getTypedefNameForAnonDecl()->getIdentifier();
        if (!::IsStructurallyEquivalent(Name1, Name2) ||
            !::IsStructurallyEquivalent(*this, Record1, Record2))
          Equivalent = false;
      } else {
        // Record/non-record mismatch.
        Equivalent = false;
      }
    } else if (EnumDecl *Enum1 = dyn_cast<EnumDecl>(D1)) {
      if (EnumDecl *Enum2 = dyn_cast<EnumDecl>(D2)) {
        // Check for equivalent enum names.
        IdentifierInfo *Name1 = Enum1->getIdentifier();
        if (!Name1 && Enum1->getTypedefNameForAnonDecl())
          Name1 = Enum1->getTypedefNameForAnonDecl()->getIdentifier();
        IdentifierInfo *Name2 = Enum2->getIdentifier();
        if (!Name2 && Enum2->getTypedefNameForAnonDecl())
          Name2 = Enum2->getTypedefNameForAnonDecl()->getIdentifier();
        if (!::IsStructurallyEquivalent(Name1, Name2) ||
            !::IsStructurallyEquivalent(*this, Enum1, Enum2))
          Equivalent = false;
      } else {
        // Enum/non-enum mismatch
        Equivalent = false;
      }
    } else if (TypedefNameDecl *Typedef1 = dyn_cast<TypedefNameDecl>(D1)) {
      if (TypedefNameDecl *Typedef2 = dyn_cast<TypedefNameDecl>(D2)) {
        if (!::IsStructurallyEquivalent(Typedef1->getIdentifier(),
                                        Typedef2->getIdentifier()) ||
            !::IsStructurallyEquivalent(*this,
                                        Typedef1->getUnderlyingType(),
                                        Typedef2->getUnderlyingType()))
          Equivalent = false;
      } else {
        // Typedef/non-typedef mismatch.
        Equivalent = false;
      }
    } else if (ClassTemplateDecl *ClassTemplate1 
                                           = dyn_cast<ClassTemplateDecl>(D1)) {
      if (ClassTemplateDecl *ClassTemplate2 = dyn_cast<ClassTemplateDecl>(D2)) {
        if (!::IsStructurallyEquivalent(ClassTemplate1->getIdentifier(),
                                        ClassTemplate2->getIdentifier()) ||
            !::IsStructurallyEquivalent(*this, ClassTemplate1, ClassTemplate2))
          Equivalent = false;
      } else {
        // Class template/non-class-template mismatch.
        Equivalent = false;
      }
    } else if (TemplateTypeParmDecl *TTP1= dyn_cast<TemplateTypeParmDecl>(D1)) {
      if (TemplateTypeParmDecl *TTP2 = dyn_cast<TemplateTypeParmDecl>(D2)) {
        if (!::IsStructurallyEquivalent(*this, TTP1, TTP2))
          Equivalent = false;
      } else {
        // Kind mismatch.
        Equivalent = false;
      }
    } else if (NonTypeTemplateParmDecl *NTTP1
                                     = dyn_cast<NonTypeTemplateParmDecl>(D1)) {
      if (NonTypeTemplateParmDecl *NTTP2
                                      = dyn_cast<NonTypeTemplateParmDecl>(D2)) {
        if (!::IsStructurallyEquivalent(*this, NTTP1, NTTP2))
          Equivalent = false;
      } else {
        // Kind mismatch.
        Equivalent = false;
      }
    } else if (TemplateTemplateParmDecl *TTP1
                                  = dyn_cast<TemplateTemplateParmDecl>(D1)) {
      if (TemplateTemplateParmDecl *TTP2
                                    = dyn_cast<TemplateTemplateParmDecl>(D2)) {
        if (!::IsStructurallyEquivalent(*this, TTP1, TTP2))
          Equivalent = false;
      } else {
        // Kind mismatch.
        Equivalent = false;
      }
    }
    
    if (!Equivalent) {
      // Note that these two declarations are not equivalent (and we already
      // know about it).
      NonEquivalentDecls.insert(std::make_pair(D1->getCanonicalDecl(),
                                               D2->getCanonicalDecl()));
      return true;
    }
    // FIXME: Check other declaration kinds!
  }
  
  return false;
}

//----------------------------------------------------------------------------
// Import Types
//----------------------------------------------------------------------------

QualType ASTNodeImporter::VisitType(const Type *T) {
  Importer.FromDiag(SourceLocation(), diag::err_unsupported_ast_node)
    << T->getTypeClassName();
  return QualType();
}

QualType ASTNodeImporter::VisitBuiltinType(const BuiltinType *T) {
  switch (T->getKind()) {
  case BuiltinType::Void: return Importer.getToContext().VoidTy;
  case BuiltinType::Bool: return Importer.getToContext().BoolTy;
    
  case BuiltinType::Char_U:
    // The context we're importing from has an unsigned 'char'. If we're 
    // importing into a context with a signed 'char', translate to 
    // 'unsigned char' instead.
    if (Importer.getToContext().getLangOptions().CharIsSigned)
      return Importer.getToContext().UnsignedCharTy;
    
    return Importer.getToContext().CharTy;

  case BuiltinType::UChar: return Importer.getToContext().UnsignedCharTy;
    
  case BuiltinType::Char16:
    // FIXME: Make sure that the "to" context supports C++!
    return Importer.getToContext().Char16Ty;
    
  case BuiltinType::Char32: 
    // FIXME: Make sure that the "to" context supports C++!
    return Importer.getToContext().Char32Ty;

  case BuiltinType::UShort: return Importer.getToContext().UnsignedShortTy;
  case BuiltinType::UInt: return Importer.getToContext().UnsignedIntTy;
  case BuiltinType::ULong: return Importer.getToContext().UnsignedLongTy;
  case BuiltinType::ULongLong: 
    return Importer.getToContext().UnsignedLongLongTy;
  case BuiltinType::UInt128: return Importer.getToContext().UnsignedInt128Ty;
    
  case BuiltinType::Char_S:
    // The context we're importing from has an unsigned 'char'. If we're 
    // importing into a context with a signed 'char', translate to 
    // 'unsigned char' instead.
    if (!Importer.getToContext().getLangOptions().CharIsSigned)
      return Importer.getToContext().SignedCharTy;
    
    return Importer.getToContext().CharTy;

  case BuiltinType::SChar: return Importer.getToContext().SignedCharTy;
  case BuiltinType::WChar_S:
  case BuiltinType::WChar_U:
    // FIXME: If not in C++, shall we translate to the C equivalent of
    // wchar_t?
    return Importer.getToContext().WCharTy;
    
  case BuiltinType::Short : return Importer.getToContext().ShortTy;
  case BuiltinType::Int : return Importer.getToContext().IntTy;
  case BuiltinType::Long : return Importer.getToContext().LongTy;
  case BuiltinType::LongLong : return Importer.getToContext().LongLongTy;
  case BuiltinType::Int128 : return Importer.getToContext().Int128Ty;
  case BuiltinType::Float: return Importer.getToContext().FloatTy;
  case BuiltinType::Double: return Importer.getToContext().DoubleTy;
  case BuiltinType::LongDouble: return Importer.getToContext().LongDoubleTy;

  case BuiltinType::NullPtr:
    // FIXME: Make sure that the "to" context supports C++0x!
    return Importer.getToContext().NullPtrTy;
    
  case BuiltinType::Overload: return Importer.getToContext().OverloadTy;
  case BuiltinType::Dependent: return Importer.getToContext().DependentTy;
  case BuiltinType::UnknownAny: return Importer.getToContext().UnknownAnyTy;
  case BuiltinType::BoundMember: return Importer.getToContext().BoundMemberTy;

  case BuiltinType::ObjCId:
    // FIXME: Make sure that the "to" context supports Objective-C!
    return Importer.getToContext().ObjCBuiltinIdTy;
    
  case BuiltinType::ObjCClass:
    return Importer.getToContext().ObjCBuiltinClassTy;

  case BuiltinType::ObjCSel:
    return Importer.getToContext().ObjCBuiltinSelTy;
  }
  
  return QualType();
}

QualType ASTNodeImporter::VisitComplexType(const ComplexType *T) {
  QualType ToElementType = Importer.Import(T->getElementType());
  if (ToElementType.isNull())
    return QualType();
  
  return Importer.getToContext().getComplexType(ToElementType);
}

QualType ASTNodeImporter::VisitPointerType(const PointerType *T) {
  QualType ToPointeeType = Importer.Import(T->getPointeeType());
  if (ToPointeeType.isNull())
    return QualType();
  
  return Importer.getToContext().getPointerType(ToPointeeType);
}

QualType ASTNodeImporter::VisitBlockPointerType(const BlockPointerType *T) {
  // FIXME: Check for blocks support in "to" context.
  QualType ToPointeeType = Importer.Import(T->getPointeeType());
  if (ToPointeeType.isNull())
    return QualType();
  
  return Importer.getToContext().getBlockPointerType(ToPointeeType);
}

QualType
ASTNodeImporter::VisitLValueReferenceType(const LValueReferenceType *T) {
  // FIXME: Check for C++ support in "to" context.
  QualType ToPointeeType = Importer.Import(T->getPointeeTypeAsWritten());
  if (ToPointeeType.isNull())
    return QualType();
  
  return Importer.getToContext().getLValueReferenceType(ToPointeeType);
}

QualType
ASTNodeImporter::VisitRValueReferenceType(const RValueReferenceType *T) {
  // FIXME: Check for C++0x support in "to" context.
  QualType ToPointeeType = Importer.Import(T->getPointeeTypeAsWritten());
  if (ToPointeeType.isNull())
    return QualType();
  
  return Importer.getToContext().getRValueReferenceType(ToPointeeType);  
}

QualType ASTNodeImporter::VisitMemberPointerType(const MemberPointerType *T) {
  // FIXME: Check for C++ support in "to" context.
  QualType ToPointeeType = Importer.Import(T->getPointeeType());
  if (ToPointeeType.isNull())
    return QualType();
  
  QualType ClassType = Importer.Import(QualType(T->getClass(), 0));
  return Importer.getToContext().getMemberPointerType(ToPointeeType, 
                                                      ClassType.getTypePtr());
}

QualType ASTNodeImporter::VisitConstantArrayType(const ConstantArrayType *T) {
  QualType ToElementType = Importer.Import(T->getElementType());
  if (ToElementType.isNull())
    return QualType();
  
  return Importer.getToContext().getConstantArrayType(ToElementType, 
                                                      T->getSize(),
                                                      T->getSizeModifier(),
                                               T->getIndexTypeCVRQualifiers());
}

QualType
ASTNodeImporter::VisitIncompleteArrayType(const IncompleteArrayType *T) {
  QualType ToElementType = Importer.Import(T->getElementType());
  if (ToElementType.isNull())
    return QualType();
  
  return Importer.getToContext().getIncompleteArrayType(ToElementType, 
                                                        T->getSizeModifier(),
                                                T->getIndexTypeCVRQualifiers());
}

QualType ASTNodeImporter::VisitVariableArrayType(const VariableArrayType *T) {
  QualType ToElementType = Importer.Import(T->getElementType());
  if (ToElementType.isNull())
    return QualType();

  Expr *Size = Importer.Import(T->getSizeExpr());
  if (!Size)
    return QualType();
  
  SourceRange Brackets = Importer.Import(T->getBracketsRange());
  return Importer.getToContext().getVariableArrayType(ToElementType, Size,
                                                      T->getSizeModifier(),
                                                T->getIndexTypeCVRQualifiers(),
                                                      Brackets);
}

QualType ASTNodeImporter::VisitVectorType(const VectorType *T) {
  QualType ToElementType = Importer.Import(T->getElementType());
  if (ToElementType.isNull())
    return QualType();
  
  return Importer.getToContext().getVectorType(ToElementType, 
                                               T->getNumElements(),
                                               T->getVectorKind());
}

QualType ASTNodeImporter::VisitExtVectorType(const ExtVectorType *T) {
  QualType ToElementType = Importer.Import(T->getElementType());
  if (ToElementType.isNull())
    return QualType();
  
  return Importer.getToContext().getExtVectorType(ToElementType, 
                                                  T->getNumElements());
}

QualType
ASTNodeImporter::VisitFunctionNoProtoType(const FunctionNoProtoType *T) {
  // FIXME: What happens if we're importing a function without a prototype 
  // into C++? Should we make it variadic?
  QualType ToResultType = Importer.Import(T->getResultType());
  if (ToResultType.isNull())
    return QualType();

  return Importer.getToContext().getFunctionNoProtoType(ToResultType,
                                                        T->getExtInfo());
}

QualType ASTNodeImporter::VisitFunctionProtoType(const FunctionProtoType *T) {
  QualType ToResultType = Importer.Import(T->getResultType());
  if (ToResultType.isNull())
    return QualType();
  
  // Import argument types
  llvm::SmallVector<QualType, 4> ArgTypes;
  for (FunctionProtoType::arg_type_iterator A = T->arg_type_begin(),
                                         AEnd = T->arg_type_end();
       A != AEnd; ++A) {
    QualType ArgType = Importer.Import(*A);
    if (ArgType.isNull())
      return QualType();
    ArgTypes.push_back(ArgType);
  }
  
  // Import exception types
  llvm::SmallVector<QualType, 4> ExceptionTypes;
  for (FunctionProtoType::exception_iterator E = T->exception_begin(),
                                          EEnd = T->exception_end();
       E != EEnd; ++E) {
    QualType ExceptionType = Importer.Import(*E);
    if (ExceptionType.isNull())
      return QualType();
    ExceptionTypes.push_back(ExceptionType);
  }

  FunctionProtoType::ExtProtoInfo EPI = T->getExtProtoInfo();
  EPI.Exceptions = ExceptionTypes.data();
       
  return Importer.getToContext().getFunctionType(ToResultType, ArgTypes.data(),
                                                 ArgTypes.size(), EPI);
}

QualType ASTNodeImporter::VisitTypedefType(const TypedefType *T) {
  TypedefNameDecl *ToDecl
             = dyn_cast_or_null<TypedefNameDecl>(Importer.Import(T->getDecl()));
  if (!ToDecl)
    return QualType();
  
  return Importer.getToContext().getTypeDeclType(ToDecl);
}

QualType ASTNodeImporter::VisitTypeOfExprType(const TypeOfExprType *T) {
  Expr *ToExpr = Importer.Import(T->getUnderlyingExpr());
  if (!ToExpr)
    return QualType();
  
  return Importer.getToContext().getTypeOfExprType(ToExpr);
}

QualType ASTNodeImporter::VisitTypeOfType(const TypeOfType *T) {
  QualType ToUnderlyingType = Importer.Import(T->getUnderlyingType());
  if (ToUnderlyingType.isNull())
    return QualType();
  
  return Importer.getToContext().getTypeOfType(ToUnderlyingType);
}

QualType ASTNodeImporter::VisitDecltypeType(const DecltypeType *T) {
  // FIXME: Make sure that the "to" context supports C++0x!
  Expr *ToExpr = Importer.Import(T->getUnderlyingExpr());
  if (!ToExpr)
    return QualType();
  
  return Importer.getToContext().getDecltypeType(ToExpr);
}

QualType ASTNodeImporter::VisitUnaryTransformType(const UnaryTransformType *T) {
  QualType ToBaseType = Importer.Import(T->getBaseType());
  QualType ToUnderlyingType = Importer.Import(T->getUnderlyingType());
  if (ToBaseType.isNull() || ToUnderlyingType.isNull())
    return QualType();

  return Importer.getToContext().getUnaryTransformType(ToBaseType,
                                                       ToUnderlyingType,
                                                       T->getUTTKind());
}

QualType ASTNodeImporter::VisitAutoType(const AutoType *T) {
  // FIXME: Make sure that the "to" context supports C++0x!
  QualType FromDeduced = T->getDeducedType();
  QualType ToDeduced;
  if (!FromDeduced.isNull()) {
    ToDeduced = Importer.Import(FromDeduced);
    if (ToDeduced.isNull())
      return QualType();
  }
  
  return Importer.getToContext().getAutoType(ToDeduced);
}

QualType ASTNodeImporter::VisitRecordType(const RecordType *T) {
  RecordDecl *ToDecl
    = dyn_cast_or_null<RecordDecl>(Importer.Import(T->getDecl()));
  if (!ToDecl)
    return QualType();

  return Importer.getToContext().getTagDeclType(ToDecl);
}

QualType ASTNodeImporter::VisitEnumType(const EnumType *T) {
  EnumDecl *ToDecl
    = dyn_cast_or_null<EnumDecl>(Importer.Import(T->getDecl()));
  if (!ToDecl)
    return QualType();

  return Importer.getToContext().getTagDeclType(ToDecl);
}

QualType ASTNodeImporter::VisitTemplateSpecializationType(
                                       const TemplateSpecializationType *T) {
  TemplateName ToTemplate = Importer.Import(T->getTemplateName());
  if (ToTemplate.isNull())
    return QualType();
  
  llvm::SmallVector<TemplateArgument, 2> ToTemplateArgs;
  if (ImportTemplateArguments(T->getArgs(), T->getNumArgs(), ToTemplateArgs))
    return QualType();
  
  QualType ToCanonType;
  if (!QualType(T, 0).isCanonical()) {
    QualType FromCanonType 
      = Importer.getFromContext().getCanonicalType(QualType(T, 0));
    ToCanonType =Importer.Import(FromCanonType);
    if (ToCanonType.isNull())
      return QualType();
  }
  return Importer.getToContext().getTemplateSpecializationType(ToTemplate, 
                                                         ToTemplateArgs.data(), 
                                                         ToTemplateArgs.size(),
                                                               ToCanonType);
}

QualType ASTNodeImporter::VisitElaboratedType(const ElaboratedType *T) {
  NestedNameSpecifier *ToQualifier = 0;
  // Note: the qualifier in an ElaboratedType is optional.
  if (T->getQualifier()) {
    ToQualifier = Importer.Import(T->getQualifier());
    if (!ToQualifier)
      return QualType();
  }

  QualType ToNamedType = Importer.Import(T->getNamedType());
  if (ToNamedType.isNull())
    return QualType();

  return Importer.getToContext().getElaboratedType(T->getKeyword(),
                                                   ToQualifier, ToNamedType);
}

QualType ASTNodeImporter::VisitObjCInterfaceType(const ObjCInterfaceType *T) {
  ObjCInterfaceDecl *Class
    = dyn_cast_or_null<ObjCInterfaceDecl>(Importer.Import(T->getDecl()));
  if (!Class)
    return QualType();

  return Importer.getToContext().getObjCInterfaceType(Class);
}

QualType ASTNodeImporter::VisitObjCObjectType(const ObjCObjectType *T) {
  QualType ToBaseType = Importer.Import(T->getBaseType());
  if (ToBaseType.isNull())
    return QualType();

  llvm::SmallVector<ObjCProtocolDecl *, 4> Protocols;
  for (ObjCObjectType::qual_iterator P = T->qual_begin(), 
                                     PEnd = T->qual_end();
       P != PEnd; ++P) {
    ObjCProtocolDecl *Protocol
      = dyn_cast_or_null<ObjCProtocolDecl>(Importer.Import(*P));
    if (!Protocol)
      return QualType();
    Protocols.push_back(Protocol);
  }

  return Importer.getToContext().getObjCObjectType(ToBaseType,
                                                   Protocols.data(),
                                                   Protocols.size());
}

QualType
ASTNodeImporter::VisitObjCObjectPointerType(const ObjCObjectPointerType *T) {
  QualType ToPointeeType = Importer.Import(T->getPointeeType());
  if (ToPointeeType.isNull())
    return QualType();

  return Importer.getToContext().getObjCObjectPointerType(ToPointeeType);
}

//----------------------------------------------------------------------------
// Import Declarations
//----------------------------------------------------------------------------
bool ASTNodeImporter::ImportDeclParts(NamedDecl *D, DeclContext *&DC, 
                                      DeclContext *&LexicalDC, 
                                      DeclarationName &Name, 
                                      SourceLocation &Loc) {
  // Import the context of this declaration.
  DC = Importer.ImportContext(D->getDeclContext());
  if (!DC)
    return true;
  
  LexicalDC = DC;
  if (D->getDeclContext() != D->getLexicalDeclContext()) {
    LexicalDC = Importer.ImportContext(D->getLexicalDeclContext());
    if (!LexicalDC)
      return true;
  }
  
  // Import the name of this declaration.
  Name = Importer.Import(D->getDeclName());
  if (D->getDeclName() && !Name)
    return true;
  
  // Import the location of this declaration.
  Loc = Importer.Import(D->getLocation());
  return false;
}

void
ASTNodeImporter::ImportDeclarationNameLoc(const DeclarationNameInfo &From,
                                          DeclarationNameInfo& To) {
  // NOTE: To.Name and To.Loc are already imported.
  // We only have to import To.LocInfo.
  switch (To.getName().getNameKind()) {
  case DeclarationName::Identifier:
  case DeclarationName::ObjCZeroArgSelector:
  case DeclarationName::ObjCOneArgSelector:
  case DeclarationName::ObjCMultiArgSelector:
  case DeclarationName::CXXUsingDirective:
    return;

  case DeclarationName::CXXOperatorName: {
    SourceRange Range = From.getCXXOperatorNameRange();
    To.setCXXOperatorNameRange(Importer.Import(Range));
    return;
  }
  case DeclarationName::CXXLiteralOperatorName: {
    SourceLocation Loc = From.getCXXLiteralOperatorNameLoc();
    To.setCXXLiteralOperatorNameLoc(Importer.Import(Loc));
    return;
  }
  case DeclarationName::CXXConstructorName:
  case DeclarationName::CXXDestructorName:
  case DeclarationName::CXXConversionFunctionName: {
    TypeSourceInfo *FromTInfo = From.getNamedTypeInfo();
    To.setNamedTypeInfo(Importer.Import(FromTInfo));
    return;
  }
    assert(0 && "Unknown name kind.");
  }
}

void ASTNodeImporter::ImportDeclContext(DeclContext *FromDC, bool ForceImport) {
  if (Importer.isMinimalImport() && !ForceImport) {
    if (DeclContext *ToDC = Importer.ImportContext(FromDC)) {
      ToDC->setHasExternalLexicalStorage();
      ToDC->setHasExternalVisibleStorage();
    }
    return;
  }
  
  for (DeclContext::decl_iterator From = FromDC->decls_begin(),
                               FromEnd = FromDC->decls_end();
       From != FromEnd;
       ++From)
    Importer.Import(*From);
}

bool ASTNodeImporter::ImportDefinition(RecordDecl *From, RecordDecl *To) {
  if (To->getDefinition())
    return false;
  
  To->startDefinition();
  
  // Add base classes.
  if (CXXRecordDecl *ToCXX = dyn_cast<CXXRecordDecl>(To)) {
    CXXRecordDecl *FromCXX = cast<CXXRecordDecl>(From);
    
    llvm::SmallVector<CXXBaseSpecifier *, 4> Bases;
    for (CXXRecordDecl::base_class_iterator 
                  Base1 = FromCXX->bases_begin(),
            FromBaseEnd = FromCXX->bases_end();
         Base1 != FromBaseEnd;
         ++Base1) {
      QualType T = Importer.Import(Base1->getType());
      if (T.isNull())
        return true;

      SourceLocation EllipsisLoc;
      if (Base1->isPackExpansion())
        EllipsisLoc = Importer.Import(Base1->getEllipsisLoc());
      
      Bases.push_back(
                    new (Importer.getToContext()) 
                      CXXBaseSpecifier(Importer.Import(Base1->getSourceRange()),
                                       Base1->isVirtual(),
                                       Base1->isBaseOfClass(),
                                       Base1->getAccessSpecifierAsWritten(),
                                   Importer.Import(Base1->getTypeSourceInfo()),
                                       EllipsisLoc));
    }
    if (!Bases.empty())
      ToCXX->setBases(Bases.data(), Bases.size());
  }
  
  ImportDeclContext(From);
  To->completeDefinition();
  return false;
}

TemplateParameterList *ASTNodeImporter::ImportTemplateParameterList(
                                                TemplateParameterList *Params) {
  llvm::SmallVector<NamedDecl *, 4> ToParams;
  ToParams.reserve(Params->size());
  for (TemplateParameterList::iterator P = Params->begin(), 
                                    PEnd = Params->end();
       P != PEnd; ++P) {
    Decl *To = Importer.Import(*P);
    if (!To)
      return 0;
    
    ToParams.push_back(cast<NamedDecl>(To));
  }
  
  return TemplateParameterList::Create(Importer.getToContext(),
                                       Importer.Import(Params->getTemplateLoc()),
                                       Importer.Import(Params->getLAngleLoc()),
                                       ToParams.data(), ToParams.size(),
                                       Importer.Import(Params->getRAngleLoc()));
}

TemplateArgument 
ASTNodeImporter::ImportTemplateArgument(const TemplateArgument &From) {
  switch (From.getKind()) {
  case TemplateArgument::Null:
    return TemplateArgument();
     
  case TemplateArgument::Type: {
    QualType ToType = Importer.Import(From.getAsType());
    if (ToType.isNull())
      return TemplateArgument();
    return TemplateArgument(ToType);
  }
      
  case TemplateArgument::Integral: {
    QualType ToType = Importer.Import(From.getIntegralType());
    if (ToType.isNull())
      return TemplateArgument();
    return TemplateArgument(*From.getAsIntegral(), ToType);
  }

  case TemplateArgument::Declaration:
    if (Decl *To = Importer.Import(From.getAsDecl()))
      return TemplateArgument(To);
    return TemplateArgument();
      
  case TemplateArgument::Template: {
    TemplateName ToTemplate = Importer.Import(From.getAsTemplate());
    if (ToTemplate.isNull())
      return TemplateArgument();
    
    return TemplateArgument(ToTemplate);
  }

  case TemplateArgument::TemplateExpansion: {
    TemplateName ToTemplate 
      = Importer.Import(From.getAsTemplateOrTemplatePattern());
    if (ToTemplate.isNull())
      return TemplateArgument();
    
    return TemplateArgument(ToTemplate, From.getNumTemplateExpansions());
  }

  case TemplateArgument::Expression:
    if (Expr *ToExpr = Importer.Import(From.getAsExpr()))
      return TemplateArgument(ToExpr);
    return TemplateArgument();
      
  case TemplateArgument::Pack: {
    llvm::SmallVector<TemplateArgument, 2> ToPack;
    ToPack.reserve(From.pack_size());
    if (ImportTemplateArguments(From.pack_begin(), From.pack_size(), ToPack))
      return TemplateArgument();
    
    TemplateArgument *ToArgs 
      = new (Importer.getToContext()) TemplateArgument[ToPack.size()];
    std::copy(ToPack.begin(), ToPack.end(), ToArgs);
    return TemplateArgument(ToArgs, ToPack.size());
  }
  }
  
  llvm_unreachable("Invalid template argument kind");
  return TemplateArgument();
}

bool ASTNodeImporter::ImportTemplateArguments(const TemplateArgument *FromArgs,
                                              unsigned NumFromArgs,
                              llvm::SmallVectorImpl<TemplateArgument> &ToArgs) {
  for (unsigned I = 0; I != NumFromArgs; ++I) {
    TemplateArgument To = ImportTemplateArgument(FromArgs[I]);
    if (To.isNull() && !FromArgs[I].isNull())
      return true;
    
    ToArgs.push_back(To);
  }
  
  return false;
}

bool ASTNodeImporter::IsStructuralMatch(RecordDecl *FromRecord, 
                                        RecordDecl *ToRecord) {
  StructuralEquivalenceContext Ctx(Importer.getFromContext(),
                                   Importer.getToContext(),
                                   Importer.getNonEquivalentDecls());
  return Ctx.IsStructurallyEquivalent(FromRecord, ToRecord);
}

bool ASTNodeImporter::IsStructuralMatch(EnumDecl *FromEnum, EnumDecl *ToEnum) {
  StructuralEquivalenceContext Ctx(Importer.getFromContext(),
                                   Importer.getToContext(),
                                   Importer.getNonEquivalentDecls());
  return Ctx.IsStructurallyEquivalent(FromEnum, ToEnum);
}

bool ASTNodeImporter::IsStructuralMatch(ClassTemplateDecl *From, 
                                        ClassTemplateDecl *To) {
  StructuralEquivalenceContext Ctx(Importer.getFromContext(),
                                   Importer.getToContext(),
                                   Importer.getNonEquivalentDecls());
  return Ctx.IsStructurallyEquivalent(From, To);  
}

Decl *ASTNodeImporter::VisitDecl(Decl *D) {
  Importer.FromDiag(D->getLocation(), diag::err_unsupported_ast_node)
    << D->getDeclKindName();
  return 0;
}

Decl *ASTNodeImporter::VisitNamespaceDecl(NamespaceDecl *D) {
  // Import the major distinguishing characteristics of this namespace.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  if (ImportDeclParts(D, DC, LexicalDC, Name, Loc))
    return 0;
  
  NamespaceDecl *MergeWithNamespace = 0;
  if (!Name) {
    // This is an anonymous namespace. Adopt an existing anonymous
    // namespace if we can.
    // FIXME: Not testable.
    if (TranslationUnitDecl *TU = dyn_cast<TranslationUnitDecl>(DC))
      MergeWithNamespace = TU->getAnonymousNamespace();
    else
      MergeWithNamespace = cast<NamespaceDecl>(DC)->getAnonymousNamespace();
  } else {
    llvm::SmallVector<NamedDecl *, 4> ConflictingDecls;
    for (DeclContext::lookup_result Lookup = DC->lookup(Name);
         Lookup.first != Lookup.second; 
         ++Lookup.first) {
      if (!(*Lookup.first)->isInIdentifierNamespace(Decl::IDNS_Namespace))
        continue;
      
      if (NamespaceDecl *FoundNS = dyn_cast<NamespaceDecl>(*Lookup.first)) {
        MergeWithNamespace = FoundNS;
        ConflictingDecls.clear();
        break;
      }
      
      ConflictingDecls.push_back(*Lookup.first);
    }
    
    if (!ConflictingDecls.empty()) {
      Name = Importer.HandleNameConflict(Name, DC, Decl::IDNS_Namespace,
                                         ConflictingDecls.data(), 
                                         ConflictingDecls.size());
    }
  }
  
  // Create the "to" namespace, if needed.
  NamespaceDecl *ToNamespace = MergeWithNamespace;
  if (!ToNamespace) {
    ToNamespace = NamespaceDecl::Create(Importer.getToContext(), DC,
                                        Importer.Import(D->getLocStart()),
                                        Loc, Name.getAsIdentifierInfo());
    ToNamespace->setLexicalDeclContext(LexicalDC);
    LexicalDC->addDecl(ToNamespace);
    
    // If this is an anonymous namespace, register it as the anonymous
    // namespace within its context.
    if (!Name) {
      if (TranslationUnitDecl *TU = dyn_cast<TranslationUnitDecl>(DC))
        TU->setAnonymousNamespace(ToNamespace);
      else
        cast<NamespaceDecl>(DC)->setAnonymousNamespace(ToNamespace);
    }
  }
  Importer.Imported(D, ToNamespace);
  
  ImportDeclContext(D);
  
  return ToNamespace;
}

Decl *ASTNodeImporter::VisitTypedefNameDecl(TypedefNameDecl *D, bool IsAlias) {
  // Import the major distinguishing characteristics of this typedef.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  if (ImportDeclParts(D, DC, LexicalDC, Name, Loc))
    return 0;
  
  // If this typedef is not in block scope, determine whether we've
  // seen a typedef with the same name (that we can merge with) or any
  // other entity by that name (which name lookup could conflict with).
  if (!DC->isFunctionOrMethod()) {
    llvm::SmallVector<NamedDecl *, 4> ConflictingDecls;
    unsigned IDNS = Decl::IDNS_Ordinary;
    for (DeclContext::lookup_result Lookup = DC->lookup(Name);
         Lookup.first != Lookup.second; 
         ++Lookup.first) {
      if (!(*Lookup.first)->isInIdentifierNamespace(IDNS))
        continue;
      if (TypedefNameDecl *FoundTypedef =
            dyn_cast<TypedefNameDecl>(*Lookup.first)) {
        if (Importer.IsStructurallyEquivalent(D->getUnderlyingType(),
                                            FoundTypedef->getUnderlyingType()))
          return Importer.Imported(D, FoundTypedef);
      }
      
      ConflictingDecls.push_back(*Lookup.first);
    }
    
    if (!ConflictingDecls.empty()) {
      Name = Importer.HandleNameConflict(Name, DC, IDNS,
                                         ConflictingDecls.data(), 
                                         ConflictingDecls.size());
      if (!Name)
        return 0;
    }
  }
  
  // Import the underlying type of this typedef;
  QualType T = Importer.Import(D->getUnderlyingType());
  if (T.isNull())
    return 0;
  
  // Create the new typedef node.
  TypeSourceInfo *TInfo = Importer.Import(D->getTypeSourceInfo());
  SourceLocation StartL = Importer.Import(D->getLocStart());
  TypedefNameDecl *ToTypedef;
  if (IsAlias)
    ToTypedef = TypedefDecl::Create(Importer.getToContext(), DC,
                                    StartL, Loc,
                                    Name.getAsIdentifierInfo(),
                                    TInfo);
  else
    ToTypedef = TypeAliasDecl::Create(Importer.getToContext(), DC,
                                  StartL, Loc,
                                  Name.getAsIdentifierInfo(),
                                  TInfo);
  ToTypedef->setAccess(D->getAccess());
  ToTypedef->setLexicalDeclContext(LexicalDC);
  Importer.Imported(D, ToTypedef);
  LexicalDC->addDecl(ToTypedef);
  
  return ToTypedef;
}

Decl *ASTNodeImporter::VisitTypedefDecl(TypedefDecl *D) {
  return VisitTypedefNameDecl(D, /*IsAlias=*/false);
}

Decl *ASTNodeImporter::VisitTypeAliasDecl(TypeAliasDecl *D) {
  return VisitTypedefNameDecl(D, /*IsAlias=*/true);
}

Decl *ASTNodeImporter::VisitEnumDecl(EnumDecl *D) {
  // Import the major distinguishing characteristics of this enum.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  if (ImportDeclParts(D, DC, LexicalDC, Name, Loc))
    return 0;
  
  // Figure out what enum name we're looking for.
  unsigned IDNS = Decl::IDNS_Tag;
  DeclarationName SearchName = Name;
  if (!SearchName && D->getTypedefNameForAnonDecl()) {
    SearchName = Importer.Import(D->getTypedefNameForAnonDecl()->getDeclName());
    IDNS = Decl::IDNS_Ordinary;
  } else if (Importer.getToContext().getLangOptions().CPlusPlus)
    IDNS |= Decl::IDNS_Ordinary;
  
  // We may already have an enum of the same name; try to find and match it.
  if (!DC->isFunctionOrMethod() && SearchName) {
    llvm::SmallVector<NamedDecl *, 4> ConflictingDecls;
    for (DeclContext::lookup_result Lookup = DC->lookup(Name);
         Lookup.first != Lookup.second; 
         ++Lookup.first) {
      if (!(*Lookup.first)->isInIdentifierNamespace(IDNS))
        continue;
      
      Decl *Found = *Lookup.first;
      if (TypedefNameDecl *Typedef = dyn_cast<TypedefNameDecl>(Found)) {
        if (const TagType *Tag = Typedef->getUnderlyingType()->getAs<TagType>())
          Found = Tag->getDecl();
      }
      
      if (EnumDecl *FoundEnum = dyn_cast<EnumDecl>(Found)) {
        if (IsStructuralMatch(D, FoundEnum))
          return Importer.Imported(D, FoundEnum);
      }
      
      ConflictingDecls.push_back(*Lookup.first);
    }
    
    if (!ConflictingDecls.empty()) {
      Name = Importer.HandleNameConflict(Name, DC, IDNS,
                                         ConflictingDecls.data(), 
                                         ConflictingDecls.size());
    }
  }
  
  // Create the enum declaration.
  EnumDecl *D2 = EnumDecl::Create(Importer.getToContext(), DC,
                                  Importer.Import(D->getLocStart()),
                                  Loc, Name.getAsIdentifierInfo(), 0,
                                  D->isScoped(), D->isScopedUsingClassTag(),
                                  D->isFixed());
  // Import the qualifier, if any.
  D2->setQualifierInfo(Importer.Import(D->getQualifierLoc()));
  D2->setAccess(D->getAccess());
  D2->setLexicalDeclContext(LexicalDC);
  Importer.Imported(D, D2);
  LexicalDC->addDecl(D2);

  // Import the integer type.
  QualType ToIntegerType = Importer.Import(D->getIntegerType());
  if (ToIntegerType.isNull())
    return 0;
  D2->setIntegerType(ToIntegerType);
  
  // Import the definition
  if (D->isDefinition()) {
    QualType T = Importer.Import(Importer.getFromContext().getTypeDeclType(D));
    if (T.isNull())
      return 0;

    QualType ToPromotionType = Importer.Import(D->getPromotionType());
    if (ToPromotionType.isNull())
      return 0;
    
    D2->startDefinition();
    ImportDeclContext(D);

    // FIXME: we might need to merge the number of positive or negative bits
    // if the enumerator lists don't match.
    D2->completeDefinition(T, ToPromotionType,
                           D->getNumPositiveBits(),
                           D->getNumNegativeBits());
  }
  
  return D2;
}

Decl *ASTNodeImporter::VisitRecordDecl(RecordDecl *D) {
  // If this record has a definition in the translation unit we're coming from,
  // but this particular declaration is not that definition, import the
  // definition and map to that.
  TagDecl *Definition = D->getDefinition();
  if (Definition && Definition != D) {
    Decl *ImportedDef = Importer.Import(Definition);
    if (!ImportedDef)
      return 0;
    
    return Importer.Imported(D, ImportedDef);
  }
  
  // Import the major distinguishing characteristics of this record.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  if (ImportDeclParts(D, DC, LexicalDC, Name, Loc))
    return 0;
      
  // Figure out what structure name we're looking for.
  unsigned IDNS = Decl::IDNS_Tag;
  DeclarationName SearchName = Name;
  if (!SearchName && D->getTypedefNameForAnonDecl()) {
    SearchName = Importer.Import(D->getTypedefNameForAnonDecl()->getDeclName());
    IDNS = Decl::IDNS_Ordinary;
  } else if (Importer.getToContext().getLangOptions().CPlusPlus)
    IDNS |= Decl::IDNS_Ordinary;

  // We may already have a record of the same name; try to find and match it.
  RecordDecl *AdoptDecl = 0;
  if (!DC->isFunctionOrMethod() && SearchName) {
    llvm::SmallVector<NamedDecl *, 4> ConflictingDecls;
    for (DeclContext::lookup_result Lookup = DC->lookup(Name);
         Lookup.first != Lookup.second; 
         ++Lookup.first) {
      if (!(*Lookup.first)->isInIdentifierNamespace(IDNS))
        continue;
      
      Decl *Found = *Lookup.first;
      if (TypedefNameDecl *Typedef = dyn_cast<TypedefNameDecl>(Found)) {
        if (const TagType *Tag = Typedef->getUnderlyingType()->getAs<TagType>())
          Found = Tag->getDecl();
      }
      
      if (RecordDecl *FoundRecord = dyn_cast<RecordDecl>(Found)) {
        if (RecordDecl *FoundDef = FoundRecord->getDefinition()) {
          if (!D->isDefinition() || IsStructuralMatch(D, FoundDef)) {
            // The record types structurally match, or the "from" translation
            // unit only had a forward declaration anyway; call it the same
            // function.
            // FIXME: For C++, we should also merge methods here.
            return Importer.Imported(D, FoundDef);
          }
        } else {
          // We have a forward declaration of this type, so adopt that forward
          // declaration rather than building a new one.
          AdoptDecl = FoundRecord;
          continue;
        }          
      }
      
      ConflictingDecls.push_back(*Lookup.first);
    }
    
    if (!ConflictingDecls.empty()) {
      Name = Importer.HandleNameConflict(Name, DC, IDNS,
                                         ConflictingDecls.data(), 
                                         ConflictingDecls.size());
    }
  }
  
  // Create the record declaration.
  RecordDecl *D2 = AdoptDecl;
  SourceLocation StartLoc = Importer.Import(D->getLocStart());
  if (!D2) {
    if (isa<CXXRecordDecl>(D)) {
      CXXRecordDecl *D2CXX = CXXRecordDecl::Create(Importer.getToContext(), 
                                                   D->getTagKind(),
                                                   DC, StartLoc, Loc,
                                                   Name.getAsIdentifierInfo());
      D2 = D2CXX;
      D2->setAccess(D->getAccess());
    } else {
      D2 = RecordDecl::Create(Importer.getToContext(), D->getTagKind(),
                              DC, StartLoc, Loc, Name.getAsIdentifierInfo());
    }
    
    D2->setQualifierInfo(Importer.Import(D->getQualifierLoc()));
    D2->setLexicalDeclContext(LexicalDC);
    LexicalDC->addDecl(D2);
  }
  
  Importer.Imported(D, D2);

  if (D->isDefinition() && ImportDefinition(D, D2))
    return 0;
  
  return D2;
}

Decl *ASTNodeImporter::VisitEnumConstantDecl(EnumConstantDecl *D) {
  // Import the major distinguishing characteristics of this enumerator.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  if (ImportDeclParts(D, DC, LexicalDC, Name, Loc))
    return 0;

  QualType T = Importer.Import(D->getType());
  if (T.isNull())
    return 0;

  // Determine whether there are any other declarations with the same name and 
  // in the same context.
  if (!LexicalDC->isFunctionOrMethod()) {
    llvm::SmallVector<NamedDecl *, 4> ConflictingDecls;
    unsigned IDNS = Decl::IDNS_Ordinary;
    for (DeclContext::lookup_result Lookup = DC->lookup(Name);
         Lookup.first != Lookup.second; 
         ++Lookup.first) {
      if (!(*Lookup.first)->isInIdentifierNamespace(IDNS))
        continue;
      
      ConflictingDecls.push_back(*Lookup.first);
    }
    
    if (!ConflictingDecls.empty()) {
      Name = Importer.HandleNameConflict(Name, DC, IDNS,
                                         ConflictingDecls.data(), 
                                         ConflictingDecls.size());
      if (!Name)
        return 0;
    }
  }
  
  Expr *Init = Importer.Import(D->getInitExpr());
  if (D->getInitExpr() && !Init)
    return 0;
  
  EnumConstantDecl *ToEnumerator
    = EnumConstantDecl::Create(Importer.getToContext(), cast<EnumDecl>(DC), Loc, 
                               Name.getAsIdentifierInfo(), T, 
                               Init, D->getInitVal());
  ToEnumerator->setAccess(D->getAccess());
  ToEnumerator->setLexicalDeclContext(LexicalDC);
  Importer.Imported(D, ToEnumerator);
  LexicalDC->addDecl(ToEnumerator);
  return ToEnumerator;
}

Decl *ASTNodeImporter::VisitFunctionDecl(FunctionDecl *D) {
  // Import the major distinguishing characteristics of this function.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  if (ImportDeclParts(D, DC, LexicalDC, Name, Loc))
    return 0;

  // Try to find a function in our own ("to") context with the same name, same
  // type, and in the same context as the function we're importing.
  if (!LexicalDC->isFunctionOrMethod()) {
    llvm::SmallVector<NamedDecl *, 4> ConflictingDecls;
    unsigned IDNS = Decl::IDNS_Ordinary;
    for (DeclContext::lookup_result Lookup = DC->lookup(Name);
         Lookup.first != Lookup.second; 
         ++Lookup.first) {
      if (!(*Lookup.first)->isInIdentifierNamespace(IDNS))
        continue;
    
      if (FunctionDecl *FoundFunction = dyn_cast<FunctionDecl>(*Lookup.first)) {
        if (isExternalLinkage(FoundFunction->getLinkage()) &&
            isExternalLinkage(D->getLinkage())) {
          if (Importer.IsStructurallyEquivalent(D->getType(), 
                                                FoundFunction->getType())) {
            // FIXME: Actually try to merge the body and other attributes.
            return Importer.Imported(D, FoundFunction);
          }
        
          // FIXME: Check for overloading more carefully, e.g., by boosting
          // Sema::IsOverload out to the AST library.
          
          // Function overloading is okay in C++.
          if (Importer.getToContext().getLangOptions().CPlusPlus)
            continue;
          
          // Complain about inconsistent function types.
          Importer.ToDiag(Loc, diag::err_odr_function_type_inconsistent)
            << Name << D->getType() << FoundFunction->getType();
          Importer.ToDiag(FoundFunction->getLocation(), 
                          diag::note_odr_value_here)
            << FoundFunction->getType();
        }
      }
      
      ConflictingDecls.push_back(*Lookup.first);
    }
    
    if (!ConflictingDecls.empty()) {
      Name = Importer.HandleNameConflict(Name, DC, IDNS,
                                         ConflictingDecls.data(), 
                                         ConflictingDecls.size());
      if (!Name)
        return 0;
    }    
  }

  DeclarationNameInfo NameInfo(Name, Loc);
  // Import additional name location/type info.
  ImportDeclarationNameLoc(D->getNameInfo(), NameInfo);

  // Import the type.
  QualType T = Importer.Import(D->getType());
  if (T.isNull())
    return 0;
  
  // Import the function parameters.
  llvm::SmallVector<ParmVarDecl *, 8> Parameters;
  for (FunctionDecl::param_iterator P = D->param_begin(), PEnd = D->param_end();
       P != PEnd; ++P) {
    ParmVarDecl *ToP = cast_or_null<ParmVarDecl>(Importer.Import(*P));
    if (!ToP)
      return 0;
    
    Parameters.push_back(ToP);
  }
  
  // Create the imported function.
  TypeSourceInfo *TInfo = Importer.Import(D->getTypeSourceInfo());
  FunctionDecl *ToFunction = 0;
  if (CXXConstructorDecl *FromConstructor = dyn_cast<CXXConstructorDecl>(D)) {
    ToFunction = CXXConstructorDecl::Create(Importer.getToContext(),
                                            cast<CXXRecordDecl>(DC),
                                            D->getInnerLocStart(),
                                            NameInfo, T, TInfo, 
                                            FromConstructor->isExplicit(),
                                            D->isInlineSpecified(), 
                                            D->isImplicit());
  } else if (isa<CXXDestructorDecl>(D)) {
    ToFunction = CXXDestructorDecl::Create(Importer.getToContext(),
                                           cast<CXXRecordDecl>(DC),
                                           D->getInnerLocStart(),
                                           NameInfo, T, TInfo,
                                           D->isInlineSpecified(),
                                           D->isImplicit());
  } else if (CXXConversionDecl *FromConversion
                                           = dyn_cast<CXXConversionDecl>(D)) {
    ToFunction = CXXConversionDecl::Create(Importer.getToContext(), 
                                           cast<CXXRecordDecl>(DC),
                                           D->getInnerLocStart(),
                                           NameInfo, T, TInfo,
                                           D->isInlineSpecified(),
                                           FromConversion->isExplicit(),
                                           Importer.Import(D->getLocEnd()));
  } else if (CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(D)) {
    ToFunction = CXXMethodDecl::Create(Importer.getToContext(), 
                                       cast<CXXRecordDecl>(DC),
                                       D->getInnerLocStart(),
                                       NameInfo, T, TInfo,
                                       Method->isStatic(),
                                       Method->getStorageClassAsWritten(),
                                       Method->isInlineSpecified(),
                                       Importer.Import(D->getLocEnd()));
  } else {
    ToFunction = FunctionDecl::Create(Importer.getToContext(), DC,
                                      D->getInnerLocStart(),
                                      NameInfo, T, TInfo, D->getStorageClass(),
                                      D->getStorageClassAsWritten(),
                                      D->isInlineSpecified(),
                                      D->hasWrittenPrototype());
  }

  // Import the qualifier, if any.
  ToFunction->setQualifierInfo(Importer.Import(D->getQualifierLoc()));
  ToFunction->setAccess(D->getAccess());
  ToFunction->setLexicalDeclContext(LexicalDC);
  ToFunction->setVirtualAsWritten(D->isVirtualAsWritten());
  ToFunction->setTrivial(D->isTrivial());
  ToFunction->setPure(D->isPure());
  Importer.Imported(D, ToFunction);

  // Set the parameters.
  for (unsigned I = 0, N = Parameters.size(); I != N; ++I) {
    Parameters[I]->setOwningFunction(ToFunction);
    ToFunction->addDecl(Parameters[I]);
  }
  ToFunction->setParams(Parameters.data(), Parameters.size());

  // FIXME: Other bits to merge?

  // Add this function to the lexical context.
  LexicalDC->addDecl(ToFunction);

  return ToFunction;
}

Decl *ASTNodeImporter::VisitCXXMethodDecl(CXXMethodDecl *D) {
  return VisitFunctionDecl(D);
}

Decl *ASTNodeImporter::VisitCXXConstructorDecl(CXXConstructorDecl *D) {
  return VisitCXXMethodDecl(D);
}

Decl *ASTNodeImporter::VisitCXXDestructorDecl(CXXDestructorDecl *D) {
  return VisitCXXMethodDecl(D);
}

Decl *ASTNodeImporter::VisitCXXConversionDecl(CXXConversionDecl *D) {
  return VisitCXXMethodDecl(D);
}

Decl *ASTNodeImporter::VisitFieldDecl(FieldDecl *D) {
  // Import the major distinguishing characteristics of a variable.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  if (ImportDeclParts(D, DC, LexicalDC, Name, Loc))
    return 0;
  
  // Import the type.
  QualType T = Importer.Import(D->getType());
  if (T.isNull())
    return 0;
  
  TypeSourceInfo *TInfo = Importer.Import(D->getTypeSourceInfo());
  Expr *BitWidth = Importer.Import(D->getBitWidth());
  if (!BitWidth && D->getBitWidth())
    return 0;
  
  FieldDecl *ToField = FieldDecl::Create(Importer.getToContext(), DC,
                                         Importer.Import(D->getInnerLocStart()),
                                         Loc, Name.getAsIdentifierInfo(),
                                         T, TInfo, BitWidth, D->isMutable(),
                                         D->hasInClassInitializer());
  ToField->setAccess(D->getAccess());
  ToField->setLexicalDeclContext(LexicalDC);
  if (ToField->hasInClassInitializer())
    ToField->setInClassInitializer(D->getInClassInitializer());
  Importer.Imported(D, ToField);
  LexicalDC->addDecl(ToField);
  return ToField;
}

Decl *ASTNodeImporter::VisitIndirectFieldDecl(IndirectFieldDecl *D) {
  // Import the major distinguishing characteristics of a variable.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  if (ImportDeclParts(D, DC, LexicalDC, Name, Loc))
    return 0;

  // Import the type.
  QualType T = Importer.Import(D->getType());
  if (T.isNull())
    return 0;

  NamedDecl **NamedChain =
    new (Importer.getToContext())NamedDecl*[D->getChainingSize()];

  unsigned i = 0;
  for (IndirectFieldDecl::chain_iterator PI = D->chain_begin(),
       PE = D->chain_end(); PI != PE; ++PI) {
    Decl* D = Importer.Import(*PI);
    if (!D)
      return 0;
    NamedChain[i++] = cast<NamedDecl>(D);
  }

  IndirectFieldDecl *ToIndirectField = IndirectFieldDecl::Create(
                                         Importer.getToContext(), DC,
                                         Loc, Name.getAsIdentifierInfo(), T,
                                         NamedChain, D->getChainingSize());
  ToIndirectField->setAccess(D->getAccess());
  ToIndirectField->setLexicalDeclContext(LexicalDC);
  Importer.Imported(D, ToIndirectField);
  LexicalDC->addDecl(ToIndirectField);
  return ToIndirectField;
}

Decl *ASTNodeImporter::VisitObjCIvarDecl(ObjCIvarDecl *D) {
  // Import the major distinguishing characteristics of an ivar.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  if (ImportDeclParts(D, DC, LexicalDC, Name, Loc))
    return 0;
  
  // Determine whether we've already imported this ivar 
  for (DeclContext::lookup_result Lookup = DC->lookup(Name);
       Lookup.first != Lookup.second; 
       ++Lookup.first) {
    if (ObjCIvarDecl *FoundIvar = dyn_cast<ObjCIvarDecl>(*Lookup.first)) {
      if (Importer.IsStructurallyEquivalent(D->getType(), 
                                            FoundIvar->getType())) {
        Importer.Imported(D, FoundIvar);
        return FoundIvar;
      }

      Importer.ToDiag(Loc, diag::err_odr_ivar_type_inconsistent)
        << Name << D->getType() << FoundIvar->getType();
      Importer.ToDiag(FoundIvar->getLocation(), diag::note_odr_value_here)
        << FoundIvar->getType();
      return 0;
    }
  }

  // Import the type.
  QualType T = Importer.Import(D->getType());
  if (T.isNull())
    return 0;
  
  TypeSourceInfo *TInfo = Importer.Import(D->getTypeSourceInfo());
  Expr *BitWidth = Importer.Import(D->getBitWidth());
  if (!BitWidth && D->getBitWidth())
    return 0;
  
  ObjCIvarDecl *ToIvar = ObjCIvarDecl::Create(Importer.getToContext(),
                                              cast<ObjCContainerDecl>(DC),
                                       Importer.Import(D->getInnerLocStart()),
                                              Loc, Name.getAsIdentifierInfo(),
                                              T, TInfo, D->getAccessControl(),
                                              BitWidth, D->getSynthesize());
  ToIvar->setLexicalDeclContext(LexicalDC);
  Importer.Imported(D, ToIvar);
  LexicalDC->addDecl(ToIvar);
  return ToIvar;
  
}

Decl *ASTNodeImporter::VisitVarDecl(VarDecl *D) {
  // Import the major distinguishing characteristics of a variable.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  if (ImportDeclParts(D, DC, LexicalDC, Name, Loc))
    return 0;
  
  // Try to find a variable in our own ("to") context with the same name and
  // in the same context as the variable we're importing.
  if (D->isFileVarDecl()) {
    VarDecl *MergeWithVar = 0;
    llvm::SmallVector<NamedDecl *, 4> ConflictingDecls;
    unsigned IDNS = Decl::IDNS_Ordinary;
    for (DeclContext::lookup_result Lookup = DC->lookup(Name);
         Lookup.first != Lookup.second; 
         ++Lookup.first) {
      if (!(*Lookup.first)->isInIdentifierNamespace(IDNS))
        continue;
      
      if (VarDecl *FoundVar = dyn_cast<VarDecl>(*Lookup.first)) {
        // We have found a variable that we may need to merge with. Check it.
        if (isExternalLinkage(FoundVar->getLinkage()) &&
            isExternalLinkage(D->getLinkage())) {
          if (Importer.IsStructurallyEquivalent(D->getType(), 
                                                FoundVar->getType())) {
            MergeWithVar = FoundVar;
            break;
          }

          const ArrayType *FoundArray
            = Importer.getToContext().getAsArrayType(FoundVar->getType());
          const ArrayType *TArray
            = Importer.getToContext().getAsArrayType(D->getType());
          if (FoundArray && TArray) {
            if (isa<IncompleteArrayType>(FoundArray) &&
                isa<ConstantArrayType>(TArray)) {
              // Import the type.
              QualType T = Importer.Import(D->getType());
              if (T.isNull())
                return 0;
              
              FoundVar->setType(T);
              MergeWithVar = FoundVar;
              break;
            } else if (isa<IncompleteArrayType>(TArray) &&
                       isa<ConstantArrayType>(FoundArray)) {
              MergeWithVar = FoundVar;
              break;
            }
          }

          Importer.ToDiag(Loc, diag::err_odr_variable_type_inconsistent)
            << Name << D->getType() << FoundVar->getType();
          Importer.ToDiag(FoundVar->getLocation(), diag::note_odr_value_here)
            << FoundVar->getType();
        }
      }
      
      ConflictingDecls.push_back(*Lookup.first);
    }

    if (MergeWithVar) {
      // An equivalent variable with external linkage has been found. Link 
      // the two declarations, then merge them.
      Importer.Imported(D, MergeWithVar);
      
      if (VarDecl *DDef = D->getDefinition()) {
        if (VarDecl *ExistingDef = MergeWithVar->getDefinition()) {
          Importer.ToDiag(ExistingDef->getLocation(), 
                          diag::err_odr_variable_multiple_def)
            << Name;
          Importer.FromDiag(DDef->getLocation(), diag::note_odr_defined_here);
        } else {
          Expr *Init = Importer.Import(DDef->getInit());
          MergeWithVar->setInit(Init);
        }
      }
      
      return MergeWithVar;
    }
    
    if (!ConflictingDecls.empty()) {
      Name = Importer.HandleNameConflict(Name, DC, IDNS,
                                         ConflictingDecls.data(), 
                                         ConflictingDecls.size());
      if (!Name)
        return 0;
    }
  }
    
  // Import the type.
  QualType T = Importer.Import(D->getType());
  if (T.isNull())
    return 0;
  
  // Create the imported variable.
  TypeSourceInfo *TInfo = Importer.Import(D->getTypeSourceInfo());
  VarDecl *ToVar = VarDecl::Create(Importer.getToContext(), DC,
                                   Importer.Import(D->getInnerLocStart()),
                                   Loc, Name.getAsIdentifierInfo(),
                                   T, TInfo,
                                   D->getStorageClass(),
                                   D->getStorageClassAsWritten());
  ToVar->setQualifierInfo(Importer.Import(D->getQualifierLoc()));
  ToVar->setAccess(D->getAccess());
  ToVar->setLexicalDeclContext(LexicalDC);
  Importer.Imported(D, ToVar);
  LexicalDC->addDecl(ToVar);

  // Merge the initializer.
  // FIXME: Can we really import any initializer? Alternatively, we could force
  // ourselves to import every declaration of a variable and then only use
  // getInit() here.
  ToVar->setInit(Importer.Import(const_cast<Expr *>(D->getAnyInitializer())));

  // FIXME: Other bits to merge?
  
  return ToVar;
}

Decl *ASTNodeImporter::VisitImplicitParamDecl(ImplicitParamDecl *D) {
  // Parameters are created in the translation unit's context, then moved
  // into the function declaration's context afterward.
  DeclContext *DC = Importer.getToContext().getTranslationUnitDecl();
  
  // Import the name of this declaration.
  DeclarationName Name = Importer.Import(D->getDeclName());
  if (D->getDeclName() && !Name)
    return 0;
  
  // Import the location of this declaration.
  SourceLocation Loc = Importer.Import(D->getLocation());
  
  // Import the parameter's type.
  QualType T = Importer.Import(D->getType());
  if (T.isNull())
    return 0;
  
  // Create the imported parameter.
  ImplicitParamDecl *ToParm
    = ImplicitParamDecl::Create(Importer.getToContext(), DC,
                                Loc, Name.getAsIdentifierInfo(),
                                T);
  return Importer.Imported(D, ToParm);
}

Decl *ASTNodeImporter::VisitParmVarDecl(ParmVarDecl *D) {
  // Parameters are created in the translation unit's context, then moved
  // into the function declaration's context afterward.
  DeclContext *DC = Importer.getToContext().getTranslationUnitDecl();
  
  // Import the name of this declaration.
  DeclarationName Name = Importer.Import(D->getDeclName());
  if (D->getDeclName() && !Name)
    return 0;
  
  // Import the location of this declaration.
  SourceLocation Loc = Importer.Import(D->getLocation());
  
  // Import the parameter's type.
  QualType T = Importer.Import(D->getType());
  if (T.isNull())
    return 0;
  
  // Create the imported parameter.
  TypeSourceInfo *TInfo = Importer.Import(D->getTypeSourceInfo());
  ParmVarDecl *ToParm = ParmVarDecl::Create(Importer.getToContext(), DC,
                                     Importer.Import(D->getInnerLocStart()),
                                            Loc, Name.getAsIdentifierInfo(),
                                            T, TInfo, D->getStorageClass(),
                                             D->getStorageClassAsWritten(),
                                            /*FIXME: Default argument*/ 0);
  ToParm->setHasInheritedDefaultArg(D->hasInheritedDefaultArg());
  return Importer.Imported(D, ToParm);
}

Decl *ASTNodeImporter::VisitObjCMethodDecl(ObjCMethodDecl *D) {
  // Import the major distinguishing characteristics of a method.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  if (ImportDeclParts(D, DC, LexicalDC, Name, Loc))
    return 0;
  
  for (DeclContext::lookup_result Lookup = DC->lookup(Name);
       Lookup.first != Lookup.second; 
       ++Lookup.first) {
    if (ObjCMethodDecl *FoundMethod = dyn_cast<ObjCMethodDecl>(*Lookup.first)) {
      if (FoundMethod->isInstanceMethod() != D->isInstanceMethod())
        continue;

      // Check return types.
      if (!Importer.IsStructurallyEquivalent(D->getResultType(),
                                             FoundMethod->getResultType())) {
        Importer.ToDiag(Loc, diag::err_odr_objc_method_result_type_inconsistent)
          << D->isInstanceMethod() << Name
          << D->getResultType() << FoundMethod->getResultType();
        Importer.ToDiag(FoundMethod->getLocation(), 
                        diag::note_odr_objc_method_here)
          << D->isInstanceMethod() << Name;
        return 0;
      }

      // Check the number of parameters.
      if (D->param_size() != FoundMethod->param_size()) {
        Importer.ToDiag(Loc, diag::err_odr_objc_method_num_params_inconsistent)
          << D->isInstanceMethod() << Name
          << D->param_size() << FoundMethod->param_size();
        Importer.ToDiag(FoundMethod->getLocation(), 
                        diag::note_odr_objc_method_here)
          << D->isInstanceMethod() << Name;
        return 0;
      }

      // Check parameter types.
      for (ObjCMethodDecl::param_iterator P = D->param_begin(), 
             PEnd = D->param_end(), FoundP = FoundMethod->param_begin();
           P != PEnd; ++P, ++FoundP) {
        if (!Importer.IsStructurallyEquivalent((*P)->getType(), 
                                               (*FoundP)->getType())) {
          Importer.FromDiag((*P)->getLocation(), 
                            diag::err_odr_objc_method_param_type_inconsistent)
            << D->isInstanceMethod() << Name
            << (*P)->getType() << (*FoundP)->getType();
          Importer.ToDiag((*FoundP)->getLocation(), diag::note_odr_value_here)
            << (*FoundP)->getType();
          return 0;
        }
      }

      // Check variadic/non-variadic.
      // Check the number of parameters.
      if (D->isVariadic() != FoundMethod->isVariadic()) {
        Importer.ToDiag(Loc, diag::err_odr_objc_method_variadic_inconsistent)
          << D->isInstanceMethod() << Name;
        Importer.ToDiag(FoundMethod->getLocation(), 
                        diag::note_odr_objc_method_here)
          << D->isInstanceMethod() << Name;
        return 0;
      }

      // FIXME: Any other bits we need to merge?
      return Importer.Imported(D, FoundMethod);
    }
  }

  // Import the result type.
  QualType ResultTy = Importer.Import(D->getResultType());
  if (ResultTy.isNull())
    return 0;

  TypeSourceInfo *ResultTInfo = Importer.Import(D->getResultTypeSourceInfo());

  ObjCMethodDecl *ToMethod
    = ObjCMethodDecl::Create(Importer.getToContext(),
                             Loc,
                             Importer.Import(D->getLocEnd()),
                             Name.getObjCSelector(),
                             ResultTy, ResultTInfo, DC,
                             D->isInstanceMethod(),
                             D->isVariadic(),
                             D->isSynthesized(),
                             D->isDefined(),
                             D->getImplementationControl(),
                             D->hasRelatedResultType());

  // FIXME: When we decide to merge method definitions, we'll need to
  // deal with implicit parameters.

  // Import the parameters
  llvm::SmallVector<ParmVarDecl *, 5> ToParams;
  for (ObjCMethodDecl::param_iterator FromP = D->param_begin(),
                                   FromPEnd = D->param_end();
       FromP != FromPEnd; 
       ++FromP) {
    ParmVarDecl *ToP = cast_or_null<ParmVarDecl>(Importer.Import(*FromP));
    if (!ToP)
      return 0;
    
    ToParams.push_back(ToP);
  }
  
  // Set the parameters.
  for (unsigned I = 0, N = ToParams.size(); I != N; ++I) {
    ToParams[I]->setOwningFunction(ToMethod);
    ToMethod->addDecl(ToParams[I]);
  }
  ToMethod->setMethodParams(Importer.getToContext(), 
                            ToParams.data(), ToParams.size(),
                            ToParams.size());

  ToMethod->setLexicalDeclContext(LexicalDC);
  Importer.Imported(D, ToMethod);
  LexicalDC->addDecl(ToMethod);
  return ToMethod;
}

Decl *ASTNodeImporter::VisitObjCCategoryDecl(ObjCCategoryDecl *D) {
  // Import the major distinguishing characteristics of a category.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  if (ImportDeclParts(D, DC, LexicalDC, Name, Loc))
    return 0;
  
  ObjCInterfaceDecl *ToInterface
    = cast_or_null<ObjCInterfaceDecl>(Importer.Import(D->getClassInterface()));
  if (!ToInterface)
    return 0;
  
  // Determine if we've already encountered this category.
  ObjCCategoryDecl *MergeWithCategory
    = ToInterface->FindCategoryDeclaration(Name.getAsIdentifierInfo());
  ObjCCategoryDecl *ToCategory = MergeWithCategory;
  if (!ToCategory) {
    ToCategory = ObjCCategoryDecl::Create(Importer.getToContext(), DC,
                                          Importer.Import(D->getAtLoc()),
                                          Loc, 
                                       Importer.Import(D->getCategoryNameLoc()), 
                                          Name.getAsIdentifierInfo());
    ToCategory->setLexicalDeclContext(LexicalDC);
    LexicalDC->addDecl(ToCategory);
    Importer.Imported(D, ToCategory);
    
    // Link this category into its class's category list.
    ToCategory->setClassInterface(ToInterface);
    ToCategory->insertNextClassCategory();
    
    // Import protocols
    llvm::SmallVector<ObjCProtocolDecl *, 4> Protocols;
    llvm::SmallVector<SourceLocation, 4> ProtocolLocs;
    ObjCCategoryDecl::protocol_loc_iterator FromProtoLoc
      = D->protocol_loc_begin();
    for (ObjCCategoryDecl::protocol_iterator FromProto = D->protocol_begin(),
                                          FromProtoEnd = D->protocol_end();
         FromProto != FromProtoEnd;
         ++FromProto, ++FromProtoLoc) {
      ObjCProtocolDecl *ToProto
        = cast_or_null<ObjCProtocolDecl>(Importer.Import(*FromProto));
      if (!ToProto)
        return 0;
      Protocols.push_back(ToProto);
      ProtocolLocs.push_back(Importer.Import(*FromProtoLoc));
    }
    
    // FIXME: If we're merging, make sure that the protocol list is the same.
    ToCategory->setProtocolList(Protocols.data(), Protocols.size(),
                                ProtocolLocs.data(), Importer.getToContext());
    
  } else {
    Importer.Imported(D, ToCategory);
  }
  
  // Import all of the members of this category.
  ImportDeclContext(D);
 
  // If we have an implementation, import it as well.
  if (D->getImplementation()) {
    ObjCCategoryImplDecl *Impl
      = cast_or_null<ObjCCategoryImplDecl>(
                                       Importer.Import(D->getImplementation()));
    if (!Impl)
      return 0;
    
    ToCategory->setImplementation(Impl);
  }
  
  return ToCategory;
}

Decl *ASTNodeImporter::VisitObjCProtocolDecl(ObjCProtocolDecl *D) {
  // Import the major distinguishing characteristics of a protocol.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  if (ImportDeclParts(D, DC, LexicalDC, Name, Loc))
    return 0;

  ObjCProtocolDecl *MergeWithProtocol = 0;
  for (DeclContext::lookup_result Lookup = DC->lookup(Name);
       Lookup.first != Lookup.second; 
       ++Lookup.first) {
    if (!(*Lookup.first)->isInIdentifierNamespace(Decl::IDNS_ObjCProtocol))
      continue;
    
    if ((MergeWithProtocol = dyn_cast<ObjCProtocolDecl>(*Lookup.first)))
      break;
  }
  
  ObjCProtocolDecl *ToProto = MergeWithProtocol;
  if (!ToProto || ToProto->isForwardDecl()) {
    if (!ToProto) {
      ToProto = ObjCProtocolDecl::Create(Importer.getToContext(), DC, Loc,
                                         Name.getAsIdentifierInfo());
      ToProto->setForwardDecl(D->isForwardDecl());
      ToProto->setLexicalDeclContext(LexicalDC);
      LexicalDC->addDecl(ToProto);
    }
    Importer.Imported(D, ToProto);

    // Import protocols
    llvm::SmallVector<ObjCProtocolDecl *, 4> Protocols;
    llvm::SmallVector<SourceLocation, 4> ProtocolLocs;
    ObjCProtocolDecl::protocol_loc_iterator 
      FromProtoLoc = D->protocol_loc_begin();
    for (ObjCProtocolDecl::protocol_iterator FromProto = D->protocol_begin(),
                                          FromProtoEnd = D->protocol_end();
       FromProto != FromProtoEnd;
       ++FromProto, ++FromProtoLoc) {
      ObjCProtocolDecl *ToProto
        = cast_or_null<ObjCProtocolDecl>(Importer.Import(*FromProto));
      if (!ToProto)
        return 0;
      Protocols.push_back(ToProto);
      ProtocolLocs.push_back(Importer.Import(*FromProtoLoc));
    }
    
    // FIXME: If we're merging, make sure that the protocol list is the same.
    ToProto->setProtocolList(Protocols.data(), Protocols.size(),
                             ProtocolLocs.data(), Importer.getToContext());
  } else {
    Importer.Imported(D, ToProto);
  }

  // Import all of the members of this protocol.
  ImportDeclContext(D);

  return ToProto;
}

Decl *ASTNodeImporter::VisitObjCInterfaceDecl(ObjCInterfaceDecl *D) {
  // Import the major distinguishing characteristics of an @interface.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  if (ImportDeclParts(D, DC, LexicalDC, Name, Loc))
    return 0;

  ObjCInterfaceDecl *MergeWithIface = 0;
  for (DeclContext::lookup_result Lookup = DC->lookup(Name);
       Lookup.first != Lookup.second; 
       ++Lookup.first) {
    if (!(*Lookup.first)->isInIdentifierNamespace(Decl::IDNS_Ordinary))
      continue;
    
    if ((MergeWithIface = dyn_cast<ObjCInterfaceDecl>(*Lookup.first)))
      break;
  }
  
  ObjCInterfaceDecl *ToIface = MergeWithIface;
  if (!ToIface || ToIface->isForwardDecl()) {
    if (!ToIface) {
      ToIface = ObjCInterfaceDecl::Create(Importer.getToContext(),
                                          DC, Loc,
                                          Name.getAsIdentifierInfo(),
                                          Importer.Import(D->getClassLoc()),
                                          D->isForwardDecl(),
                                          D->isImplicitInterfaceDecl());
      ToIface->setForwardDecl(D->isForwardDecl());
      ToIface->setLexicalDeclContext(LexicalDC);
      LexicalDC->addDecl(ToIface);
    }
    Importer.Imported(D, ToIface);

    if (D->getSuperClass()) {
      ObjCInterfaceDecl *Super
        = cast_or_null<ObjCInterfaceDecl>(Importer.Import(D->getSuperClass()));
      if (!Super)
        return 0;
      
      ToIface->setSuperClass(Super);
      ToIface->setSuperClassLoc(Importer.Import(D->getSuperClassLoc()));
    }
    
    // Import protocols
    llvm::SmallVector<ObjCProtocolDecl *, 4> Protocols;
    llvm::SmallVector<SourceLocation, 4> ProtocolLocs;
    ObjCInterfaceDecl::protocol_loc_iterator 
      FromProtoLoc = D->protocol_loc_begin();
    
    // FIXME: Should we be usng all_referenced_protocol_begin() here?
    for (ObjCInterfaceDecl::protocol_iterator FromProto = D->protocol_begin(),
                                           FromProtoEnd = D->protocol_end();
       FromProto != FromProtoEnd;
       ++FromProto, ++FromProtoLoc) {
      ObjCProtocolDecl *ToProto
        = cast_or_null<ObjCProtocolDecl>(Importer.Import(*FromProto));
      if (!ToProto)
        return 0;
      Protocols.push_back(ToProto);
      ProtocolLocs.push_back(Importer.Import(*FromProtoLoc));
    }
    
    // FIXME: If we're merging, make sure that the protocol list is the same.
    ToIface->setProtocolList(Protocols.data(), Protocols.size(),
                             ProtocolLocs.data(), Importer.getToContext());
    
    // Import @end range
    ToIface->setAtEndRange(Importer.Import(D->getAtEndRange()));
  } else {
    Importer.Imported(D, ToIface);

    // Check for consistency of superclasses.
    DeclarationName FromSuperName, ToSuperName;
    if (D->getSuperClass())
      FromSuperName = Importer.Import(D->getSuperClass()->getDeclName());
    if (ToIface->getSuperClass())
      ToSuperName = ToIface->getSuperClass()->getDeclName();
    if (FromSuperName != ToSuperName) {
      Importer.ToDiag(ToIface->getLocation(), 
                      diag::err_odr_objc_superclass_inconsistent)
        << ToIface->getDeclName();
      if (ToIface->getSuperClass())
        Importer.ToDiag(ToIface->getSuperClassLoc(), 
                        diag::note_odr_objc_superclass)
          << ToIface->getSuperClass()->getDeclName();
      else
        Importer.ToDiag(ToIface->getLocation(), 
                        diag::note_odr_objc_missing_superclass);
      if (D->getSuperClass())
        Importer.FromDiag(D->getSuperClassLoc(), 
                          diag::note_odr_objc_superclass)
          << D->getSuperClass()->getDeclName();
      else
        Importer.FromDiag(D->getLocation(), 
                          diag::note_odr_objc_missing_superclass);
      return 0;
    }
  }
  
  // Import categories. When the categories themselves are imported, they'll
  // hook themselves into this interface.
  for (ObjCCategoryDecl *FromCat = D->getCategoryList(); FromCat;
       FromCat = FromCat->getNextClassCategory())
    Importer.Import(FromCat);
  
  // Import all of the members of this class.
  ImportDeclContext(D);
  
  // If we have an @implementation, import it as well.
  if (D->getImplementation()) {
    ObjCImplementationDecl *Impl = cast_or_null<ObjCImplementationDecl>(
                                       Importer.Import(D->getImplementation()));
    if (!Impl)
      return 0;
    
    ToIface->setImplementation(Impl);
  }
  
  return ToIface;
}

Decl *ASTNodeImporter::VisitObjCCategoryImplDecl(ObjCCategoryImplDecl *D) {
  ObjCCategoryDecl *Category = cast_or_null<ObjCCategoryDecl>(
                                        Importer.Import(D->getCategoryDecl()));
  if (!Category)
    return 0;
  
  ObjCCategoryImplDecl *ToImpl = Category->getImplementation();
  if (!ToImpl) {
    DeclContext *DC = Importer.ImportContext(D->getDeclContext());
    if (!DC)
      return 0;
    
    ToImpl = ObjCCategoryImplDecl::Create(Importer.getToContext(), DC,
                                          Importer.Import(D->getLocation()),
                                          Importer.Import(D->getIdentifier()),
                                          Category->getClassInterface());
    
    DeclContext *LexicalDC = DC;
    if (D->getDeclContext() != D->getLexicalDeclContext()) {
      LexicalDC = Importer.ImportContext(D->getLexicalDeclContext());
      if (!LexicalDC)
        return 0;
      
      ToImpl->setLexicalDeclContext(LexicalDC);
    }
    
    LexicalDC->addDecl(ToImpl);
    Category->setImplementation(ToImpl);
  }
  
  Importer.Imported(D, ToImpl);
  ImportDeclContext(D);
  return ToImpl;
}

Decl *ASTNodeImporter::VisitObjCImplementationDecl(ObjCImplementationDecl *D) {
  // Find the corresponding interface.
  ObjCInterfaceDecl *Iface = cast_or_null<ObjCInterfaceDecl>(
                                       Importer.Import(D->getClassInterface()));
  if (!Iface)
    return 0;

  // Import the superclass, if any.
  ObjCInterfaceDecl *Super = 0;
  if (D->getSuperClass()) {
    Super = cast_or_null<ObjCInterfaceDecl>(
                                          Importer.Import(D->getSuperClass()));
    if (!Super)
      return 0;
  }

  ObjCImplementationDecl *Impl = Iface->getImplementation();
  if (!Impl) {
    // We haven't imported an implementation yet. Create a new @implementation
    // now.
    Impl = ObjCImplementationDecl::Create(Importer.getToContext(),
                                  Importer.ImportContext(D->getDeclContext()),
                                          Importer.Import(D->getLocation()),
                                          Iface, Super);
    
    if (D->getDeclContext() != D->getLexicalDeclContext()) {
      DeclContext *LexicalDC
        = Importer.ImportContext(D->getLexicalDeclContext());
      if (!LexicalDC)
        return 0;
      Impl->setLexicalDeclContext(LexicalDC);
    }
    
    // Associate the implementation with the class it implements.
    Iface->setImplementation(Impl);
    Importer.Imported(D, Iface->getImplementation());
  } else {
    Importer.Imported(D, Iface->getImplementation());

    // Verify that the existing @implementation has the same superclass.
    if ((Super && !Impl->getSuperClass()) ||
        (!Super && Impl->getSuperClass()) ||
        (Super && Impl->getSuperClass() && 
         Super->getCanonicalDecl() != Impl->getSuperClass())) {
        Importer.ToDiag(Impl->getLocation(), 
                        diag::err_odr_objc_superclass_inconsistent)
          << Iface->getDeclName();
        // FIXME: It would be nice to have the location of the superclass
        // below.
        if (Impl->getSuperClass())
          Importer.ToDiag(Impl->getLocation(), 
                          diag::note_odr_objc_superclass)
          << Impl->getSuperClass()->getDeclName();
        else
          Importer.ToDiag(Impl->getLocation(), 
                          diag::note_odr_objc_missing_superclass);
        if (D->getSuperClass())
          Importer.FromDiag(D->getLocation(), 
                            diag::note_odr_objc_superclass)
          << D->getSuperClass()->getDeclName();
        else
          Importer.FromDiag(D->getLocation(), 
                            diag::note_odr_objc_missing_superclass);
      return 0;
    }
  }
    
  // Import all of the members of this @implementation.
  ImportDeclContext(D);

  return Impl;
}

Decl *ASTNodeImporter::VisitObjCPropertyDecl(ObjCPropertyDecl *D) {
  // Import the major distinguishing characteristics of an @property.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  if (ImportDeclParts(D, DC, LexicalDC, Name, Loc))
    return 0;

  // Check whether we have already imported this property.
  for (DeclContext::lookup_result Lookup = DC->lookup(Name);
       Lookup.first != Lookup.second; 
       ++Lookup.first) {
    if (ObjCPropertyDecl *FoundProp
                                = dyn_cast<ObjCPropertyDecl>(*Lookup.first)) {
      // Check property types.
      if (!Importer.IsStructurallyEquivalent(D->getType(), 
                                             FoundProp->getType())) {
        Importer.ToDiag(Loc, diag::err_odr_objc_property_type_inconsistent)
          << Name << D->getType() << FoundProp->getType();
        Importer.ToDiag(FoundProp->getLocation(), diag::note_odr_value_here)
          << FoundProp->getType();
        return 0;
      }

      // FIXME: Check property attributes, getters, setters, etc.?

      // Consider these properties to be equivalent.
      Importer.Imported(D, FoundProp);
      return FoundProp;
    }
  }

  // Import the type.
  TypeSourceInfo *T = Importer.Import(D->getTypeSourceInfo());
  if (!T)
    return 0;

  // Create the new property.
  ObjCPropertyDecl *ToProperty
    = ObjCPropertyDecl::Create(Importer.getToContext(), DC, Loc,
                               Name.getAsIdentifierInfo(), 
                               Importer.Import(D->getAtLoc()),
                               T,
                               D->getPropertyImplementation());
  Importer.Imported(D, ToProperty);
  ToProperty->setLexicalDeclContext(LexicalDC);
  LexicalDC->addDecl(ToProperty);

  ToProperty->setPropertyAttributes(D->getPropertyAttributes());
  ToProperty->setPropertyAttributesAsWritten(
                                      D->getPropertyAttributesAsWritten());
  ToProperty->setGetterName(Importer.Import(D->getGetterName()));
  ToProperty->setSetterName(Importer.Import(D->getSetterName()));
  ToProperty->setGetterMethodDecl(
     cast_or_null<ObjCMethodDecl>(Importer.Import(D->getGetterMethodDecl())));
  ToProperty->setSetterMethodDecl(
     cast_or_null<ObjCMethodDecl>(Importer.Import(D->getSetterMethodDecl())));
  ToProperty->setPropertyIvarDecl(
       cast_or_null<ObjCIvarDecl>(Importer.Import(D->getPropertyIvarDecl())));
  return ToProperty;
}

Decl *ASTNodeImporter::VisitObjCPropertyImplDecl(ObjCPropertyImplDecl *D) {
  ObjCPropertyDecl *Property = cast_or_null<ObjCPropertyDecl>(
                                        Importer.Import(D->getPropertyDecl()));
  if (!Property)
    return 0;

  DeclContext *DC = Importer.ImportContext(D->getDeclContext());
  if (!DC)
    return 0;
  
  // Import the lexical declaration context.
  DeclContext *LexicalDC = DC;
  if (D->getDeclContext() != D->getLexicalDeclContext()) {
    LexicalDC = Importer.ImportContext(D->getLexicalDeclContext());
    if (!LexicalDC)
      return 0;
  }

  ObjCImplDecl *InImpl = dyn_cast<ObjCImplDecl>(LexicalDC);
  if (!InImpl)
    return 0;

  // Import the ivar (for an @synthesize).
  ObjCIvarDecl *Ivar = 0;
  if (D->getPropertyIvarDecl()) {
    Ivar = cast_or_null<ObjCIvarDecl>(
                                    Importer.Import(D->getPropertyIvarDecl()));
    if (!Ivar)
      return 0;
  }

  ObjCPropertyImplDecl *ToImpl
    = InImpl->FindPropertyImplDecl(Property->getIdentifier());
  if (!ToImpl) {    
    ToImpl = ObjCPropertyImplDecl::Create(Importer.getToContext(), DC,
                                          Importer.Import(D->getLocStart()),
                                          Importer.Import(D->getLocation()),
                                          Property,
                                          D->getPropertyImplementation(),
                                          Ivar, 
                                  Importer.Import(D->getPropertyIvarDeclLoc()));
    ToImpl->setLexicalDeclContext(LexicalDC);
    Importer.Imported(D, ToImpl);
    LexicalDC->addDecl(ToImpl);
  } else {
    // Check that we have the same kind of property implementation (@synthesize
    // vs. @dynamic).
    if (D->getPropertyImplementation() != ToImpl->getPropertyImplementation()) {
      Importer.ToDiag(ToImpl->getLocation(), 
                      diag::err_odr_objc_property_impl_kind_inconsistent)
        << Property->getDeclName() 
        << (ToImpl->getPropertyImplementation() 
                                              == ObjCPropertyImplDecl::Dynamic);
      Importer.FromDiag(D->getLocation(),
                        diag::note_odr_objc_property_impl_kind)
        << D->getPropertyDecl()->getDeclName()
        << (D->getPropertyImplementation() == ObjCPropertyImplDecl::Dynamic);
      return 0;
    }
    
    // For @synthesize, check that we have the same 
    if (D->getPropertyImplementation() == ObjCPropertyImplDecl::Synthesize &&
        Ivar != ToImpl->getPropertyIvarDecl()) {
      Importer.ToDiag(ToImpl->getPropertyIvarDeclLoc(), 
                      diag::err_odr_objc_synthesize_ivar_inconsistent)
        << Property->getDeclName()
        << ToImpl->getPropertyIvarDecl()->getDeclName()
        << Ivar->getDeclName();
      Importer.FromDiag(D->getPropertyIvarDeclLoc(), 
                        diag::note_odr_objc_synthesize_ivar_here)
        << D->getPropertyIvarDecl()->getDeclName();
      return 0;
    }
    
    // Merge the existing implementation with the new implementation.
    Importer.Imported(D, ToImpl);
  }
  
  return ToImpl;
}

Decl *
ASTNodeImporter::VisitObjCForwardProtocolDecl(ObjCForwardProtocolDecl *D) {
  // Import the context of this declaration.
  DeclContext *DC = Importer.ImportContext(D->getDeclContext());
  if (!DC)
    return 0;
  
  DeclContext *LexicalDC = DC;
  if (D->getDeclContext() != D->getLexicalDeclContext()) {
    LexicalDC = Importer.ImportContext(D->getLexicalDeclContext());
    if (!LexicalDC)
      return 0;
  }
  
  // Import the location of this declaration.
  SourceLocation Loc = Importer.Import(D->getLocation());
  
  llvm::SmallVector<ObjCProtocolDecl *, 4> Protocols;
  llvm::SmallVector<SourceLocation, 4> Locations;
  ObjCForwardProtocolDecl::protocol_loc_iterator FromProtoLoc
    = D->protocol_loc_begin();
  for (ObjCForwardProtocolDecl::protocol_iterator FromProto
         = D->protocol_begin(), FromProtoEnd = D->protocol_end();
       FromProto != FromProtoEnd; 
       ++FromProto, ++FromProtoLoc) {
    ObjCProtocolDecl *ToProto
      = cast_or_null<ObjCProtocolDecl>(Importer.Import(*FromProto));
    if (!ToProto)
      continue;
    
    Protocols.push_back(ToProto);
    Locations.push_back(Importer.Import(*FromProtoLoc));
  }
  
  ObjCForwardProtocolDecl *ToForward
    = ObjCForwardProtocolDecl::Create(Importer.getToContext(), DC, Loc, 
                                      Protocols.data(), Protocols.size(),
                                      Locations.data());
  ToForward->setLexicalDeclContext(LexicalDC);
  LexicalDC->addDecl(ToForward);
  Importer.Imported(D, ToForward);
  return ToForward;
}

Decl *ASTNodeImporter::VisitObjCClassDecl(ObjCClassDecl *D) {
  // Import the context of this declaration.
  DeclContext *DC = Importer.ImportContext(D->getDeclContext());
  if (!DC)
    return 0;
  
  DeclContext *LexicalDC = DC;
  if (D->getDeclContext() != D->getLexicalDeclContext()) {
    LexicalDC = Importer.ImportContext(D->getLexicalDeclContext());
    if (!LexicalDC)
      return 0;
  }
  
  // Import the location of this declaration.
  SourceLocation Loc = Importer.Import(D->getLocation());

  llvm::SmallVector<ObjCInterfaceDecl *, 4> Interfaces;
  llvm::SmallVector<SourceLocation, 4> Locations;
  for (ObjCClassDecl::iterator From = D->begin(), FromEnd = D->end();
       From != FromEnd; ++From) {
    ObjCInterfaceDecl *ToIface
      = cast_or_null<ObjCInterfaceDecl>(Importer.Import(From->getInterface()));
    if (!ToIface)
      continue;
    
    Interfaces.push_back(ToIface);
    Locations.push_back(Importer.Import(From->getLocation()));
  }
  
  ObjCClassDecl *ToClass = ObjCClassDecl::Create(Importer.getToContext(), DC,
                                                 Loc, 
                                                 Interfaces.data(),
                                                 Locations.data(),
                                                 Interfaces.size());
  ToClass->setLexicalDeclContext(LexicalDC);
  LexicalDC->addDecl(ToClass);
  Importer.Imported(D, ToClass);
  return ToClass;
}

Decl *ASTNodeImporter::VisitTemplateTypeParmDecl(TemplateTypeParmDecl *D) {
  // For template arguments, we adopt the translation unit as our declaration
  // context. This context will be fixed when the actual template declaration
  // is created.
  
  // FIXME: Import default argument.
  return TemplateTypeParmDecl::Create(Importer.getToContext(),
                              Importer.getToContext().getTranslationUnitDecl(),
                                      Importer.Import(D->getLocStart()),
                                      Importer.Import(D->getLocation()),
                                      D->getDepth(),
                                      D->getIndex(), 
                                      Importer.Import(D->getIdentifier()),
                                      D->wasDeclaredWithTypename(),
                                      D->isParameterPack());
}

Decl *
ASTNodeImporter::VisitNonTypeTemplateParmDecl(NonTypeTemplateParmDecl *D) {
  // Import the name of this declaration.
  DeclarationName Name = Importer.Import(D->getDeclName());
  if (D->getDeclName() && !Name)
    return 0;
  
  // Import the location of this declaration.
  SourceLocation Loc = Importer.Import(D->getLocation());

  // Import the type of this declaration.
  QualType T = Importer.Import(D->getType());
  if (T.isNull())
    return 0;
  
  // Import type-source information.
  TypeSourceInfo *TInfo = Importer.Import(D->getTypeSourceInfo());
  if (D->getTypeSourceInfo() && !TInfo)
    return 0;
  
  // FIXME: Import default argument.
  
  return NonTypeTemplateParmDecl::Create(Importer.getToContext(),
                               Importer.getToContext().getTranslationUnitDecl(),
                                         Importer.Import(D->getInnerLocStart()),
                                         Loc, D->getDepth(), D->getPosition(),
                                         Name.getAsIdentifierInfo(),
                                         T, D->isParameterPack(), TInfo);
}

Decl *
ASTNodeImporter::VisitTemplateTemplateParmDecl(TemplateTemplateParmDecl *D) {
  // Import the name of this declaration.
  DeclarationName Name = Importer.Import(D->getDeclName());
  if (D->getDeclName() && !Name)
    return 0;
  
  // Import the location of this declaration.
  SourceLocation Loc = Importer.Import(D->getLocation());
  
  // Import template parameters.
  TemplateParameterList *TemplateParams
    = ImportTemplateParameterList(D->getTemplateParameters());
  if (!TemplateParams)
    return 0;
  
  // FIXME: Import default argument.
  
  return TemplateTemplateParmDecl::Create(Importer.getToContext(), 
                              Importer.getToContext().getTranslationUnitDecl(), 
                                          Loc, D->getDepth(), D->getPosition(),
                                          D->isParameterPack(),
                                          Name.getAsIdentifierInfo(), 
                                          TemplateParams);
}

Decl *ASTNodeImporter::VisitClassTemplateDecl(ClassTemplateDecl *D) {
  // If this record has a definition in the translation unit we're coming from,
  // but this particular declaration is not that definition, import the
  // definition and map to that.
  CXXRecordDecl *Definition 
    = cast_or_null<CXXRecordDecl>(D->getTemplatedDecl()->getDefinition());
  if (Definition && Definition != D->getTemplatedDecl()) {
    Decl *ImportedDef
      = Importer.Import(Definition->getDescribedClassTemplate());
    if (!ImportedDef)
      return 0;
    
    return Importer.Imported(D, ImportedDef);
  }
  
  // Import the major distinguishing characteristics of this class template.
  DeclContext *DC, *LexicalDC;
  DeclarationName Name;
  SourceLocation Loc;
  if (ImportDeclParts(D, DC, LexicalDC, Name, Loc))
    return 0;
  
  // We may already have a template of the same name; try to find and match it.
  if (!DC->isFunctionOrMethod()) {
    llvm::SmallVector<NamedDecl *, 4> ConflictingDecls;
    for (DeclContext::lookup_result Lookup = DC->lookup(Name);
         Lookup.first != Lookup.second; 
         ++Lookup.first) {
      if (!(*Lookup.first)->isInIdentifierNamespace(Decl::IDNS_Ordinary))
        continue;
      
      Decl *Found = *Lookup.first;
      if (ClassTemplateDecl *FoundTemplate 
                                        = dyn_cast<ClassTemplateDecl>(Found)) {
        if (IsStructuralMatch(D, FoundTemplate)) {
          // The class templates structurally match; call it the same template.
          // FIXME: We may be filling in a forward declaration here. Handle
          // this case!
          Importer.Imported(D->getTemplatedDecl(), 
                            FoundTemplate->getTemplatedDecl());
          return Importer.Imported(D, FoundTemplate);
        }         
      }
      
      ConflictingDecls.push_back(*Lookup.first);
    }
    
    if (!ConflictingDecls.empty()) {
      Name = Importer.HandleNameConflict(Name, DC, Decl::IDNS_Ordinary,
                                         ConflictingDecls.data(), 
                                         ConflictingDecls.size());
    }
    
    if (!Name)
      return 0;
  }

  CXXRecordDecl *DTemplated = D->getTemplatedDecl();
  
  // Create the declaration that is being templated.
  SourceLocation StartLoc = Importer.Import(DTemplated->getLocStart());
  SourceLocation IdLoc = Importer.Import(DTemplated->getLocation());
  CXXRecordDecl *D2Templated = CXXRecordDecl::Create(Importer.getToContext(),
                                                     DTemplated->getTagKind(),
                                                     DC, StartLoc, IdLoc,
                                                   Name.getAsIdentifierInfo());
  D2Templated->setAccess(DTemplated->getAccess());
  D2Templated->setQualifierInfo(Importer.Import(DTemplated->getQualifierLoc()));
  D2Templated->setLexicalDeclContext(LexicalDC);
  
  // Create the class template declaration itself.
  TemplateParameterList *TemplateParams
    = ImportTemplateParameterList(D->getTemplateParameters());
  if (!TemplateParams)
    return 0;
  
  ClassTemplateDecl *D2 = ClassTemplateDecl::Create(Importer.getToContext(), DC, 
                                                    Loc, Name, TemplateParams, 
                                                    D2Templated, 
  /*PrevDecl=*/0);
  D2Templated->setDescribedClassTemplate(D2);    
  
  D2->setAccess(D->getAccess());
  D2->setLexicalDeclContext(LexicalDC);
  LexicalDC->addDecl(D2);
  
  // Note the relationship between the class templates.
  Importer.Imported(D, D2);
  Importer.Imported(DTemplated, D2Templated);

  if (DTemplated->isDefinition() && !D2Templated->isDefinition()) {
    // FIXME: Import definition!
  }
  
  return D2;
}

Decl *ASTNodeImporter::VisitClassTemplateSpecializationDecl(
                                          ClassTemplateSpecializationDecl *D) {
  // If this record has a definition in the translation unit we're coming from,
  // but this particular declaration is not that definition, import the
  // definition and map to that.
  TagDecl *Definition = D->getDefinition();
  if (Definition && Definition != D) {
    Decl *ImportedDef = Importer.Import(Definition);
    if (!ImportedDef)
      return 0;
    
    return Importer.Imported(D, ImportedDef);
  }

  ClassTemplateDecl *ClassTemplate
    = cast_or_null<ClassTemplateDecl>(Importer.Import(
                                                 D->getSpecializedTemplate()));
  if (!ClassTemplate)
    return 0;
  
  // Import the context of this declaration.
  DeclContext *DC = ClassTemplate->getDeclContext();
  if (!DC)
    return 0;
  
  DeclContext *LexicalDC = DC;
  if (D->getDeclContext() != D->getLexicalDeclContext()) {
    LexicalDC = Importer.ImportContext(D->getLexicalDeclContext());
    if (!LexicalDC)
      return 0;
  }
  
  // Import the location of this declaration.
  SourceLocation StartLoc = Importer.Import(D->getLocStart());
  SourceLocation IdLoc = Importer.Import(D->getLocation());

  // Import template arguments.
  llvm::SmallVector<TemplateArgument, 2> TemplateArgs;
  if (ImportTemplateArguments(D->getTemplateArgs().data(), 
                              D->getTemplateArgs().size(),
                              TemplateArgs))
    return 0;
  
  // Try to find an existing specialization with these template arguments.
  void *InsertPos = 0;
  ClassTemplateSpecializationDecl *D2
    = ClassTemplate->findSpecialization(TemplateArgs.data(), 
                                        TemplateArgs.size(), InsertPos);
  if (D2) {
    // We already have a class template specialization with these template
    // arguments.
    
    // FIXME: Check for specialization vs. instantiation errors.
    
    if (RecordDecl *FoundDef = D2->getDefinition()) {
      if (!D->isDefinition() || IsStructuralMatch(D, FoundDef)) {
        // The record types structurally match, or the "from" translation
        // unit only had a forward declaration anyway; call it the same
        // function.
        return Importer.Imported(D, FoundDef);
      }
    }
  } else {
    // Create a new specialization.
    D2 = ClassTemplateSpecializationDecl::Create(Importer.getToContext(), 
                                                 D->getTagKind(), DC, 
                                                 StartLoc, IdLoc,
                                                 ClassTemplate,
                                                 TemplateArgs.data(), 
                                                 TemplateArgs.size(), 
                                                 /*PrevDecl=*/0);
    D2->setSpecializationKind(D->getSpecializationKind());

    // Add this specialization to the class template.
    ClassTemplate->AddSpecialization(D2, InsertPos);
    
    // Import the qualifier, if any.
    D2->setQualifierInfo(Importer.Import(D->getQualifierLoc()));
    
    // Add the specialization to this context.
    D2->setLexicalDeclContext(LexicalDC);
    LexicalDC->addDecl(D2);
  }
  Importer.Imported(D, D2);
  
  if (D->isDefinition() && ImportDefinition(D, D2))
    return 0;
  
  return D2;
}

//----------------------------------------------------------------------------
// Import Statements
//----------------------------------------------------------------------------

Stmt *ASTNodeImporter::VisitStmt(Stmt *S) {
  Importer.FromDiag(S->getLocStart(), diag::err_unsupported_ast_node)
    << S->getStmtClassName();
  return 0;
}

//----------------------------------------------------------------------------
// Import Expressions
//----------------------------------------------------------------------------
Expr *ASTNodeImporter::VisitExpr(Expr *E) {
  Importer.FromDiag(E->getLocStart(), diag::err_unsupported_ast_node)
    << E->getStmtClassName();
  return 0;
}

Expr *ASTNodeImporter::VisitDeclRefExpr(DeclRefExpr *E) {
  ValueDecl *ToD = cast_or_null<ValueDecl>(Importer.Import(E->getDecl()));
  if (!ToD)
    return 0;

  NamedDecl *FoundD = 0;
  if (E->getDecl() != E->getFoundDecl()) {
    FoundD = cast_or_null<NamedDecl>(Importer.Import(E->getFoundDecl()));
    if (!FoundD)
      return 0;
  }
  
  QualType T = Importer.Import(E->getType());
  if (T.isNull())
    return 0;
  
  return DeclRefExpr::Create(Importer.getToContext(), 
                             Importer.Import(E->getQualifierLoc()),
                             ToD,
                             Importer.Import(E->getLocation()),
                             T, E->getValueKind(),
                             FoundD,
                             /*FIXME:TemplateArgs=*/0);
}

Expr *ASTNodeImporter::VisitIntegerLiteral(IntegerLiteral *E) {
  QualType T = Importer.Import(E->getType());
  if (T.isNull())
    return 0;

  return IntegerLiteral::Create(Importer.getToContext(), 
                                E->getValue(), T,
                                Importer.Import(E->getLocation()));
}

Expr *ASTNodeImporter::VisitCharacterLiteral(CharacterLiteral *E) {
  QualType T = Importer.Import(E->getType());
  if (T.isNull())
    return 0;
  
  return new (Importer.getToContext()) CharacterLiteral(E->getValue(), 
                                                        E->isWide(), T,
                                          Importer.Import(E->getLocation()));
}

Expr *ASTNodeImporter::VisitParenExpr(ParenExpr *E) {
  Expr *SubExpr = Importer.Import(E->getSubExpr());
  if (!SubExpr)
    return 0;
  
  return new (Importer.getToContext()) 
                                  ParenExpr(Importer.Import(E->getLParen()),
                                            Importer.Import(E->getRParen()),
                                            SubExpr);
}

Expr *ASTNodeImporter::VisitUnaryOperator(UnaryOperator *E) {
  QualType T = Importer.Import(E->getType());
  if (T.isNull())
    return 0;

  Expr *SubExpr = Importer.Import(E->getSubExpr());
  if (!SubExpr)
    return 0;
  
  return new (Importer.getToContext()) UnaryOperator(SubExpr, E->getOpcode(),
                                                     T, E->getValueKind(),
                                                     E->getObjectKind(),
                                         Importer.Import(E->getOperatorLoc()));                                        
}

Expr *ASTNodeImporter::VisitUnaryExprOrTypeTraitExpr(
                                            UnaryExprOrTypeTraitExpr *E) {
  QualType ResultType = Importer.Import(E->getType());
  
  if (E->isArgumentType()) {
    TypeSourceInfo *TInfo = Importer.Import(E->getArgumentTypeInfo());
    if (!TInfo)
      return 0;
    
    return new (Importer.getToContext()) UnaryExprOrTypeTraitExpr(E->getKind(),
                                           TInfo, ResultType,
                                           Importer.Import(E->getOperatorLoc()),
                                           Importer.Import(E->getRParenLoc()));
  }
  
  Expr *SubExpr = Importer.Import(E->getArgumentExpr());
  if (!SubExpr)
    return 0;
  
  return new (Importer.getToContext()) UnaryExprOrTypeTraitExpr(E->getKind(),
                                          SubExpr, ResultType,
                                          Importer.Import(E->getOperatorLoc()),
                                          Importer.Import(E->getRParenLoc()));
}

Expr *ASTNodeImporter::VisitBinaryOperator(BinaryOperator *E) {
  QualType T = Importer.Import(E->getType());
  if (T.isNull())
    return 0;

  Expr *LHS = Importer.Import(E->getLHS());
  if (!LHS)
    return 0;
  
  Expr *RHS = Importer.Import(E->getRHS());
  if (!RHS)
    return 0;
  
  return new (Importer.getToContext()) BinaryOperator(LHS, RHS, E->getOpcode(),
                                                      T, E->getValueKind(),
                                                      E->getObjectKind(),
                                          Importer.Import(E->getOperatorLoc()));
}

Expr *ASTNodeImporter::VisitCompoundAssignOperator(CompoundAssignOperator *E) {
  QualType T = Importer.Import(E->getType());
  if (T.isNull())
    return 0;
  
  QualType CompLHSType = Importer.Import(E->getComputationLHSType());
  if (CompLHSType.isNull())
    return 0;
  
  QualType CompResultType = Importer.Import(E->getComputationResultType());
  if (CompResultType.isNull())
    return 0;
  
  Expr *LHS = Importer.Import(E->getLHS());
  if (!LHS)
    return 0;
  
  Expr *RHS = Importer.Import(E->getRHS());
  if (!RHS)
    return 0;
  
  return new (Importer.getToContext()) 
                        CompoundAssignOperator(LHS, RHS, E->getOpcode(),
                                               T, E->getValueKind(),
                                               E->getObjectKind(),
                                               CompLHSType, CompResultType,
                                          Importer.Import(E->getOperatorLoc()));
}

static bool ImportCastPath(CastExpr *E, CXXCastPath &Path) {
  if (E->path_empty()) return false;

  // TODO: import cast paths
  return true;
}

Expr *ASTNodeImporter::VisitImplicitCastExpr(ImplicitCastExpr *E) {
  QualType T = Importer.Import(E->getType());
  if (T.isNull())
    return 0;

  Expr *SubExpr = Importer.Import(E->getSubExpr());
  if (!SubExpr)
    return 0;

  CXXCastPath BasePath;
  if (ImportCastPath(E, BasePath))
    return 0;

  return ImplicitCastExpr::Create(Importer.getToContext(), T, E->getCastKind(),
                                  SubExpr, &BasePath, E->getValueKind());
}

Expr *ASTNodeImporter::VisitCStyleCastExpr(CStyleCastExpr *E) {
  QualType T = Importer.Import(E->getType());
  if (T.isNull())
    return 0;
  
  Expr *SubExpr = Importer.Import(E->getSubExpr());
  if (!SubExpr)
    return 0;

  TypeSourceInfo *TInfo = Importer.Import(E->getTypeInfoAsWritten());
  if (!TInfo && E->getTypeInfoAsWritten())
    return 0;
  
  CXXCastPath BasePath;
  if (ImportCastPath(E, BasePath))
    return 0;

  return CStyleCastExpr::Create(Importer.getToContext(), T,
                                E->getValueKind(), E->getCastKind(),
                                SubExpr, &BasePath, TInfo,
                                Importer.Import(E->getLParenLoc()),
                                Importer.Import(E->getRParenLoc()));
}

ASTImporter::ASTImporter(ASTContext &ToContext, FileManager &ToFileManager,
                         ASTContext &FromContext, FileManager &FromFileManager,
                         bool MinimalImport)
  : ToContext(ToContext), FromContext(FromContext),
    ToFileManager(ToFileManager), FromFileManager(FromFileManager),
    Minimal(MinimalImport) 
{
  ImportedDecls[FromContext.getTranslationUnitDecl()]
    = ToContext.getTranslationUnitDecl();
}

ASTImporter::~ASTImporter() { }

QualType ASTImporter::Import(QualType FromT) {
  if (FromT.isNull())
    return QualType();

  const Type *fromTy = FromT.getTypePtr();
  
  // Check whether we've already imported this type.  
  llvm::DenseMap<const Type *, const Type *>::iterator Pos
    = ImportedTypes.find(fromTy);
  if (Pos != ImportedTypes.end())
    return ToContext.getQualifiedType(Pos->second, FromT.getLocalQualifiers());
  
  // Import the type
  ASTNodeImporter Importer(*this);
  QualType ToT = Importer.Visit(fromTy);
  if (ToT.isNull())
    return ToT;
  
  // Record the imported type.
  ImportedTypes[fromTy] = ToT.getTypePtr();
  
  return ToContext.getQualifiedType(ToT, FromT.getLocalQualifiers());
}

TypeSourceInfo *ASTImporter::Import(TypeSourceInfo *FromTSI) {
  if (!FromTSI)
    return FromTSI;

  // FIXME: For now we just create a "trivial" type source info based
  // on the type and a single location. Implement a real version of this.
  QualType T = Import(FromTSI->getType());
  if (T.isNull())
    return 0;

  return ToContext.getTrivialTypeSourceInfo(T, 
                        FromTSI->getTypeLoc().getSourceRange().getBegin());
}

Decl *ASTImporter::Import(Decl *FromD) {
  if (!FromD)
    return 0;

  // Check whether we've already imported this declaration.  
  llvm::DenseMap<Decl *, Decl *>::iterator Pos = ImportedDecls.find(FromD);
  if (Pos != ImportedDecls.end())
    return Pos->second;
  
  // Import the type
  ASTNodeImporter Importer(*this);
  Decl *ToD = Importer.Visit(FromD);
  if (!ToD)
    return 0;
  
  // Record the imported declaration.
  ImportedDecls[FromD] = ToD;
  
  if (TagDecl *FromTag = dyn_cast<TagDecl>(FromD)) {
    // Keep track of anonymous tags that have an associated typedef.
    if (FromTag->getTypedefNameForAnonDecl())
      AnonTagsWithPendingTypedefs.push_back(FromTag);
  } else if (TypedefNameDecl *FromTypedef = dyn_cast<TypedefNameDecl>(FromD)) {
    // When we've finished transforming a typedef, see whether it was the
    // typedef for an anonymous tag.
    for (llvm::SmallVector<TagDecl *, 4>::iterator
               FromTag = AnonTagsWithPendingTypedefs.begin(), 
            FromTagEnd = AnonTagsWithPendingTypedefs.end();
         FromTag != FromTagEnd; ++FromTag) {
      if ((*FromTag)->getTypedefNameForAnonDecl() == FromTypedef) {
        if (TagDecl *ToTag = cast_or_null<TagDecl>(Import(*FromTag))) {
          // We found the typedef for an anonymous tag; link them.
          ToTag->setTypedefNameForAnonDecl(cast<TypedefNameDecl>(ToD));
          AnonTagsWithPendingTypedefs.erase(FromTag);
          break;
        }
      }
    }
  }
  
  return ToD;
}

DeclContext *ASTImporter::ImportContext(DeclContext *FromDC) {
  if (!FromDC)
    return FromDC;

  return cast_or_null<DeclContext>(Import(cast<Decl>(FromDC)));
}

Expr *ASTImporter::Import(Expr *FromE) {
  if (!FromE)
    return 0;

  return cast_or_null<Expr>(Import(cast<Stmt>(FromE)));
}

Stmt *ASTImporter::Import(Stmt *FromS) {
  if (!FromS)
    return 0;

  // Check whether we've already imported this declaration.  
  llvm::DenseMap<Stmt *, Stmt *>::iterator Pos = ImportedStmts.find(FromS);
  if (Pos != ImportedStmts.end())
    return Pos->second;
  
  // Import the type
  ASTNodeImporter Importer(*this);
  Stmt *ToS = Importer.Visit(FromS);
  if (!ToS)
    return 0;
  
  // Record the imported declaration.
  ImportedStmts[FromS] = ToS;
  return ToS;
}

NestedNameSpecifier *ASTImporter::Import(NestedNameSpecifier *FromNNS) {
  if (!FromNNS)
    return 0;

  NestedNameSpecifier *prefix = Import(FromNNS->getPrefix());

  switch (FromNNS->getKind()) {
  case NestedNameSpecifier::Identifier:
    if (IdentifierInfo *II = Import(FromNNS->getAsIdentifier())) {
      return NestedNameSpecifier::Create(ToContext, prefix, II);
    }
    return 0;

  case NestedNameSpecifier::Namespace:
    if (NamespaceDecl *NS = 
          cast<NamespaceDecl>(Import(FromNNS->getAsNamespace()))) {
      return NestedNameSpecifier::Create(ToContext, prefix, NS);
    }
    return 0;

  case NestedNameSpecifier::NamespaceAlias:
    if (NamespaceAliasDecl *NSAD = 
          cast<NamespaceAliasDecl>(Import(FromNNS->getAsNamespaceAlias()))) {
      return NestedNameSpecifier::Create(ToContext, prefix, NSAD);
    }
    return 0;

  case NestedNameSpecifier::Global:
    return NestedNameSpecifier::GlobalSpecifier(ToContext);

  case NestedNameSpecifier::TypeSpec:
  case NestedNameSpecifier::TypeSpecWithTemplate: {
      QualType T = Import(QualType(FromNNS->getAsType(), 0u));
      if (!T.isNull()) {
        bool bTemplate = FromNNS->getKind() == 
                         NestedNameSpecifier::TypeSpecWithTemplate;
        return NestedNameSpecifier::Create(ToContext, prefix, 
                                           bTemplate, T.getTypePtr());
      }
    }
    return 0;
  }

  llvm_unreachable("Invalid nested name specifier kind");
  return 0;
}

NestedNameSpecifierLoc ASTImporter::Import(NestedNameSpecifierLoc FromNNS) {
  // FIXME: Implement!
  return NestedNameSpecifierLoc();
}

TemplateName ASTImporter::Import(TemplateName From) {
  switch (From.getKind()) {
  case TemplateName::Template:
    if (TemplateDecl *ToTemplate
                = cast_or_null<TemplateDecl>(Import(From.getAsTemplateDecl())))
      return TemplateName(ToTemplate);
      
    return TemplateName();
      
  case TemplateName::OverloadedTemplate: {
    OverloadedTemplateStorage *FromStorage = From.getAsOverloadedTemplate();
    UnresolvedSet<2> ToTemplates;
    for (OverloadedTemplateStorage::iterator I = FromStorage->begin(),
                                             E = FromStorage->end();
         I != E; ++I) {
      if (NamedDecl *To = cast_or_null<NamedDecl>(Import(*I))) 
        ToTemplates.addDecl(To);
      else
        return TemplateName();
    }
    return ToContext.getOverloadedTemplateName(ToTemplates.begin(), 
                                               ToTemplates.end());
  }
      
  case TemplateName::QualifiedTemplate: {
    QualifiedTemplateName *QTN = From.getAsQualifiedTemplateName();
    NestedNameSpecifier *Qualifier = Import(QTN->getQualifier());
    if (!Qualifier)
      return TemplateName();
    
    if (TemplateDecl *ToTemplate
        = cast_or_null<TemplateDecl>(Import(From.getAsTemplateDecl())))
      return ToContext.getQualifiedTemplateName(Qualifier, 
                                                QTN->hasTemplateKeyword(), 
                                                ToTemplate);
    
    return TemplateName();
  }
  
  case TemplateName::DependentTemplate: {
    DependentTemplateName *DTN = From.getAsDependentTemplateName();
    NestedNameSpecifier *Qualifier = Import(DTN->getQualifier());
    if (!Qualifier)
      return TemplateName();
    
    if (DTN->isIdentifier()) {
      return ToContext.getDependentTemplateName(Qualifier, 
                                                Import(DTN->getIdentifier()));
    }
    
    return ToContext.getDependentTemplateName(Qualifier, DTN->getOperator());
  }

  case TemplateName::SubstTemplateTemplateParm: {
    SubstTemplateTemplateParmStorage *subst
      = From.getAsSubstTemplateTemplateParm();
    TemplateTemplateParmDecl *param
      = cast_or_null<TemplateTemplateParmDecl>(Import(subst->getParameter()));
    if (!param)
      return TemplateName();

    TemplateName replacement = Import(subst->getReplacement());
    if (replacement.isNull()) return TemplateName();
    
    return ToContext.getSubstTemplateTemplateParm(param, replacement);
  }
      
  case TemplateName::SubstTemplateTemplateParmPack: {
    SubstTemplateTemplateParmPackStorage *SubstPack
      = From.getAsSubstTemplateTemplateParmPack();
    TemplateTemplateParmDecl *Param
      = cast_or_null<TemplateTemplateParmDecl>(
                                        Import(SubstPack->getParameterPack()));
    if (!Param)
      return TemplateName();
    
    ASTNodeImporter Importer(*this);
    TemplateArgument ArgPack 
      = Importer.ImportTemplateArgument(SubstPack->getArgumentPack());
    if (ArgPack.isNull())
      return TemplateName();
    
    return ToContext.getSubstTemplateTemplateParmPack(Param, ArgPack);
  }
  }
  
  llvm_unreachable("Invalid template name kind");
  return TemplateName();
}

SourceLocation ASTImporter::Import(SourceLocation FromLoc) {
  if (FromLoc.isInvalid())
    return SourceLocation();

  SourceManager &FromSM = FromContext.getSourceManager();
  
  // For now, map everything down to its spelling location, so that we
  // don't have to import macro instantiations.
  // FIXME: Import macro instantiations!
  FromLoc = FromSM.getSpellingLoc(FromLoc);
  std::pair<FileID, unsigned> Decomposed = FromSM.getDecomposedLoc(FromLoc);
  SourceManager &ToSM = ToContext.getSourceManager();
  return ToSM.getLocForStartOfFile(Import(Decomposed.first))
             .getFileLocWithOffset(Decomposed.second);
}

SourceRange ASTImporter::Import(SourceRange FromRange) {
  return SourceRange(Import(FromRange.getBegin()), Import(FromRange.getEnd()));
}

FileID ASTImporter::Import(FileID FromID) {
  llvm::DenseMap<FileID, FileID>::iterator Pos
    = ImportedFileIDs.find(FromID);
  if (Pos != ImportedFileIDs.end())
    return Pos->second;
  
  SourceManager &FromSM = FromContext.getSourceManager();
  SourceManager &ToSM = ToContext.getSourceManager();
  const SrcMgr::SLocEntry &FromSLoc = FromSM.getSLocEntry(FromID);
  assert(FromSLoc.isFile() && "Cannot handle macro instantiations yet");
  
  // Include location of this file.
  SourceLocation ToIncludeLoc = Import(FromSLoc.getFile().getIncludeLoc());
  
  // Map the FileID for to the "to" source manager.
  FileID ToID;
  const SrcMgr::ContentCache *Cache = FromSLoc.getFile().getContentCache();
  if (Cache->OrigEntry) {
    // FIXME: We probably want to use getVirtualFile(), so we don't hit the
    // disk again
    // FIXME: We definitely want to re-use the existing MemoryBuffer, rather
    // than mmap the files several times.
    const FileEntry *Entry = ToFileManager.getFile(Cache->OrigEntry->getName());
    ToID = ToSM.createFileID(Entry, ToIncludeLoc, 
                             FromSLoc.getFile().getFileCharacteristic());
  } else {
    // FIXME: We want to re-use the existing MemoryBuffer!
    const llvm::MemoryBuffer *
        FromBuf = Cache->getBuffer(FromContext.getDiagnostics(), FromSM);
    llvm::MemoryBuffer *ToBuf
      = llvm::MemoryBuffer::getMemBufferCopy(FromBuf->getBuffer(),
                                             FromBuf->getBufferIdentifier());
    ToID = ToSM.createFileIDForMemBuffer(ToBuf);
  }
  
  
  ImportedFileIDs[FromID] = ToID;
  return ToID;
}

void ASTImporter::ImportDefinition(Decl *From) {
  Decl *To = Import(From);
  if (!To)
    return;
  
  if (DeclContext *FromDC = cast<DeclContext>(From)) {
    ASTNodeImporter Importer(*this);
    Importer.ImportDeclContext(FromDC, true);
  }
}

DeclarationName ASTImporter::Import(DeclarationName FromName) {
  if (!FromName)
    return DeclarationName();

  switch (FromName.getNameKind()) {
  case DeclarationName::Identifier:
    return Import(FromName.getAsIdentifierInfo());

  case DeclarationName::ObjCZeroArgSelector:
  case DeclarationName::ObjCOneArgSelector:
  case DeclarationName::ObjCMultiArgSelector:
    return Import(FromName.getObjCSelector());

  case DeclarationName::CXXConstructorName: {
    QualType T = Import(FromName.getCXXNameType());
    if (T.isNull())
      return DeclarationName();

    return ToContext.DeclarationNames.getCXXConstructorName(
                                               ToContext.getCanonicalType(T));
  }

  case DeclarationName::CXXDestructorName: {
    QualType T = Import(FromName.getCXXNameType());
    if (T.isNull())
      return DeclarationName();

    return ToContext.DeclarationNames.getCXXDestructorName(
                                               ToContext.getCanonicalType(T));
  }

  case DeclarationName::CXXConversionFunctionName: {
    QualType T = Import(FromName.getCXXNameType());
    if (T.isNull())
      return DeclarationName();

    return ToContext.DeclarationNames.getCXXConversionFunctionName(
                                               ToContext.getCanonicalType(T));
  }

  case DeclarationName::CXXOperatorName:
    return ToContext.DeclarationNames.getCXXOperatorName(
                                          FromName.getCXXOverloadedOperator());

  case DeclarationName::CXXLiteralOperatorName:
    return ToContext.DeclarationNames.getCXXLiteralOperatorName(
                                   Import(FromName.getCXXLiteralIdentifier()));

  case DeclarationName::CXXUsingDirective:
    // FIXME: STATICS!
    return DeclarationName::getUsingDirectiveName();
  }

  // Silence bogus GCC warning
  return DeclarationName();
}

IdentifierInfo *ASTImporter::Import(const IdentifierInfo *FromId) {
  if (!FromId)
    return 0;

  return &ToContext.Idents.get(FromId->getName());
}

Selector ASTImporter::Import(Selector FromSel) {
  if (FromSel.isNull())
    return Selector();

  llvm::SmallVector<IdentifierInfo *, 4> Idents;
  Idents.push_back(Import(FromSel.getIdentifierInfoForSlot(0)));
  for (unsigned I = 1, N = FromSel.getNumArgs(); I < N; ++I)
    Idents.push_back(Import(FromSel.getIdentifierInfoForSlot(I)));
  return ToContext.Selectors.getSelector(FromSel.getNumArgs(), Idents.data());
}

DeclarationName ASTImporter::HandleNameConflict(DeclarationName Name,
                                                DeclContext *DC,
                                                unsigned IDNS,
                                                NamedDecl **Decls,
                                                unsigned NumDecls) {
  return Name;
}

DiagnosticBuilder ASTImporter::ToDiag(SourceLocation Loc, unsigned DiagID) {
  return ToContext.getDiagnostics().Report(Loc, DiagID);
}

DiagnosticBuilder ASTImporter::FromDiag(SourceLocation Loc, unsigned DiagID) {
  return FromContext.getDiagnostics().Report(Loc, DiagID);
}

Decl *ASTImporter::Imported(Decl *From, Decl *To) {
  ImportedDecls[From] = To;
  return To;
}

bool ASTImporter::IsStructurallyEquivalent(QualType From, QualType To) {
  llvm::DenseMap<const Type *, const Type *>::iterator Pos
   = ImportedTypes.find(From.getTypePtr());
  if (Pos != ImportedTypes.end() && ToContext.hasSameType(Import(From), To))
    return true;
      
  StructuralEquivalenceContext Ctx(FromContext, ToContext, NonEquivalentDecls);
  return Ctx.IsStructurallyEquivalent(From, To);
}
