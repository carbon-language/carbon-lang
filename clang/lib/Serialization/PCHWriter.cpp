//===--- PCHWriter.cpp - AST File Writer ----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the ASTWriter class, which writes AST files.
//
//===----------------------------------------------------------------------===//

#include "clang/Serialization/PCHWriter.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/IdentifierResolver.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclContextInternals.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Type.h"
#include "clang/AST/TypeLocVisitor.h"
#include "clang/Serialization/PCHReader.h"
#include "clang/Lex/MacroInfo.h"
#include "clang/Lex/PreprocessingRecord.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/OnDiskHashTable.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/SourceManagerInternals.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/Version.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Bitcode/BitstreamWriter.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/System/Path.h"
#include <cstdio>
using namespace clang;

template <typename T, typename Allocator>
T *data(std::vector<T, Allocator> &v) {
  return v.empty() ? 0 : &v.front();
}
template <typename T, typename Allocator>
const T *data(const std::vector<T, Allocator> &v) {
  return v.empty() ? 0 : &v.front();
}

//===----------------------------------------------------------------------===//
// Type serialization
//===----------------------------------------------------------------------===//

namespace {
  class ASTTypeWriter {
    ASTWriter &Writer;
    ASTWriter::RecordData &Record;

  public:
    /// \brief Type code that corresponds to the record generated.
    pch::TypeCode Code;

    ASTTypeWriter(ASTWriter &Writer, ASTWriter::RecordData &Record)
      : Writer(Writer), Record(Record), Code(pch::TYPE_EXT_QUAL) { }

    void VisitArrayType(const ArrayType *T);
    void VisitFunctionType(const FunctionType *T);
    void VisitTagType(const TagType *T);

#define TYPE(Class, Base) void Visit##Class##Type(const Class##Type *T);
#define ABSTRACT_TYPE(Class, Base)
#include "clang/AST/TypeNodes.def"
  };
}

void ASTTypeWriter::VisitBuiltinType(const BuiltinType *T) {
  assert(false && "Built-in types are never serialized");
}

void ASTTypeWriter::VisitComplexType(const ComplexType *T) {
  Writer.AddTypeRef(T->getElementType(), Record);
  Code = pch::TYPE_COMPLEX;
}

void ASTTypeWriter::VisitPointerType(const PointerType *T) {
  Writer.AddTypeRef(T->getPointeeType(), Record);
  Code = pch::TYPE_POINTER;
}

void ASTTypeWriter::VisitBlockPointerType(const BlockPointerType *T) {
  Writer.AddTypeRef(T->getPointeeType(), Record);
  Code = pch::TYPE_BLOCK_POINTER;
}

void ASTTypeWriter::VisitLValueReferenceType(const LValueReferenceType *T) {
  Writer.AddTypeRef(T->getPointeeType(), Record);
  Code = pch::TYPE_LVALUE_REFERENCE;
}

void ASTTypeWriter::VisitRValueReferenceType(const RValueReferenceType *T) {
  Writer.AddTypeRef(T->getPointeeType(), Record);
  Code = pch::TYPE_RVALUE_REFERENCE;
}

void ASTTypeWriter::VisitMemberPointerType(const MemberPointerType *T) {
  Writer.AddTypeRef(T->getPointeeType(), Record);
  Writer.AddTypeRef(QualType(T->getClass(), 0), Record);
  Code = pch::TYPE_MEMBER_POINTER;
}

void ASTTypeWriter::VisitArrayType(const ArrayType *T) {
  Writer.AddTypeRef(T->getElementType(), Record);
  Record.push_back(T->getSizeModifier()); // FIXME: stable values
  Record.push_back(T->getIndexTypeCVRQualifiers()); // FIXME: stable values
}

void ASTTypeWriter::VisitConstantArrayType(const ConstantArrayType *T) {
  VisitArrayType(T);
  Writer.AddAPInt(T->getSize(), Record);
  Code = pch::TYPE_CONSTANT_ARRAY;
}

void ASTTypeWriter::VisitIncompleteArrayType(const IncompleteArrayType *T) {
  VisitArrayType(T);
  Code = pch::TYPE_INCOMPLETE_ARRAY;
}

void ASTTypeWriter::VisitVariableArrayType(const VariableArrayType *T) {
  VisitArrayType(T);
  Writer.AddSourceLocation(T->getLBracketLoc(), Record);
  Writer.AddSourceLocation(T->getRBracketLoc(), Record);
  Writer.AddStmt(T->getSizeExpr());
  Code = pch::TYPE_VARIABLE_ARRAY;
}

void ASTTypeWriter::VisitVectorType(const VectorType *T) {
  Writer.AddTypeRef(T->getElementType(), Record);
  Record.push_back(T->getNumElements());
  Record.push_back(T->getAltiVecSpecific());
  Code = pch::TYPE_VECTOR;
}

void ASTTypeWriter::VisitExtVectorType(const ExtVectorType *T) {
  VisitVectorType(T);
  Code = pch::TYPE_EXT_VECTOR;
}

void ASTTypeWriter::VisitFunctionType(const FunctionType *T) {
  Writer.AddTypeRef(T->getResultType(), Record);
  FunctionType::ExtInfo C = T->getExtInfo();
  Record.push_back(C.getNoReturn());
  Record.push_back(C.getRegParm());
  // FIXME: need to stabilize encoding of calling convention...
  Record.push_back(C.getCC());
}

void ASTTypeWriter::VisitFunctionNoProtoType(const FunctionNoProtoType *T) {
  VisitFunctionType(T);
  Code = pch::TYPE_FUNCTION_NO_PROTO;
}

void ASTTypeWriter::VisitFunctionProtoType(const FunctionProtoType *T) {
  VisitFunctionType(T);
  Record.push_back(T->getNumArgs());
  for (unsigned I = 0, N = T->getNumArgs(); I != N; ++I)
    Writer.AddTypeRef(T->getArgType(I), Record);
  Record.push_back(T->isVariadic());
  Record.push_back(T->getTypeQuals());
  Record.push_back(T->hasExceptionSpec());
  Record.push_back(T->hasAnyExceptionSpec());
  Record.push_back(T->getNumExceptions());
  for (unsigned I = 0, N = T->getNumExceptions(); I != N; ++I)
    Writer.AddTypeRef(T->getExceptionType(I), Record);
  Code = pch::TYPE_FUNCTION_PROTO;
}

void ASTTypeWriter::VisitUnresolvedUsingType(const UnresolvedUsingType *T) {
  Writer.AddDeclRef(T->getDecl(), Record);
  Code = pch::TYPE_UNRESOLVED_USING;
}

void ASTTypeWriter::VisitTypedefType(const TypedefType *T) {
  Writer.AddDeclRef(T->getDecl(), Record);
  assert(!T->isCanonicalUnqualified() && "Invalid typedef ?");
  Writer.AddTypeRef(T->getCanonicalTypeInternal(), Record);
  Code = pch::TYPE_TYPEDEF;
}

void ASTTypeWriter::VisitTypeOfExprType(const TypeOfExprType *T) {
  Writer.AddStmt(T->getUnderlyingExpr());
  Code = pch::TYPE_TYPEOF_EXPR;
}

void ASTTypeWriter::VisitTypeOfType(const TypeOfType *T) {
  Writer.AddTypeRef(T->getUnderlyingType(), Record);
  Code = pch::TYPE_TYPEOF;
}

void ASTTypeWriter::VisitDecltypeType(const DecltypeType *T) {
  Writer.AddStmt(T->getUnderlyingExpr());
  Code = pch::TYPE_DECLTYPE;
}

void ASTTypeWriter::VisitTagType(const TagType *T) {
  Record.push_back(T->isDependentType());
  Writer.AddDeclRef(T->getDecl(), Record);
  assert(!T->isBeingDefined() &&
         "Cannot serialize in the middle of a type definition");
}

void ASTTypeWriter::VisitRecordType(const RecordType *T) {
  VisitTagType(T);
  Code = pch::TYPE_RECORD;
}

void ASTTypeWriter::VisitEnumType(const EnumType *T) {
  VisitTagType(T);
  Code = pch::TYPE_ENUM;
}

void
ASTTypeWriter::VisitSubstTemplateTypeParmType(
                                        const SubstTemplateTypeParmType *T) {
  Writer.AddTypeRef(QualType(T->getReplacedParameter(), 0), Record);
  Writer.AddTypeRef(T->getReplacementType(), Record);
  Code = pch::TYPE_SUBST_TEMPLATE_TYPE_PARM;
}

void
ASTTypeWriter::VisitTemplateSpecializationType(
                                       const TemplateSpecializationType *T) {
  Record.push_back(T->isDependentType());
  Writer.AddTemplateName(T->getTemplateName(), Record);
  Record.push_back(T->getNumArgs());
  for (TemplateSpecializationType::iterator ArgI = T->begin(), ArgE = T->end();
         ArgI != ArgE; ++ArgI)
    Writer.AddTemplateArgument(*ArgI, Record);
  Writer.AddTypeRef(T->isCanonicalUnqualified() ? QualType()
                                                : T->getCanonicalTypeInternal(),
                    Record);
  Code = pch::TYPE_TEMPLATE_SPECIALIZATION;
}

void
ASTTypeWriter::VisitDependentSizedArrayType(const DependentSizedArrayType *T) {
  VisitArrayType(T);
  Writer.AddStmt(T->getSizeExpr());
  Writer.AddSourceRange(T->getBracketsRange(), Record);
  Code = pch::TYPE_DEPENDENT_SIZED_ARRAY;
}

void
ASTTypeWriter::VisitDependentSizedExtVectorType(
                                        const DependentSizedExtVectorType *T) {
  // FIXME: Serialize this type (C++ only)
  assert(false && "Cannot serialize dependent sized extended vector types");
}

void
ASTTypeWriter::VisitTemplateTypeParmType(const TemplateTypeParmType *T) {
  Record.push_back(T->getDepth());
  Record.push_back(T->getIndex());
  Record.push_back(T->isParameterPack());
  Writer.AddIdentifierRef(T->getName(), Record);
  Code = pch::TYPE_TEMPLATE_TYPE_PARM;
}

void
ASTTypeWriter::VisitDependentNameType(const DependentNameType *T) {
  Record.push_back(T->getKeyword());
  Writer.AddNestedNameSpecifier(T->getQualifier(), Record);
  Writer.AddIdentifierRef(T->getIdentifier(), Record);
  Writer.AddTypeRef(T->isCanonicalUnqualified() ? QualType()
                                                : T->getCanonicalTypeInternal(),
                    Record);
  Code = pch::TYPE_DEPENDENT_NAME;
}

void
ASTTypeWriter::VisitDependentTemplateSpecializationType(
                                const DependentTemplateSpecializationType *T) {
  Record.push_back(T->getKeyword());
  Writer.AddNestedNameSpecifier(T->getQualifier(), Record);
  Writer.AddIdentifierRef(T->getIdentifier(), Record);
  Record.push_back(T->getNumArgs());
  for (DependentTemplateSpecializationType::iterator
         I = T->begin(), E = T->end(); I != E; ++I)
    Writer.AddTemplateArgument(*I, Record);
  Code = pch::TYPE_DEPENDENT_TEMPLATE_SPECIALIZATION;
}

void ASTTypeWriter::VisitElaboratedType(const ElaboratedType *T) {
  Record.push_back(T->getKeyword());
  Writer.AddNestedNameSpecifier(T->getQualifier(), Record);
  Writer.AddTypeRef(T->getNamedType(), Record);
  Code = pch::TYPE_ELABORATED;
}

void ASTTypeWriter::VisitInjectedClassNameType(const InjectedClassNameType *T) {
  Writer.AddDeclRef(T->getDecl(), Record);
  Writer.AddTypeRef(T->getInjectedSpecializationType(), Record);
  Code = pch::TYPE_INJECTED_CLASS_NAME;
}

void ASTTypeWriter::VisitObjCInterfaceType(const ObjCInterfaceType *T) {
  Writer.AddDeclRef(T->getDecl(), Record);
  Code = pch::TYPE_OBJC_INTERFACE;
}

void ASTTypeWriter::VisitObjCObjectType(const ObjCObjectType *T) {
  Writer.AddTypeRef(T->getBaseType(), Record);
  Record.push_back(T->getNumProtocols());
  for (ObjCObjectType::qual_iterator I = T->qual_begin(),
       E = T->qual_end(); I != E; ++I)
    Writer.AddDeclRef(*I, Record);
  Code = pch::TYPE_OBJC_OBJECT;
}

void
ASTTypeWriter::VisitObjCObjectPointerType(const ObjCObjectPointerType *T) {
  Writer.AddTypeRef(T->getPointeeType(), Record);
  Code = pch::TYPE_OBJC_OBJECT_POINTER;
}

namespace {

class TypeLocWriter : public TypeLocVisitor<TypeLocWriter> {
  ASTWriter &Writer;
  ASTWriter::RecordData &Record;

public:
  TypeLocWriter(ASTWriter &Writer, ASTWriter::RecordData &Record)
    : Writer(Writer), Record(Record) { }

#define ABSTRACT_TYPELOC(CLASS, PARENT)
#define TYPELOC(CLASS, PARENT) \
    void Visit##CLASS##TypeLoc(CLASS##TypeLoc TyLoc);
#include "clang/AST/TypeLocNodes.def"

  void VisitArrayTypeLoc(ArrayTypeLoc TyLoc);
  void VisitFunctionTypeLoc(FunctionTypeLoc TyLoc);
};

}

void TypeLocWriter::VisitQualifiedTypeLoc(QualifiedTypeLoc TL) {
  // nothing to do
}
void TypeLocWriter::VisitBuiltinTypeLoc(BuiltinTypeLoc TL) {
  Writer.AddSourceLocation(TL.getBuiltinLoc(), Record);
  if (TL.needsExtraLocalData()) {
    Record.push_back(TL.getWrittenTypeSpec());
    Record.push_back(TL.getWrittenSignSpec());
    Record.push_back(TL.getWrittenWidthSpec());
    Record.push_back(TL.hasModeAttr());
  }
}
void TypeLocWriter::VisitComplexTypeLoc(ComplexTypeLoc TL) {
  Writer.AddSourceLocation(TL.getNameLoc(), Record);
}
void TypeLocWriter::VisitPointerTypeLoc(PointerTypeLoc TL) {
  Writer.AddSourceLocation(TL.getStarLoc(), Record);
}
void TypeLocWriter::VisitBlockPointerTypeLoc(BlockPointerTypeLoc TL) {
  Writer.AddSourceLocation(TL.getCaretLoc(), Record);
}
void TypeLocWriter::VisitLValueReferenceTypeLoc(LValueReferenceTypeLoc TL) {
  Writer.AddSourceLocation(TL.getAmpLoc(), Record);
}
void TypeLocWriter::VisitRValueReferenceTypeLoc(RValueReferenceTypeLoc TL) {
  Writer.AddSourceLocation(TL.getAmpAmpLoc(), Record);
}
void TypeLocWriter::VisitMemberPointerTypeLoc(MemberPointerTypeLoc TL) {
  Writer.AddSourceLocation(TL.getStarLoc(), Record);
}
void TypeLocWriter::VisitArrayTypeLoc(ArrayTypeLoc TL) {
  Writer.AddSourceLocation(TL.getLBracketLoc(), Record);
  Writer.AddSourceLocation(TL.getRBracketLoc(), Record);
  Record.push_back(TL.getSizeExpr() ? 1 : 0);
  if (TL.getSizeExpr())
    Writer.AddStmt(TL.getSizeExpr());
}
void TypeLocWriter::VisitConstantArrayTypeLoc(ConstantArrayTypeLoc TL) {
  VisitArrayTypeLoc(TL);
}
void TypeLocWriter::VisitIncompleteArrayTypeLoc(IncompleteArrayTypeLoc TL) {
  VisitArrayTypeLoc(TL);
}
void TypeLocWriter::VisitVariableArrayTypeLoc(VariableArrayTypeLoc TL) {
  VisitArrayTypeLoc(TL);
}
void TypeLocWriter::VisitDependentSizedArrayTypeLoc(
                                            DependentSizedArrayTypeLoc TL) {
  VisitArrayTypeLoc(TL);
}
void TypeLocWriter::VisitDependentSizedExtVectorTypeLoc(
                                        DependentSizedExtVectorTypeLoc TL) {
  Writer.AddSourceLocation(TL.getNameLoc(), Record);
}
void TypeLocWriter::VisitVectorTypeLoc(VectorTypeLoc TL) {
  Writer.AddSourceLocation(TL.getNameLoc(), Record);
}
void TypeLocWriter::VisitExtVectorTypeLoc(ExtVectorTypeLoc TL) {
  Writer.AddSourceLocation(TL.getNameLoc(), Record);
}
void TypeLocWriter::VisitFunctionTypeLoc(FunctionTypeLoc TL) {
  Writer.AddSourceLocation(TL.getLParenLoc(), Record);
  Writer.AddSourceLocation(TL.getRParenLoc(), Record);
  for (unsigned i = 0, e = TL.getNumArgs(); i != e; ++i)
    Writer.AddDeclRef(TL.getArg(i), Record);
}
void TypeLocWriter::VisitFunctionProtoTypeLoc(FunctionProtoTypeLoc TL) {
  VisitFunctionTypeLoc(TL);
}
void TypeLocWriter::VisitFunctionNoProtoTypeLoc(FunctionNoProtoTypeLoc TL) {
  VisitFunctionTypeLoc(TL);
}
void TypeLocWriter::VisitUnresolvedUsingTypeLoc(UnresolvedUsingTypeLoc TL) {
  Writer.AddSourceLocation(TL.getNameLoc(), Record);
}
void TypeLocWriter::VisitTypedefTypeLoc(TypedefTypeLoc TL) {
  Writer.AddSourceLocation(TL.getNameLoc(), Record);
}
void TypeLocWriter::VisitTypeOfExprTypeLoc(TypeOfExprTypeLoc TL) {
  Writer.AddSourceLocation(TL.getTypeofLoc(), Record);
  Writer.AddSourceLocation(TL.getLParenLoc(), Record);
  Writer.AddSourceLocation(TL.getRParenLoc(), Record);
}
void TypeLocWriter::VisitTypeOfTypeLoc(TypeOfTypeLoc TL) {
  Writer.AddSourceLocation(TL.getTypeofLoc(), Record);
  Writer.AddSourceLocation(TL.getLParenLoc(), Record);
  Writer.AddSourceLocation(TL.getRParenLoc(), Record);
  Writer.AddTypeSourceInfo(TL.getUnderlyingTInfo(), Record);
}
void TypeLocWriter::VisitDecltypeTypeLoc(DecltypeTypeLoc TL) {
  Writer.AddSourceLocation(TL.getNameLoc(), Record);
}
void TypeLocWriter::VisitRecordTypeLoc(RecordTypeLoc TL) {
  Writer.AddSourceLocation(TL.getNameLoc(), Record);
}
void TypeLocWriter::VisitEnumTypeLoc(EnumTypeLoc TL) {
  Writer.AddSourceLocation(TL.getNameLoc(), Record);
}
void TypeLocWriter::VisitTemplateTypeParmTypeLoc(TemplateTypeParmTypeLoc TL) {
  Writer.AddSourceLocation(TL.getNameLoc(), Record);
}
void TypeLocWriter::VisitSubstTemplateTypeParmTypeLoc(
                                            SubstTemplateTypeParmTypeLoc TL) {
  Writer.AddSourceLocation(TL.getNameLoc(), Record);
}
void TypeLocWriter::VisitTemplateSpecializationTypeLoc(
                                           TemplateSpecializationTypeLoc TL) {
  Writer.AddSourceLocation(TL.getTemplateNameLoc(), Record);
  Writer.AddSourceLocation(TL.getLAngleLoc(), Record);
  Writer.AddSourceLocation(TL.getRAngleLoc(), Record);
  for (unsigned i = 0, e = TL.getNumArgs(); i != e; ++i)
    Writer.AddTemplateArgumentLocInfo(TL.getArgLoc(i).getArgument().getKind(),
                                      TL.getArgLoc(i).getLocInfo(), Record);
}
void TypeLocWriter::VisitElaboratedTypeLoc(ElaboratedTypeLoc TL) {
  Writer.AddSourceLocation(TL.getKeywordLoc(), Record);
  Writer.AddSourceRange(TL.getQualifierRange(), Record);
}
void TypeLocWriter::VisitInjectedClassNameTypeLoc(InjectedClassNameTypeLoc TL) {
  Writer.AddSourceLocation(TL.getNameLoc(), Record);
}
void TypeLocWriter::VisitDependentNameTypeLoc(DependentNameTypeLoc TL) {
  Writer.AddSourceLocation(TL.getKeywordLoc(), Record);
  Writer.AddSourceRange(TL.getQualifierRange(), Record);
  Writer.AddSourceLocation(TL.getNameLoc(), Record);
}
void TypeLocWriter::VisitDependentTemplateSpecializationTypeLoc(
       DependentTemplateSpecializationTypeLoc TL) {
  Writer.AddSourceLocation(TL.getKeywordLoc(), Record);
  Writer.AddSourceRange(TL.getQualifierRange(), Record);
  Writer.AddSourceLocation(TL.getNameLoc(), Record);
  Writer.AddSourceLocation(TL.getLAngleLoc(), Record);
  Writer.AddSourceLocation(TL.getRAngleLoc(), Record);
  for (unsigned I = 0, E = TL.getNumArgs(); I != E; ++I)
    Writer.AddTemplateArgumentLocInfo(TL.getArgLoc(I).getArgument().getKind(),
                                      TL.getArgLoc(I).getLocInfo(), Record);
}
void TypeLocWriter::VisitObjCInterfaceTypeLoc(ObjCInterfaceTypeLoc TL) {
  Writer.AddSourceLocation(TL.getNameLoc(), Record);
}
void TypeLocWriter::VisitObjCObjectTypeLoc(ObjCObjectTypeLoc TL) {
  Record.push_back(TL.hasBaseTypeAsWritten());
  Writer.AddSourceLocation(TL.getLAngleLoc(), Record);
  Writer.AddSourceLocation(TL.getRAngleLoc(), Record);
  for (unsigned i = 0, e = TL.getNumProtocols(); i != e; ++i)
    Writer.AddSourceLocation(TL.getProtocolLoc(i), Record);
}
void TypeLocWriter::VisitObjCObjectPointerTypeLoc(ObjCObjectPointerTypeLoc TL) {
  Writer.AddSourceLocation(TL.getStarLoc(), Record);
}

//===----------------------------------------------------------------------===//
// ASTWriter Implementation
//===----------------------------------------------------------------------===//

static void EmitBlockID(unsigned ID, const char *Name,
                        llvm::BitstreamWriter &Stream,
                        ASTWriter::RecordData &Record) {
  Record.clear();
  Record.push_back(ID);
  Stream.EmitRecord(llvm::bitc::BLOCKINFO_CODE_SETBID, Record);

  // Emit the block name if present.
  if (Name == 0 || Name[0] == 0) return;
  Record.clear();
  while (*Name)
    Record.push_back(*Name++);
  Stream.EmitRecord(llvm::bitc::BLOCKINFO_CODE_BLOCKNAME, Record);
}

static void EmitRecordID(unsigned ID, const char *Name,
                         llvm::BitstreamWriter &Stream,
                         ASTWriter::RecordData &Record) {
  Record.clear();
  Record.push_back(ID);
  while (*Name)
    Record.push_back(*Name++);
  Stream.EmitRecord(llvm::bitc::BLOCKINFO_CODE_SETRECORDNAME, Record);
}

static void AddStmtsExprs(llvm::BitstreamWriter &Stream,
                          ASTWriter::RecordData &Record) {
#define RECORD(X) EmitRecordID(pch::X, #X, Stream, Record)
  RECORD(STMT_STOP);
  RECORD(STMT_NULL_PTR);
  RECORD(STMT_NULL);
  RECORD(STMT_COMPOUND);
  RECORD(STMT_CASE);
  RECORD(STMT_DEFAULT);
  RECORD(STMT_LABEL);
  RECORD(STMT_IF);
  RECORD(STMT_SWITCH);
  RECORD(STMT_WHILE);
  RECORD(STMT_DO);
  RECORD(STMT_FOR);
  RECORD(STMT_GOTO);
  RECORD(STMT_INDIRECT_GOTO);
  RECORD(STMT_CONTINUE);
  RECORD(STMT_BREAK);
  RECORD(STMT_RETURN);
  RECORD(STMT_DECL);
  RECORD(STMT_ASM);
  RECORD(EXPR_PREDEFINED);
  RECORD(EXPR_DECL_REF);
  RECORD(EXPR_INTEGER_LITERAL);
  RECORD(EXPR_FLOATING_LITERAL);
  RECORD(EXPR_IMAGINARY_LITERAL);
  RECORD(EXPR_STRING_LITERAL);
  RECORD(EXPR_CHARACTER_LITERAL);
  RECORD(EXPR_PAREN);
  RECORD(EXPR_UNARY_OPERATOR);
  RECORD(EXPR_SIZEOF_ALIGN_OF);
  RECORD(EXPR_ARRAY_SUBSCRIPT);
  RECORD(EXPR_CALL);
  RECORD(EXPR_MEMBER);
  RECORD(EXPR_BINARY_OPERATOR);
  RECORD(EXPR_COMPOUND_ASSIGN_OPERATOR);
  RECORD(EXPR_CONDITIONAL_OPERATOR);
  RECORD(EXPR_IMPLICIT_CAST);
  RECORD(EXPR_CSTYLE_CAST);
  RECORD(EXPR_COMPOUND_LITERAL);
  RECORD(EXPR_EXT_VECTOR_ELEMENT);
  RECORD(EXPR_INIT_LIST);
  RECORD(EXPR_DESIGNATED_INIT);
  RECORD(EXPR_IMPLICIT_VALUE_INIT);
  RECORD(EXPR_VA_ARG);
  RECORD(EXPR_ADDR_LABEL);
  RECORD(EXPR_STMT);
  RECORD(EXPR_TYPES_COMPATIBLE);
  RECORD(EXPR_CHOOSE);
  RECORD(EXPR_GNU_NULL);
  RECORD(EXPR_SHUFFLE_VECTOR);
  RECORD(EXPR_BLOCK);
  RECORD(EXPR_BLOCK_DECL_REF);
  RECORD(EXPR_OBJC_STRING_LITERAL);
  RECORD(EXPR_OBJC_ENCODE);
  RECORD(EXPR_OBJC_SELECTOR_EXPR);
  RECORD(EXPR_OBJC_PROTOCOL_EXPR);
  RECORD(EXPR_OBJC_IVAR_REF_EXPR);
  RECORD(EXPR_OBJC_PROPERTY_REF_EXPR);
  RECORD(EXPR_OBJC_KVC_REF_EXPR);
  RECORD(EXPR_OBJC_MESSAGE_EXPR);
  RECORD(EXPR_OBJC_SUPER_EXPR);
  RECORD(STMT_OBJC_FOR_COLLECTION);
  RECORD(STMT_OBJC_CATCH);
  RECORD(STMT_OBJC_FINALLY);
  RECORD(STMT_OBJC_AT_TRY);
  RECORD(STMT_OBJC_AT_SYNCHRONIZED);
  RECORD(STMT_OBJC_AT_THROW);
  RECORD(EXPR_CXX_OPERATOR_CALL);
  RECORD(EXPR_CXX_CONSTRUCT);
  RECORD(EXPR_CXX_STATIC_CAST);
  RECORD(EXPR_CXX_DYNAMIC_CAST);
  RECORD(EXPR_CXX_REINTERPRET_CAST);
  RECORD(EXPR_CXX_CONST_CAST);
  RECORD(EXPR_CXX_FUNCTIONAL_CAST);
  RECORD(EXPR_CXX_BOOL_LITERAL);
  RECORD(EXPR_CXX_NULL_PTR_LITERAL);
#undef RECORD
}

void ASTWriter::WriteBlockInfoBlock() {
  RecordData Record;
  Stream.EnterSubblock(llvm::bitc::BLOCKINFO_BLOCK_ID, 3);

#define BLOCK(X) EmitBlockID(pch::X ## _ID, #X, Stream, Record)
#define RECORD(X) EmitRecordID(pch::X, #X, Stream, Record)

  // AST Top-Level Block.
  BLOCK(PCH_BLOCK);
  RECORD(ORIGINAL_FILE_NAME);
  RECORD(TYPE_OFFSET);
  RECORD(DECL_OFFSET);
  RECORD(LANGUAGE_OPTIONS);
  RECORD(METADATA);
  RECORD(IDENTIFIER_OFFSET);
  RECORD(IDENTIFIER_TABLE);
  RECORD(EXTERNAL_DEFINITIONS);
  RECORD(SPECIAL_TYPES);
  RECORD(STATISTICS);
  RECORD(TENTATIVE_DEFINITIONS);
  RECORD(UNUSED_FILESCOPED_DECLS);
  RECORD(LOCALLY_SCOPED_EXTERNAL_DECLS);
  RECORD(SELECTOR_OFFSETS);
  RECORD(METHOD_POOL);
  RECORD(PP_COUNTER_VALUE);
  RECORD(SOURCE_LOCATION_OFFSETS);
  RECORD(SOURCE_LOCATION_PRELOADS);
  RECORD(STAT_CACHE);
  RECORD(EXT_VECTOR_DECLS);
  RECORD(VERSION_CONTROL_BRANCH_REVISION);
  RECORD(MACRO_DEFINITION_OFFSETS);
  RECORD(CHAINED_METADATA);
  RECORD(REFERENCED_SELECTOR_POOL);
  
  // SourceManager Block.
  BLOCK(SOURCE_MANAGER_BLOCK);
  RECORD(SM_SLOC_FILE_ENTRY);
  RECORD(SM_SLOC_BUFFER_ENTRY);
  RECORD(SM_SLOC_BUFFER_BLOB);
  RECORD(SM_SLOC_INSTANTIATION_ENTRY);
  RECORD(SM_LINE_TABLE);

  // Preprocessor Block.
  BLOCK(PREPROCESSOR_BLOCK);
  RECORD(PP_MACRO_OBJECT_LIKE);
  RECORD(PP_MACRO_FUNCTION_LIKE);
  RECORD(PP_TOKEN);
  RECORD(PP_MACRO_INSTANTIATION);
  RECORD(PP_MACRO_DEFINITION);
  
  // Decls and Types block.
  BLOCK(DECLTYPES_BLOCK);
  RECORD(TYPE_EXT_QUAL);
  RECORD(TYPE_COMPLEX);
  RECORD(TYPE_POINTER);
  RECORD(TYPE_BLOCK_POINTER);
  RECORD(TYPE_LVALUE_REFERENCE);
  RECORD(TYPE_RVALUE_REFERENCE);
  RECORD(TYPE_MEMBER_POINTER);
  RECORD(TYPE_CONSTANT_ARRAY);
  RECORD(TYPE_INCOMPLETE_ARRAY);
  RECORD(TYPE_VARIABLE_ARRAY);
  RECORD(TYPE_VECTOR);
  RECORD(TYPE_EXT_VECTOR);
  RECORD(TYPE_FUNCTION_PROTO);
  RECORD(TYPE_FUNCTION_NO_PROTO);
  RECORD(TYPE_TYPEDEF);
  RECORD(TYPE_TYPEOF_EXPR);
  RECORD(TYPE_TYPEOF);
  RECORD(TYPE_RECORD);
  RECORD(TYPE_ENUM);
  RECORD(TYPE_OBJC_INTERFACE);
  RECORD(TYPE_OBJC_OBJECT);
  RECORD(TYPE_OBJC_OBJECT_POINTER);
  RECORD(DECL_ATTR);
  RECORD(DECL_TRANSLATION_UNIT);
  RECORD(DECL_TYPEDEF);
  RECORD(DECL_ENUM);
  RECORD(DECL_RECORD);
  RECORD(DECL_ENUM_CONSTANT);
  RECORD(DECL_FUNCTION);
  RECORD(DECL_OBJC_METHOD);
  RECORD(DECL_OBJC_INTERFACE);
  RECORD(DECL_OBJC_PROTOCOL);
  RECORD(DECL_OBJC_IVAR);
  RECORD(DECL_OBJC_AT_DEFS_FIELD);
  RECORD(DECL_OBJC_CLASS);
  RECORD(DECL_OBJC_FORWARD_PROTOCOL);
  RECORD(DECL_OBJC_CATEGORY);
  RECORD(DECL_OBJC_CATEGORY_IMPL);
  RECORD(DECL_OBJC_IMPLEMENTATION);
  RECORD(DECL_OBJC_COMPATIBLE_ALIAS);
  RECORD(DECL_OBJC_PROPERTY);
  RECORD(DECL_OBJC_PROPERTY_IMPL);
  RECORD(DECL_FIELD);
  RECORD(DECL_VAR);
  RECORD(DECL_IMPLICIT_PARAM);
  RECORD(DECL_PARM_VAR);
  RECORD(DECL_FILE_SCOPE_ASM);
  RECORD(DECL_BLOCK);
  RECORD(DECL_CONTEXT_LEXICAL);
  RECORD(DECL_CONTEXT_VISIBLE);
  // Statements and Exprs can occur in the Decls and Types block.
  AddStmtsExprs(Stream, Record);
#undef RECORD
#undef BLOCK
  Stream.ExitBlock();
}

/// \brief Adjusts the given filename to only write out the portion of the
/// filename that is not part of the system root directory.
///
/// \param Filename the file name to adjust.
///
/// \param isysroot When non-NULL, the PCH file is a relocatable PCH file and
/// the returned filename will be adjusted by this system root.
///
/// \returns either the original filename (if it needs no adjustment) or the
/// adjusted filename (which points into the @p Filename parameter).
static const char *
adjustFilenameForRelocatablePCH(const char *Filename, const char *isysroot) {
  assert(Filename && "No file name to adjust?");

  if (!isysroot)
    return Filename;

  // Verify that the filename and the system root have the same prefix.
  unsigned Pos = 0;
  for (; Filename[Pos] && isysroot[Pos]; ++Pos)
    if (Filename[Pos] != isysroot[Pos])
      return Filename; // Prefixes don't match.

  // We hit the end of the filename before we hit the end of the system root.
  if (!Filename[Pos])
    return Filename;

  // If the file name has a '/' at the current position, skip over the '/'.
  // We distinguish sysroot-based includes from absolute includes by the
  // absence of '/' at the beginning of sysroot-based includes.
  if (Filename[Pos] == '/')
    ++Pos;

  return Filename + Pos;
}

/// \brief Write the AST metadata (e.g., i686-apple-darwin9).
void ASTWriter::WriteMetadata(ASTContext &Context, const char *isysroot) {
  using namespace llvm;

  // Metadata
  const TargetInfo &Target = Context.Target;
  BitCodeAbbrev *MetaAbbrev = new BitCodeAbbrev();
  MetaAbbrev->Add(BitCodeAbbrevOp(
                    Chain ? pch::CHAINED_METADATA : pch::METADATA));
  MetaAbbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 16)); // AST major
  MetaAbbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 16)); // AST minor
  MetaAbbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 16)); // Clang major
  MetaAbbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 16)); // Clang minor
  MetaAbbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // Relocatable
  // Target triple or chained PCH name
  MetaAbbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob));
  unsigned MetaAbbrevCode = Stream.EmitAbbrev(MetaAbbrev);

  RecordData Record;
  Record.push_back(Chain ? pch::CHAINED_METADATA : pch::METADATA);
  Record.push_back(pch::VERSION_MAJOR);
  Record.push_back(pch::VERSION_MINOR);
  Record.push_back(CLANG_VERSION_MAJOR);
  Record.push_back(CLANG_VERSION_MINOR);
  Record.push_back(isysroot != 0);
  // FIXME: This writes the absolute path for chained headers.
  const std::string &BlobStr = Chain ? Chain->getFileName() : Target.getTriple().getTriple();
  Stream.EmitRecordWithBlob(MetaAbbrevCode, Record, BlobStr);

  // Original file name
  SourceManager &SM = Context.getSourceManager();
  if (const FileEntry *MainFile = SM.getFileEntryForID(SM.getMainFileID())) {
    BitCodeAbbrev *FileAbbrev = new BitCodeAbbrev();
    FileAbbrev->Add(BitCodeAbbrevOp(pch::ORIGINAL_FILE_NAME));
    FileAbbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob)); // File name
    unsigned FileAbbrevCode = Stream.EmitAbbrev(FileAbbrev);

    llvm::sys::Path MainFilePath(MainFile->getName());

    MainFilePath.makeAbsolute();

    const char *MainFileNameStr = MainFilePath.c_str();
    MainFileNameStr = adjustFilenameForRelocatablePCH(MainFileNameStr,
                                                      isysroot);
    RecordData Record;
    Record.push_back(pch::ORIGINAL_FILE_NAME);
    Stream.EmitRecordWithBlob(FileAbbrevCode, Record, MainFileNameStr);
  }

  // Repository branch/version information.
  BitCodeAbbrev *RepoAbbrev = new BitCodeAbbrev();
  RepoAbbrev->Add(BitCodeAbbrevOp(pch::VERSION_CONTROL_BRANCH_REVISION));
  RepoAbbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob)); // SVN branch/tag
  unsigned RepoAbbrevCode = Stream.EmitAbbrev(RepoAbbrev);
  Record.clear();
  Record.push_back(pch::VERSION_CONTROL_BRANCH_REVISION);
  Stream.EmitRecordWithBlob(RepoAbbrevCode, Record,
                            getClangFullRepositoryVersion());
}

/// \brief Write the LangOptions structure.
void ASTWriter::WriteLanguageOptions(const LangOptions &LangOpts) {
  RecordData Record;
  Record.push_back(LangOpts.Trigraphs);
  Record.push_back(LangOpts.BCPLComment);  // BCPL-style '//' comments.
  Record.push_back(LangOpts.DollarIdents);  // '$' allowed in identifiers.
  Record.push_back(LangOpts.AsmPreprocessor);  // Preprocessor in asm mode.
  Record.push_back(LangOpts.GNUMode);  // True in gnu99 mode false in c99 mode (etc)
  Record.push_back(LangOpts.GNUKeywords);  // Allow GNU-extension keywords
  Record.push_back(LangOpts.ImplicitInt);  // C89 implicit 'int'.
  Record.push_back(LangOpts.Digraphs);  // C94, C99 and C++
  Record.push_back(LangOpts.HexFloats);  // C99 Hexadecimal float constants.
  Record.push_back(LangOpts.C99);  // C99 Support
  Record.push_back(LangOpts.Microsoft);  // Microsoft extensions.
  Record.push_back(LangOpts.CPlusPlus);  // C++ Support
  Record.push_back(LangOpts.CPlusPlus0x);  // C++0x Support
  Record.push_back(LangOpts.CXXOperatorNames);  // Treat C++ operator names as keywords.

  Record.push_back(LangOpts.ObjC1);  // Objective-C 1 support enabled.
  Record.push_back(LangOpts.ObjC2);  // Objective-C 2 support enabled.
  Record.push_back(LangOpts.ObjCNonFragileABI);  // Objective-C
                                                 // modern abi enabled.
  Record.push_back(LangOpts.ObjCNonFragileABI2); // Objective-C enhanced
                                                 // modern abi enabled.
  Record.push_back(LangOpts.NoConstantCFStrings); // non cfstring generation enabled..

  Record.push_back(LangOpts.PascalStrings);  // Allow Pascal strings
  Record.push_back(LangOpts.WritableStrings);  // Allow writable strings
  Record.push_back(LangOpts.LaxVectorConversions);
  Record.push_back(LangOpts.AltiVec);
  Record.push_back(LangOpts.Exceptions);  // Support exception handling.
  Record.push_back(LangOpts.SjLjExceptions);

  Record.push_back(LangOpts.NeXTRuntime); // Use NeXT runtime.
  Record.push_back(LangOpts.Freestanding); // Freestanding implementation
  Record.push_back(LangOpts.NoBuiltin); // Do not use builtin functions (-fno-builtin)

  // Whether static initializers are protected by locks.
  Record.push_back(LangOpts.ThreadsafeStatics);
  Record.push_back(LangOpts.POSIXThreads);
  Record.push_back(LangOpts.Blocks); // block extension to C
  Record.push_back(LangOpts.EmitAllDecls); // Emit all declarations, even if
                                  // they are unused.
  Record.push_back(LangOpts.MathErrno); // Math functions must respect errno
                                  // (modulo the platform support).

  Record.push_back(LangOpts.getSignedOverflowBehavior());
  Record.push_back(LangOpts.HeinousExtensions);

  Record.push_back(LangOpts.Optimize); // Whether __OPTIMIZE__ should be defined.
  Record.push_back(LangOpts.OptimizeSize); // Whether __OPTIMIZE_SIZE__ should be
                                  // defined.
  Record.push_back(LangOpts.Static); // Should __STATIC__ be defined (as
                                  // opposed to __DYNAMIC__).
  Record.push_back(LangOpts.PICLevel); // The value for __PIC__, if non-zero.

  Record.push_back(LangOpts.GNUInline); // Should GNU inline semantics be
                                  // used (instead of C99 semantics).
  Record.push_back(LangOpts.NoInline); // Should __NO_INLINE__ be defined.
  Record.push_back(LangOpts.AccessControl); // Whether C++ access control should
                                            // be enabled.
  Record.push_back(LangOpts.CharIsSigned); // Whether char is a signed or
                                           // unsigned type
  Record.push_back(LangOpts.ShortWChar);  // force wchar_t to be unsigned short
  Record.push_back(LangOpts.getGCMode());
  Record.push_back(LangOpts.getVisibilityMode());
  Record.push_back(LangOpts.getStackProtectorMode());
  Record.push_back(LangOpts.InstantiationDepth);
  Record.push_back(LangOpts.OpenCL);
  Record.push_back(LangOpts.CatchUndefined);
  Record.push_back(LangOpts.ElideConstructors);
  Record.push_back(LangOpts.SpellChecking);
  Stream.EmitRecord(pch::LANGUAGE_OPTIONS, Record);
}

//===----------------------------------------------------------------------===//
// stat cache Serialization
//===----------------------------------------------------------------------===//

namespace {
// Trait used for the on-disk hash table of stat cache results.
class ASTStatCacheTrait {
public:
  typedef const char * key_type;
  typedef key_type key_type_ref;

  typedef std::pair<int, struct stat> data_type;
  typedef const data_type& data_type_ref;

  static unsigned ComputeHash(const char *path) {
    return llvm::HashString(path);
  }

  std::pair<unsigned,unsigned>
    EmitKeyDataLength(llvm::raw_ostream& Out, const char *path,
                      data_type_ref Data) {
    unsigned StrLen = strlen(path);
    clang::io::Emit16(Out, StrLen);
    unsigned DataLen = 1; // result value
    if (Data.first == 0)
      DataLen += 4 + 4 + 2 + 8 + 8;
    clang::io::Emit8(Out, DataLen);
    return std::make_pair(StrLen + 1, DataLen);
  }

  void EmitKey(llvm::raw_ostream& Out, const char *path, unsigned KeyLen) {
    Out.write(path, KeyLen);
  }

  void EmitData(llvm::raw_ostream& Out, key_type_ref,
                data_type_ref Data, unsigned DataLen) {
    using namespace clang::io;
    uint64_t Start = Out.tell(); (void)Start;

    // Result of stat()
    Emit8(Out, Data.first? 1 : 0);

    if (Data.first == 0) {
      Emit32(Out, (uint32_t) Data.second.st_ino);
      Emit32(Out, (uint32_t) Data.second.st_dev);
      Emit16(Out, (uint16_t) Data.second.st_mode);
      Emit64(Out, (uint64_t) Data.second.st_mtime);
      Emit64(Out, (uint64_t) Data.second.st_size);
    }

    assert(Out.tell() - Start == DataLen && "Wrong data length");
  }
};
} // end anonymous namespace

/// \brief Write the stat() system call cache to the AST file.
void ASTWriter::WriteStatCache(MemorizeStatCalls &StatCalls) {
  // Build the on-disk hash table containing information about every
  // stat() call.
  OnDiskChainedHashTableGenerator<ASTStatCacheTrait> Generator;
  unsigned NumStatEntries = 0;
  for (MemorizeStatCalls::iterator Stat = StatCalls.begin(),
                                StatEnd = StatCalls.end();
       Stat != StatEnd; ++Stat, ++NumStatEntries) {
    const char *Filename = Stat->first();
    Generator.insert(Filename, Stat->second);
  }

  // Create the on-disk hash table in a buffer.
  llvm::SmallString<4096> StatCacheData;
  uint32_t BucketOffset;
  {
    llvm::raw_svector_ostream Out(StatCacheData);
    // Make sure that no bucket is at offset 0
    clang::io::Emit32(Out, 0);
    BucketOffset = Generator.Emit(Out);
  }

  // Create a blob abbreviation
  using namespace llvm;
  BitCodeAbbrev *Abbrev = new BitCodeAbbrev();
  Abbrev->Add(BitCodeAbbrevOp(pch::STAT_CACHE));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 32));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 32));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob));
  unsigned StatCacheAbbrev = Stream.EmitAbbrev(Abbrev);

  // Write the stat cache
  RecordData Record;
  Record.push_back(pch::STAT_CACHE);
  Record.push_back(BucketOffset);
  Record.push_back(NumStatEntries);
  Stream.EmitRecordWithBlob(StatCacheAbbrev, Record, StatCacheData.str());
}

//===----------------------------------------------------------------------===//
// Source Manager Serialization
//===----------------------------------------------------------------------===//

/// \brief Create an abbreviation for the SLocEntry that refers to a
/// file.
static unsigned CreateSLocFileAbbrev(llvm::BitstreamWriter &Stream) {
  using namespace llvm;
  BitCodeAbbrev *Abbrev = new BitCodeAbbrev();
  Abbrev->Add(BitCodeAbbrevOp(pch::SM_SLOC_FILE_ENTRY));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 8)); // Offset
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 8)); // Include location
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 2)); // Characteristic
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // Line directives
  // FileEntry fields.
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 12)); // Size
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 32)); // Modification time
  // HeaderFileInfo fields.
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // isImport
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 2)); // DirInfo
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 8)); // NumIncludes
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 8)); // ControllingMacro
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob)); // File name
  return Stream.EmitAbbrev(Abbrev);
}

/// \brief Create an abbreviation for the SLocEntry that refers to a
/// buffer.
static unsigned CreateSLocBufferAbbrev(llvm::BitstreamWriter &Stream) {
  using namespace llvm;
  BitCodeAbbrev *Abbrev = new BitCodeAbbrev();
  Abbrev->Add(BitCodeAbbrevOp(pch::SM_SLOC_BUFFER_ENTRY));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 8)); // Offset
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 8)); // Include location
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 2)); // Characteristic
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 1)); // Line directives
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob)); // Buffer name blob
  return Stream.EmitAbbrev(Abbrev);
}

/// \brief Create an abbreviation for the SLocEntry that refers to a
/// buffer's blob.
static unsigned CreateSLocBufferBlobAbbrev(llvm::BitstreamWriter &Stream) {
  using namespace llvm;
  BitCodeAbbrev *Abbrev = new BitCodeAbbrev();
  Abbrev->Add(BitCodeAbbrevOp(pch::SM_SLOC_BUFFER_BLOB));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob)); // Blob
  return Stream.EmitAbbrev(Abbrev);
}

/// \brief Create an abbreviation for the SLocEntry that refers to an
/// buffer.
static unsigned CreateSLocInstantiationAbbrev(llvm::BitstreamWriter &Stream) {
  using namespace llvm;
  BitCodeAbbrev *Abbrev = new BitCodeAbbrev();
  Abbrev->Add(BitCodeAbbrevOp(pch::SM_SLOC_INSTANTIATION_ENTRY));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 8)); // Offset
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 8)); // Spelling location
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 8)); // Start location
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 8)); // End location
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 6)); // Token length
  return Stream.EmitAbbrev(Abbrev);
}

/// \brief Writes the block containing the serialized form of the
/// source manager.
///
/// TODO: We should probably use an on-disk hash table (stored in a
/// blob), indexed based on the file name, so that we only create
/// entries for files that we actually need. In the common case (no
/// errors), we probably won't have to create file entries for any of
/// the files in the AST.
void ASTWriter::WriteSourceManagerBlock(SourceManager &SourceMgr,
                                        const Preprocessor &PP,
                                        const char *isysroot) {
  RecordData Record;

  // Enter the source manager block.
  Stream.EnterSubblock(pch::SOURCE_MANAGER_BLOCK_ID, 3);

  // Abbreviations for the various kinds of source-location entries.
  unsigned SLocFileAbbrv = CreateSLocFileAbbrev(Stream);
  unsigned SLocBufferAbbrv = CreateSLocBufferAbbrev(Stream);
  unsigned SLocBufferBlobAbbrv = CreateSLocBufferBlobAbbrev(Stream);
  unsigned SLocInstantiationAbbrv = CreateSLocInstantiationAbbrev(Stream);

  // Write the line table.
  if (SourceMgr.hasLineTable()) {
    LineTableInfo &LineTable = SourceMgr.getLineTable();

    // Emit the file names
    Record.push_back(LineTable.getNumFilenames());
    for (unsigned I = 0, N = LineTable.getNumFilenames(); I != N; ++I) {
      // Emit the file name
      const char *Filename = LineTable.getFilename(I);
      Filename = adjustFilenameForRelocatablePCH(Filename, isysroot);
      unsigned FilenameLen = Filename? strlen(Filename) : 0;
      Record.push_back(FilenameLen);
      if (FilenameLen)
        Record.insert(Record.end(), Filename, Filename + FilenameLen);
    }

    // Emit the line entries
    for (LineTableInfo::iterator L = LineTable.begin(), LEnd = LineTable.end();
         L != LEnd; ++L) {
      // Emit the file ID
      Record.push_back(L->first);

      // Emit the line entries
      Record.push_back(L->second.size());
      for (std::vector<LineEntry>::iterator LE = L->second.begin(),
                                         LEEnd = L->second.end();
           LE != LEEnd; ++LE) {
        Record.push_back(LE->FileOffset);
        Record.push_back(LE->LineNo);
        Record.push_back(LE->FilenameID);
        Record.push_back((unsigned)LE->FileKind);
        Record.push_back(LE->IncludeOffset);
      }
    }
    Stream.EmitRecord(pch::SM_LINE_TABLE, Record);
  }

  // Write out the source location entry table. We skip the first
  // entry, which is always the same dummy entry.
  std::vector<uint32_t> SLocEntryOffsets;
  RecordData PreloadSLocs;
  unsigned BaseSLocID = Chain ? Chain->getTotalNumSLocs() : 0;
  SLocEntryOffsets.reserve(SourceMgr.sloc_entry_size() - 1 - BaseSLocID);
  for (unsigned I = BaseSLocID + 1, N = SourceMgr.sloc_entry_size();
       I != N; ++I) {
    // Get this source location entry.
    const SrcMgr::SLocEntry *SLoc = &SourceMgr.getSLocEntry(I);

    // Record the offset of this source-location entry.
    SLocEntryOffsets.push_back(Stream.GetCurrentBitNo());

    // Figure out which record code to use.
    unsigned Code;
    if (SLoc->isFile()) {
      if (SLoc->getFile().getContentCache()->Entry)
        Code = pch::SM_SLOC_FILE_ENTRY;
      else
        Code = pch::SM_SLOC_BUFFER_ENTRY;
    } else
      Code = pch::SM_SLOC_INSTANTIATION_ENTRY;
    Record.clear();
    Record.push_back(Code);

    Record.push_back(SLoc->getOffset());
    if (SLoc->isFile()) {
      const SrcMgr::FileInfo &File = SLoc->getFile();
      Record.push_back(File.getIncludeLoc().getRawEncoding());
      Record.push_back(File.getFileCharacteristic()); // FIXME: stable encoding
      Record.push_back(File.hasLineDirectives());

      const SrcMgr::ContentCache *Content = File.getContentCache();
      if (Content->Entry) {
        // The source location entry is a file. The blob associated
        // with this entry is the file name.

        // Emit size/modification time for this file.
        Record.push_back(Content->Entry->getSize());
        Record.push_back(Content->Entry->getModificationTime());

        // Emit header-search information associated with this file.
        HeaderFileInfo HFI;
        HeaderSearch &HS = PP.getHeaderSearchInfo();
        if (Content->Entry->getUID() < HS.header_file_size())
          HFI = HS.header_file_begin()[Content->Entry->getUID()];
        Record.push_back(HFI.isImport);
        Record.push_back(HFI.DirInfo);
        Record.push_back(HFI.NumIncludes);
        AddIdentifierRef(HFI.ControllingMacro, Record);

        // Turn the file name into an absolute path, if it isn't already.
        const char *Filename = Content->Entry->getName();
        llvm::sys::Path FilePath(Filename, strlen(Filename));
        FilePath.makeAbsolute();
        Filename = FilePath.c_str();

        Filename = adjustFilenameForRelocatablePCH(Filename, isysroot);
        Stream.EmitRecordWithBlob(SLocFileAbbrv, Record, Filename);

        // FIXME: For now, preload all file source locations, so that
        // we get the appropriate File entries in the reader. This is
        // a temporary measure.
        PreloadSLocs.push_back(BaseSLocID + SLocEntryOffsets.size());
      } else {
        // The source location entry is a buffer. The blob associated
        // with this entry contains the contents of the buffer.

        // We add one to the size so that we capture the trailing NULL
        // that is required by llvm::MemoryBuffer::getMemBuffer (on
        // the reader side).
        const llvm::MemoryBuffer *Buffer
          = Content->getBuffer(PP.getDiagnostics(), PP.getSourceManager());
        const char *Name = Buffer->getBufferIdentifier();
        Stream.EmitRecordWithBlob(SLocBufferAbbrv, Record,
                                  llvm::StringRef(Name, strlen(Name) + 1));
        Record.clear();
        Record.push_back(pch::SM_SLOC_BUFFER_BLOB);
        Stream.EmitRecordWithBlob(SLocBufferBlobAbbrv, Record,
                                  llvm::StringRef(Buffer->getBufferStart(),
                                                  Buffer->getBufferSize() + 1));

        if (strcmp(Name, "<built-in>") == 0)
          PreloadSLocs.push_back(BaseSLocID + SLocEntryOffsets.size());
      }
    } else {
      // The source location entry is an instantiation.
      const SrcMgr::InstantiationInfo &Inst = SLoc->getInstantiation();
      Record.push_back(Inst.getSpellingLoc().getRawEncoding());
      Record.push_back(Inst.getInstantiationLocStart().getRawEncoding());
      Record.push_back(Inst.getInstantiationLocEnd().getRawEncoding());

      // Compute the token length for this macro expansion.
      unsigned NextOffset = SourceMgr.getNextOffset();
      if (I + 1 != N)
        NextOffset = SourceMgr.getSLocEntry(I + 1).getOffset();
      Record.push_back(NextOffset - SLoc->getOffset() - 1);
      Stream.EmitRecordWithAbbrev(SLocInstantiationAbbrv, Record);
    }
  }

  Stream.ExitBlock();

  if (SLocEntryOffsets.empty())
    return;

  // Write the source-location offsets table into the AST block. This
  // table is used for lazily loading source-location information.
  using namespace llvm;
  BitCodeAbbrev *Abbrev = new BitCodeAbbrev();
  Abbrev->Add(BitCodeAbbrevOp(pch::SOURCE_LOCATION_OFFSETS));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 16)); // # of slocs
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::VBR, 16)); // next offset
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob)); // offsets
  unsigned SLocOffsetsAbbrev = Stream.EmitAbbrev(Abbrev);

  Record.clear();
  Record.push_back(pch::SOURCE_LOCATION_OFFSETS);
  Record.push_back(SLocEntryOffsets.size());
  Record.push_back(SourceMgr.getNextOffset());
  Stream.EmitRecordWithBlob(SLocOffsetsAbbrev, Record,
                            (const char *)data(SLocEntryOffsets),
                           SLocEntryOffsets.size()*sizeof(SLocEntryOffsets[0]));

  // Write the source location entry preloads array, telling the AST
  // reader which source locations entries it should load eagerly.
  Stream.EmitRecord(pch::SOURCE_LOCATION_PRELOADS, PreloadSLocs);
}

//===----------------------------------------------------------------------===//
// Preprocessor Serialization
//===----------------------------------------------------------------------===//

/// \brief Writes the block containing the serialized form of the
/// preprocessor.
///
void ASTWriter::WritePreprocessor(const Preprocessor &PP) {
  RecordData Record;

  // If the preprocessor __COUNTER__ value has been bumped, remember it.
  if (PP.getCounterValue() != 0) {
    Record.push_back(PP.getCounterValue());
    Stream.EmitRecord(pch::PP_COUNTER_VALUE, Record);
    Record.clear();
  }

  // Enter the preprocessor block.
  Stream.EnterSubblock(pch::PREPROCESSOR_BLOCK_ID, 2);

  // If the AST file contains __DATE__ or __TIME__ emit a warning about this.
  // FIXME: use diagnostics subsystem for localization etc.
  if (PP.SawDateOrTime())
    fprintf(stderr, "warning: precompiled header used __DATE__ or __TIME__.\n");

  // Loop over all the macro definitions that are live at the end of the file,
  // emitting each to the PP section.
  PreprocessingRecord *PPRec = PP.getPreprocessingRecord();
  for (Preprocessor::macro_iterator I = PP.macro_begin(), E = PP.macro_end();
       I != E; ++I) {
    // FIXME: This emits macros in hash table order, we should do it in a stable
    // order so that output is reproducible.
    MacroInfo *MI = I->second;

    // Don't emit builtin macros like __LINE__ to the AST file unless they have
    // been redefined by the header (in which case they are not isBuiltinMacro).
    // Also skip macros from a AST file if we're chaining.
    if (MI->isBuiltinMacro() || (Chain && MI->isFromPCH()))
      continue;

    AddIdentifierRef(I->first, Record);
    MacroOffsets[I->first] = Stream.GetCurrentBitNo();
    Record.push_back(MI->getDefinitionLoc().getRawEncoding());
    Record.push_back(MI->isUsed());

    unsigned Code;
    if (MI->isObjectLike()) {
      Code = pch::PP_MACRO_OBJECT_LIKE;
    } else {
      Code = pch::PP_MACRO_FUNCTION_LIKE;

      Record.push_back(MI->isC99Varargs());
      Record.push_back(MI->isGNUVarargs());
      Record.push_back(MI->getNumArgs());
      for (MacroInfo::arg_iterator I = MI->arg_begin(), E = MI->arg_end();
           I != E; ++I)
        AddIdentifierRef(*I, Record);
    }
    
    // If we have a detailed preprocessing record, record the macro definition
    // ID that corresponds to this macro.
    if (PPRec)
      Record.push_back(getMacroDefinitionID(PPRec->findMacroDefinition(MI)));
    
    Stream.EmitRecord(Code, Record);
    Record.clear();

    // Emit the tokens array.
    for (unsigned TokNo = 0, e = MI->getNumTokens(); TokNo != e; ++TokNo) {
      // Note that we know that the preprocessor does not have any annotation
      // tokens in it because they are created by the parser, and thus can't be
      // in a macro definition.
      const Token &Tok = MI->getReplacementToken(TokNo);

      Record.push_back(Tok.getLocation().getRawEncoding());
      Record.push_back(Tok.getLength());

      // FIXME: When reading literal tokens, reconstruct the literal pointer if
      // it is needed.
      AddIdentifierRef(Tok.getIdentifierInfo(), Record);

      // FIXME: Should translate token kind to a stable encoding.
      Record.push_back(Tok.getKind());
      // FIXME: Should translate token flags to a stable encoding.
      Record.push_back(Tok.getFlags());

      Stream.EmitRecord(pch::PP_TOKEN, Record);
      Record.clear();
    }
    ++NumMacros;
  }
  
  // If the preprocessor has a preprocessing record, emit it.
  unsigned NumPreprocessingRecords = 0;
  if (PPRec) {
    for (PreprocessingRecord::iterator E = PPRec->begin(), EEnd = PPRec->end();
         E != EEnd; ++E) {
      Record.clear();
      
      if (MacroInstantiation *MI = dyn_cast<MacroInstantiation>(*E)) {
        Record.push_back(NumPreprocessingRecords++);
        AddSourceLocation(MI->getSourceRange().getBegin(), Record);
        AddSourceLocation(MI->getSourceRange().getEnd(), Record);
        AddIdentifierRef(MI->getName(), Record);
        Record.push_back(getMacroDefinitionID(MI->getDefinition()));
        Stream.EmitRecord(pch::PP_MACRO_INSTANTIATION, Record);
        continue;
      }
      
      if (MacroDefinition *MD = dyn_cast<MacroDefinition>(*E)) {
        // Record this macro definition's location.
        pch::IdentID ID = getMacroDefinitionID(MD);
        if (ID != MacroDefinitionOffsets.size()) {
          if (ID > MacroDefinitionOffsets.size())
            MacroDefinitionOffsets.resize(ID + 1);
          
          MacroDefinitionOffsets[ID] = Stream.GetCurrentBitNo();            
        } else
          MacroDefinitionOffsets.push_back(Stream.GetCurrentBitNo());
        
        Record.push_back(NumPreprocessingRecords++);
        Record.push_back(ID);
        AddSourceLocation(MD->getSourceRange().getBegin(), Record);
        AddSourceLocation(MD->getSourceRange().getEnd(), Record);
        AddIdentifierRef(MD->getName(), Record);
        AddSourceLocation(MD->getLocation(), Record);
        Stream.EmitRecord(pch::PP_MACRO_DEFINITION, Record);
        continue;
      }
    }
  }
  
  Stream.ExitBlock();
  
  // Write the offsets table for the preprocessing record.
  if (NumPreprocessingRecords > 0) {
    // Write the offsets table for identifier IDs.
    using namespace llvm;
    BitCodeAbbrev *Abbrev = new BitCodeAbbrev();
    Abbrev->Add(BitCodeAbbrevOp(pch::MACRO_DEFINITION_OFFSETS));
    Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 32)); // # of records
    Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 32)); // # of macro defs
    Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob));
    unsigned MacroDefOffsetAbbrev = Stream.EmitAbbrev(Abbrev);
    
    Record.clear();
    Record.push_back(pch::MACRO_DEFINITION_OFFSETS);
    Record.push_back(NumPreprocessingRecords);
    Record.push_back(MacroDefinitionOffsets.size());
    Stream.EmitRecordWithBlob(MacroDefOffsetAbbrev, Record,
                              (const char *)data(MacroDefinitionOffsets),
                              MacroDefinitionOffsets.size() * sizeof(uint32_t));
  }
}

//===----------------------------------------------------------------------===//
// Type Serialization
//===----------------------------------------------------------------------===//

/// \brief Write the representation of a type to the AST stream.
void ASTWriter::WriteType(QualType T) {
  pch::TypeID &ID = TypeIDs[T];
  if (ID == 0) // we haven't seen this type before.
    ID = NextTypeID++;

  // Record the offset for this type.
  unsigned Index = ID - FirstTypeID;
  if (TypeOffsets.size() == Index)
    TypeOffsets.push_back(Stream.GetCurrentBitNo());
  else if (TypeOffsets.size() < Index) {
    TypeOffsets.resize(Index + 1);
    TypeOffsets[Index] = Stream.GetCurrentBitNo();
  }

  RecordData Record;

  // Emit the type's representation.
  ASTTypeWriter W(*this, Record);

  if (T.hasLocalNonFastQualifiers()) {
    Qualifiers Qs = T.getLocalQualifiers();
    AddTypeRef(T.getLocalUnqualifiedType(), Record);
    Record.push_back(Qs.getAsOpaqueValue());
    W.Code = pch::TYPE_EXT_QUAL;
  } else {
    switch (T->getTypeClass()) {
      // For all of the concrete, non-dependent types, call the
      // appropriate visitor function.
#define TYPE(Class, Base) \
    case Type::Class: W.Visit##Class##Type(cast<Class##Type>(T)); break;
#define ABSTRACT_TYPE(Class, Base)
#include "clang/AST/TypeNodes.def"
    }
  }

  // Emit the serialized record.
  Stream.EmitRecord(W.Code, Record);

  // Flush any expressions that were written as part of this type.
  FlushStmts();
}

//===----------------------------------------------------------------------===//
// Declaration Serialization
//===----------------------------------------------------------------------===//

/// \brief Write the block containing all of the declaration IDs
/// lexically declared within the given DeclContext.
///
/// \returns the offset of the DECL_CONTEXT_LEXICAL block within the
/// bistream, or 0 if no block was written.
uint64_t ASTWriter::WriteDeclContextLexicalBlock(ASTContext &Context,
                                                 DeclContext *DC) {
  if (DC->decls_empty())
    return 0;

  uint64_t Offset = Stream.GetCurrentBitNo();
  RecordData Record;
  Record.push_back(pch::DECL_CONTEXT_LEXICAL);
  llvm::SmallVector<pch::DeclID, 64> Decls;
  for (DeclContext::decl_iterator D = DC->decls_begin(), DEnd = DC->decls_end();
         D != DEnd; ++D)
    Decls.push_back(GetDeclRef(*D));

  ++NumLexicalDeclContexts;
  Stream.EmitRecordWithBlob(DeclContextLexicalAbbrev, Record,
     reinterpret_cast<char*>(Decls.data()), Decls.size() * sizeof(pch::DeclID));
  return Offset;
}

/// \brief Write the block containing all of the declaration IDs
/// visible from the given DeclContext.
///
/// \returns the offset of the DECL_CONTEXT_VISIBLE block within the
/// bistream, or 0 if no block was written.
uint64_t ASTWriter::WriteDeclContextVisibleBlock(ASTContext &Context,
                                                 DeclContext *DC) {
  if (DC->getPrimaryContext() != DC)
    return 0;

  // Since there is no name lookup into functions or methods, don't bother to
  // build a visible-declarations table for these entities.
  if (DC->isFunctionOrMethod())
    return 0;

  // If not in C++, we perform name lookup for the translation unit via the
  // IdentifierInfo chains, don't bother to build a visible-declarations table.
  // FIXME: In C++ we need the visible declarations in order to "see" the
  // friend declarations, is there a way to do this without writing the table ?
  if (DC->isTranslationUnit() && !Context.getLangOptions().CPlusPlus)
    return 0;

  // Force the DeclContext to build a its name-lookup table.
  DC->lookup(DeclarationName());

  // Serialize the contents of the mapping used for lookup. Note that,
  // although we have two very different code paths, the serialized
  // representation is the same for both cases: a declaration name,
  // followed by a size, followed by references to the visible
  // declarations that have that name.
  uint64_t Offset = Stream.GetCurrentBitNo();
  RecordData Record;
  StoredDeclsMap *Map = static_cast<StoredDeclsMap*>(DC->getLookupPtr());
  if (!Map)
    return 0;

  for (StoredDeclsMap::iterator D = Map->begin(), DEnd = Map->end();
       D != DEnd; ++D) {
    AddDeclarationName(D->first, Record);
    DeclContext::lookup_result Result = D->second.getLookupResult(Context);
    Record.push_back(Result.second - Result.first);
    for (; Result.first != Result.second; ++Result.first)
      AddDeclRef(*Result.first, Record);
  }

  if (Record.size() == 0)
    return 0;

  Stream.EmitRecord(pch::DECL_CONTEXT_VISIBLE, Record);
  ++NumVisibleDeclContexts;
  return Offset;
}

void ASTWriter::WriteTypeDeclOffsets() {
  using namespace llvm;
  RecordData Record;

  // Write the type offsets array
  BitCodeAbbrev *Abbrev = new BitCodeAbbrev();
  Abbrev->Add(BitCodeAbbrevOp(pch::TYPE_OFFSET));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 32)); // # of types
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob)); // types block
  unsigned TypeOffsetAbbrev = Stream.EmitAbbrev(Abbrev);
  Record.clear();
  Record.push_back(pch::TYPE_OFFSET);
  Record.push_back(TypeOffsets.size());
  Stream.EmitRecordWithBlob(TypeOffsetAbbrev, Record,
                            (const char *)data(TypeOffsets),
                            TypeOffsets.size() * sizeof(TypeOffsets[0]));

  // Write the declaration offsets array
  Abbrev = new BitCodeAbbrev();
  Abbrev->Add(BitCodeAbbrevOp(pch::DECL_OFFSET));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 32)); // # of declarations
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob)); // declarations block
  unsigned DeclOffsetAbbrev = Stream.EmitAbbrev(Abbrev);
  Record.clear();
  Record.push_back(pch::DECL_OFFSET);
  Record.push_back(DeclOffsets.size());
  Stream.EmitRecordWithBlob(DeclOffsetAbbrev, Record,
                            (const char *)data(DeclOffsets),
                            DeclOffsets.size() * sizeof(DeclOffsets[0]));
}

//===----------------------------------------------------------------------===//
// Global Method Pool and Selector Serialization
//===----------------------------------------------------------------------===//

namespace {
// Trait used for the on-disk hash table used in the method pool.
class ASTMethodPoolTrait {
  ASTWriter &Writer;

public:
  typedef Selector key_type;
  typedef key_type key_type_ref;

  struct data_type {
    pch::SelectorID ID;
    ObjCMethodList Instance, Factory;
  };
  typedef const data_type& data_type_ref;

  explicit ASTMethodPoolTrait(ASTWriter &Writer) : Writer(Writer) { }

  static unsigned ComputeHash(Selector Sel) {
    unsigned N = Sel.getNumArgs();
    if (N == 0)
      ++N;
    unsigned R = 5381;
    for (unsigned I = 0; I != N; ++I)
      if (IdentifierInfo *II = Sel.getIdentifierInfoForSlot(I))
        R = llvm::HashString(II->getName(), R);
    return R;
  }

  std::pair<unsigned,unsigned>
    EmitKeyDataLength(llvm::raw_ostream& Out, Selector Sel,
                      data_type_ref Methods) {
    unsigned KeyLen = 2 + (Sel.getNumArgs()? Sel.getNumArgs() * 4 : 4);
    clang::io::Emit16(Out, KeyLen);
    unsigned DataLen = 4 + 2 + 2; // 2 bytes for each of the method counts
    for (const ObjCMethodList *Method = &Methods.Instance; Method;
         Method = Method->Next)
      if (Method->Method)
        DataLen += 4;
    for (const ObjCMethodList *Method = &Methods.Factory; Method;
         Method = Method->Next)
      if (Method->Method)
        DataLen += 4;
    clang::io::Emit16(Out, DataLen);
    return std::make_pair(KeyLen, DataLen);
  }

  void EmitKey(llvm::raw_ostream& Out, Selector Sel, unsigned) {
    uint64_t Start = Out.tell();
    assert((Start >> 32) == 0 && "Selector key offset too large");
    Writer.SetSelectorOffset(Sel, Start);
    unsigned N = Sel.getNumArgs();
    clang::io::Emit16(Out, N);
    if (N == 0)
      N = 1;
    for (unsigned I = 0; I != N; ++I)
      clang::io::Emit32(Out,
                    Writer.getIdentifierRef(Sel.getIdentifierInfoForSlot(I)));
  }

  void EmitData(llvm::raw_ostream& Out, key_type_ref,
                data_type_ref Methods, unsigned DataLen) {
    uint64_t Start = Out.tell(); (void)Start;
    clang::io::Emit32(Out, Methods.ID);
    unsigned NumInstanceMethods = 0;
    for (const ObjCMethodList *Method = &Methods.Instance; Method;
         Method = Method->Next)
      if (Method->Method)
        ++NumInstanceMethods;

    unsigned NumFactoryMethods = 0;
    for (const ObjCMethodList *Method = &Methods.Factory; Method;
         Method = Method->Next)
      if (Method->Method)
        ++NumFactoryMethods;

    clang::io::Emit16(Out, NumInstanceMethods);
    clang::io::Emit16(Out, NumFactoryMethods);
    for (const ObjCMethodList *Method = &Methods.Instance; Method;
         Method = Method->Next)
      if (Method->Method)
        clang::io::Emit32(Out, Writer.getDeclID(Method->Method));
    for (const ObjCMethodList *Method = &Methods.Factory; Method;
         Method = Method->Next)
      if (Method->Method)
        clang::io::Emit32(Out, Writer.getDeclID(Method->Method));

    assert(Out.tell() - Start == DataLen && "Data length is wrong");
  }
};
} // end anonymous namespace

/// \brief Write ObjC data: selectors and the method pool.
///
/// The method pool contains both instance and factory methods, stored
/// in an on-disk hash table indexed by the selector. The hash table also
/// contains an empty entry for every other selector known to Sema.
void ASTWriter::WriteSelectors(Sema &SemaRef) {
  using namespace llvm;

  // Do we have to do anything at all?
  if (SemaRef.MethodPool.empty() && SelectorIDs.empty())
    return;
  unsigned NumTableEntries = 0;
  // Create and write out the blob that contains selectors and the method pool.
  {
    OnDiskChainedHashTableGenerator<ASTMethodPoolTrait> Generator;

    // Create the on-disk hash table representation. We walk through every
    // selector we've seen and look it up in the method pool.
    SelectorOffsets.resize(NextSelectorID - FirstSelectorID);
    for (llvm::DenseMap<Selector, pch::SelectorID>::iterator
             I = SelectorIDs.begin(), E = SelectorIDs.end();
         I != E; ++I) {
      Selector S = I->first;
      Sema::GlobalMethodPool::iterator F = SemaRef.MethodPool.find(S);
      ASTMethodPoolTrait::data_type Data = {
        I->second,
        ObjCMethodList(),
        ObjCMethodList()
      };
      if (F != SemaRef.MethodPool.end()) {
        Data.Instance = F->second.first;
        Data.Factory = F->second.second;
      }
      // Only write this selector if it's not in an existing AST or something
      // changed.
      if (Chain && I->second < FirstSelectorID) {
        // Selector already exists. Did it change?
        bool changed = false;
        for (ObjCMethodList *M = &Data.Instance; !changed && M && M->Method;
             M = M->Next) {
          if (M->Method->getPCHLevel() == 0)
            changed = true;
        }
        for (ObjCMethodList *M = &Data.Factory; !changed && M && M->Method;
             M = M->Next) {
          if (M->Method->getPCHLevel() == 0)
            changed = true;
        }
        if (!changed)
          continue;
      } else if (Data.Instance.Method || Data.Factory.Method) {
        // A new method pool entry.
        ++NumTableEntries;
      }
      Generator.insert(S, Data);
    }

    // Create the on-disk hash table in a buffer.
    llvm::SmallString<4096> MethodPool;
    uint32_t BucketOffset;
    {
      ASTMethodPoolTrait Trait(*this);
      llvm::raw_svector_ostream Out(MethodPool);
      // Make sure that no bucket is at offset 0
      clang::io::Emit32(Out, 0);
      BucketOffset = Generator.Emit(Out, Trait);
    }

    // Create a blob abbreviation
    BitCodeAbbrev *Abbrev = new BitCodeAbbrev();
    Abbrev->Add(BitCodeAbbrevOp(pch::METHOD_POOL));
    Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 32));
    Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 32));
    Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob));
    unsigned MethodPoolAbbrev = Stream.EmitAbbrev(Abbrev);

    // Write the method pool
    RecordData Record;
    Record.push_back(pch::METHOD_POOL);
    Record.push_back(BucketOffset);
    Record.push_back(NumTableEntries);
    Stream.EmitRecordWithBlob(MethodPoolAbbrev, Record, MethodPool.str());

    // Create a blob abbreviation for the selector table offsets.
    Abbrev = new BitCodeAbbrev();
    Abbrev->Add(BitCodeAbbrevOp(pch::SELECTOR_OFFSETS));
    Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 32)); // index
    Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob));
    unsigned SelectorOffsetAbbrev = Stream.EmitAbbrev(Abbrev);

    // Write the selector offsets table.
    Record.clear();
    Record.push_back(pch::SELECTOR_OFFSETS);
    Record.push_back(SelectorOffsets.size());
    Stream.EmitRecordWithBlob(SelectorOffsetAbbrev, Record,
                              (const char *)data(SelectorOffsets),
                              SelectorOffsets.size() * 4);
  }
}

/// \brief Write the selectors referenced in @selector expression into AST file.
void ASTWriter::WriteReferencedSelectorsPool(Sema &SemaRef) {
  using namespace llvm;
  if (SemaRef.ReferencedSelectors.empty())
    return;

  RecordData Record;

  // Note: this writes out all references even for a dependent AST. But it is
  // very tricky to fix, and given that @selector shouldn't really appear in
  // headers, probably not worth it. It's not a correctness issue.
  for (DenseMap<Selector, SourceLocation>::iterator S =
       SemaRef.ReferencedSelectors.begin(),
       E = SemaRef.ReferencedSelectors.end(); S != E; ++S) {
    Selector Sel = (*S).first;
    SourceLocation Loc = (*S).second;
    AddSelectorRef(Sel, Record);
    AddSourceLocation(Loc, Record);
  }
  Stream.EmitRecord(pch::REFERENCED_SELECTOR_POOL, Record);
}

//===----------------------------------------------------------------------===//
// Identifier Table Serialization
//===----------------------------------------------------------------------===//

namespace {
class ASTIdentifierTableTrait {
  ASTWriter &Writer;
  Preprocessor &PP;

  /// \brief Determines whether this is an "interesting" identifier
  /// that needs a full IdentifierInfo structure written into the hash
  /// table.
  static bool isInterestingIdentifier(const IdentifierInfo *II) {
    return II->isPoisoned() ||
      II->isExtensionToken() ||
      II->hasMacroDefinition() ||
      II->getObjCOrBuiltinID() ||
      II->getFETokenInfo<void>();
  }

public:
  typedef const IdentifierInfo* key_type;
  typedef key_type  key_type_ref;

  typedef pch::IdentID data_type;
  typedef data_type data_type_ref;

  ASTIdentifierTableTrait(ASTWriter &Writer, Preprocessor &PP)
    : Writer(Writer), PP(PP) { }

  static unsigned ComputeHash(const IdentifierInfo* II) {
    return llvm::HashString(II->getName());
  }

  std::pair<unsigned,unsigned>
    EmitKeyDataLength(llvm::raw_ostream& Out, const IdentifierInfo* II,
                      pch::IdentID ID) {
    unsigned KeyLen = II->getLength() + 1;
    unsigned DataLen = 4; // 4 bytes for the persistent ID << 1
    if (isInterestingIdentifier(II)) {
      DataLen += 2; // 2 bytes for builtin ID, flags
      if (II->hasMacroDefinition() &&
          !PP.getMacroInfo(const_cast<IdentifierInfo *>(II))->isBuiltinMacro())
        DataLen += 4;
      for (IdentifierResolver::iterator D = IdentifierResolver::begin(II),
                                     DEnd = IdentifierResolver::end();
           D != DEnd; ++D)
        DataLen += sizeof(pch::DeclID);
    }
    clang::io::Emit16(Out, DataLen);
    // We emit the key length after the data length so that every
    // string is preceded by a 16-bit length. This matches the PTH
    // format for storing identifiers.
    clang::io::Emit16(Out, KeyLen);
    return std::make_pair(KeyLen, DataLen);
  }

  void EmitKey(llvm::raw_ostream& Out, const IdentifierInfo* II,
               unsigned KeyLen) {
    // Record the location of the key data.  This is used when generating
    // the mapping from persistent IDs to strings.
    Writer.SetIdentifierOffset(II, Out.tell());
    Out.write(II->getNameStart(), KeyLen);
  }

  void EmitData(llvm::raw_ostream& Out, const IdentifierInfo* II,
                pch::IdentID ID, unsigned) {
    if (!isInterestingIdentifier(II)) {
      clang::io::Emit32(Out, ID << 1);
      return;
    }

    clang::io::Emit32(Out, (ID << 1) | 0x01);
    uint32_t Bits = 0;
    bool hasMacroDefinition =
      II->hasMacroDefinition() &&
      !PP.getMacroInfo(const_cast<IdentifierInfo *>(II))->isBuiltinMacro();
    Bits = (uint32_t)II->getObjCOrBuiltinID();
    Bits = (Bits << 1) | unsigned(hasMacroDefinition);
    Bits = (Bits << 1) | unsigned(II->isExtensionToken());
    Bits = (Bits << 1) | unsigned(II->isPoisoned());
    Bits = (Bits << 1) | unsigned(II->hasRevertedTokenIDToIdentifier());
    Bits = (Bits << 1) | unsigned(II->isCPlusPlusOperatorKeyword());
    clang::io::Emit16(Out, Bits);

    if (hasMacroDefinition)
      clang::io::Emit32(Out, Writer.getMacroOffset(II));

    // Emit the declaration IDs in reverse order, because the
    // IdentifierResolver provides the declarations as they would be
    // visible (e.g., the function "stat" would come before the struct
    // "stat"), but IdentifierResolver::AddDeclToIdentifierChain()
    // adds declarations to the end of the list (so we need to see the
    // struct "status" before the function "status").
    // Only emit declarations that aren't from a chained PCH, though.
    llvm::SmallVector<Decl *, 16> Decls(IdentifierResolver::begin(II),
                                        IdentifierResolver::end());
    for (llvm::SmallVector<Decl *, 16>::reverse_iterator D = Decls.rbegin(),
                                                      DEnd = Decls.rend();
         D != DEnd; ++D)
      clang::io::Emit32(Out, Writer.getDeclID(*D));
  }
};
} // end anonymous namespace

/// \brief Write the identifier table into the AST file.
///
/// The identifier table consists of a blob containing string data
/// (the actual identifiers themselves) and a separate "offsets" index
/// that maps identifier IDs to locations within the blob.
void ASTWriter::WriteIdentifierTable(Preprocessor &PP) {
  using namespace llvm;

  // Create and write out the blob that contains the identifier
  // strings.
  {
    OnDiskChainedHashTableGenerator<ASTIdentifierTableTrait> Generator;

    // Look for any identifiers that were named while processing the
    // headers, but are otherwise not needed. We add these to the hash
    // table to enable checking of the predefines buffer in the case
    // where the user adds new macro definitions when building the AST
    // file.
    for (IdentifierTable::iterator ID = PP.getIdentifierTable().begin(),
                                IDEnd = PP.getIdentifierTable().end();
         ID != IDEnd; ++ID)
      getIdentifierRef(ID->second);

    // Create the on-disk hash table representation. We only store offsets
    // for identifiers that appear here for the first time.
    IdentifierOffsets.resize(NextIdentID - FirstIdentID);
    for (llvm::DenseMap<const IdentifierInfo *, pch::IdentID>::iterator
           ID = IdentifierIDs.begin(), IDEnd = IdentifierIDs.end();
         ID != IDEnd; ++ID) {
      assert(ID->first && "NULL identifier in identifier table");
      if (!Chain || !ID->first->isFromPCH())
        Generator.insert(ID->first, ID->second);
    }

    // Create the on-disk hash table in a buffer.
    llvm::SmallString<4096> IdentifierTable;
    uint32_t BucketOffset;
    {
      ASTIdentifierTableTrait Trait(*this, PP);
      llvm::raw_svector_ostream Out(IdentifierTable);
      // Make sure that no bucket is at offset 0
      clang::io::Emit32(Out, 0);
      BucketOffset = Generator.Emit(Out, Trait);
    }

    // Create a blob abbreviation
    BitCodeAbbrev *Abbrev = new BitCodeAbbrev();
    Abbrev->Add(BitCodeAbbrevOp(pch::IDENTIFIER_TABLE));
    Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 32));
    Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob));
    unsigned IDTableAbbrev = Stream.EmitAbbrev(Abbrev);

    // Write the identifier table
    RecordData Record;
    Record.push_back(pch::IDENTIFIER_TABLE);
    Record.push_back(BucketOffset);
    Stream.EmitRecordWithBlob(IDTableAbbrev, Record, IdentifierTable.str());
  }

  // Write the offsets table for identifier IDs.
  BitCodeAbbrev *Abbrev = new BitCodeAbbrev();
  Abbrev->Add(BitCodeAbbrevOp(pch::IDENTIFIER_OFFSET));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Fixed, 32)); // # of identifiers
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob));
  unsigned IdentifierOffsetAbbrev = Stream.EmitAbbrev(Abbrev);

  RecordData Record;
  Record.push_back(pch::IDENTIFIER_OFFSET);
  Record.push_back(IdentifierOffsets.size());
  Stream.EmitRecordWithBlob(IdentifierOffsetAbbrev, Record,
                            (const char *)data(IdentifierOffsets),
                            IdentifierOffsets.size() * sizeof(uint32_t));
}

//===----------------------------------------------------------------------===//
// General Serialization Routines
//===----------------------------------------------------------------------===//

/// \brief Write a record containing the given attributes.
void ASTWriter::WriteAttributeRecord(const AttrVec &Attrs) {
  RecordData Record;
  for (AttrVec::const_iterator i = Attrs.begin(), e = Attrs.end(); i != e; ++i){
    const Attr * A = *i;
    Record.push_back(A->getKind()); // FIXME: stable encoding, target attrs
    AddSourceLocation(A->getLocation(), Record);
    Record.push_back(A->isInherited());

#include "clang/Serialization/AttrPCHWrite.inc"

  }

  Stream.EmitRecord(pch::DECL_ATTR, Record);
}

void ASTWriter::AddString(const std::string &Str, RecordData &Record) {
  Record.push_back(Str.size());
  Record.insert(Record.end(), Str.begin(), Str.end());
}

/// \brief Note that the identifier II occurs at the given offset
/// within the identifier table.
void ASTWriter::SetIdentifierOffset(const IdentifierInfo *II, uint32_t Offset) {
  pch::IdentID ID = IdentifierIDs[II];
  // Only store offsets new to this AST file. Other identifier names are looked
  // up earlier in the chain and thus don't need an offset.
  if (ID >= FirstIdentID)
    IdentifierOffsets[ID - FirstIdentID] = Offset;
}

/// \brief Note that the selector Sel occurs at the given offset
/// within the method pool/selector table.
void ASTWriter::SetSelectorOffset(Selector Sel, uint32_t Offset) {
  unsigned ID = SelectorIDs[Sel];
  assert(ID && "Unknown selector");
  // Don't record offsets for selectors that are also available in a different
  // file.
  if (ID < FirstSelectorID)
    return;
  SelectorOffsets[ID - FirstSelectorID] = Offset;
}

ASTWriter::ASTWriter(llvm::BitstreamWriter &Stream)
  : Stream(Stream), Chain(0), FirstDeclID(1), NextDeclID(FirstDeclID),
    FirstTypeID(pch::NUM_PREDEF_TYPE_IDS), NextTypeID(FirstTypeID),
    FirstIdentID(1), NextIdentID(FirstIdentID), FirstSelectorID(1),
    NextSelectorID(FirstSelectorID), CollectedStmts(&StmtsToEmit),
    NumStatements(0), NumMacros(0), NumLexicalDeclContexts(0),
    NumVisibleDeclContexts(0) {
}

void ASTWriter::WriteAST(Sema &SemaRef, MemorizeStatCalls *StatCalls,
                         const char *isysroot) {
  // Emit the file header.
  Stream.Emit((unsigned)'C', 8);
  Stream.Emit((unsigned)'P', 8);
  Stream.Emit((unsigned)'C', 8);
  Stream.Emit((unsigned)'H', 8);

  WriteBlockInfoBlock();

  if (Chain)
    WriteASTChain(SemaRef, StatCalls, isysroot);
  else
    WriteASTCore(SemaRef, StatCalls, isysroot);
}

void ASTWriter::WriteASTCore(Sema &SemaRef, MemorizeStatCalls *StatCalls,
                             const char *isysroot) {
  using namespace llvm;

  ASTContext &Context = SemaRef.Context;
  Preprocessor &PP = SemaRef.PP;

  // The translation unit is the first declaration we'll emit.
  DeclIDs[Context.getTranslationUnitDecl()] = 1;
  ++NextDeclID;
  DeclTypesToEmit.push(Context.getTranslationUnitDecl());

  // Make sure that we emit IdentifierInfos (and any attached
  // declarations) for builtins.
  {
    IdentifierTable &Table = PP.getIdentifierTable();
    llvm::SmallVector<const char *, 32> BuiltinNames;
    Context.BuiltinInfo.GetBuiltinNames(BuiltinNames,
                                        Context.getLangOptions().NoBuiltin);
    for (unsigned I = 0, N = BuiltinNames.size(); I != N; ++I)
      getIdentifierRef(&Table.get(BuiltinNames[I]));
  }

  // Build a record containing all of the tentative definitions in this file, in
  // TentativeDefinitions order.  Generally, this record will be empty for
  // headers.
  RecordData TentativeDefinitions;
  for (unsigned i = 0, e = SemaRef.TentativeDefinitions.size(); i != e; ++i) {
    AddDeclRef(SemaRef.TentativeDefinitions[i], TentativeDefinitions);
  }

  // Build a record containing all of the file scoped decls in this file.
  RecordData UnusedFileScopedDecls;
  for (unsigned i=0, e = SemaRef.UnusedFileScopedDecls.size(); i !=e; ++i)
    AddDeclRef(SemaRef.UnusedFileScopedDecls[i], UnusedFileScopedDecls);

  RecordData WeakUndeclaredIdentifiers;
  if (!SemaRef.WeakUndeclaredIdentifiers.empty()) {
    WeakUndeclaredIdentifiers.push_back(
                                      SemaRef.WeakUndeclaredIdentifiers.size());
    for (llvm::DenseMap<IdentifierInfo*,Sema::WeakInfo>::iterator
         I = SemaRef.WeakUndeclaredIdentifiers.begin(),
         E = SemaRef.WeakUndeclaredIdentifiers.end(); I != E; ++I) {
      AddIdentifierRef(I->first, WeakUndeclaredIdentifiers);
      AddIdentifierRef(I->second.getAlias(), WeakUndeclaredIdentifiers);
      AddSourceLocation(I->second.getLocation(), WeakUndeclaredIdentifiers);
      WeakUndeclaredIdentifiers.push_back(I->second.getUsed());
    }
  }

  // Build a record containing all of the locally-scoped external
  // declarations in this header file. Generally, this record will be
  // empty.
  RecordData LocallyScopedExternalDecls;
  // FIXME: This is filling in the AST file in densemap order which is
  // nondeterminstic!
  for (llvm::DenseMap<DeclarationName, NamedDecl *>::iterator
         TD = SemaRef.LocallyScopedExternalDecls.begin(),
         TDEnd = SemaRef.LocallyScopedExternalDecls.end();
       TD != TDEnd; ++TD)
    AddDeclRef(TD->second, LocallyScopedExternalDecls);

  // Build a record containing all of the ext_vector declarations.
  RecordData ExtVectorDecls;
  for (unsigned I = 0, N = SemaRef.ExtVectorDecls.size(); I != N; ++I)
    AddDeclRef(SemaRef.ExtVectorDecls[I], ExtVectorDecls);

  // Build a record containing all of the VTable uses information.
  RecordData VTableUses;
  if (!SemaRef.VTableUses.empty()) {
    VTableUses.push_back(SemaRef.VTableUses.size());
    for (unsigned I = 0, N = SemaRef.VTableUses.size(); I != N; ++I) {
      AddDeclRef(SemaRef.VTableUses[I].first, VTableUses);
      AddSourceLocation(SemaRef.VTableUses[I].second, VTableUses);
      VTableUses.push_back(SemaRef.VTablesUsed[SemaRef.VTableUses[I].first]);
    }
  }

  // Build a record containing all of dynamic classes declarations.
  RecordData DynamicClasses;
  for (unsigned I = 0, N = SemaRef.DynamicClasses.size(); I != N; ++I)
    AddDeclRef(SemaRef.DynamicClasses[I], DynamicClasses);

  // Build a record containing all of pending implicit instantiations.
  RecordData PendingImplicitInstantiations;
  for (std::deque<Sema::PendingImplicitInstantiation>::iterator
         I = SemaRef.PendingImplicitInstantiations.begin(),
         N = SemaRef.PendingImplicitInstantiations.end(); I != N; ++I) {
    AddDeclRef(I->first, PendingImplicitInstantiations);
    AddSourceLocation(I->second, PendingImplicitInstantiations);
  }
  assert(SemaRef.PendingLocalImplicitInstantiations.empty() &&
         "There are local ones at end of translation unit!");

  // Build a record containing some declaration references.
  RecordData SemaDeclRefs;
  if (SemaRef.StdNamespace || SemaRef.StdBadAlloc) {
    AddDeclRef(SemaRef.getStdNamespace(), SemaDeclRefs);
    AddDeclRef(SemaRef.getStdBadAlloc(), SemaDeclRefs);
  }

  // Write the remaining AST contents.
  RecordData Record;
  Stream.EnterSubblock(pch::PCH_BLOCK_ID, 5);
  WriteMetadata(Context, isysroot);
  WriteLanguageOptions(Context.getLangOptions());
  if (StatCalls && !isysroot)
    WriteStatCache(*StatCalls);
  WriteSourceManagerBlock(Context.getSourceManager(), PP, isysroot);
  // Write the record of special types.
  Record.clear();

  AddTypeRef(Context.getBuiltinVaListType(), Record);
  AddTypeRef(Context.getObjCIdType(), Record);
  AddTypeRef(Context.getObjCSelType(), Record);
  AddTypeRef(Context.getObjCProtoType(), Record);
  AddTypeRef(Context.getObjCClassType(), Record);
  AddTypeRef(Context.getRawCFConstantStringType(), Record);
  AddTypeRef(Context.getRawObjCFastEnumerationStateType(), Record);
  AddTypeRef(Context.getFILEType(), Record);
  AddTypeRef(Context.getjmp_bufType(), Record);
  AddTypeRef(Context.getsigjmp_bufType(), Record);
  AddTypeRef(Context.ObjCIdRedefinitionType, Record);
  AddTypeRef(Context.ObjCClassRedefinitionType, Record);
  AddTypeRef(Context.getRawBlockdescriptorType(), Record);
  AddTypeRef(Context.getRawBlockdescriptorExtendedType(), Record);
  AddTypeRef(Context.ObjCSelRedefinitionType, Record);
  AddTypeRef(Context.getRawNSConstantStringType(), Record);
  Record.push_back(Context.isInt128Installed());
  Stream.EmitRecord(pch::SPECIAL_TYPES, Record);

  // Keep writing types and declarations until all types and
  // declarations have been written.
  Stream.EnterSubblock(pch::DECLTYPES_BLOCK_ID, 3);
  WriteDeclsBlockAbbrevs();
  while (!DeclTypesToEmit.empty()) {
    DeclOrType DOT = DeclTypesToEmit.front();
    DeclTypesToEmit.pop();
    if (DOT.isType())
      WriteType(DOT.getType());
    else
      WriteDecl(Context, DOT.getDecl());
  }
  Stream.ExitBlock();

  WritePreprocessor(PP);
  WriteSelectors(SemaRef);
  WriteReferencedSelectorsPool(SemaRef);
  WriteIdentifierTable(PP);

  WriteTypeDeclOffsets();

  // Write the record containing external, unnamed definitions.
  if (!ExternalDefinitions.empty())
    Stream.EmitRecord(pch::EXTERNAL_DEFINITIONS, ExternalDefinitions);

  // Write the record containing tentative definitions.
  if (!TentativeDefinitions.empty())
    Stream.EmitRecord(pch::TENTATIVE_DEFINITIONS, TentativeDefinitions);

  // Write the record containing unused file scoped decls.
  if (!UnusedFileScopedDecls.empty())
    Stream.EmitRecord(pch::UNUSED_FILESCOPED_DECLS, UnusedFileScopedDecls);

  // Write the record containing weak undeclared identifiers.
  if (!WeakUndeclaredIdentifiers.empty())
    Stream.EmitRecord(pch::WEAK_UNDECLARED_IDENTIFIERS,
                      WeakUndeclaredIdentifiers);

  // Write the record containing locally-scoped external definitions.
  if (!LocallyScopedExternalDecls.empty())
    Stream.EmitRecord(pch::LOCALLY_SCOPED_EXTERNAL_DECLS,
                      LocallyScopedExternalDecls);

  // Write the record containing ext_vector type names.
  if (!ExtVectorDecls.empty())
    Stream.EmitRecord(pch::EXT_VECTOR_DECLS, ExtVectorDecls);

  // Write the record containing VTable uses information.
  if (!VTableUses.empty())
    Stream.EmitRecord(pch::VTABLE_USES, VTableUses);

  // Write the record containing dynamic classes declarations.
  if (!DynamicClasses.empty())
    Stream.EmitRecord(pch::DYNAMIC_CLASSES, DynamicClasses);

  // Write the record containing pending implicit instantiations.
  if (!PendingImplicitInstantiations.empty())
    Stream.EmitRecord(pch::PENDING_IMPLICIT_INSTANTIATIONS,
                      PendingImplicitInstantiations);

  // Write the record containing declaration references of Sema.
  if (!SemaDeclRefs.empty())
    Stream.EmitRecord(pch::SEMA_DECL_REFS, SemaDeclRefs);

  // Some simple statistics
  Record.clear();
  Record.push_back(NumStatements);
  Record.push_back(NumMacros);
  Record.push_back(NumLexicalDeclContexts);
  Record.push_back(NumVisibleDeclContexts);
  Stream.EmitRecord(pch::STATISTICS, Record);
  Stream.ExitBlock();
}

void ASTWriter::WriteASTChain(Sema &SemaRef, MemorizeStatCalls *StatCalls,
                              const char *isysroot) {
  using namespace llvm;

  FirstDeclID += Chain->getTotalNumDecls();
  FirstTypeID += Chain->getTotalNumTypes();
  FirstIdentID += Chain->getTotalNumIdentifiers();
  FirstSelectorID += Chain->getTotalNumSelectors();
  NextDeclID = FirstDeclID;
  NextTypeID = FirstTypeID;
  NextIdentID = FirstIdentID;
  NextSelectorID = FirstSelectorID;

  ASTContext &Context = SemaRef.Context;
  Preprocessor &PP = SemaRef.PP;

  RecordData Record;
  Stream.EnterSubblock(pch::PCH_BLOCK_ID, 5);
  WriteMetadata(Context, isysroot);
  if (StatCalls && !isysroot)
    WriteStatCache(*StatCalls);
  // FIXME: Source manager block should only write new stuff, which could be
  // done by tracking the largest ID in the chain
  WriteSourceManagerBlock(Context.getSourceManager(), PP, isysroot);

  // The special types are in the chained PCH.

  // We don't start with the translation unit, but with its decls that
  // don't come from the other PCH.
  const TranslationUnitDecl *TU = Context.getTranslationUnitDecl();
  llvm::SmallVector<pch::DeclID, 64> NewGlobalDecls;
  for (DeclContext::decl_iterator I = TU->noload_decls_begin(),
                                  E = TU->noload_decls_end();
       I != E; ++I) {
    if ((*I)->getPCHLevel() == 0)
      NewGlobalDecls.push_back(GetDeclRef(*I));
    else if ((*I)->isChangedSinceDeserialization())
      (void)GetDeclRef(*I); // Make sure it's written, but don't record it.
  }
  // We also need to write a lexical updates block for the TU.
  llvm::BitCodeAbbrev *Abv = new llvm::BitCodeAbbrev();
  Abv->Add(llvm::BitCodeAbbrevOp(pch::TU_UPDATE_LEXICAL));
  Abv->Add(llvm::BitCodeAbbrevOp(llvm::BitCodeAbbrevOp::Blob));
  unsigned TuUpdateLexicalAbbrev = Stream.EmitAbbrev(Abv);
  Record.clear();
  Record.push_back(pch::TU_UPDATE_LEXICAL);
  Stream.EmitRecordWithBlob(TuUpdateLexicalAbbrev, Record,
                          reinterpret_cast<const char*>(NewGlobalDecls.data()),
                          NewGlobalDecls.size() * sizeof(pch::DeclID));

  // Build a record containing all of the new tentative definitions in this
  // file, in TentativeDefinitions order.
  RecordData TentativeDefinitions;
  for (unsigned i = 0, e = SemaRef.TentativeDefinitions.size(); i != e; ++i) {
    if (SemaRef.TentativeDefinitions[i]->getPCHLevel() == 0)
      AddDeclRef(SemaRef.TentativeDefinitions[i], TentativeDefinitions);
  }

  // Build a record containing all of the file scoped decls in this file.
  RecordData UnusedFileScopedDecls;
  for (unsigned i=0, e = SemaRef.UnusedFileScopedDecls.size(); i !=e; ++i) {
    if (SemaRef.UnusedFileScopedDecls[i]->getPCHLevel() == 0)
      AddDeclRef(SemaRef.UnusedFileScopedDecls[i], UnusedFileScopedDecls);
  }

  // We write the entire table, overwriting the tables from the chain.
  RecordData WeakUndeclaredIdentifiers;
  if (!SemaRef.WeakUndeclaredIdentifiers.empty()) {
    WeakUndeclaredIdentifiers.push_back(
                                      SemaRef.WeakUndeclaredIdentifiers.size());
    for (llvm::DenseMap<IdentifierInfo*,Sema::WeakInfo>::iterator
         I = SemaRef.WeakUndeclaredIdentifiers.begin(),
         E = SemaRef.WeakUndeclaredIdentifiers.end(); I != E; ++I) {
      AddIdentifierRef(I->first, WeakUndeclaredIdentifiers);
      AddIdentifierRef(I->second.getAlias(), WeakUndeclaredIdentifiers);
      AddSourceLocation(I->second.getLocation(), WeakUndeclaredIdentifiers);
      WeakUndeclaredIdentifiers.push_back(I->second.getUsed());
    }
  }

  // Build a record containing all of the locally-scoped external
  // declarations in this header file. Generally, this record will be
  // empty.
  RecordData LocallyScopedExternalDecls;
  // FIXME: This is filling in the AST file in densemap order which is
  // nondeterminstic!
  for (llvm::DenseMap<DeclarationName, NamedDecl *>::iterator
         TD = SemaRef.LocallyScopedExternalDecls.begin(),
         TDEnd = SemaRef.LocallyScopedExternalDecls.end();
       TD != TDEnd; ++TD) {
    if (TD->second->getPCHLevel() == 0)
      AddDeclRef(TD->second, LocallyScopedExternalDecls);
  }

  // Build a record containing all of the ext_vector declarations.
  RecordData ExtVectorDecls;
  for (unsigned I = 0, N = SemaRef.ExtVectorDecls.size(); I != N; ++I) {
    if (SemaRef.ExtVectorDecls[I]->getPCHLevel() == 0)
      AddDeclRef(SemaRef.ExtVectorDecls[I], ExtVectorDecls);
  }

  // Build a record containing all of the VTable uses information.
  // We write everything here, because it's too hard to determine whether
  // a use is new to this part.
  RecordData VTableUses;
  if (!SemaRef.VTableUses.empty()) {
    VTableUses.push_back(SemaRef.VTableUses.size());
    for (unsigned I = 0, N = SemaRef.VTableUses.size(); I != N; ++I) {
      AddDeclRef(SemaRef.VTableUses[I].first, VTableUses);
      AddSourceLocation(SemaRef.VTableUses[I].second, VTableUses);
      VTableUses.push_back(SemaRef.VTablesUsed[SemaRef.VTableUses[I].first]);
    }
  }

  // Build a record containing all of dynamic classes declarations.
  RecordData DynamicClasses;
  for (unsigned I = 0, N = SemaRef.DynamicClasses.size(); I != N; ++I)
    if (SemaRef.DynamicClasses[I]->getPCHLevel() == 0)
      AddDeclRef(SemaRef.DynamicClasses[I], DynamicClasses);

  // Build a record containing all of pending implicit instantiations.
  RecordData PendingImplicitInstantiations;
  for (std::deque<Sema::PendingImplicitInstantiation>::iterator
         I = SemaRef.PendingImplicitInstantiations.begin(),
         N = SemaRef.PendingImplicitInstantiations.end(); I != N; ++I) {
    if (I->first->getPCHLevel() == 0) {
      AddDeclRef(I->first, PendingImplicitInstantiations);
      AddSourceLocation(I->second, PendingImplicitInstantiations);
    }
  }
  assert(SemaRef.PendingLocalImplicitInstantiations.empty() &&
         "There are local ones at end of translation unit!");

  // Build a record containing some declaration references.
  // It's not worth the effort to avoid duplication here.
  RecordData SemaDeclRefs;
  if (SemaRef.StdNamespace || SemaRef.StdBadAlloc) {
    AddDeclRef(SemaRef.getStdNamespace(), SemaDeclRefs);
    AddDeclRef(SemaRef.getStdBadAlloc(), SemaDeclRefs);
  }

  Stream.EnterSubblock(pch::DECLTYPES_BLOCK_ID, 3);
  WriteDeclsBlockAbbrevs();
  while (!DeclTypesToEmit.empty()) {
    DeclOrType DOT = DeclTypesToEmit.front();
    DeclTypesToEmit.pop();
    if (DOT.isType())
      WriteType(DOT.getType());
    else
      WriteDecl(Context, DOT.getDecl());
  }
  Stream.ExitBlock();

  WritePreprocessor(PP);
  WriteSelectors(SemaRef);
  WriteReferencedSelectorsPool(SemaRef);
  WriteIdentifierTable(PP);
  WriteTypeDeclOffsets();

  /// Build a record containing first declarations from a chained PCH and the
  /// most recent declarations in this AST that they point to.
  RecordData FirstLatestDeclIDs;
  for (FirstLatestDeclMap::iterator
        I = FirstLatestDecls.begin(), E = FirstLatestDecls.end(); I != E; ++I) {
    assert(I->first->getPCHLevel() > I->second->getPCHLevel() &&
           "Expected first & second to be in different PCHs");
    AddDeclRef(I->first, FirstLatestDeclIDs);
    AddDeclRef(I->second, FirstLatestDeclIDs);
  }
  if (!FirstLatestDeclIDs.empty())
    Stream.EmitRecord(pch::REDECLS_UPDATE_LATEST, FirstLatestDeclIDs);

  // Write the record containing external, unnamed definitions.
  if (!ExternalDefinitions.empty())
    Stream.EmitRecord(pch::EXTERNAL_DEFINITIONS, ExternalDefinitions);

  // Write the record containing tentative definitions.
  if (!TentativeDefinitions.empty())
    Stream.EmitRecord(pch::TENTATIVE_DEFINITIONS, TentativeDefinitions);

  // Write the record containing unused file scoped decls.
  if (!UnusedFileScopedDecls.empty())
    Stream.EmitRecord(pch::UNUSED_FILESCOPED_DECLS, UnusedFileScopedDecls);

  // Write the record containing weak undeclared identifiers.
  if (!WeakUndeclaredIdentifiers.empty())
    Stream.EmitRecord(pch::WEAK_UNDECLARED_IDENTIFIERS,
                      WeakUndeclaredIdentifiers);

  // Write the record containing locally-scoped external definitions.
  if (!LocallyScopedExternalDecls.empty())
    Stream.EmitRecord(pch::LOCALLY_SCOPED_EXTERNAL_DECLS,
                      LocallyScopedExternalDecls);

  // Write the record containing ext_vector type names.
  if (!ExtVectorDecls.empty())
    Stream.EmitRecord(pch::EXT_VECTOR_DECLS, ExtVectorDecls);

  // Write the record containing VTable uses information.
  if (!VTableUses.empty())
    Stream.EmitRecord(pch::VTABLE_USES, VTableUses);

  // Write the record containing dynamic classes declarations.
  if (!DynamicClasses.empty())
    Stream.EmitRecord(pch::DYNAMIC_CLASSES, DynamicClasses);

  // Write the record containing pending implicit instantiations.
  if (!PendingImplicitInstantiations.empty())
    Stream.EmitRecord(pch::PENDING_IMPLICIT_INSTANTIATIONS,
                      PendingImplicitInstantiations);

  // Write the record containing declaration references of Sema.
  if (!SemaDeclRefs.empty())
    Stream.EmitRecord(pch::SEMA_DECL_REFS, SemaDeclRefs);

  Record.clear();
  Record.push_back(NumStatements);
  Record.push_back(NumMacros);
  Record.push_back(NumLexicalDeclContexts);
  Record.push_back(NumVisibleDeclContexts);
  WriteDeclUpdateBlock();
  Stream.EmitRecord(pch::STATISTICS, Record);
  Stream.ExitBlock();
}

void ASTWriter::WriteDeclUpdateBlock() {
  if (ReplacedDecls.empty())
    return;

  RecordData Record;
  for (llvm::SmallVector<std::pair<pch::DeclID, uint64_t>, 16>::iterator
           I = ReplacedDecls.begin(), E = ReplacedDecls.end(); I != E; ++I) {
    Record.push_back(I->first);
    Record.push_back(I->second);
  }
  Stream.EmitRecord(pch::DECL_REPLACEMENTS, Record);
}

void ASTWriter::AddSourceLocation(SourceLocation Loc, RecordData &Record) {
  Record.push_back(Loc.getRawEncoding());
}

void ASTWriter::AddSourceRange(SourceRange Range, RecordData &Record) {
  AddSourceLocation(Range.getBegin(), Record);
  AddSourceLocation(Range.getEnd(), Record);
}

void ASTWriter::AddAPInt(const llvm::APInt &Value, RecordData &Record) {
  Record.push_back(Value.getBitWidth());
  unsigned N = Value.getNumWords();
  const uint64_t* Words = Value.getRawData();
  for (unsigned I = 0; I != N; ++I)
    Record.push_back(Words[I]);
}

void ASTWriter::AddAPSInt(const llvm::APSInt &Value, RecordData &Record) {
  Record.push_back(Value.isUnsigned());
  AddAPInt(Value, Record);
}

void ASTWriter::AddAPFloat(const llvm::APFloat &Value, RecordData &Record) {
  AddAPInt(Value.bitcastToAPInt(), Record);
}

void ASTWriter::AddIdentifierRef(const IdentifierInfo *II, RecordData &Record) {
  Record.push_back(getIdentifierRef(II));
}

pch::IdentID ASTWriter::getIdentifierRef(const IdentifierInfo *II) {
  if (II == 0)
    return 0;

  pch::IdentID &ID = IdentifierIDs[II];
  if (ID == 0)
    ID = NextIdentID++;
  return ID;
}

pch::IdentID ASTWriter::getMacroDefinitionID(MacroDefinition *MD) {
  if (MD == 0)
    return 0;
  
  pch::IdentID &ID = MacroDefinitions[MD];
  if (ID == 0)
    ID = MacroDefinitions.size();
  return ID;
}

void ASTWriter::AddSelectorRef(const Selector SelRef, RecordData &Record) {
  Record.push_back(getSelectorRef(SelRef));
}

pch::SelectorID ASTWriter::getSelectorRef(Selector Sel) {
  if (Sel.getAsOpaquePtr() == 0) {
    return 0;
  }

  pch::SelectorID &SID = SelectorIDs[Sel];
  if (SID == 0 && Chain) {
    // This might trigger a ReadSelector callback, which will set the ID for
    // this selector.
    Chain->LoadSelector(Sel);
  }
  if (SID == 0) {
    SID = NextSelectorID++;
  }
  return SID;
}

void ASTWriter::AddCXXTemporary(const CXXTemporary *Temp, RecordData &Record) {
  AddDeclRef(Temp->getDestructor(), Record);
}

void ASTWriter::AddTemplateArgumentLocInfo(TemplateArgument::ArgKind Kind,
                                           const TemplateArgumentLocInfo &Arg,
                                           RecordData &Record) {
  switch (Kind) {
  case TemplateArgument::Expression:
    AddStmt(Arg.getAsExpr());
    break;
  case TemplateArgument::Type:
    AddTypeSourceInfo(Arg.getAsTypeSourceInfo(), Record);
    break;
  case TemplateArgument::Template:
    AddSourceRange(Arg.getTemplateQualifierRange(), Record);
    AddSourceLocation(Arg.getTemplateNameLoc(), Record);
    break;
  case TemplateArgument::Null:
  case TemplateArgument::Integral:
  case TemplateArgument::Declaration:
  case TemplateArgument::Pack:
    break;
  }
}

void ASTWriter::AddTemplateArgumentLoc(const TemplateArgumentLoc &Arg,
                                       RecordData &Record) {
  AddTemplateArgument(Arg.getArgument(), Record);

  if (Arg.getArgument().getKind() == TemplateArgument::Expression) {
    bool InfoHasSameExpr
      = Arg.getArgument().getAsExpr() == Arg.getLocInfo().getAsExpr();
    Record.push_back(InfoHasSameExpr);
    if (InfoHasSameExpr)
      return; // Avoid storing the same expr twice.
  }
  AddTemplateArgumentLocInfo(Arg.getArgument().getKind(), Arg.getLocInfo(),
                             Record);
}

void ASTWriter::AddTypeSourceInfo(TypeSourceInfo *TInfo, RecordData &Record) {
  if (TInfo == 0) {
    AddTypeRef(QualType(), Record);
    return;
  }

  AddTypeRef(TInfo->getType(), Record);
  TypeLocWriter TLW(*this, Record);
  for (TypeLoc TL = TInfo->getTypeLoc(); !TL.isNull(); TL = TL.getNextTypeLoc())
    TLW.Visit(TL);
}

void ASTWriter::AddTypeRef(QualType T, RecordData &Record) {
  if (T.isNull()) {
    Record.push_back(pch::PREDEF_TYPE_NULL_ID);
    return;
  }

  unsigned FastQuals = T.getLocalFastQualifiers();
  T.removeFastQualifiers();

  if (T.hasLocalNonFastQualifiers()) {
    pch::TypeID &ID = TypeIDs[T];
    if (ID == 0) {
      // We haven't seen these qualifiers applied to this type before.
      // Assign it a new ID.  This is the only time we enqueue a
      // qualified type, and it has no CV qualifiers.
      ID = NextTypeID++;
      DeclTypesToEmit.push(T);
    }

    // Encode the type qualifiers in the type reference.
    Record.push_back((ID << Qualifiers::FastWidth) | FastQuals);
    return;
  }

  assert(!T.hasLocalQualifiers());

  if (const BuiltinType *BT = dyn_cast<BuiltinType>(T.getTypePtr())) {
    pch::TypeID ID = 0;
    switch (BT->getKind()) {
    case BuiltinType::Void:       ID = pch::PREDEF_TYPE_VOID_ID;       break;
    case BuiltinType::Bool:       ID = pch::PREDEF_TYPE_BOOL_ID;       break;
    case BuiltinType::Char_U:     ID = pch::PREDEF_TYPE_CHAR_U_ID;     break;
    case BuiltinType::UChar:      ID = pch::PREDEF_TYPE_UCHAR_ID;      break;
    case BuiltinType::UShort:     ID = pch::PREDEF_TYPE_USHORT_ID;     break;
    case BuiltinType::UInt:       ID = pch::PREDEF_TYPE_UINT_ID;       break;
    case BuiltinType::ULong:      ID = pch::PREDEF_TYPE_ULONG_ID;      break;
    case BuiltinType::ULongLong:  ID = pch::PREDEF_TYPE_ULONGLONG_ID;  break;
    case BuiltinType::UInt128:    ID = pch::PREDEF_TYPE_UINT128_ID;    break;
    case BuiltinType::Char_S:     ID = pch::PREDEF_TYPE_CHAR_S_ID;     break;
    case BuiltinType::SChar:      ID = pch::PREDEF_TYPE_SCHAR_ID;      break;
    case BuiltinType::WChar:      ID = pch::PREDEF_TYPE_WCHAR_ID;      break;
    case BuiltinType::Short:      ID = pch::PREDEF_TYPE_SHORT_ID;      break;
    case BuiltinType::Int:        ID = pch::PREDEF_TYPE_INT_ID;        break;
    case BuiltinType::Long:       ID = pch::PREDEF_TYPE_LONG_ID;       break;
    case BuiltinType::LongLong:   ID = pch::PREDEF_TYPE_LONGLONG_ID;   break;
    case BuiltinType::Int128:     ID = pch::PREDEF_TYPE_INT128_ID;     break;
    case BuiltinType::Float:      ID = pch::PREDEF_TYPE_FLOAT_ID;      break;
    case BuiltinType::Double:     ID = pch::PREDEF_TYPE_DOUBLE_ID;     break;
    case BuiltinType::LongDouble: ID = pch::PREDEF_TYPE_LONGDOUBLE_ID; break;
    case BuiltinType::NullPtr:    ID = pch::PREDEF_TYPE_NULLPTR_ID;    break;
    case BuiltinType::Char16:     ID = pch::PREDEF_TYPE_CHAR16_ID;     break;
    case BuiltinType::Char32:     ID = pch::PREDEF_TYPE_CHAR32_ID;     break;
    case BuiltinType::Overload:   ID = pch::PREDEF_TYPE_OVERLOAD_ID;   break;
    case BuiltinType::Dependent:  ID = pch::PREDEF_TYPE_DEPENDENT_ID;  break;
    case BuiltinType::ObjCId:     ID = pch::PREDEF_TYPE_OBJC_ID;       break;
    case BuiltinType::ObjCClass:  ID = pch::PREDEF_TYPE_OBJC_CLASS;    break;
    case BuiltinType::ObjCSel:    ID = pch::PREDEF_TYPE_OBJC_SEL;      break;
    case BuiltinType::UndeducedAuto:
      assert(0 && "Should not see undeduced auto here");
      break;
    }

    Record.push_back((ID << Qualifiers::FastWidth) | FastQuals);
    return;
  }

  pch::TypeID &ID = TypeIDs[T];
  if (ID == 0) {
    // We haven't seen this type before. Assign it a new ID and put it
    // into the queue of types to emit.
    ID = NextTypeID++;
    DeclTypesToEmit.push(T);
  }

  // Encode the type qualifiers in the type reference.
  Record.push_back((ID << Qualifiers::FastWidth) | FastQuals);
}

void ASTWriter::AddDeclRef(const Decl *D, RecordData &Record) {
  Record.push_back(GetDeclRef(D));
}

pch::DeclID ASTWriter::GetDeclRef(const Decl *D) {
  if (D == 0) {
    return 0;
  }

  pch::DeclID &ID = DeclIDs[D];
  if (ID == 0) {
    // We haven't seen this declaration before. Give it a new ID and
    // enqueue it in the list of declarations to emit.
    ID = NextDeclID++;
    DeclTypesToEmit.push(const_cast<Decl *>(D));
  } else if (ID < FirstDeclID && D->isChangedSinceDeserialization()) {
    // We don't add it to the replacement collection here, because we don't
    // have the offset yet.
    DeclTypesToEmit.push(const_cast<Decl *>(D));
    // Reset the flag, so that we don't add this decl multiple times.
    const_cast<Decl *>(D)->setChangedSinceDeserialization(false);
  }

  return ID;
}

pch::DeclID ASTWriter::getDeclID(const Decl *D) {
  if (D == 0)
    return 0;

  assert(DeclIDs.find(D) != DeclIDs.end() && "Declaration not emitted!");
  return DeclIDs[D];
}

void ASTWriter::AddDeclarationName(DeclarationName Name, RecordData &Record) {
  // FIXME: Emit a stable enum for NameKind.  0 = Identifier etc.
  Record.push_back(Name.getNameKind());
  switch (Name.getNameKind()) {
  case DeclarationName::Identifier:
    AddIdentifierRef(Name.getAsIdentifierInfo(), Record);
    break;

  case DeclarationName::ObjCZeroArgSelector:
  case DeclarationName::ObjCOneArgSelector:
  case DeclarationName::ObjCMultiArgSelector:
    AddSelectorRef(Name.getObjCSelector(), Record);
    break;

  case DeclarationName::CXXConstructorName:
  case DeclarationName::CXXDestructorName:
  case DeclarationName::CXXConversionFunctionName:
    AddTypeRef(Name.getCXXNameType(), Record);
    break;

  case DeclarationName::CXXOperatorName:
    Record.push_back(Name.getCXXOverloadedOperator());
    break;

  case DeclarationName::CXXLiteralOperatorName:
    AddIdentifierRef(Name.getCXXLiteralIdentifier(), Record);
    break;

  case DeclarationName::CXXUsingDirective:
    // No extra data to emit
    break;
  }
}

void ASTWriter::AddNestedNameSpecifier(NestedNameSpecifier *NNS,
                                       RecordData &Record) {
  // Nested name specifiers usually aren't too long. I think that 8 would
  // typically accomodate the vast majority.
  llvm::SmallVector<NestedNameSpecifier *, 8> NestedNames;

  // Push each of the NNS's onto a stack for serialization in reverse order.
  while (NNS) {
    NestedNames.push_back(NNS);
    NNS = NNS->getPrefix();
  }

  Record.push_back(NestedNames.size());
  while(!NestedNames.empty()) {
    NNS = NestedNames.pop_back_val();
    NestedNameSpecifier::SpecifierKind Kind = NNS->getKind();
    Record.push_back(Kind);
    switch (Kind) {
    case NestedNameSpecifier::Identifier:
      AddIdentifierRef(NNS->getAsIdentifier(), Record);
      break;

    case NestedNameSpecifier::Namespace:
      AddDeclRef(NNS->getAsNamespace(), Record);
      break;

    case NestedNameSpecifier::TypeSpec:
    case NestedNameSpecifier::TypeSpecWithTemplate:
      AddTypeRef(QualType(NNS->getAsType(), 0), Record);
      Record.push_back(Kind == NestedNameSpecifier::TypeSpecWithTemplate);
      break;

    case NestedNameSpecifier::Global:
      // Don't need to write an associated value.
      break;
    }
  }
}

void ASTWriter::AddTemplateName(TemplateName Name, RecordData &Record) {
  TemplateName::NameKind Kind = Name.getKind(); 
  Record.push_back(Kind);
  switch (Kind) {
  case TemplateName::Template:
    AddDeclRef(Name.getAsTemplateDecl(), Record);
    break;

  case TemplateName::OverloadedTemplate: {
    OverloadedTemplateStorage *OvT = Name.getAsOverloadedTemplate();
    Record.push_back(OvT->size());
    for (OverloadedTemplateStorage::iterator I = OvT->begin(), E = OvT->end();
           I != E; ++I)
      AddDeclRef(*I, Record);
    break;
  }
    
  case TemplateName::QualifiedTemplate: {
    QualifiedTemplateName *QualT = Name.getAsQualifiedTemplateName();
    AddNestedNameSpecifier(QualT->getQualifier(), Record);
    Record.push_back(QualT->hasTemplateKeyword());
    AddDeclRef(QualT->getTemplateDecl(), Record);
    break;
  }
    
  case TemplateName::DependentTemplate: {
    DependentTemplateName *DepT = Name.getAsDependentTemplateName();
    AddNestedNameSpecifier(DepT->getQualifier(), Record);
    Record.push_back(DepT->isIdentifier());
    if (DepT->isIdentifier())
      AddIdentifierRef(DepT->getIdentifier(), Record);
    else
      Record.push_back(DepT->getOperator());
    break;
  }
  }
}

void ASTWriter::AddTemplateArgument(const TemplateArgument &Arg, 
                                    RecordData &Record) {
  Record.push_back(Arg.getKind());
  switch (Arg.getKind()) {
  case TemplateArgument::Null:
    break;
  case TemplateArgument::Type:
    AddTypeRef(Arg.getAsType(), Record);
    break;
  case TemplateArgument::Declaration:
    AddDeclRef(Arg.getAsDecl(), Record);
    break;
  case TemplateArgument::Integral:
    AddAPSInt(*Arg.getAsIntegral(), Record);
    AddTypeRef(Arg.getIntegralType(), Record);
    break;
  case TemplateArgument::Template:
    AddTemplateName(Arg.getAsTemplate(), Record);
    break;
  case TemplateArgument::Expression:
    AddStmt(Arg.getAsExpr());
    break;
  case TemplateArgument::Pack:
    Record.push_back(Arg.pack_size());
    for (TemplateArgument::pack_iterator I=Arg.pack_begin(), E=Arg.pack_end();
           I != E; ++I)
      AddTemplateArgument(*I, Record);
    break;
  }
}

void
ASTWriter::AddTemplateParameterList(const TemplateParameterList *TemplateParams,
                                    RecordData &Record) {
  assert(TemplateParams && "No TemplateParams!");
  AddSourceLocation(TemplateParams->getTemplateLoc(), Record);
  AddSourceLocation(TemplateParams->getLAngleLoc(), Record);
  AddSourceLocation(TemplateParams->getRAngleLoc(), Record);
  Record.push_back(TemplateParams->size());
  for (TemplateParameterList::const_iterator
         P = TemplateParams->begin(), PEnd = TemplateParams->end();
         P != PEnd; ++P)
    AddDeclRef(*P, Record);
}

/// \brief Emit a template argument list.
void
ASTWriter::AddTemplateArgumentList(const TemplateArgumentList *TemplateArgs,
                                   RecordData &Record) {
  assert(TemplateArgs && "No TemplateArgs!");
  Record.push_back(TemplateArgs->flat_size());
  for (int i=0, e = TemplateArgs->flat_size(); i != e; ++i)
    AddTemplateArgument(TemplateArgs->get(i), Record);
}


void
ASTWriter::AddUnresolvedSet(const UnresolvedSetImpl &Set, RecordData &Record) {
  Record.push_back(Set.size());
  for (UnresolvedSetImpl::const_iterator
         I = Set.begin(), E = Set.end(); I != E; ++I) {
    AddDeclRef(I.getDecl(), Record);
    Record.push_back(I.getAccess());
  }
}

void ASTWriter::AddCXXBaseSpecifier(const CXXBaseSpecifier &Base,
                                    RecordData &Record) {
  Record.push_back(Base.isVirtual());
  Record.push_back(Base.isBaseOfClass());
  Record.push_back(Base.getAccessSpecifierAsWritten());
  AddTypeSourceInfo(Base.getTypeSourceInfo(), Record);
  AddSourceRange(Base.getSourceRange(), Record);
}

void ASTWriter::AddCXXBaseOrMemberInitializers(
                        const CXXBaseOrMemberInitializer * const *BaseOrMembers,
                        unsigned NumBaseOrMembers, RecordData &Record) {
  Record.push_back(NumBaseOrMembers);
  for (unsigned i=0; i != NumBaseOrMembers; ++i) {
    const CXXBaseOrMemberInitializer *Init = BaseOrMembers[i];

    Record.push_back(Init->isBaseInitializer());
    if (Init->isBaseInitializer()) {
      AddTypeSourceInfo(Init->getBaseClassInfo(), Record);
      Record.push_back(Init->isBaseVirtual());
    } else {
      AddDeclRef(Init->getMember(), Record);
    }
    AddSourceLocation(Init->getMemberLocation(), Record);
    AddStmt(Init->getInit());
    AddDeclRef(Init->getAnonUnionMember(), Record);
    AddSourceLocation(Init->getLParenLoc(), Record);
    AddSourceLocation(Init->getRParenLoc(), Record);
    Record.push_back(Init->isWritten());
    if (Init->isWritten()) {
      Record.push_back(Init->getSourceOrder());
    } else {
      Record.push_back(Init->getNumArrayIndices());
      for (unsigned i=0, e=Init->getNumArrayIndices(); i != e; ++i)
        AddDeclRef(Init->getArrayIndex(i), Record);
    }
  }
}

void ASTWriter::SetReader(PCHReader *Reader) {
  assert(Reader && "Cannot remove chain");
  assert(FirstDeclID == NextDeclID &&
         FirstTypeID == NextTypeID &&
         FirstIdentID == NextIdentID &&
         FirstSelectorID == NextSelectorID &&
         "Setting chain after writing has started.");
  Chain = Reader;
}

void ASTWriter::IdentifierRead(pch::IdentID ID, IdentifierInfo *II) {
  IdentifierIDs[II] = ID;
}

void ASTWriter::TypeRead(pch::TypeID ID, QualType T) {
  TypeIDs[T] = ID;
}

void ASTWriter::DeclRead(pch::DeclID ID, const Decl *D) {
  DeclIDs[D] = ID;
}

void ASTWriter::SelectorRead(pch::SelectorID ID, Selector S) {
  SelectorIDs[S] = ID;
}
