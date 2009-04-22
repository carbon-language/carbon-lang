//===--- PCHWriter.h - Precompiled Headers Writer ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the PCHWriter class, which writes a precompiled header.
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/PCHWriter.h"
#include "../Sema/Sema.h" // FIXME: move header into include/clang/Sema
#include "../Sema/IdentifierResolver.h" // FIXME: move header 
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclContextInternals.h"
#include "clang/AST/DeclVisitor.h"
#include "clang/AST/Expr.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/AST/Type.h"
#include "clang/Lex/MacroInfo.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/OnDiskHashTable.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/SourceManagerInternals.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Bitcode/BitstreamWriter.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/MemoryBuffer.h"
#include <cstdio>
using namespace clang;

//===----------------------------------------------------------------------===//
// Type serialization
//===----------------------------------------------------------------------===//
namespace {
  class VISIBILITY_HIDDEN PCHTypeWriter {
    PCHWriter &Writer;
    PCHWriter::RecordData &Record;

  public:
    /// \brief Type code that corresponds to the record generated.
    pch::TypeCode Code;

    PCHTypeWriter(PCHWriter &Writer, PCHWriter::RecordData &Record) 
      : Writer(Writer), Record(Record) { }

    void VisitArrayType(const ArrayType *T);
    void VisitFunctionType(const FunctionType *T);
    void VisitTagType(const TagType *T);

#define TYPE(Class, Base) void Visit##Class##Type(const Class##Type *T);
#define ABSTRACT_TYPE(Class, Base)
#define DEPENDENT_TYPE(Class, Base)
#include "clang/AST/TypeNodes.def"
  };
}

void PCHTypeWriter::VisitExtQualType(const ExtQualType *T) {
  Writer.AddTypeRef(QualType(T->getBaseType(), 0), Record);
  Record.push_back(T->getObjCGCAttr()); // FIXME: use stable values
  Record.push_back(T->getAddressSpace());
  Code = pch::TYPE_EXT_QUAL;
}

void PCHTypeWriter::VisitBuiltinType(const BuiltinType *T) {
  assert(false && "Built-in types are never serialized");
}

void PCHTypeWriter::VisitFixedWidthIntType(const FixedWidthIntType *T) {
  Record.push_back(T->getWidth());
  Record.push_back(T->isSigned());
  Code = pch::TYPE_FIXED_WIDTH_INT;
}

void PCHTypeWriter::VisitComplexType(const ComplexType *T) {
  Writer.AddTypeRef(T->getElementType(), Record);
  Code = pch::TYPE_COMPLEX;
}

void PCHTypeWriter::VisitPointerType(const PointerType *T) {
  Writer.AddTypeRef(T->getPointeeType(), Record);
  Code = pch::TYPE_POINTER;
}

void PCHTypeWriter::VisitBlockPointerType(const BlockPointerType *T) {
  Writer.AddTypeRef(T->getPointeeType(), Record);  
  Code = pch::TYPE_BLOCK_POINTER;
}

void PCHTypeWriter::VisitLValueReferenceType(const LValueReferenceType *T) {
  Writer.AddTypeRef(T->getPointeeType(), Record);
  Code = pch::TYPE_LVALUE_REFERENCE;
}

void PCHTypeWriter::VisitRValueReferenceType(const RValueReferenceType *T) {
  Writer.AddTypeRef(T->getPointeeType(), Record);
  Code = pch::TYPE_RVALUE_REFERENCE;
}

void PCHTypeWriter::VisitMemberPointerType(const MemberPointerType *T) {
  Writer.AddTypeRef(T->getPointeeType(), Record);  
  Writer.AddTypeRef(QualType(T->getClass(), 0), Record);  
  Code = pch::TYPE_MEMBER_POINTER;
}

void PCHTypeWriter::VisitArrayType(const ArrayType *T) {
  Writer.AddTypeRef(T->getElementType(), Record);
  Record.push_back(T->getSizeModifier()); // FIXME: stable values
  Record.push_back(T->getIndexTypeQualifier()); // FIXME: stable values
}

void PCHTypeWriter::VisitConstantArrayType(const ConstantArrayType *T) {
  VisitArrayType(T);
  Writer.AddAPInt(T->getSize(), Record);
  Code = pch::TYPE_CONSTANT_ARRAY;
}

void PCHTypeWriter::VisitIncompleteArrayType(const IncompleteArrayType *T) {
  VisitArrayType(T);
  Code = pch::TYPE_INCOMPLETE_ARRAY;
}

void PCHTypeWriter::VisitVariableArrayType(const VariableArrayType *T) {
  VisitArrayType(T);
  Writer.AddStmt(T->getSizeExpr());
  Code = pch::TYPE_VARIABLE_ARRAY;
}

void PCHTypeWriter::VisitVectorType(const VectorType *T) {
  Writer.AddTypeRef(T->getElementType(), Record);
  Record.push_back(T->getNumElements());
  Code = pch::TYPE_VECTOR;
}

void PCHTypeWriter::VisitExtVectorType(const ExtVectorType *T) {
  VisitVectorType(T);
  Code = pch::TYPE_EXT_VECTOR;
}

void PCHTypeWriter::VisitFunctionType(const FunctionType *T) {
  Writer.AddTypeRef(T->getResultType(), Record);
}

void PCHTypeWriter::VisitFunctionNoProtoType(const FunctionNoProtoType *T) {
  VisitFunctionType(T);
  Code = pch::TYPE_FUNCTION_NO_PROTO;
}

void PCHTypeWriter::VisitFunctionProtoType(const FunctionProtoType *T) {
  VisitFunctionType(T);
  Record.push_back(T->getNumArgs());
  for (unsigned I = 0, N = T->getNumArgs(); I != N; ++I)
    Writer.AddTypeRef(T->getArgType(I), Record);
  Record.push_back(T->isVariadic());
  Record.push_back(T->getTypeQuals());
  Code = pch::TYPE_FUNCTION_PROTO;
}

void PCHTypeWriter::VisitTypedefType(const TypedefType *T) {
  Writer.AddDeclRef(T->getDecl(), Record);
  Code = pch::TYPE_TYPEDEF;
}

void PCHTypeWriter::VisitTypeOfExprType(const TypeOfExprType *T) {
  Writer.AddStmt(T->getUnderlyingExpr());
  Code = pch::TYPE_TYPEOF_EXPR;
}

void PCHTypeWriter::VisitTypeOfType(const TypeOfType *T) {
  Writer.AddTypeRef(T->getUnderlyingType(), Record);
  Code = pch::TYPE_TYPEOF;
}

void PCHTypeWriter::VisitTagType(const TagType *T) {
  Writer.AddDeclRef(T->getDecl(), Record);
  assert(!T->isBeingDefined() && 
         "Cannot serialize in the middle of a type definition");
}

void PCHTypeWriter::VisitRecordType(const RecordType *T) {
  VisitTagType(T);
  Code = pch::TYPE_RECORD;
}

void PCHTypeWriter::VisitEnumType(const EnumType *T) {
  VisitTagType(T);
  Code = pch::TYPE_ENUM;
}

void 
PCHTypeWriter::VisitTemplateSpecializationType(
                                       const TemplateSpecializationType *T) {
  // FIXME: Serialize this type (C++ only)
  assert(false && "Cannot serialize template specialization types");
}

void PCHTypeWriter::VisitQualifiedNameType(const QualifiedNameType *T) {
  // FIXME: Serialize this type (C++ only)
  assert(false && "Cannot serialize qualified name types");
}

void PCHTypeWriter::VisitObjCInterfaceType(const ObjCInterfaceType *T) {
  Writer.AddDeclRef(T->getDecl(), Record);
  Code = pch::TYPE_OBJC_INTERFACE;
}

void 
PCHTypeWriter::VisitObjCQualifiedInterfaceType(
                                      const ObjCQualifiedInterfaceType *T) {
  VisitObjCInterfaceType(T);
  Record.push_back(T->getNumProtocols());
  for (unsigned I = 0, N = T->getNumProtocols(); I != N; ++I)
    Writer.AddDeclRef(T->getProtocol(I), Record);
  Code = pch::TYPE_OBJC_QUALIFIED_INTERFACE;
}

void PCHTypeWriter::VisitObjCQualifiedIdType(const ObjCQualifiedIdType *T) {
  Record.push_back(T->getNumProtocols());
  for (unsigned I = 0, N = T->getNumProtocols(); I != N; ++I)
    Writer.AddDeclRef(T->getProtocols(I), Record);
  Code = pch::TYPE_OBJC_QUALIFIED_ID;
}

//===----------------------------------------------------------------------===//
// Declaration serialization
//===----------------------------------------------------------------------===//
namespace {
  class VISIBILITY_HIDDEN PCHDeclWriter
    : public DeclVisitor<PCHDeclWriter, void> {

    PCHWriter &Writer;
    ASTContext &Context;
    PCHWriter::RecordData &Record;

  public:
    pch::DeclCode Code;

    PCHDeclWriter(PCHWriter &Writer, ASTContext &Context, 
                  PCHWriter::RecordData &Record) 
      : Writer(Writer), Context(Context), Record(Record) { }

    void VisitDecl(Decl *D);
    void VisitTranslationUnitDecl(TranslationUnitDecl *D);
    void VisitNamedDecl(NamedDecl *D);
    void VisitTypeDecl(TypeDecl *D);
    void VisitTypedefDecl(TypedefDecl *D);
    void VisitTagDecl(TagDecl *D);
    void VisitEnumDecl(EnumDecl *D);
    void VisitRecordDecl(RecordDecl *D);
    void VisitValueDecl(ValueDecl *D);
    void VisitEnumConstantDecl(EnumConstantDecl *D);
    void VisitFunctionDecl(FunctionDecl *D);
    void VisitFieldDecl(FieldDecl *D);
    void VisitVarDecl(VarDecl *D);
    void VisitParmVarDecl(ParmVarDecl *D);
    void VisitOriginalParmVarDecl(OriginalParmVarDecl *D);
    void VisitFileScopeAsmDecl(FileScopeAsmDecl *D);
    void VisitBlockDecl(BlockDecl *D);
    void VisitDeclContext(DeclContext *DC, uint64_t LexicalOffset, 
                          uint64_t VisibleOffset);
    void VisitObjCMethodDecl(ObjCMethodDecl *D);
    void VisitObjCContainerDecl(ObjCContainerDecl *D);
    void VisitObjCInterfaceDecl(ObjCInterfaceDecl *D);
    void VisitObjCIvarDecl(ObjCIvarDecl *D);
    void VisitObjCProtocolDecl(ObjCProtocolDecl *D);
    void VisitObjCAtDefsFieldDecl(ObjCAtDefsFieldDecl *D);
    void VisitObjCClassDecl(ObjCClassDecl *D);
    void VisitObjCForwardProtocolDecl(ObjCForwardProtocolDecl *D);
    void VisitObjCCategoryDecl(ObjCCategoryDecl *D);
    void VisitObjCImplDecl(ObjCImplDecl *D);
    void VisitObjCCategoryImplDecl(ObjCCategoryImplDecl *D);
    void VisitObjCImplementationDecl(ObjCImplementationDecl *D);
    void VisitObjCCompatibleAliasDecl(ObjCCompatibleAliasDecl *D);
    void VisitObjCPropertyDecl(ObjCPropertyDecl *D);
    void VisitObjCPropertyImplDecl(ObjCPropertyImplDecl *D);
  };
}

void PCHDeclWriter::VisitDecl(Decl *D) {
  Writer.AddDeclRef(cast_or_null<Decl>(D->getDeclContext()), Record);
  Writer.AddDeclRef(cast_or_null<Decl>(D->getLexicalDeclContext()), Record);
  Writer.AddSourceLocation(D->getLocation(), Record);
  Record.push_back(D->isInvalidDecl());
  Record.push_back(D->hasAttrs());
  Record.push_back(D->isImplicit());
  Record.push_back(D->getAccess());
}

void PCHDeclWriter::VisitTranslationUnitDecl(TranslationUnitDecl *D) {
  VisitDecl(D);
  Code = pch::DECL_TRANSLATION_UNIT;
}

void PCHDeclWriter::VisitNamedDecl(NamedDecl *D) {
  VisitDecl(D);
  Writer.AddDeclarationName(D->getDeclName(), Record);
}

void PCHDeclWriter::VisitTypeDecl(TypeDecl *D) {
  VisitNamedDecl(D);
  Writer.AddTypeRef(QualType(D->getTypeForDecl(), 0), Record);
}

void PCHDeclWriter::VisitTypedefDecl(TypedefDecl *D) {
  VisitTypeDecl(D);
  Writer.AddTypeRef(D->getUnderlyingType(), Record);
  Code = pch::DECL_TYPEDEF;
}

void PCHDeclWriter::VisitTagDecl(TagDecl *D) {
  VisitTypeDecl(D);
  Record.push_back((unsigned)D->getTagKind()); // FIXME: stable encoding
  Record.push_back(D->isDefinition());
  Writer.AddDeclRef(D->getTypedefForAnonDecl(), Record);
}

void PCHDeclWriter::VisitEnumDecl(EnumDecl *D) {
  VisitTagDecl(D);
  Writer.AddTypeRef(D->getIntegerType(), Record);
  Code = pch::DECL_ENUM;
}

void PCHDeclWriter::VisitRecordDecl(RecordDecl *D) {
  VisitTagDecl(D);
  Record.push_back(D->hasFlexibleArrayMember());
  Record.push_back(D->isAnonymousStructOrUnion());
  Code = pch::DECL_RECORD;
}

void PCHDeclWriter::VisitValueDecl(ValueDecl *D) {
  VisitNamedDecl(D);
  Writer.AddTypeRef(D->getType(), Record);
}

void PCHDeclWriter::VisitEnumConstantDecl(EnumConstantDecl *D) {
  VisitValueDecl(D);
  Record.push_back(D->getInitExpr()? 1 : 0);
  if (D->getInitExpr())
    Writer.AddStmt(D->getInitExpr());
  Writer.AddAPSInt(D->getInitVal(), Record);
  Code = pch::DECL_ENUM_CONSTANT;
}

void PCHDeclWriter::VisitFunctionDecl(FunctionDecl *D) {
  VisitValueDecl(D);
  Record.push_back(D->isThisDeclarationADefinition());
  if (D->isThisDeclarationADefinition())
    Writer.AddStmt(D->getBody(Context));
  Writer.AddDeclRef(D->getPreviousDeclaration(), Record);
  Record.push_back(D->getStorageClass()); // FIXME: stable encoding
  Record.push_back(D->isInline());
  Record.push_back(D->isVirtual());
  Record.push_back(D->isPure());
  Record.push_back(D->inheritedPrototype());
  Record.push_back(D->hasPrototype() && !D->inheritedPrototype());
  Record.push_back(D->isDeleted());
  Writer.AddSourceLocation(D->getTypeSpecStartLoc(), Record);
  Record.push_back(D->param_size());
  for (FunctionDecl::param_iterator P = D->param_begin(), PEnd = D->param_end();
       P != PEnd; ++P)
    Writer.AddDeclRef(*P, Record);
  Code = pch::DECL_FUNCTION;
}

void PCHDeclWriter::VisitObjCMethodDecl(ObjCMethodDecl *D) {
  VisitNamedDecl(D);
  // FIXME: convert to LazyStmtPtr?
  // Unlike C/C++, method bodies will never be in header files. 
  Record.push_back(D->getBody() != 0);
  if (D->getBody() != 0) {
    Writer.AddStmt(D->getBody(Context));
    Writer.AddDeclRef(D->getSelfDecl(), Record);
    Writer.AddDeclRef(D->getCmdDecl(), Record);
  }
  Record.push_back(D->isInstanceMethod());
  Record.push_back(D->isVariadic());
  Record.push_back(D->isSynthesized());
  // FIXME: stable encoding for @required/@optional
  Record.push_back(D->getImplementationControl()); 
  // FIXME: stable encoding for in/out/inout/bycopy/byref/oneway
  Record.push_back(D->getObjCDeclQualifier()); 
  Writer.AddTypeRef(D->getResultType(), Record);
  Writer.AddSourceLocation(D->getLocEnd(), Record);
  Record.push_back(D->param_size());
  for (ObjCMethodDecl::param_iterator P = D->param_begin(), 
                                   PEnd = D->param_end(); P != PEnd; ++P)
    Writer.AddDeclRef(*P, Record);
  Code = pch::DECL_OBJC_METHOD;
}

void PCHDeclWriter::VisitObjCContainerDecl(ObjCContainerDecl *D) {
  VisitNamedDecl(D);
  Writer.AddSourceLocation(D->getAtEndLoc(), Record);
  // Abstract class (no need to define a stable pch::DECL code).
}

void PCHDeclWriter::VisitObjCInterfaceDecl(ObjCInterfaceDecl *D) {
  VisitObjCContainerDecl(D);
  Writer.AddTypeRef(QualType(D->getTypeForDecl(), 0), Record);
  Writer.AddDeclRef(D->getSuperClass(), Record);
  Record.push_back(D->ivar_size());
  for (ObjCInterfaceDecl::ivar_iterator I = D->ivar_begin(), 
                                     IEnd = D->ivar_end(); I != IEnd; ++I)
    Writer.AddDeclRef(*I, Record);
  Record.push_back(D->isForwardDecl());
  Record.push_back(D->isImplicitInterfaceDecl());
  Writer.AddSourceLocation(D->getClassLoc(), Record);
  Writer.AddSourceLocation(D->getSuperClassLoc(), Record);
  Writer.AddSourceLocation(D->getLocEnd(), Record);
  // FIXME: add protocols, categories.
  Code = pch::DECL_OBJC_INTERFACE;
}

void PCHDeclWriter::VisitObjCIvarDecl(ObjCIvarDecl *D) {
  VisitFieldDecl(D);
  // FIXME: stable encoding for @public/@private/@protected/@package
  Record.push_back(D->getAccessControl()); 
  Code = pch::DECL_OBJC_IVAR;
}

void PCHDeclWriter::VisitObjCProtocolDecl(ObjCProtocolDecl *D) {
  VisitObjCContainerDecl(D);
  Record.push_back(D->isForwardDecl());
  Writer.AddSourceLocation(D->getLocEnd(), Record);
  Record.push_back(D->protocol_size());
  for (ObjCProtocolDecl::protocol_iterator 
       I = D->protocol_begin(), IEnd = D->protocol_end(); I != IEnd; ++I)
    Writer.AddDeclRef(*I, Record);
  Code = pch::DECL_OBJC_PROTOCOL;
}

void PCHDeclWriter::VisitObjCAtDefsFieldDecl(ObjCAtDefsFieldDecl *D) {
  VisitFieldDecl(D);
  Code = pch::DECL_OBJC_AT_DEFS_FIELD;
}

void PCHDeclWriter::VisitObjCClassDecl(ObjCClassDecl *D) {
  VisitDecl(D);
  Record.push_back(D->size());
  for (ObjCClassDecl::iterator I = D->begin(), IEnd = D->end(); I != IEnd; ++I)
    Writer.AddDeclRef(*I, Record);
  Code = pch::DECL_OBJC_CLASS;
}

void PCHDeclWriter::VisitObjCForwardProtocolDecl(ObjCForwardProtocolDecl *D) {
  VisitDecl(D);
  Record.push_back(D->protocol_size());
  for (ObjCProtocolDecl::protocol_iterator 
       I = D->protocol_begin(), IEnd = D->protocol_end(); I != IEnd; ++I)
    Writer.AddDeclRef(*I, Record);
  Code = pch::DECL_OBJC_FORWARD_PROTOCOL;
}

void PCHDeclWriter::VisitObjCCategoryDecl(ObjCCategoryDecl *D) {
  VisitObjCContainerDecl(D);
  Writer.AddDeclRef(D->getClassInterface(), Record);
  Record.push_back(D->protocol_size());
  for (ObjCProtocolDecl::protocol_iterator 
       I = D->protocol_begin(), IEnd = D->protocol_end(); I != IEnd; ++I)
    Writer.AddDeclRef(*I, Record);
  Writer.AddDeclRef(D->getNextClassCategory(), Record);
  Writer.AddSourceLocation(D->getLocEnd(), Record);
  Code = pch::DECL_OBJC_CATEGORY;
}

void PCHDeclWriter::VisitObjCCompatibleAliasDecl(ObjCCompatibleAliasDecl *D) {
  VisitNamedDecl(D);
  Writer.AddDeclRef(D->getClassInterface(), Record);
  Code = pch::DECL_OBJC_COMPATIBLE_ALIAS;
}

void PCHDeclWriter::VisitObjCPropertyDecl(ObjCPropertyDecl *D) {
  VisitNamedDecl(D);
  // FIXME: Implement.
  Code = pch::DECL_OBJC_PROPERTY;
}

void PCHDeclWriter::VisitObjCImplDecl(ObjCImplDecl *D) {
  VisitDecl(D);
  // FIXME: Implement.
  // Abstract class (no need to define a stable pch::DECL code).
}

void PCHDeclWriter::VisitObjCCategoryImplDecl(ObjCCategoryImplDecl *D) {
  VisitObjCImplDecl(D);
  // FIXME: Implement.
  Code = pch::DECL_OBJC_CATEGORY_IMPL;
}

void PCHDeclWriter::VisitObjCImplementationDecl(ObjCImplementationDecl *D) {
  VisitObjCImplDecl(D);
  // FIXME: Implement.
  Code = pch::DECL_OBJC_IMPLEMENTATION;
}

void PCHDeclWriter::VisitObjCPropertyImplDecl(ObjCPropertyImplDecl *D) {
  VisitDecl(D);
  // FIXME: Implement.
  Code = pch::DECL_OBJC_PROPERTY_IMPL;
}

void PCHDeclWriter::VisitFieldDecl(FieldDecl *D) {
  VisitValueDecl(D);
  Record.push_back(D->isMutable());
  Record.push_back(D->getBitWidth()? 1 : 0);
  if (D->getBitWidth())
    Writer.AddStmt(D->getBitWidth());
  Code = pch::DECL_FIELD;
}

void PCHDeclWriter::VisitVarDecl(VarDecl *D) {
  VisitValueDecl(D);
  Record.push_back(D->getStorageClass()); // FIXME: stable encoding
  Record.push_back(D->isThreadSpecified());
  Record.push_back(D->hasCXXDirectInitializer());
  Record.push_back(D->isDeclaredInCondition());
  Writer.AddDeclRef(D->getPreviousDeclaration(), Record);
  Writer.AddSourceLocation(D->getTypeSpecStartLoc(), Record);
  Record.push_back(D->getInit()? 1 : 0);
  if (D->getInit())
    Writer.AddStmt(D->getInit());
  Code = pch::DECL_VAR;
}

void PCHDeclWriter::VisitParmVarDecl(ParmVarDecl *D) {
  VisitVarDecl(D);
  Record.push_back(D->getObjCDeclQualifier()); // FIXME: stable encoding
  // FIXME: emit default argument (C++)
  // FIXME: why isn't the "default argument" just stored as the initializer
  // in VarDecl?
  Code = pch::DECL_PARM_VAR;
}

void PCHDeclWriter::VisitOriginalParmVarDecl(OriginalParmVarDecl *D) {
  VisitParmVarDecl(D);
  Writer.AddTypeRef(D->getOriginalType(), Record);
  Code = pch::DECL_ORIGINAL_PARM_VAR;
}

void PCHDeclWriter::VisitFileScopeAsmDecl(FileScopeAsmDecl *D) {
  VisitDecl(D);
  Writer.AddStmt(D->getAsmString());
  Code = pch::DECL_FILE_SCOPE_ASM;
}

void PCHDeclWriter::VisitBlockDecl(BlockDecl *D) {
  VisitDecl(D);
  Writer.AddStmt(D->getBody());
  Record.push_back(D->param_size());
  for (FunctionDecl::param_iterator P = D->param_begin(), PEnd = D->param_end();
       P != PEnd; ++P)
    Writer.AddDeclRef(*P, Record);
  Code = pch::DECL_BLOCK;
}

/// \brief Emit the DeclContext part of a declaration context decl.
///
/// \param LexicalOffset the offset at which the DECL_CONTEXT_LEXICAL
/// block for this declaration context is stored. May be 0 to indicate
/// that there are no declarations stored within this context.
///
/// \param VisibleOffset the offset at which the DECL_CONTEXT_VISIBLE
/// block for this declaration context is stored. May be 0 to indicate
/// that there are no declarations visible from this context. Note
/// that this value will not be emitted for non-primary declaration
/// contexts.
void PCHDeclWriter::VisitDeclContext(DeclContext *DC, uint64_t LexicalOffset, 
                                     uint64_t VisibleOffset) {
  Record.push_back(LexicalOffset);
  Record.push_back(VisibleOffset);
}

//===----------------------------------------------------------------------===//
// Statement/expression serialization
//===----------------------------------------------------------------------===//
namespace {
  class VISIBILITY_HIDDEN PCHStmtWriter
    : public StmtVisitor<PCHStmtWriter, void> {

    PCHWriter &Writer;
    PCHWriter::RecordData &Record;

  public:
    pch::StmtCode Code;

    PCHStmtWriter(PCHWriter &Writer, PCHWriter::RecordData &Record)
      : Writer(Writer), Record(Record) { }

    void VisitStmt(Stmt *S);
    void VisitNullStmt(NullStmt *S);
    void VisitCompoundStmt(CompoundStmt *S);
    void VisitSwitchCase(SwitchCase *S);
    void VisitCaseStmt(CaseStmt *S);
    void VisitDefaultStmt(DefaultStmt *S);
    void VisitLabelStmt(LabelStmt *S);
    void VisitIfStmt(IfStmt *S);
    void VisitSwitchStmt(SwitchStmt *S);
    void VisitWhileStmt(WhileStmt *S);
    void VisitDoStmt(DoStmt *S);
    void VisitForStmt(ForStmt *S);
    void VisitGotoStmt(GotoStmt *S);
    void VisitIndirectGotoStmt(IndirectGotoStmt *S);
    void VisitContinueStmt(ContinueStmt *S);
    void VisitBreakStmt(BreakStmt *S);
    void VisitReturnStmt(ReturnStmt *S);
    void VisitDeclStmt(DeclStmt *S);
    void VisitAsmStmt(AsmStmt *S);
    void VisitExpr(Expr *E);
    void VisitPredefinedExpr(PredefinedExpr *E);
    void VisitDeclRefExpr(DeclRefExpr *E);
    void VisitIntegerLiteral(IntegerLiteral *E);
    void VisitFloatingLiteral(FloatingLiteral *E);
    void VisitImaginaryLiteral(ImaginaryLiteral *E);
    void VisitStringLiteral(StringLiteral *E);
    void VisitCharacterLiteral(CharacterLiteral *E);
    void VisitParenExpr(ParenExpr *E);
    void VisitUnaryOperator(UnaryOperator *E);
    void VisitSizeOfAlignOfExpr(SizeOfAlignOfExpr *E);
    void VisitArraySubscriptExpr(ArraySubscriptExpr *E);
    void VisitCallExpr(CallExpr *E);
    void VisitMemberExpr(MemberExpr *E);
    void VisitCastExpr(CastExpr *E);
    void VisitBinaryOperator(BinaryOperator *E);
    void VisitCompoundAssignOperator(CompoundAssignOperator *E);
    void VisitConditionalOperator(ConditionalOperator *E);
    void VisitImplicitCastExpr(ImplicitCastExpr *E);
    void VisitExplicitCastExpr(ExplicitCastExpr *E);
    void VisitCStyleCastExpr(CStyleCastExpr *E);
    void VisitCompoundLiteralExpr(CompoundLiteralExpr *E);
    void VisitExtVectorElementExpr(ExtVectorElementExpr *E);
    void VisitInitListExpr(InitListExpr *E);
    void VisitDesignatedInitExpr(DesignatedInitExpr *E);
    void VisitImplicitValueInitExpr(ImplicitValueInitExpr *E);
    void VisitVAArgExpr(VAArgExpr *E);
    void VisitAddrLabelExpr(AddrLabelExpr *E);
    void VisitStmtExpr(StmtExpr *E);
    void VisitTypesCompatibleExpr(TypesCompatibleExpr *E);
    void VisitChooseExpr(ChooseExpr *E);
    void VisitGNUNullExpr(GNUNullExpr *E);
    void VisitShuffleVectorExpr(ShuffleVectorExpr *E);
    void VisitBlockExpr(BlockExpr *E);
    void VisitBlockDeclRefExpr(BlockDeclRefExpr *E);
      
    // Objective-C
    void VisitObjCStringLiteral(ObjCStringLiteral *E);
    void VisitObjCEncodeExpr(ObjCEncodeExpr *E);
    void VisitObjCSelectorExpr(ObjCSelectorExpr *E);
    void VisitObjCProtocolExpr(ObjCProtocolExpr *E);
  };
}

void PCHStmtWriter::VisitStmt(Stmt *S) { 
}

void PCHStmtWriter::VisitNullStmt(NullStmt *S) {
  VisitStmt(S);
  Writer.AddSourceLocation(S->getSemiLoc(), Record);
  Code = pch::STMT_NULL;
}

void PCHStmtWriter::VisitCompoundStmt(CompoundStmt *S) {
  VisitStmt(S);
  Record.push_back(S->size());
  for (CompoundStmt::body_iterator CS = S->body_begin(), CSEnd = S->body_end();
       CS != CSEnd; ++CS)
    Writer.WriteSubStmt(*CS);
  Writer.AddSourceLocation(S->getLBracLoc(), Record);
  Writer.AddSourceLocation(S->getRBracLoc(), Record);
  Code = pch::STMT_COMPOUND;
}

void PCHStmtWriter::VisitSwitchCase(SwitchCase *S) {
  VisitStmt(S);
  Record.push_back(Writer.RecordSwitchCaseID(S));
}

void PCHStmtWriter::VisitCaseStmt(CaseStmt *S) {
  VisitSwitchCase(S);
  Writer.WriteSubStmt(S->getLHS());
  Writer.WriteSubStmt(S->getRHS());
  Writer.WriteSubStmt(S->getSubStmt());
  Writer.AddSourceLocation(S->getCaseLoc(), Record);
  Code = pch::STMT_CASE;
}

void PCHStmtWriter::VisitDefaultStmt(DefaultStmt *S) {
  VisitSwitchCase(S);
  Writer.WriteSubStmt(S->getSubStmt());
  Writer.AddSourceLocation(S->getDefaultLoc(), Record);
  Code = pch::STMT_DEFAULT;
}

void PCHStmtWriter::VisitLabelStmt(LabelStmt *S) {
  VisitStmt(S);
  Writer.AddIdentifierRef(S->getID(), Record);
  Writer.WriteSubStmt(S->getSubStmt());
  Writer.AddSourceLocation(S->getIdentLoc(), Record);
  Record.push_back(Writer.GetLabelID(S));
  Code = pch::STMT_LABEL;
}

void PCHStmtWriter::VisitIfStmt(IfStmt *S) {
  VisitStmt(S);
  Writer.WriteSubStmt(S->getCond());
  Writer.WriteSubStmt(S->getThen());
  Writer.WriteSubStmt(S->getElse());
  Writer.AddSourceLocation(S->getIfLoc(), Record);
  Code = pch::STMT_IF;
}

void PCHStmtWriter::VisitSwitchStmt(SwitchStmt *S) {
  VisitStmt(S);
  Writer.WriteSubStmt(S->getCond());
  Writer.WriteSubStmt(S->getBody());
  Writer.AddSourceLocation(S->getSwitchLoc(), Record);
  for (SwitchCase *SC = S->getSwitchCaseList(); SC; 
       SC = SC->getNextSwitchCase())
    Record.push_back(Writer.getSwitchCaseID(SC));
  Code = pch::STMT_SWITCH;
}

void PCHStmtWriter::VisitWhileStmt(WhileStmt *S) {
  VisitStmt(S);
  Writer.WriteSubStmt(S->getCond());
  Writer.WriteSubStmt(S->getBody());
  Writer.AddSourceLocation(S->getWhileLoc(), Record);
  Code = pch::STMT_WHILE;
}

void PCHStmtWriter::VisitDoStmt(DoStmt *S) {
  VisitStmt(S);
  Writer.WriteSubStmt(S->getCond());
  Writer.WriteSubStmt(S->getBody());
  Writer.AddSourceLocation(S->getDoLoc(), Record);
  Code = pch::STMT_DO;
}

void PCHStmtWriter::VisitForStmt(ForStmt *S) {
  VisitStmt(S);
  Writer.WriteSubStmt(S->getInit());
  Writer.WriteSubStmt(S->getCond());
  Writer.WriteSubStmt(S->getInc());
  Writer.WriteSubStmt(S->getBody());
  Writer.AddSourceLocation(S->getForLoc(), Record);
  Code = pch::STMT_FOR;
}

void PCHStmtWriter::VisitGotoStmt(GotoStmt *S) {
  VisitStmt(S);
  Record.push_back(Writer.GetLabelID(S->getLabel()));
  Writer.AddSourceLocation(S->getGotoLoc(), Record);
  Writer.AddSourceLocation(S->getLabelLoc(), Record);
  Code = pch::STMT_GOTO;
}

void PCHStmtWriter::VisitIndirectGotoStmt(IndirectGotoStmt *S) {
  VisitStmt(S);
  Writer.AddSourceLocation(S->getGotoLoc(), Record);
  Writer.WriteSubStmt(S->getTarget());
  Code = pch::STMT_INDIRECT_GOTO;
}

void PCHStmtWriter::VisitContinueStmt(ContinueStmt *S) {
  VisitStmt(S);
  Writer.AddSourceLocation(S->getContinueLoc(), Record);
  Code = pch::STMT_CONTINUE;
}

void PCHStmtWriter::VisitBreakStmt(BreakStmt *S) {
  VisitStmt(S);
  Writer.AddSourceLocation(S->getBreakLoc(), Record);
  Code = pch::STMT_BREAK;
}

void PCHStmtWriter::VisitReturnStmt(ReturnStmt *S) {
  VisitStmt(S);
  Writer.WriteSubStmt(S->getRetValue());
  Writer.AddSourceLocation(S->getReturnLoc(), Record);
  Code = pch::STMT_RETURN;
}

void PCHStmtWriter::VisitDeclStmt(DeclStmt *S) {
  VisitStmt(S);
  Writer.AddSourceLocation(S->getStartLoc(), Record);
  Writer.AddSourceLocation(S->getEndLoc(), Record);
  DeclGroupRef DG = S->getDeclGroup();
  for (DeclGroupRef::iterator D = DG.begin(), DEnd = DG.end(); D != DEnd; ++D)
    Writer.AddDeclRef(*D, Record);
  Code = pch::STMT_DECL;
}

void PCHStmtWriter::VisitAsmStmt(AsmStmt *S) {
  VisitStmt(S);
  Record.push_back(S->getNumOutputs());
  Record.push_back(S->getNumInputs());
  Record.push_back(S->getNumClobbers());
  Writer.AddSourceLocation(S->getAsmLoc(), Record);
  Writer.AddSourceLocation(S->getRParenLoc(), Record);
  Record.push_back(S->isVolatile());
  Record.push_back(S->isSimple());
  Writer.WriteSubStmt(S->getAsmString());

  // Outputs
  for (unsigned I = 0, N = S->getNumOutputs(); I != N; ++I) {
    Writer.AddString(S->getOutputName(I), Record);
    Writer.WriteSubStmt(S->getOutputConstraintLiteral(I));
    Writer.WriteSubStmt(S->getOutputExpr(I));
  }

  // Inputs
  for (unsigned I = 0, N = S->getNumInputs(); I != N; ++I) {
    Writer.AddString(S->getInputName(I), Record);
    Writer.WriteSubStmt(S->getInputConstraintLiteral(I));
    Writer.WriteSubStmt(S->getInputExpr(I));
  }

  // Clobbers
  for (unsigned I = 0, N = S->getNumClobbers(); I != N; ++I)
    Writer.WriteSubStmt(S->getClobber(I));

  Code = pch::STMT_ASM;
}

void PCHStmtWriter::VisitExpr(Expr *E) {
  VisitStmt(E);
  Writer.AddTypeRef(E->getType(), Record);
  Record.push_back(E->isTypeDependent());
  Record.push_back(E->isValueDependent());
}

void PCHStmtWriter::VisitPredefinedExpr(PredefinedExpr *E) {
  VisitExpr(E);
  Writer.AddSourceLocation(E->getLocation(), Record);
  Record.push_back(E->getIdentType()); // FIXME: stable encoding
  Code = pch::EXPR_PREDEFINED;
}

void PCHStmtWriter::VisitDeclRefExpr(DeclRefExpr *E) {
  VisitExpr(E);
  Writer.AddDeclRef(E->getDecl(), Record);
  Writer.AddSourceLocation(E->getLocation(), Record);
  Code = pch::EXPR_DECL_REF;
}

void PCHStmtWriter::VisitIntegerLiteral(IntegerLiteral *E) {
  VisitExpr(E);
  Writer.AddSourceLocation(E->getLocation(), Record);
  Writer.AddAPInt(E->getValue(), Record);
  Code = pch::EXPR_INTEGER_LITERAL;
}

void PCHStmtWriter::VisitFloatingLiteral(FloatingLiteral *E) {
  VisitExpr(E);
  Writer.AddAPFloat(E->getValue(), Record);
  Record.push_back(E->isExact());
  Writer.AddSourceLocation(E->getLocation(), Record);
  Code = pch::EXPR_FLOATING_LITERAL;
}

void PCHStmtWriter::VisitImaginaryLiteral(ImaginaryLiteral *E) {
  VisitExpr(E);
  Writer.WriteSubStmt(E->getSubExpr());
  Code = pch::EXPR_IMAGINARY_LITERAL;
}

void PCHStmtWriter::VisitStringLiteral(StringLiteral *E) {
  VisitExpr(E);
  Record.push_back(E->getByteLength());
  Record.push_back(E->getNumConcatenated());
  Record.push_back(E->isWide());
  // FIXME: String data should be stored as a blob at the end of the
  // StringLiteral. However, we can't do so now because we have no
  // provision for coping with abbreviations when we're jumping around
  // the PCH file during deserialization.
  Record.insert(Record.end(), 
                E->getStrData(), E->getStrData() + E->getByteLength());
  for (unsigned I = 0, N = E->getNumConcatenated(); I != N; ++I)
    Writer.AddSourceLocation(E->getStrTokenLoc(I), Record);
  Code = pch::EXPR_STRING_LITERAL;
}

void PCHStmtWriter::VisitCharacterLiteral(CharacterLiteral *E) {
  VisitExpr(E);
  Record.push_back(E->getValue());
  Writer.AddSourceLocation(E->getLoc(), Record);
  Record.push_back(E->isWide());
  Code = pch::EXPR_CHARACTER_LITERAL;
}

void PCHStmtWriter::VisitParenExpr(ParenExpr *E) {
  VisitExpr(E);
  Writer.AddSourceLocation(E->getLParen(), Record);
  Writer.AddSourceLocation(E->getRParen(), Record);
  Writer.WriteSubStmt(E->getSubExpr());
  Code = pch::EXPR_PAREN;
}

void PCHStmtWriter::VisitUnaryOperator(UnaryOperator *E) {
  VisitExpr(E);
  Writer.WriteSubStmt(E->getSubExpr());
  Record.push_back(E->getOpcode()); // FIXME: stable encoding
  Writer.AddSourceLocation(E->getOperatorLoc(), Record);
  Code = pch::EXPR_UNARY_OPERATOR;
}

void PCHStmtWriter::VisitSizeOfAlignOfExpr(SizeOfAlignOfExpr *E) { 
  VisitExpr(E);
  Record.push_back(E->isSizeOf());
  if (E->isArgumentType())
    Writer.AddTypeRef(E->getArgumentType(), Record);
  else {
    Record.push_back(0);
    Writer.WriteSubStmt(E->getArgumentExpr());
  }
  Writer.AddSourceLocation(E->getOperatorLoc(), Record);
  Writer.AddSourceLocation(E->getRParenLoc(), Record);
  Code = pch::EXPR_SIZEOF_ALIGN_OF;
}

void PCHStmtWriter::VisitArraySubscriptExpr(ArraySubscriptExpr *E) {
  VisitExpr(E);
  Writer.WriteSubStmt(E->getLHS());
  Writer.WriteSubStmt(E->getRHS());
  Writer.AddSourceLocation(E->getRBracketLoc(), Record);
  Code = pch::EXPR_ARRAY_SUBSCRIPT;
}

void PCHStmtWriter::VisitCallExpr(CallExpr *E) {
  VisitExpr(E);
  Record.push_back(E->getNumArgs());
  Writer.AddSourceLocation(E->getRParenLoc(), Record);
  Writer.WriteSubStmt(E->getCallee());
  for (CallExpr::arg_iterator Arg = E->arg_begin(), ArgEnd = E->arg_end();
       Arg != ArgEnd; ++Arg)
    Writer.WriteSubStmt(*Arg);
  Code = pch::EXPR_CALL;
}

void PCHStmtWriter::VisitMemberExpr(MemberExpr *E) {
  VisitExpr(E);
  Writer.WriteSubStmt(E->getBase());
  Writer.AddDeclRef(E->getMemberDecl(), Record);
  Writer.AddSourceLocation(E->getMemberLoc(), Record);
  Record.push_back(E->isArrow());
  Code = pch::EXPR_MEMBER;
}

void PCHStmtWriter::VisitCastExpr(CastExpr *E) {
  VisitExpr(E);
  Writer.WriteSubStmt(E->getSubExpr());
}

void PCHStmtWriter::VisitBinaryOperator(BinaryOperator *E) {
  VisitExpr(E);
  Writer.WriteSubStmt(E->getLHS());
  Writer.WriteSubStmt(E->getRHS());
  Record.push_back(E->getOpcode()); // FIXME: stable encoding
  Writer.AddSourceLocation(E->getOperatorLoc(), Record);
  Code = pch::EXPR_BINARY_OPERATOR;
}

void PCHStmtWriter::VisitCompoundAssignOperator(CompoundAssignOperator *E) {
  VisitBinaryOperator(E);
  Writer.AddTypeRef(E->getComputationLHSType(), Record);
  Writer.AddTypeRef(E->getComputationResultType(), Record);
  Code = pch::EXPR_COMPOUND_ASSIGN_OPERATOR;
}

void PCHStmtWriter::VisitConditionalOperator(ConditionalOperator *E) {
  VisitExpr(E);
  Writer.WriteSubStmt(E->getCond());
  Writer.WriteSubStmt(E->getLHS());
  Writer.WriteSubStmt(E->getRHS());
  Code = pch::EXPR_CONDITIONAL_OPERATOR;
}

void PCHStmtWriter::VisitImplicitCastExpr(ImplicitCastExpr *E) {
  VisitCastExpr(E);
  Record.push_back(E->isLvalueCast());
  Code = pch::EXPR_IMPLICIT_CAST;
}

void PCHStmtWriter::VisitExplicitCastExpr(ExplicitCastExpr *E) {
  VisitCastExpr(E);
  Writer.AddTypeRef(E->getTypeAsWritten(), Record);
}

void PCHStmtWriter::VisitCStyleCastExpr(CStyleCastExpr *E) {
  VisitExplicitCastExpr(E);
  Writer.AddSourceLocation(E->getLParenLoc(), Record);
  Writer.AddSourceLocation(E->getRParenLoc(), Record);
  Code = pch::EXPR_CSTYLE_CAST;
}

void PCHStmtWriter::VisitCompoundLiteralExpr(CompoundLiteralExpr *E) {
  VisitExpr(E);
  Writer.AddSourceLocation(E->getLParenLoc(), Record);
  Writer.WriteSubStmt(E->getInitializer());
  Record.push_back(E->isFileScope());
  Code = pch::EXPR_COMPOUND_LITERAL;
}

void PCHStmtWriter::VisitExtVectorElementExpr(ExtVectorElementExpr *E) {
  VisitExpr(E);
  Writer.WriteSubStmt(E->getBase());
  Writer.AddIdentifierRef(&E->getAccessor(), Record);
  Writer.AddSourceLocation(E->getAccessorLoc(), Record);
  Code = pch::EXPR_EXT_VECTOR_ELEMENT;
}

void PCHStmtWriter::VisitInitListExpr(InitListExpr *E) {
  VisitExpr(E);
  Record.push_back(E->getNumInits());
  for (unsigned I = 0, N = E->getNumInits(); I != N; ++I)
    Writer.WriteSubStmt(E->getInit(I));
  Writer.WriteSubStmt(E->getSyntacticForm());
  Writer.AddSourceLocation(E->getLBraceLoc(), Record);
  Writer.AddSourceLocation(E->getRBraceLoc(), Record);
  Writer.AddDeclRef(E->getInitializedFieldInUnion(), Record);
  Record.push_back(E->hadArrayRangeDesignator());
  Code = pch::EXPR_INIT_LIST;
}

void PCHStmtWriter::VisitDesignatedInitExpr(DesignatedInitExpr *E) {
  VisitExpr(E);
  Record.push_back(E->getNumSubExprs());
  for (unsigned I = 0, N = E->getNumSubExprs(); I != N; ++I)
    Writer.WriteSubStmt(E->getSubExpr(I));
  Writer.AddSourceLocation(E->getEqualOrColonLoc(), Record);
  Record.push_back(E->usesGNUSyntax());
  for (DesignatedInitExpr::designators_iterator D = E->designators_begin(),
                                             DEnd = E->designators_end();
       D != DEnd; ++D) {
    if (D->isFieldDesignator()) {
      if (FieldDecl *Field = D->getField()) {
        Record.push_back(pch::DESIG_FIELD_DECL);
        Writer.AddDeclRef(Field, Record);
      } else {
        Record.push_back(pch::DESIG_FIELD_NAME);
        Writer.AddIdentifierRef(D->getFieldName(), Record);
      }
      Writer.AddSourceLocation(D->getDotLoc(), Record);
      Writer.AddSourceLocation(D->getFieldLoc(), Record);
    } else if (D->isArrayDesignator()) {
      Record.push_back(pch::DESIG_ARRAY);
      Record.push_back(D->getFirstExprIndex());
      Writer.AddSourceLocation(D->getLBracketLoc(), Record);
      Writer.AddSourceLocation(D->getRBracketLoc(), Record);
    } else {
      assert(D->isArrayRangeDesignator() && "Unknown designator");
      Record.push_back(pch::DESIG_ARRAY_RANGE);
      Record.push_back(D->getFirstExprIndex());
      Writer.AddSourceLocation(D->getLBracketLoc(), Record);
      Writer.AddSourceLocation(D->getEllipsisLoc(), Record);
      Writer.AddSourceLocation(D->getRBracketLoc(), Record);
    }
  }
  Code = pch::EXPR_DESIGNATED_INIT;
}

void PCHStmtWriter::VisitImplicitValueInitExpr(ImplicitValueInitExpr *E) {
  VisitExpr(E);
  Code = pch::EXPR_IMPLICIT_VALUE_INIT;
}

void PCHStmtWriter::VisitVAArgExpr(VAArgExpr *E) {
  VisitExpr(E);
  Writer.WriteSubStmt(E->getSubExpr());
  Writer.AddSourceLocation(E->getBuiltinLoc(), Record);
  Writer.AddSourceLocation(E->getRParenLoc(), Record);
  Code = pch::EXPR_VA_ARG;
}

void PCHStmtWriter::VisitAddrLabelExpr(AddrLabelExpr *E) {
  VisitExpr(E);
  Writer.AddSourceLocation(E->getAmpAmpLoc(), Record);
  Writer.AddSourceLocation(E->getLabelLoc(), Record);
  Record.push_back(Writer.GetLabelID(E->getLabel()));
  Code = pch::EXPR_ADDR_LABEL;
}

void PCHStmtWriter::VisitStmtExpr(StmtExpr *E) {
  VisitExpr(E);
  Writer.WriteSubStmt(E->getSubStmt());
  Writer.AddSourceLocation(E->getLParenLoc(), Record);
  Writer.AddSourceLocation(E->getRParenLoc(), Record);
  Code = pch::EXPR_STMT;
}

void PCHStmtWriter::VisitTypesCompatibleExpr(TypesCompatibleExpr *E) {
  VisitExpr(E);
  Writer.AddTypeRef(E->getArgType1(), Record);
  Writer.AddTypeRef(E->getArgType2(), Record);
  Writer.AddSourceLocation(E->getBuiltinLoc(), Record);
  Writer.AddSourceLocation(E->getRParenLoc(), Record);
  Code = pch::EXPR_TYPES_COMPATIBLE;
}

void PCHStmtWriter::VisitChooseExpr(ChooseExpr *E) {
  VisitExpr(E);
  Writer.WriteSubStmt(E->getCond());
  Writer.WriteSubStmt(E->getLHS());
  Writer.WriteSubStmt(E->getRHS());
  Writer.AddSourceLocation(E->getBuiltinLoc(), Record);
  Writer.AddSourceLocation(E->getRParenLoc(), Record);
  Code = pch::EXPR_CHOOSE;
}

void PCHStmtWriter::VisitGNUNullExpr(GNUNullExpr *E) {
  VisitExpr(E);
  Writer.AddSourceLocation(E->getTokenLocation(), Record);
  Code = pch::EXPR_GNU_NULL;
}

void PCHStmtWriter::VisitShuffleVectorExpr(ShuffleVectorExpr *E) {
  VisitExpr(E);
  Record.push_back(E->getNumSubExprs());
  for (unsigned I = 0, N = E->getNumSubExprs(); I != N; ++I)
    Writer.WriteSubStmt(E->getExpr(I));
  Writer.AddSourceLocation(E->getBuiltinLoc(), Record);
  Writer.AddSourceLocation(E->getRParenLoc(), Record);
  Code = pch::EXPR_SHUFFLE_VECTOR;
}

void PCHStmtWriter::VisitBlockExpr(BlockExpr *E) {
  VisitExpr(E);
  Writer.AddDeclRef(E->getBlockDecl(), Record);
  Record.push_back(E->hasBlockDeclRefExprs());
  Code = pch::EXPR_BLOCK;
}

void PCHStmtWriter::VisitBlockDeclRefExpr(BlockDeclRefExpr *E) {
  VisitExpr(E);
  Writer.AddDeclRef(E->getDecl(), Record);
  Writer.AddSourceLocation(E->getLocation(), Record);
  Record.push_back(E->isByRef());
  Code = pch::EXPR_BLOCK_DECL_REF;
}

//===----------------------------------------------------------------------===//
// Objective-C Expressions and Statements.
//===----------------------------------------------------------------------===//

void PCHStmtWriter::VisitObjCStringLiteral(ObjCStringLiteral *E) {
  VisitExpr(E);
  Writer.WriteSubStmt(E->getString());
  Writer.AddSourceLocation(E->getAtLoc(), Record);
  Code = pch::EXPR_OBJC_STRING_LITERAL;
}

void PCHStmtWriter::VisitObjCEncodeExpr(ObjCEncodeExpr *E) { 
  VisitExpr(E);
  Writer.AddTypeRef(E->getEncodedType(), Record);
  Writer.AddSourceLocation(E->getAtLoc(), Record);
  Writer.AddSourceLocation(E->getRParenLoc(), Record);
  Code = pch::EXPR_OBJC_ENCODE;
}

void PCHStmtWriter::VisitObjCSelectorExpr(ObjCSelectorExpr *E) {
  VisitExpr(E);
  assert(0 && "Can't write a selector yet!");
  // FIXME!  Write selectors.
  //Writer.WriteSubStmt(E->getSelector());
  Writer.AddSourceLocation(E->getAtLoc(), Record);
  Writer.AddSourceLocation(E->getRParenLoc(), Record);
  Code = pch::EXPR_OBJC_SELECTOR_EXPR;
}

void PCHStmtWriter::VisitObjCProtocolExpr(ObjCProtocolExpr *E) {
  VisitExpr(E);
  Writer.AddDeclRef(E->getProtocol(), Record);
  Writer.AddSourceLocation(E->getAtLoc(), Record);
  Writer.AddSourceLocation(E->getRParenLoc(), Record);
  Code = pch::EXPR_OBJC_PROTOCOL_EXPR;
}


//===----------------------------------------------------------------------===//
// PCHWriter Implementation
//===----------------------------------------------------------------------===//

/// \brief Write the target triple (e.g., i686-apple-darwin9).
void PCHWriter::WriteTargetTriple(const TargetInfo &Target) {
  using namespace llvm;
  BitCodeAbbrev *Abbrev = new BitCodeAbbrev();
  Abbrev->Add(BitCodeAbbrevOp(pch::TARGET_TRIPLE));
  Abbrev->Add(BitCodeAbbrevOp(BitCodeAbbrevOp::Blob)); // Triple name
  unsigned TripleAbbrev = Stream.EmitAbbrev(Abbrev);

  RecordData Record;
  Record.push_back(pch::TARGET_TRIPLE);
  const char *Triple = Target.getTargetTriple();
  Stream.EmitRecordWithBlob(TripleAbbrev, Record, Triple, strlen(Triple));
}

/// \brief Write the LangOptions structure.
void PCHWriter::WriteLanguageOptions(const LangOptions &LangOpts) {
  RecordData Record;
  Record.push_back(LangOpts.Trigraphs);
  Record.push_back(LangOpts.BCPLComment);  // BCPL-style '//' comments.
  Record.push_back(LangOpts.DollarIdents);  // '$' allowed in identifiers.
  Record.push_back(LangOpts.AsmPreprocessor);  // Preprocessor in asm mode.
  Record.push_back(LangOpts.GNUMode);  // True in gnu99 mode false in c99 mode (etc)
  Record.push_back(LangOpts.ImplicitInt);  // C89 implicit 'int'.
  Record.push_back(LangOpts.Digraphs);  // C94, C99 and C++
  Record.push_back(LangOpts.HexFloats);  // C99 Hexadecimal float constants.
  Record.push_back(LangOpts.C99);  // C99 Support
  Record.push_back(LangOpts.Microsoft);  // Microsoft extensions.
  Record.push_back(LangOpts.CPlusPlus);  // C++ Support
  Record.push_back(LangOpts.CPlusPlus0x);  // C++0x Support
  Record.push_back(LangOpts.NoExtensions);  // All extensions are disabled, strict mode.
  Record.push_back(LangOpts.CXXOperatorNames);  // Treat C++ operator names as keywords.
    
  Record.push_back(LangOpts.ObjC1);  // Objective-C 1 support enabled.
  Record.push_back(LangOpts.ObjC2);  // Objective-C 2 support enabled.
  Record.push_back(LangOpts.ObjCNonFragileABI);  // Objective-C modern abi enabled
    
  Record.push_back(LangOpts.PascalStrings);  // Allow Pascal strings
  Record.push_back(LangOpts.Boolean);  // Allow bool/true/false
  Record.push_back(LangOpts.WritableStrings);  // Allow writable strings
  Record.push_back(LangOpts.LaxVectorConversions);
  Record.push_back(LangOpts.Exceptions);  // Support exception handling.

  Record.push_back(LangOpts.NeXTRuntime); // Use NeXT runtime.
  Record.push_back(LangOpts.Freestanding); // Freestanding implementation
  Record.push_back(LangOpts.NoBuiltin); // Do not use builtin functions (-fno-builtin)

  Record.push_back(LangOpts.ThreadsafeStatics); // Whether static initializers are protected
                                  // by locks.
  Record.push_back(LangOpts.Blocks); // block extension to C
  Record.push_back(LangOpts.EmitAllDecls); // Emit all declarations, even if
                                  // they are unused.
  Record.push_back(LangOpts.MathErrno); // Math functions must respect errno
                                  // (modulo the platform support).

  Record.push_back(LangOpts.OverflowChecking); // Extension to call a handler function when
                                  // signed integer arithmetic overflows.

  Record.push_back(LangOpts.HeinousExtensions); // Extensions that we really don't like and
                                  // may be ripped out at any time.

  Record.push_back(LangOpts.Optimize); // Whether __OPTIMIZE__ should be defined.
  Record.push_back(LangOpts.OptimizeSize); // Whether __OPTIMIZE_SIZE__ should be 
                                  // defined.
  Record.push_back(LangOpts.Static); // Should __STATIC__ be defined (as
                                  // opposed to __DYNAMIC__).
  Record.push_back(LangOpts.PICLevel); // The value for __PIC__, if non-zero.

  Record.push_back(LangOpts.GNUInline); // Should GNU inline semantics be
                                  // used (instead of C99 semantics).
  Record.push_back(LangOpts.NoInline); // Should __NO_INLINE__ be defined.
  Record.push_back(LangOpts.getGCMode());
  Record.push_back(LangOpts.getVisibilityMode());
  Record.push_back(LangOpts.InstantiationDepth);
  Stream.EmitRecord(pch::LANGUAGE_OPTIONS, Record);
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
void PCHWriter::WriteSourceManagerBlock(SourceManager &SourceMgr) {
  // Enter the source manager block.
  Stream.EnterSubblock(pch::SOURCE_MANAGER_BLOCK_ID, 3);

  // Abbreviations for the various kinds of source-location entries.
  int SLocFileAbbrv = -1;
  int SLocBufferAbbrv = -1;
  int SLocBufferBlobAbbrv = -1;
  int SLocInstantiationAbbrv = -1;

  // Write out the source location entry table. We skip the first
  // entry, which is always the same dummy entry.
  RecordData Record;
  for (SourceManager::sloc_entry_iterator 
         SLoc = SourceMgr.sloc_entry_begin() + 1,
         SLocEnd = SourceMgr.sloc_entry_end();
       SLoc != SLocEnd; ++SLoc) {
    // Figure out which record code to use.
    unsigned Code;
    if (SLoc->isFile()) {
      if (SLoc->getFile().getContentCache()->Entry)
        Code = pch::SM_SLOC_FILE_ENTRY;
      else
        Code = pch::SM_SLOC_BUFFER_ENTRY;
    } else
      Code = pch::SM_SLOC_INSTANTIATION_ENTRY;
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
        if (SLocFileAbbrv == -1)
          SLocFileAbbrv = CreateSLocFileAbbrev(Stream);
        Stream.EmitRecordWithBlob(SLocFileAbbrv, Record,
                             Content->Entry->getName(),
                             strlen(Content->Entry->getName()));
      } else {
        // The source location entry is a buffer. The blob associated
        // with this entry contains the contents of the buffer.
        if (SLocBufferAbbrv == -1) {
          SLocBufferAbbrv = CreateSLocBufferAbbrev(Stream);
          SLocBufferBlobAbbrv = CreateSLocBufferBlobAbbrev(Stream);
        }

        // We add one to the size so that we capture the trailing NULL
        // that is required by llvm::MemoryBuffer::getMemBuffer (on
        // the reader side).
        const llvm::MemoryBuffer *Buffer = Content->getBuffer();
        const char *Name = Buffer->getBufferIdentifier();
        Stream.EmitRecordWithBlob(SLocBufferAbbrv, Record, Name, strlen(Name) + 1);
        Record.clear();
        Record.push_back(pch::SM_SLOC_BUFFER_BLOB);
        Stream.EmitRecordWithBlob(SLocBufferBlobAbbrv, Record,
                             Buffer->getBufferStart(),
                             Buffer->getBufferSize() + 1);
      }
    } else {
      // The source location entry is an instantiation.
      const SrcMgr::InstantiationInfo &Inst = SLoc->getInstantiation();
      Record.push_back(Inst.getSpellingLoc().getRawEncoding());
      Record.push_back(Inst.getInstantiationLocStart().getRawEncoding());
      Record.push_back(Inst.getInstantiationLocEnd().getRawEncoding());

      // Compute the token length for this macro expansion.
      unsigned NextOffset = SourceMgr.getNextOffset();
      SourceManager::sloc_entry_iterator NextSLoc = SLoc;
      if (++NextSLoc != SLocEnd)
        NextOffset = NextSLoc->getOffset();
      Record.push_back(NextOffset - SLoc->getOffset() - 1);

      if (SLocInstantiationAbbrv == -1)
        SLocInstantiationAbbrv = CreateSLocInstantiationAbbrev(Stream);
      Stream.EmitRecordWithAbbrev(SLocInstantiationAbbrv, Record);
    }

    Record.clear();
  }

  // Write the line table.
  if (SourceMgr.hasLineTable()) {
    LineTableInfo &LineTable = SourceMgr.getLineTable();

    // Emit the file names
    Record.push_back(LineTable.getNumFilenames());
    for (unsigned I = 0, N = LineTable.getNumFilenames(); I != N; ++I) {
      // Emit the file name
      const char *Filename = LineTable.getFilename(I);
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
      Stream.EmitRecord(pch::SM_LINE_TABLE, Record);
    }
  }

  Stream.ExitBlock();
}

/// \brief Writes the block containing the serialized form of the
/// preprocessor.
///
void PCHWriter::WritePreprocessor(const Preprocessor &PP) {
  // Enter the preprocessor block.
  Stream.EnterSubblock(pch::PREPROCESSOR_BLOCK_ID, 2);
  
  // If the PCH file contains __DATE__ or __TIME__ emit a warning about this.
  // FIXME: use diagnostics subsystem for localization etc.
  if (PP.SawDateOrTime())
    fprintf(stderr, "warning: precompiled header used __DATE__ or __TIME__.\n");
  
  RecordData Record;

  // If the preprocessor __COUNTER__ value has been bumped, remember it.
  if (PP.getCounterValue() != 0) {
    Record.push_back(PP.getCounterValue());
    Stream.EmitRecord(pch::PP_COUNTER_VALUE, Record);
    Record.clear();
  }  
  
  // Loop over all the macro definitions that are live at the end of the file,
  // emitting each to the PP section.
  for (Preprocessor::macro_iterator I = PP.macro_begin(), E = PP.macro_end();
       I != E; ++I) {
    // FIXME: This emits macros in hash table order, we should do it in a stable
    // order so that output is reproducible.
    MacroInfo *MI = I->second;

    // Don't emit builtin macros like __LINE__ to the PCH file unless they have
    // been redefined by the header (in which case they are not isBuiltinMacro).
    if (MI->isBuiltinMacro())
      continue;

    // FIXME: Remove this identifier reference?
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
  
  Stream.ExitBlock();
}


/// \brief Write the representation of a type to the PCH stream.
void PCHWriter::WriteType(const Type *T) {
  pch::TypeID &ID = TypeIDs[T];
  if (ID == 0) // we haven't seen this type before.
    ID = NextTypeID++;
  
  // Record the offset for this type.
  if (TypeOffsets.size() == ID - pch::NUM_PREDEF_TYPE_IDS)
    TypeOffsets.push_back(Stream.GetCurrentBitNo());
  else if (TypeOffsets.size() < ID - pch::NUM_PREDEF_TYPE_IDS) {
    TypeOffsets.resize(ID + 1 - pch::NUM_PREDEF_TYPE_IDS);
    TypeOffsets[ID - pch::NUM_PREDEF_TYPE_IDS] = Stream.GetCurrentBitNo();
  }

  RecordData Record;
  
  // Emit the type's representation.
  PCHTypeWriter W(*this, Record);
  switch (T->getTypeClass()) {
    // For all of the concrete, non-dependent types, call the
    // appropriate visitor function.
#define TYPE(Class, Base) \
    case Type::Class: W.Visit##Class##Type(cast<Class##Type>(T)); break;
#define ABSTRACT_TYPE(Class, Base)
#define DEPENDENT_TYPE(Class, Base)
#include "clang/AST/TypeNodes.def"

    // For all of the dependent type nodes (which only occur in C++
    // templates), produce an error.
#define TYPE(Class, Base)
#define DEPENDENT_TYPE(Class, Base) case Type::Class:
#include "clang/AST/TypeNodes.def"
    assert(false && "Cannot serialize dependent type nodes");
    break;
  }

  // Emit the serialized record.
  Stream.EmitRecord(W.Code, Record);

  // Flush any expressions that were written as part of this type.
  FlushStmts();
}

/// \brief Write a block containing all of the types.
void PCHWriter::WriteTypesBlock(ASTContext &Context) {
  // Enter the types block.
  Stream.EnterSubblock(pch::TYPES_BLOCK_ID, 2);

  // Emit all of the types in the ASTContext
  for (std::vector<Type*>::const_iterator T = Context.getTypes().begin(),
                                       TEnd = Context.getTypes().end();
       T != TEnd; ++T) {
    // Builtin types are never serialized.
    if (isa<BuiltinType>(*T))
      continue;

    WriteType(*T);
  }

  // Exit the types block
  Stream.ExitBlock();
}

/// \brief Write the block containing all of the declaration IDs
/// lexically declared within the given DeclContext.
///
/// \returns the offset of the DECL_CONTEXT_LEXICAL block within the
/// bistream, or 0 if no block was written.
uint64_t PCHWriter::WriteDeclContextLexicalBlock(ASTContext &Context, 
                                                 DeclContext *DC) {
  if (DC->decls_empty(Context))
    return 0;

  uint64_t Offset = Stream.GetCurrentBitNo();
  RecordData Record;
  for (DeclContext::decl_iterator D = DC->decls_begin(Context),
                               DEnd = DC->decls_end(Context);
       D != DEnd; ++D)
    AddDeclRef(*D, Record);

  Stream.EmitRecord(pch::DECL_CONTEXT_LEXICAL, Record);
  return Offset;
}

/// \brief Write the block containing all of the declaration IDs
/// visible from the given DeclContext.
///
/// \returns the offset of the DECL_CONTEXT_VISIBLE block within the
/// bistream, or 0 if no block was written.
uint64_t PCHWriter::WriteDeclContextVisibleBlock(ASTContext &Context,
                                                 DeclContext *DC) {
  if (DC->getPrimaryContext() != DC)
    return 0;

  // Since there is no name lookup into functions or methods, and we
  // perform name lookup for the translation unit via the
  // IdentifierInfo chains, don't bother to build a
  // visible-declarations table for these entities.
  if (DC->isFunctionOrMethod() || DC->isTranslationUnit())
    return 0;

  // Force the DeclContext to build a its name-lookup table.
  DC->lookup(Context, DeclarationName());

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
    for(; Result.first != Result.second; ++Result.first)
      AddDeclRef(*Result.first, Record);
  }

  if (Record.size() == 0)
    return 0;

  Stream.EmitRecord(pch::DECL_CONTEXT_VISIBLE, Record);
  return Offset;
}

/// \brief Write a block containing all of the declarations.
void PCHWriter::WriteDeclsBlock(ASTContext &Context) {
  // Enter the declarations block.
  Stream.EnterSubblock(pch::DECLS_BLOCK_ID, 2);

  // Emit all of the declarations.
  RecordData Record;
  PCHDeclWriter W(*this, Context, Record);
  while (!DeclsToEmit.empty()) {
    // Pull the next declaration off the queue
    Decl *D = DeclsToEmit.front();
    DeclsToEmit.pop();

    // If this declaration is also a DeclContext, write blocks for the
    // declarations that lexically stored inside its context and those
    // declarations that are visible from its context. These blocks
    // are written before the declaration itself so that we can put
    // their offsets into the record for the declaration.
    uint64_t LexicalOffset = 0;
    uint64_t VisibleOffset = 0;
    DeclContext *DC = dyn_cast<DeclContext>(D);
    if (DC) {
      LexicalOffset = WriteDeclContextLexicalBlock(Context, DC);
      VisibleOffset = WriteDeclContextVisibleBlock(Context, DC);
    }

    // Determine the ID for this declaration
    pch::DeclID ID = DeclIDs[D];
    if (ID == 0)
      ID = DeclIDs.size();

    unsigned Index = ID - 1;

    // Record the offset for this declaration
    if (DeclOffsets.size() == Index)
      DeclOffsets.push_back(Stream.GetCurrentBitNo());
    else if (DeclOffsets.size() < Index) {
      DeclOffsets.resize(Index+1);
      DeclOffsets[Index] = Stream.GetCurrentBitNo();
    }

    // Build and emit a record for this declaration
    Record.clear();
    W.Code = (pch::DeclCode)0;
    W.Visit(D);
    if (DC) W.VisitDeclContext(DC, LexicalOffset, VisibleOffset);
    assert(W.Code && "Unhandled declaration kind while generating PCH");
    Stream.EmitRecord(W.Code, Record);

    // If the declaration had any attributes, write them now.
    if (D->hasAttrs())
      WriteAttributeRecord(D->getAttrs());

    // Flush any expressions that were written as part of this declaration.
    FlushStmts();
    
    // Note external declarations so that we can add them to a record
    // in the PCH file later.
    if (isa<FileScopeAsmDecl>(D))
      ExternalDefinitions.push_back(ID);
  }

  // Exit the declarations block
  Stream.ExitBlock();
}

namespace {
class VISIBILITY_HIDDEN PCHIdentifierTableTrait {
  PCHWriter &Writer;
  Preprocessor &PP;

public:
  typedef const IdentifierInfo* key_type;
  typedef key_type  key_type_ref;
  
  typedef pch::IdentID data_type;
  typedef data_type data_type_ref;
  
  PCHIdentifierTableTrait(PCHWriter &Writer, Preprocessor &PP) 
    : Writer(Writer), PP(PP) { }

  static unsigned ComputeHash(const IdentifierInfo* II) {
    return clang::BernsteinHash(II->getName());
  }
  
  std::pair<unsigned,unsigned> 
    EmitKeyDataLength(llvm::raw_ostream& Out, const IdentifierInfo* II, 
                      pch::IdentID ID) {
    unsigned KeyLen = strlen(II->getName()) + 1;
    clang::io::Emit16(Out, KeyLen);
    unsigned DataLen = 4 + 4; // 4 bytes for token ID, builtin, flags
                              // 4 bytes for the persistent ID
    if (II->hasMacroDefinition() && 
        !PP.getMacroInfo(const_cast<IdentifierInfo *>(II))->isBuiltinMacro())
      DataLen += 8;
    for (IdentifierResolver::iterator D = IdentifierResolver::begin(II),
                                   DEnd = IdentifierResolver::end();
         D != DEnd; ++D)
      DataLen += sizeof(pch::DeclID);
    clang::io::Emit16(Out, DataLen);
    return std::make_pair(KeyLen, DataLen);
  }
  
  void EmitKey(llvm::raw_ostream& Out, const IdentifierInfo* II, 
               unsigned KeyLen) {
    // Record the location of the key data.  This is used when generating
    // the mapping from persistent IDs to strings.
    Writer.SetIdentifierOffset(II, Out.tell());
    Out.write(II->getName(), KeyLen);
  }
  
  void EmitData(llvm::raw_ostream& Out, const IdentifierInfo* II, 
                pch::IdentID ID, unsigned) {
    uint32_t Bits = 0;
    bool hasMacroDefinition = 
      II->hasMacroDefinition() && 
      !PP.getMacroInfo(const_cast<IdentifierInfo *>(II))->isBuiltinMacro();
    Bits = Bits | (uint32_t)II->getTokenID();
    Bits = (Bits << 10) | (uint32_t)II->getObjCOrBuiltinID();
    Bits = (Bits << 1) | hasMacroDefinition;
    Bits = (Bits << 1) | II->isExtensionToken();
    Bits = (Bits << 1) | II->isPoisoned();
    Bits = (Bits << 1) | II->isCPlusPlusOperatorKeyword();
    clang::io::Emit32(Out, Bits);
    clang::io::Emit32(Out, ID);

    if (hasMacroDefinition)
      clang::io::Emit64(Out, Writer.getMacroOffset(II));

    // Emit the declaration IDs in reverse order, because the
    // IdentifierResolver provides the declarations as they would be
    // visible (e.g., the function "stat" would come before the struct
    // "stat"), but IdentifierResolver::AddDeclToIdentifierChain()
    // adds declarations to the end of the list (so we need to see the
    // struct "status" before the function "status").
    llvm::SmallVector<Decl *, 16> Decls(IdentifierResolver::begin(II), 
                                        IdentifierResolver::end());
    for (llvm::SmallVector<Decl *, 16>::reverse_iterator D = Decls.rbegin(),
                                                      DEnd = Decls.rend();
         D != DEnd; ++D)
      clang::io::Emit32(Out, Writer.getDeclID(*D));
  }
};
} // end anonymous namespace

/// \brief Write the identifier table into the PCH file.
///
/// The identifier table consists of a blob containing string data
/// (the actual identifiers themselves) and a separate "offsets" index
/// that maps identifier IDs to locations within the blob.
void PCHWriter::WriteIdentifierTable(Preprocessor &PP) {
  using namespace llvm;

  // Create and write out the blob that contains the identifier
  // strings.
  IdentifierOffsets.resize(IdentifierIDs.size());
  {
    OnDiskChainedHashTableGenerator<PCHIdentifierTableTrait> Generator;
    
    // Create the on-disk hash table representation.
    for (llvm::DenseMap<const IdentifierInfo *, pch::IdentID>::iterator
           ID = IdentifierIDs.begin(), IDEnd = IdentifierIDs.end();
         ID != IDEnd; ++ID) {
      assert(ID->first && "NULL identifier in identifier table");
      Generator.insert(ID->first, ID->second);
    }

    // Create the on-disk hash table in a buffer.
    llvm::SmallVector<char, 4096> IdentifierTable; 
    uint32_t BucketOffset;
    {
      PCHIdentifierTableTrait Trait(*this, PP);
      llvm::raw_svector_ostream Out(IdentifierTable);
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
    Stream.EmitRecordWithBlob(IDTableAbbrev, Record, 
                              &IdentifierTable.front(), 
                              IdentifierTable.size());
  }

  // Write the offsets table for identifier IDs.
  Stream.EmitRecord(pch::IDENTIFIER_OFFSET, IdentifierOffsets);
}

/// \brief Write a record containing the given attributes.
void PCHWriter::WriteAttributeRecord(const Attr *Attr) {
  RecordData Record;
  for (; Attr; Attr = Attr->getNext()) {
    Record.push_back(Attr->getKind()); // FIXME: stable encoding
    Record.push_back(Attr->isInherited());
    switch (Attr->getKind()) {
    case Attr::Alias:
      AddString(cast<AliasAttr>(Attr)->getAliasee(), Record);
      break;

    case Attr::Aligned:
      Record.push_back(cast<AlignedAttr>(Attr)->getAlignment());
      break;

    case Attr::AlwaysInline:
      break;
     
    case Attr::AnalyzerNoReturn:
      break;

    case Attr::Annotate:
      AddString(cast<AnnotateAttr>(Attr)->getAnnotation(), Record);
      break;

    case Attr::AsmLabel:
      AddString(cast<AsmLabelAttr>(Attr)->getLabel(), Record);
      break;

    case Attr::Blocks:
      Record.push_back(cast<BlocksAttr>(Attr)->getType()); // FIXME: stable
      break;

    case Attr::Cleanup:
      AddDeclRef(cast<CleanupAttr>(Attr)->getFunctionDecl(), Record);
      break;

    case Attr::Const:
      break;

    case Attr::Constructor:
      Record.push_back(cast<ConstructorAttr>(Attr)->getPriority());
      break;

    case Attr::DLLExport:
    case Attr::DLLImport:
    case Attr::Deprecated:
      break;

    case Attr::Destructor:
      Record.push_back(cast<DestructorAttr>(Attr)->getPriority());
      break;

    case Attr::FastCall:
      break;

    case Attr::Format: {
      const FormatAttr *Format = cast<FormatAttr>(Attr);
      AddString(Format->getType(), Record);
      Record.push_back(Format->getFormatIdx());
      Record.push_back(Format->getFirstArg());
      break;
    }

    case Attr::GNUInline:
    case Attr::IBOutletKind:
    case Attr::NoReturn:
    case Attr::NoThrow:
    case Attr::Nodebug:
    case Attr::Noinline:
      break;

    case Attr::NonNull: {
      const NonNullAttr *NonNull = cast<NonNullAttr>(Attr);
      Record.push_back(NonNull->size());
      Record.insert(Record.end(), NonNull->begin(), NonNull->end());
      break;
    }

    case Attr::ObjCException:
    case Attr::ObjCNSObject:
    case Attr::Overloadable:
      break;

    case Attr::Packed:
      Record.push_back(cast<PackedAttr>(Attr)->getAlignment());
      break;

    case Attr::Pure:
      break;

    case Attr::Regparm:
      Record.push_back(cast<RegparmAttr>(Attr)->getNumParams());
      break;

    case Attr::Section:
      AddString(cast<SectionAttr>(Attr)->getName(), Record);
      break;

    case Attr::StdCall:
    case Attr::TransparentUnion:
    case Attr::Unavailable:
    case Attr::Unused:
    case Attr::Used:
      break;

    case Attr::Visibility:
      // FIXME: stable encoding
      Record.push_back(cast<VisibilityAttr>(Attr)->getVisibility()); 
      break;

    case Attr::WarnUnusedResult:
    case Attr::Weak:
    case Attr::WeakImport:
      break;
    }
  }

  Stream.EmitRecord(pch::DECL_ATTR, Record);
}

void PCHWriter::AddString(const std::string &Str, RecordData &Record) {
  Record.push_back(Str.size());
  Record.insert(Record.end(), Str.begin(), Str.end());
}

/// \brief Note that the identifier II occurs at the given offset
/// within the identifier table.
void PCHWriter::SetIdentifierOffset(const IdentifierInfo *II, uint32_t Offset) {
  IdentifierOffsets[IdentifierIDs[II] - 1] = (Offset << 1) | 0x01;
}

PCHWriter::PCHWriter(llvm::BitstreamWriter &Stream) 
  : Stream(Stream), NextTypeID(pch::NUM_PREDEF_TYPE_IDS), 
    NumStatements(0), NumMacros(0) { }

void PCHWriter::WritePCH(Sema &SemaRef) {
  ASTContext &Context = SemaRef.Context;
  Preprocessor &PP = SemaRef.PP;

  // Emit the file header.
  Stream.Emit((unsigned)'C', 8);
  Stream.Emit((unsigned)'P', 8);
  Stream.Emit((unsigned)'C', 8);
  Stream.Emit((unsigned)'H', 8);

  // The translation unit is the first declaration we'll emit.
  DeclIDs[Context.getTranslationUnitDecl()] = 1;
  DeclsToEmit.push(Context.getTranslationUnitDecl());

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

  // Build a record containing all of the tentative definitions in
  // this header file. Generally, this record will be empty.
  RecordData TentativeDefinitions;
  for (llvm::DenseMap<DeclarationName, VarDecl *>::iterator 
         TD = SemaRef.TentativeDefinitions.begin(),
         TDEnd = SemaRef.TentativeDefinitions.end();
       TD != TDEnd; ++TD)
    AddDeclRef(TD->second, TentativeDefinitions);

  // Build a record containing all of the locally-scoped external
  // declarations in this header file. Generally, this record will be
  // empty.
  RecordData LocallyScopedExternalDecls;
  for (llvm::DenseMap<DeclarationName, NamedDecl *>::iterator 
         TD = SemaRef.LocallyScopedExternalDecls.begin(),
         TDEnd = SemaRef.LocallyScopedExternalDecls.end();
       TD != TDEnd; ++TD)
    AddDeclRef(TD->second, LocallyScopedExternalDecls);

  // Write the remaining PCH contents.
  RecordData Record;
  Stream.EnterSubblock(pch::PCH_BLOCK_ID, 3);
  WriteTargetTriple(Context.Target);
  WriteLanguageOptions(Context.getLangOptions());
  WriteSourceManagerBlock(Context.getSourceManager());
  WritePreprocessor(PP);
  WriteTypesBlock(Context);
  WriteDeclsBlock(Context);
  WriteIdentifierTable(PP);
  Stream.EmitRecord(pch::TYPE_OFFSET, TypeOffsets);
  Stream.EmitRecord(pch::DECL_OFFSET, DeclOffsets);

  // Write the record of special types.
  Record.clear();
  AddTypeRef(Context.getBuiltinVaListType(), Record);
  Stream.EmitRecord(pch::SPECIAL_TYPES, Record);

  // Write the record containing external, unnamed definitions.
  if (!ExternalDefinitions.empty())
    Stream.EmitRecord(pch::EXTERNAL_DEFINITIONS, ExternalDefinitions);

  // Write the record containing tentative definitions.
  if (!TentativeDefinitions.empty())
    Stream.EmitRecord(pch::TENTATIVE_DEFINITIONS, TentativeDefinitions);

  // Write the record containing locally-scoped external definitions.
  if (!LocallyScopedExternalDecls.empty())
    Stream.EmitRecord(pch::LOCALLY_SCOPED_EXTERNAL_DECLS, 
                      LocallyScopedExternalDecls);
  
  // Some simple statistics
  Record.clear();
  Record.push_back(NumStatements);
  Record.push_back(NumMacros);
  Stream.EmitRecord(pch::STATISTICS, Record);
  Stream.ExitBlock();
}

void PCHWriter::AddSourceLocation(SourceLocation Loc, RecordData &Record) {
  Record.push_back(Loc.getRawEncoding());
}

void PCHWriter::AddAPInt(const llvm::APInt &Value, RecordData &Record) {
  Record.push_back(Value.getBitWidth());
  unsigned N = Value.getNumWords();
  const uint64_t* Words = Value.getRawData();
  for (unsigned I = 0; I != N; ++I)
    Record.push_back(Words[I]);
}

void PCHWriter::AddAPSInt(const llvm::APSInt &Value, RecordData &Record) {
  Record.push_back(Value.isUnsigned());
  AddAPInt(Value, Record);
}

void PCHWriter::AddAPFloat(const llvm::APFloat &Value, RecordData &Record) {
  AddAPInt(Value.bitcastToAPInt(), Record);
}

void PCHWriter::AddIdentifierRef(const IdentifierInfo *II, RecordData &Record) {
  Record.push_back(getIdentifierRef(II));
}

pch::IdentID PCHWriter::getIdentifierRef(const IdentifierInfo *II) {
  if (II == 0)
    return 0;

  pch::IdentID &ID = IdentifierIDs[II];
  if (ID == 0)
    ID = IdentifierIDs.size();
  return ID;
}

void PCHWriter::AddTypeRef(QualType T, RecordData &Record) {
  if (T.isNull()) {
    Record.push_back(pch::PREDEF_TYPE_NULL_ID);
    return;
  }

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
    case BuiltinType::Char_S:     ID = pch::PREDEF_TYPE_CHAR_S_ID;     break;
    case BuiltinType::SChar:      ID = pch::PREDEF_TYPE_SCHAR_ID;      break;
    case BuiltinType::WChar:      ID = pch::PREDEF_TYPE_WCHAR_ID;      break;
    case BuiltinType::Short:      ID = pch::PREDEF_TYPE_SHORT_ID;      break;
    case BuiltinType::Int:        ID = pch::PREDEF_TYPE_INT_ID;        break;
    case BuiltinType::Long:       ID = pch::PREDEF_TYPE_LONG_ID;       break;
    case BuiltinType::LongLong:   ID = pch::PREDEF_TYPE_LONGLONG_ID;   break;
    case BuiltinType::Float:      ID = pch::PREDEF_TYPE_FLOAT_ID;      break;
    case BuiltinType::Double:     ID = pch::PREDEF_TYPE_DOUBLE_ID;     break;
    case BuiltinType::LongDouble: ID = pch::PREDEF_TYPE_LONGDOUBLE_ID; break;
    case BuiltinType::Overload:   ID = pch::PREDEF_TYPE_OVERLOAD_ID;   break;
    case BuiltinType::Dependent:  ID = pch::PREDEF_TYPE_DEPENDENT_ID;  break;
    }

    Record.push_back((ID << 3) | T.getCVRQualifiers());
    return;
  }

  pch::TypeID &ID = TypeIDs[T.getTypePtr()];
  if (ID == 0) // we haven't seen this type before
    ID = NextTypeID++;

  // Encode the type qualifiers in the type reference.
  Record.push_back((ID << 3) | T.getCVRQualifiers());
}

void PCHWriter::AddDeclRef(const Decl *D, RecordData &Record) {
  if (D == 0) {
    Record.push_back(0);
    return;
  }

  pch::DeclID &ID = DeclIDs[D];
  if (ID == 0) { 
    // We haven't seen this declaration before. Give it a new ID and
    // enqueue it in the list of declarations to emit.
    ID = DeclIDs.size();
    DeclsToEmit.push(const_cast<Decl *>(D));
  }

  Record.push_back(ID);
}

pch::DeclID PCHWriter::getDeclID(const Decl *D) {
  if (D == 0)
    return 0;

  assert(DeclIDs.find(D) != DeclIDs.end() && "Declaration not emitted!");
  return DeclIDs[D];
}

void PCHWriter::AddDeclarationName(DeclarationName Name, RecordData &Record) {
  Record.push_back(Name.getNameKind());
  switch (Name.getNameKind()) {
  case DeclarationName::Identifier:
    AddIdentifierRef(Name.getAsIdentifierInfo(), Record);
    break;

  case DeclarationName::ObjCZeroArgSelector:
  case DeclarationName::ObjCOneArgSelector:
  case DeclarationName::ObjCMultiArgSelector:
    assert(false && "Serialization of Objective-C selectors unavailable");
    break;

  case DeclarationName::CXXConstructorName:
  case DeclarationName::CXXDestructorName:
  case DeclarationName::CXXConversionFunctionName:
    AddTypeRef(Name.getCXXNameType(), Record);
    break;

  case DeclarationName::CXXOperatorName:
    Record.push_back(Name.getCXXOverloadedOperator());
    break;

  case DeclarationName::CXXUsingDirective:
    // No extra data to emit
    break;
  }
}

/// \brief Write the given substatement or subexpression to the
/// bitstream.
void PCHWriter::WriteSubStmt(Stmt *S) {
  RecordData Record;
  PCHStmtWriter Writer(*this, Record);
  ++NumStatements;

  if (!S) {
    Stream.EmitRecord(pch::STMT_NULL_PTR, Record);
    return;
  }
  
  Writer.Code = pch::STMT_NULL_PTR;
  Writer.Visit(S);
  assert(Writer.Code != pch::STMT_NULL_PTR && 
         "Unhandled expression writing PCH file");
  Stream.EmitRecord(Writer.Code, Record);    
}

/// \brief Flush all of the statements that have been added to the
/// queue via AddStmt().
void PCHWriter::FlushStmts() {
  RecordData Record;
  PCHStmtWriter Writer(*this, Record);

  for (unsigned I = 0, N = StmtsToEmit.size(); I != N; ++I) {
    ++NumStatements;
    Stmt *S = StmtsToEmit[I];

    if (!S) {
      Stream.EmitRecord(pch::STMT_NULL_PTR, Record);
      continue;
    }

    Writer.Code = pch::STMT_NULL_PTR;
    Writer.Visit(S);
    assert(Writer.Code != pch::STMT_NULL_PTR && 
           "Unhandled expression writing PCH file");
    Stream.EmitRecord(Writer.Code, Record);  

    assert(N == StmtsToEmit.size() && 
           "Substatement writen via AddStmt rather than WriteSubStmt!");

    // Note that we are at the end of a full expression. Any
    // expression records that follow this one are part of a different
    // expression.
    Record.clear();
    Stream.EmitRecord(pch::STMT_STOP, Record);
  }

  StmtsToEmit.clear();
  SwitchCaseIDs.clear();
}

unsigned PCHWriter::RecordSwitchCaseID(SwitchCase *S) {
  assert(SwitchCaseIDs.find(S) == SwitchCaseIDs.end() && 
         "SwitchCase recorded twice");
  unsigned NextID = SwitchCaseIDs.size();
  SwitchCaseIDs[S] = NextID;
  return NextID;
}

unsigned PCHWriter::getSwitchCaseID(SwitchCase *S) {
  assert(SwitchCaseIDs.find(S) != SwitchCaseIDs.end() && 
         "SwitchCase hasn't been seen yet");
  return SwitchCaseIDs[S];
}

/// \brief Retrieve the ID for the given label statement, which may
/// or may not have been emitted yet.
unsigned PCHWriter::GetLabelID(LabelStmt *S) {
  std::map<LabelStmt *, unsigned>::iterator Pos = LabelIDs.find(S);
  if (Pos != LabelIDs.end())
    return Pos->second;

  unsigned NextID = LabelIDs.size();
  LabelIDs[S] = NextID;
  return NextID;
}
