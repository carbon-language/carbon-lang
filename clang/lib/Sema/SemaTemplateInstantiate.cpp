//===------- SemaTemplateInstantiate.cpp - C++ Template Instantiation ------===/
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//===----------------------------------------------------------------------===/
//
//  This file implements C++ template instantiation.
//
//===----------------------------------------------------------------------===/

#include "Sema.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/Parse/DeclSpec.h"
#include "clang/Basic/LangOptions.h"

using namespace clang;

//===----------------------------------------------------------------------===/
// Template Instantiation for Types
//===----------------------------------------------------------------------===/

static QualType PerformTypeInstantiation(Sema &SemaRef, 
                                         const ExtQualType *T, 
                                         unsigned CVR,
                                         const TemplateArgument *TemplateArgs,
                                         unsigned NumTemplateArgs,
                                         SourceLocation Loc,
                                         DeclarationName Entity) {
  // FIXME: Implement this
  assert(false && "Cannot instantiate ExtQualType yet");
  return QualType();
}

static QualType PerformTypeInstantiation(Sema &SemaRef, 
                                         const BuiltinType *T, 
                                         unsigned CVR,
                                         const TemplateArgument *TemplateArgs,
                                         unsigned NumTemplateArgs,
                                         SourceLocation Loc,
                                         DeclarationName Entity) {
  assert(false && "BuiltinType is never dependent and cannot be instantiated");
  return QualType(T, CVR);
}

static QualType PerformTypeInstantiation(Sema &SemaRef, 
                                         const FixedWidthIntType *T, 
                                         unsigned CVR,
                                         const TemplateArgument *TemplateArgs,
                                         unsigned NumTemplateArgs,
                                         SourceLocation Loc,
                                         DeclarationName Entity) {
  // FIXME: Implement this
  assert(false && "Cannot instantiate FixedWidthIntType yet");
  return QualType();
}

static QualType PerformTypeInstantiation(Sema &SemaRef, 
                                         const ComplexType *T, 
                                         unsigned CVR,
                                         const TemplateArgument *TemplateArgs,
                                         unsigned NumTemplateArgs,
                                         SourceLocation Loc,
                                         DeclarationName Entity) {
  // FIXME: Implement this
  assert(false && "Cannot instantiate ComplexType yet");
  return QualType();
}

static QualType PerformTypeInstantiation(Sema &SemaRef, 
                                         const PointerType *T, 
                                         unsigned CVR,
                                         const TemplateArgument *TemplateArgs,
                                         unsigned NumTemplateArgs,
                                         SourceLocation Loc,
                                         DeclarationName Entity) {
  // FIXME: Implement this
  assert(false && "Cannot instantiate PointerType yet");
  return QualType();
}

static QualType PerformTypeInstantiation(Sema &SemaRef, 
                                         const BlockPointerType *T,
                                         unsigned CVR,
                                         const TemplateArgument *TemplateArgs,
                                         unsigned NumTemplateArgs,
                                         SourceLocation Loc,
                                         DeclarationName Entity) {
  // FIXME: Implement this
  assert(false && "Cannot instantiate BlockPointerType yet");
  return QualType();
}

static QualType PerformTypeInstantiation(Sema &SemaRef, 
                                         const ReferenceType *T, 
                                         unsigned CVR,
                                         const TemplateArgument *TemplateArgs,
                                         unsigned NumTemplateArgs,
                                         SourceLocation Loc,
                                         DeclarationName Entity) {
  // FIXME: Implement this
  assert(false && "Cannot instantiate ReferenceType yet");
  return QualType();
}

static QualType PerformTypeInstantiation(Sema &SemaRef, 
                                         const MemberPointerType *T, 
                                         unsigned CVR,
                                         const TemplateArgument *TemplateArgs,
                                         unsigned NumTemplateArgs,
                                         SourceLocation Loc,
                                         DeclarationName Entity) {
  // FIXME: Implement this
  assert(false && "Cannot instantiate MemberPointerType yet");
  return QualType();
}

static QualType PerformTypeInstantiation(Sema &SemaRef, 
                                         const ConstantArrayType *T, 
                                         unsigned CVR,
                                         const TemplateArgument *TemplateArgs,
                                         unsigned NumTemplateArgs,
                                         SourceLocation Loc,
                                         DeclarationName Entity) {
  // FIXME: Implement this
  assert(false && "Cannot instantiate ConstantArrayType yet");
  return QualType();
}

static QualType PerformTypeInstantiation(Sema &SemaRef, 
                                         const IncompleteArrayType *T, 
                                         unsigned CVR,
                                         const TemplateArgument *TemplateArgs,
                                         unsigned NumTemplateArgs,
                                         SourceLocation Loc,
                                         DeclarationName Entity) {
  // FIXME: Implement this
  assert(false && "Cannot instantiate IncompleteArrayType yet");
  return QualType();
}

static QualType PerformTypeInstantiation(Sema &SemaRef, 
                                         const VariableArrayType *T, 
                                         unsigned CVR,
                                         const TemplateArgument *TemplateArgs,
                                         unsigned NumTemplateArgs,
                                         SourceLocation Loc,
                                         DeclarationName Entity) {
  // FIXME: Implement this
  assert(false && "Cannot instantiate VariableArrayType yet");
  return QualType();
}

static QualType PerformTypeInstantiation(Sema &SemaRef, 
                                         const DependentSizedArrayType *T, 
                                         unsigned CVR,
                                         const TemplateArgument *TemplateArgs,
                                         unsigned NumTemplateArgs,
                                         SourceLocation Loc,
                                         DeclarationName Entity) {
  // FIXME: Implement this
  assert(false && "Cannot instantiate DependentSizedArrayType yet");
  return QualType();
}

static QualType PerformTypeInstantiation(Sema &SemaRef, 
                                         const VectorType *T, 
                                         unsigned CVR,
                                         const TemplateArgument *TemplateArgs,
                                         unsigned NumTemplateArgs,
                                         SourceLocation Loc,
                                         DeclarationName Entity) {
  // FIXME: Implement this
  assert(false && "Cannot instantiate VectorType yet");
  return QualType();
}

static QualType PerformTypeInstantiation(Sema &SemaRef, 
                                         const ExtVectorType *T, 
                                         unsigned CVR,
                                         const TemplateArgument *TemplateArgs,
                                         unsigned NumTemplateArgs,
                                         SourceLocation Loc,
                                         DeclarationName Entity) {
  // FIXME: Implement this
  assert(false && "Cannot instantiate ExtVectorType yet");
  return QualType();
}

static QualType PerformTypeInstantiation(Sema &SemaRef, 
                                         const FunctionProtoType *T, 
                                         unsigned CVR,
                                         const TemplateArgument *TemplateArgs,
                                         unsigned NumTemplateArgs,
                                         SourceLocation Loc,
                                         DeclarationName Entity) {
  // FIXME: Implement this
  assert(false && "Cannot instantiate FunctionProtoType yet");
  return QualType();
}

static QualType PerformTypeInstantiation(Sema &SemaRef, 
                                         const FunctionNoProtoType *T, 
                                         unsigned CVR,
                                         const TemplateArgument *TemplateArgs,
                                         unsigned NumTemplateArgs,
                                         SourceLocation Loc,
                                         DeclarationName Entity) {
  // FIXME: Implement this
  assert(false && "Cannot instantiate FunctionNoProtoType yet");
  return QualType();
}

static QualType PerformTypeInstantiation(Sema &SemaRef, 
                                         const TypedefType *T, 
                                         unsigned CVR,
                                         const TemplateArgument *TemplateArgs,
                                         unsigned NumTemplateArgs,
                                         SourceLocation Loc,
                                         DeclarationName Entity) {
  // FIXME: Implement this
  assert(false && "Cannot instantiate TypedefType yet");
  return QualType();
}

static QualType PerformTypeInstantiation(Sema &SemaRef, 
                                         const TypeOfExprType *T, 
                                         unsigned CVR,
                                         const TemplateArgument *TemplateArgs,
                                         unsigned NumTemplateArgs,
                                         SourceLocation Loc,
                                         DeclarationName Entity) {
  // FIXME: Implement this
  assert(false && "Cannot instantiate TypeOfExprType yet");
  return QualType();
}

static QualType PerformTypeInstantiation(Sema &SemaRef, 
                                         const TypeOfType *T, 
                                         unsigned CVR,
                                         const TemplateArgument *TemplateArgs,
                                         unsigned NumTemplateArgs,
                                         SourceLocation Loc,
                                         DeclarationName Entity) {
  // FIXME: Implement this
  assert(false && "Cannot instantiate TypeOfType yet");
  return QualType();
}

static QualType PerformTypeInstantiation(Sema &SemaRef, 
                                         const RecordType *T, 
                                         unsigned CVR,
                                         const TemplateArgument *TemplateArgs,
                                         unsigned NumTemplateArgs,
                                         SourceLocation Loc,
                                         DeclarationName Entity) {
  // FIXME: Implement this
  assert(false && "Cannot instantiate RecordType yet");
  return QualType();
}

static QualType PerformTypeInstantiation(Sema &SemaRef, 
                                         const CXXRecordType *T, 
                                         unsigned CVR,
                                         const TemplateArgument *TemplateArgs,
                                         unsigned NumTemplateArgs,
                                         SourceLocation Loc,
                                         DeclarationName Entity) {
  // FIXME: Implement this
  assert(false && "Cannot instantiate CXXRecordType yet");
  return QualType();
}

static QualType PerformTypeInstantiation(Sema &SemaRef, 
                                         const EnumType *T, 
                                         unsigned CVR,
                                         const TemplateArgument *TemplateArgs,
                                         unsigned NumTemplateArgs,
                                         SourceLocation Loc,
                                         DeclarationName Entity) {
  // FIXME: Implement this
  assert(false && "Cannot instantiate EnumType yet");
  return QualType();
}

static QualType PerformTypeInstantiation(Sema &SemaRef, 
                                         const TemplateTypeParmType *T, 
                                         unsigned CVR,
                                         const TemplateArgument *TemplateArgs,
                                         unsigned NumTemplateArgs,
                                         SourceLocation Loc,
                                         DeclarationName Entity) {
  if (T->getDepth() == 0) {
    // Replace the template type parameter with its corresponding
    // template argument.
    assert(T->getIndex() < NumTemplateArgs && "Wrong # of template args");
    assert(TemplateArgs[T->getIndex()].getKind() == TemplateArgument::Type &&
           "Template argument kind mismatch");
    QualType Result = TemplateArgs[T->getIndex()].getAsType();
    if (Result.isNull() || !CVR) 
      return Result;

    // C++ [dcl.ref]p1:
    //   [...] Cv-qualified references are ill-formed except when
    //   the cv-qualifiers are introduced through the use of a
    //   typedef (7.1.3) or of a template type argument (14.3), in
    //   which case the cv-qualifiers are ignored.
    if (CVR && Result->isReferenceType())
      CVR = 0;

    return QualType(Result.getTypePtr(), CVR | Result.getCVRQualifiers());
  } 

  // The template type parameter comes from an inner template (e.g.,
  // the template parameter list of a member template inside the
  // template we are instantiating). Create a new template type
  // parameter with the template "level" reduced by one.
  return SemaRef.Context.getTemplateTypeParmType(T->getDepth() - 1,
                                                 T->getIndex(),
                                                 T->getName());
}

static QualType PerformTypeInstantiation(Sema &SemaRef, 
                                     const ClassTemplateSpecializationType *T, 
                                         unsigned CVR,
                                         const TemplateArgument *TemplateArgs,
                                         unsigned NumTemplateArgs,
                                         SourceLocation Loc,
                                         DeclarationName Entity) {
  // FIXME: Implement this
  assert(false && "Cannot instantiate ClassTemplateSpecializationType yet");
  return QualType();
}

static QualType PerformTypeInstantiation(Sema &SemaRef, 
                                         const ObjCInterfaceType *T, 
                                         unsigned CVR,
                                         const TemplateArgument *TemplateArgs,
                                         unsigned NumTemplateArgs,
                                         SourceLocation Loc,
                                         DeclarationName Entity) {
  // FIXME: Implement this
  assert(false && "Cannot instantiate ObjCInterfaceType yet");
  return QualType();
}

static QualType PerformTypeInstantiation(Sema &SemaRef, 
                                         const ObjCQualifiedInterfaceType *T, 
                                         unsigned CVR,
                                         const TemplateArgument *TemplateArgs,
                                         unsigned NumTemplateArgs,
                                         SourceLocation Loc,
                                         DeclarationName Entity) {
  // FIXME: Implement this
  assert(false && "Cannot instantiate ObjCQualifiedInterfaceType yet");
  return QualType();
}

static QualType PerformTypeInstantiation(Sema &SemaRef, 
                                         const ObjCQualifiedIdType *T, 
                                         unsigned CVR,
                                         const TemplateArgument *TemplateArgs,
                                         unsigned NumTemplateArgs,
                                         SourceLocation Loc,
                                         DeclarationName Entity) {
  // FIXME: Implement this
  assert(false && "Cannot instantiate ObjCQualifiedIdType yet");
  return QualType();
}

static QualType PerformTypeInstantiation(Sema &SemaRef, 
                                         const ObjCQualifiedClassType *T, 
                                         unsigned CVR,
                                         const TemplateArgument *TemplateArgs,
                                         unsigned NumTemplateArgs,
                                         SourceLocation Loc,
                                         DeclarationName Entity) {
  // FIXME: Implement this
  assert(false && "Cannot instantiate ObjCQualifiedClassType yet");
  return QualType();
}


/// \brief Instantiate the type T with a given set of template arguments.
///
/// This routine substitutes the given template arguments into the
/// type T and produces the instantiated type.
///
/// \param T the type into which the template arguments will be
/// substituted. If this type is not dependent, it will be returned
/// immediately.
///
/// \param TemplateArgs the template arguments that will be
/// substituted for the top-level template parameters within T.
///
/// \param NumTemplateArgs the number of template arguments provided
/// by TemplateArgs.
///
/// \param Loc the location in the source code where this substitution
/// is being performed. It will typically be the location of the
/// declarator (if we're instantiating the type of some declaration)
/// or the location of the type in the source code (if, e.g., we're
/// instantiating the type of a cast expression).
///
/// \param Entity the name of the entity associated with a declaration
/// being instantiated (if any). May be empty to indicate that there
/// is no such entity (if, e.g., this is a type that occurs as part of
/// a cast expression) or that the entity has no name (e.g., an
/// unnamed function parameter).
///
/// \returns If the instantiation succeeds, the instantiated
/// type. Otherwise, produces diagnostics and returns a NULL type.
QualType Sema::InstantiateType(QualType T, 
                               const TemplateArgument *TemplateArgs,
                               unsigned NumTemplateArgs,
                               SourceLocation Loc, DeclarationName Entity) {
  // If T is not a dependent type, there is nothing to do.
  if (!T->isDependentType())
    return T;

  switch (T->getTypeClass()) {
#define TYPE(Class, Base)                                               \
  case Type::Class:                                                     \
    return PerformTypeInstantiation(*this,                              \
                                    cast<Class##Type>(T.getTypePtr()),  \
                                    T.getCVRQualifiers(), TemplateArgs, \
                                    NumTemplateArgs, Loc, Entity);
#define ABSTRACT_TYPE(Class, Base)
#include "clang/AST/TypeNodes.def"
  }
  
  assert(false && "Not all types hav been decided for template instantiation");
  return QualType();
}
