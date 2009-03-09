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
#include "llvm/Support/Compiler.h"

using namespace clang;

//===----------------------------------------------------------------------===/
// Template Instantiation for Types
//===----------------------------------------------------------------------===/
namespace {
  class VISIBILITY_HIDDEN TemplateTypeInstantiator {
    Sema &SemaRef;
    const TemplateArgument *TemplateArgs;
    unsigned NumTemplateArgs;
    SourceLocation Loc;
    DeclarationName Entity;

  public:
    TemplateTypeInstantiator(Sema &SemaRef, 
                             const TemplateArgument *TemplateArgs,
                             unsigned NumTemplateArgs,
                             SourceLocation Loc,
                             DeclarationName Entity) 
      : SemaRef(SemaRef), TemplateArgs(TemplateArgs), 
        NumTemplateArgs(NumTemplateArgs), Loc(Loc), Entity(Entity) { }

    QualType operator()(QualType T) const { return Instantiate(T); }
    
    QualType Instantiate(QualType T) const;

    // Declare instantiate functions for each type.
#define TYPE(Class, Base)                                       \
    QualType Instantiate##Class##Type(const Class##Type *T,     \
                                      unsigned Quals) const;
#define ABSTRACT_TYPE(Class, Base)
#include "clang/AST/TypeNodes.def"
  };
}

QualType 
TemplateTypeInstantiator::InstantiateExtQualType(const ExtQualType *T,
                                                 unsigned Quals) const {
  // FIXME: Implement this
  assert(false && "Cannot instantiate ExtQualType yet");
  return QualType();
}

QualType 
TemplateTypeInstantiator::InstantiateBuiltinType(const BuiltinType *T,
                                                 unsigned Quals) const {
  assert(false && "Builtin types are not dependent and cannot be instantiated");
  return QualType(T, Quals);
}

QualType 
TemplateTypeInstantiator::
InstantiateFixedWidthIntType(const FixedWidthIntType *T, unsigned Quals) const {
  // FIXME: Implement this
  assert(false && "Cannot instantiate FixedWidthIntType yet");
  return QualType();
}

QualType 
TemplateTypeInstantiator::InstantiateComplexType(const ComplexType *T,
                                                 unsigned Quals) const {
  // FIXME: Implement this
  assert(false && "Cannot instantiate ComplexType yet");
  return QualType();
}

QualType 
TemplateTypeInstantiator::InstantiatePointerType(const PointerType *T,
                                                 unsigned Quals) const {
  QualType PointeeType = Instantiate(T->getPointeeType());
  if (PointeeType.isNull())
    return QualType();

  return SemaRef.BuildPointerType(PointeeType, Quals, Loc, Entity);
}

QualType 
TemplateTypeInstantiator::InstantiateBlockPointerType(const BlockPointerType *T,
                                                      unsigned Quals) const {
  // FIXME: Implement this
  assert(false && "Cannot instantiate BlockPointerType yet");
  return QualType();
}

QualType 
TemplateTypeInstantiator::InstantiateReferenceType(const ReferenceType *T, 
                                                   unsigned Quals) const {
  QualType ReferentType = Instantiate(T->getPointeeType());
  if (ReferentType.isNull())
    return QualType();

  return SemaRef.BuildReferenceType(ReferentType, Quals, Loc, Entity);
}

QualType 
TemplateTypeInstantiator::
InstantiateMemberPointerType(const MemberPointerType *T,
                             unsigned Quals) const {
  // FIXME: Implement this
  assert(false && "Cannot instantiate MemberPointerType yet");
  return QualType();
}

QualType 
TemplateTypeInstantiator::
InstantiateConstantArrayType(const ConstantArrayType *T, 
                             unsigned Quals) const {
  QualType ElementType = Instantiate(T->getElementType());
  if (ElementType.isNull())
    return ElementType;
  
  // Build a temporary integer literal to specify the size for
  // BuildArrayType. Since we have already checked the size as part of
  // creating the dependent array type in the first place, we know
  // there aren't any errors.
  // FIXME: Is IntTy big enough? Maybe not, but LongLongTy causes
  // problems that I have yet to investigate.
  IntegerLiteral ArraySize(T->getSize(), SemaRef.Context.IntTy, Loc);
  return SemaRef.BuildArrayType(ElementType, T->getSizeModifier(), 
                                &ArraySize, T->getIndexTypeQualifier(), 
                                Loc, Entity);
}

QualType 
TemplateTypeInstantiator::
InstantiateIncompleteArrayType(const IncompleteArrayType *T,
                               unsigned Quals) const {
  QualType ElementType = Instantiate(T->getElementType());
  if (ElementType.isNull())
    return ElementType;
  
  return SemaRef.BuildArrayType(ElementType, T->getSizeModifier(), 
                                0, T->getIndexTypeQualifier(), 
                                Loc, Entity);
}

QualType
TemplateTypeInstantiator::
InstantiateVariableArrayType(const VariableArrayType *T,
                             unsigned Quals) const {
  // FIXME: Implement this
  assert(false && "Cannot instantiate VariableArrayType yet");
  return QualType();
}

QualType 
TemplateTypeInstantiator::
InstantiateDependentSizedArrayType(const DependentSizedArrayType *T,
                                   unsigned Quals) const {
  // FIXME: Implement this
  assert(false && "Cannot instantiate DependentSizedArrayType yet");
  return QualType();
}

QualType 
TemplateTypeInstantiator::InstantiateVectorType(const VectorType *T,
                                             unsigned Quals) const {
  // FIXME: Implement this
  assert(false && "Cannot instantiate VectorType yet");
  return QualType();
}

QualType 
TemplateTypeInstantiator::InstantiateExtVectorType(const ExtVectorType *T,
                                                   unsigned Quals) const {
  // FIXME: Implement this
  assert(false && "Cannot instantiate ExtVectorType yet");
  return QualType();
}

QualType 
TemplateTypeInstantiator::
InstantiateFunctionProtoType(const FunctionProtoType *T,
                             unsigned Quals) const {
  QualType ResultType = Instantiate(T->getResultType());
  if (ResultType.isNull())
    return ResultType;

  llvm::SmallVector<QualType, 16> ParamTypes;
  for (FunctionProtoType::arg_type_iterator Param = T->arg_type_begin(),
                                         ParamEnd = T->arg_type_end(); 
       Param != ParamEnd; ++Param) {
    QualType P = Instantiate(*Param);
    if (P.isNull())
      return P;

    ParamTypes.push_back(P);
  }

  return SemaRef.BuildFunctionType(ResultType, &ParamTypes[0], 
                                   ParamTypes.size(),
                                   T->isVariadic(), T->getTypeQuals(),
                                   Loc, Entity);
}

QualType 
TemplateTypeInstantiator::
InstantiateFunctionNoProtoType(const FunctionNoProtoType *T,
                               unsigned Quals) const {
  assert(false && "Functions without prototypes cannot be dependent.");
  return QualType();
}

QualType 
TemplateTypeInstantiator::InstantiateTypedefType(const TypedefType *T,
                                                 unsigned Quals) const {
  // FIXME: Implement this
  assert(false && "Cannot instantiate TypedefType yet");
  return QualType();
}

QualType 
TemplateTypeInstantiator::InstantiateTypeOfExprType(const TypeOfExprType *T,
                                                    unsigned Quals) const {
  // FIXME: Implement this
  assert(false && "Cannot instantiate TypeOfExprType yet");
  return QualType();
}

QualType 
TemplateTypeInstantiator::InstantiateTypeOfType(const TypeOfType *T,
                                                unsigned Quals) const {
  // FIXME: Implement this
  assert(false && "Cannot instantiate TypeOfType yet");
  return QualType();
}

QualType 
TemplateTypeInstantiator::InstantiateRecordType(const RecordType *T,
                                                unsigned Quals) const {
  // FIXME: Implement this
  assert(false && "Cannot instantiate RecordType yet");
  return QualType();
}

QualType 
TemplateTypeInstantiator::InstantiateEnumType(const EnumType *T,
                                              unsigned Quals) const {
  // FIXME: Implement this
  assert(false && "Cannot instantiate EnumType yet");
  return QualType();
}

QualType 
TemplateTypeInstantiator::
InstantiateTemplateTypeParmType(const TemplateTypeParmType *T,
                                unsigned Quals) const {
  if (T->getDepth() == 0) {
    // Replace the template type parameter with its corresponding
    // template argument.
    assert(T->getIndex() < NumTemplateArgs && "Wrong # of template args");
    assert(TemplateArgs[T->getIndex()].getKind() == TemplateArgument::Type &&
           "Template argument kind mismatch");
    QualType Result = TemplateArgs[T->getIndex()].getAsType();
    if (Result.isNull() || !Quals) 
      return Result;

    // C++ [dcl.ref]p1:
    //   [...] Cv-qualified references are ill-formed except when
    //   the cv-qualifiers are introduced through the use of a
    //   typedef (7.1.3) or of a template type argument (14.3), in
    //   which case the cv-qualifiers are ignored.
    if (Quals && Result->isReferenceType())
      Quals = 0;

    return QualType(Result.getTypePtr(), Quals | Result.getCVRQualifiers());
  } 

  // The template type parameter comes from an inner template (e.g.,
  // the template parameter list of a member template inside the
  // template we are instantiating). Create a new template type
  // parameter with the template "level" reduced by one.
  return SemaRef.Context.getTemplateTypeParmType(T->getDepth() - 1,
                                                 T->getIndex(),
                                                 T->getName())
    .getQualifiedType(Quals);
}

QualType 
TemplateTypeInstantiator::
InstantiateClassTemplateSpecializationType(
                                  const ClassTemplateSpecializationType *T,
                                  unsigned Quals) const {
  // FIXME: Implement this
  assert(false && "Cannot instantiate ClassTemplateSpecializationType yet");
  return QualType();
}

QualType 
TemplateTypeInstantiator::
InstantiateObjCInterfaceType(const ObjCInterfaceType *T,
                             unsigned Quals) const {
  assert(false && "Objective-C types cannot be dependent");
  return QualType();
}

QualType 
TemplateTypeInstantiator::
InstantiateObjCQualifiedInterfaceType(const ObjCQualifiedInterfaceType *T,
                                      unsigned Quals) const {
  assert(false && "Objective-C types cannot be dependent");
  return QualType();
}

QualType 
TemplateTypeInstantiator::
InstantiateObjCQualifiedIdType(const ObjCQualifiedIdType *T,
                               unsigned Quals) const {
  assert(false && "Objective-C types cannot be dependent");
  return QualType();
}

QualType 
TemplateTypeInstantiator::
InstantiateObjCQualifiedClassType(const ObjCQualifiedClassType *T,
                                  unsigned Quals) const {
  assert(false && "Objective-C types cannot be dependent");
  return QualType();
}

/// \brief The actual implementation of Sema::InstantiateType().
QualType TemplateTypeInstantiator::Instantiate(QualType T) const {
  // If T is not a dependent type, there is nothing to do.
  if (!T->isDependentType())
    return T;

  switch (T->getTypeClass()) {
#define TYPE(Class, Base)                                               \
  case Type::Class:                                                     \
    return Instantiate##Class##Type(cast<Class##Type>(T.getTypePtr()),  \
                                    T.getCVRQualifiers());
#define ABSTRACT_TYPE(Class, Base)
#include "clang/AST/TypeNodes.def"
  }
  
  assert(false && "Not all types have been decoded for instantiation");
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

  TemplateTypeInstantiator Instantiator(*this, TemplateArgs, NumTemplateArgs,
                                        Loc, Entity);
  return Instantiator(T);
}

/// \brief Instantiate the base class specifiers of the given class
/// template specialization.
///
/// Produces a diagnostic and returns true on error, returns false and
/// attaches the instantiated base classes to the class template
/// specialization if successful.
bool 
Sema::InstantiateBaseSpecifiers(
                           ClassTemplateSpecializationDecl *ClassTemplateSpec,
                           ClassTemplateDecl *ClassTemplate) {
  bool Invalid = false;
  llvm::SmallVector<CXXBaseSpecifier*, 8> InstantiatedBases;
  for (ClassTemplateSpecializationDecl::base_class_iterator
         Base = ClassTemplate->getTemplatedDecl()->bases_begin(),
         BaseEnd = ClassTemplate->getTemplatedDecl()->bases_end();
       Base != BaseEnd && !Invalid; ++Base) {
    if (!Base->getType()->isDependentType()) {
      // FIXME: Allocate via ASTContext
      InstantiatedBases.push_back(new CXXBaseSpecifier(*Base));
      continue;
    }

    QualType BaseType = InstantiateType(Base->getType(), 
                                        ClassTemplateSpec->getTemplateArgs(),
                                        ClassTemplateSpec->getNumTemplateArgs(),
                                        Base->getSourceRange().getBegin(),
                                        DeclarationName());
    if (BaseType.isNull()) {
      Invalid = true;
      continue;
    }

    if (CXXBaseSpecifier *InstantiatedBase
          = CheckBaseSpecifier(ClassTemplateSpec,
                               Base->getSourceRange(),
                               Base->isVirtual(),
                               Base->getAccessSpecifierAsWritten(),
                               BaseType,
                               /*FIXME: Not totally accurate */
                               Base->getSourceRange().getBegin()))
      InstantiatedBases.push_back(InstantiatedBase);
    else
      Invalid = true;
  }

  if (AttachBaseSpecifiers(ClassTemplateSpec, &InstantiatedBases[0],
                           InstantiatedBases.size()))
    Invalid = true;

  return Invalid;
}

bool 
Sema::InstantiateClassTemplateSpecialization(
                           ClassTemplateSpecializationDecl *ClassTemplateSpec,
                           bool ExplicitInstantiation) {
  // Perform the actual instantiation on the canonical declaration.
  ClassTemplateSpec = cast<ClassTemplateSpecializationDecl>(
                               Context.getCanonicalDecl(ClassTemplateSpec));

  // We can only instantiate something that hasn't already been
  // instantiated or specialized. Fail without any diagnostics: our
  // caller will provide an error message.
  if (ClassTemplateSpec->getSpecializationKind() != TSK_Undeclared)
    return true;

  // FIXME: Push this class template instantiation onto the
  // instantiation stack, checking for recursion that exceeds a
  // certain depth.

  // FIXME: Perform class template partial specialization to select
  // the best template.
  ClassTemplateDecl *Template = ClassTemplateSpec->getSpecializedTemplate();

  if (!Template->getTemplatedDecl()->getDefinition(Context)) {
    Diag(ClassTemplateSpec->getLocation(), 
         diag::err_template_implicit_instantiate_undefined)
      << Context.getTypeDeclType(ClassTemplateSpec);
    Diag(Template->getTemplatedDecl()->getLocation(), 
         diag::note_template_decl_here);
    return true;
  }

  // Note that this is an instantiation.  
  ClassTemplateSpec->setSpecializationKind(
                        ExplicitInstantiation? TSK_ExplicitInstantiation 
                                             : TSK_ImplicitInstantiation);


  bool Invalid = false;
  
  // Enter the scope of this instantiation. We don't use
  // PushDeclContext because we don't have a scope.
  DeclContext *PreviousContext = CurContext;
  CurContext = ClassTemplateSpec;

  // Start the definition of this instantiation.
  ClassTemplateSpec->startDefinition();

  // FIXME: Create the injected-class-name for the
  // instantiation. Should this be a typedef or something like it?

  // Instantiate the base class specifiers.
  if (InstantiateBaseSpecifiers(ClassTemplateSpec, Template))
    Invalid = true;

  // FIXME: Instantiate all of the members.
  
  // Add any implicitly-declared members that we might need.
  AddImplicitlyDeclaredMembersToClass(ClassTemplateSpec);

  // Finish the definition of this instantiation.
  // FIXME: ActOnFields does more checking, which we'll eventually need.
  ClassTemplateSpec->completeDefinition(Context);

  // Exit the scope of this instantiation.
  CurContext = PreviousContext;

  return Invalid;
}
