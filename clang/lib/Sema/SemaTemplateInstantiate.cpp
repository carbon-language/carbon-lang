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
#include "clang/AST/DeclTemplate.h"
#include "clang/Parse/DeclSpec.h"
#include "clang/Basic/LangOptions.h"
#include "llvm/Support/Compiler.h"

using namespace clang;

//===----------------------------------------------------------------------===/
// Template Instantiation Support
//===----------------------------------------------------------------------===/

Sema::InstantiatingTemplate::
InstantiatingTemplate(Sema &SemaRef, SourceLocation PointOfInstantiation,
                      CXXRecordDecl *Entity,
                      SourceRange InstantiationRange)
  :  SemaRef(SemaRef) {

  Invalid = CheckInstantiationDepth(PointOfInstantiation,
                                    InstantiationRange);
  if (!Invalid) {
    ActiveTemplateInstantiation Inst;
    Inst.Kind = ActiveTemplateInstantiation::TemplateInstantiation;
    Inst.PointOfInstantiation = PointOfInstantiation;
    Inst.Entity = reinterpret_cast<uintptr_t>(Entity);
    Inst.TemplateArgs = 0;
    Inst.NumTemplateArgs = 0;
    Inst.InstantiationRange = InstantiationRange;
    SemaRef.ActiveTemplateInstantiations.push_back(Inst);
    Invalid = false;
  }
}

Sema::InstantiatingTemplate::InstantiatingTemplate(Sema &SemaRef, 
                                         SourceLocation PointOfInstantiation,
                                         TemplateDecl *Template,
                                         const TemplateArgument *TemplateArgs,
                                         unsigned NumTemplateArgs,
                                         SourceRange InstantiationRange)
  : SemaRef(SemaRef) {

  Invalid = CheckInstantiationDepth(PointOfInstantiation,
                                    InstantiationRange);
  if (!Invalid) {
    ActiveTemplateInstantiation Inst;
    Inst.Kind 
      = ActiveTemplateInstantiation::DefaultTemplateArgumentInstantiation;
    Inst.PointOfInstantiation = PointOfInstantiation;
    Inst.Entity = reinterpret_cast<uintptr_t>(Template);
    Inst.TemplateArgs = TemplateArgs;
    Inst.NumTemplateArgs = NumTemplateArgs;
    Inst.InstantiationRange = InstantiationRange;
    SemaRef.ActiveTemplateInstantiations.push_back(Inst);
    Invalid = false;
  }
}

Sema::InstantiatingTemplate::~InstantiatingTemplate() {
  if (!Invalid)
    SemaRef.ActiveTemplateInstantiations.pop_back();
}

bool Sema::InstantiatingTemplate::CheckInstantiationDepth(
                                        SourceLocation PointOfInstantiation,
                                           SourceRange InstantiationRange) {
  if (SemaRef.ActiveTemplateInstantiations.size() 
       <= SemaRef.getLangOptions().InstantiationDepth)
    return false;

  SemaRef.Diag(PointOfInstantiation, 
               diag::err_template_recursion_depth_exceeded)
    << SemaRef.getLangOptions().InstantiationDepth
    << InstantiationRange;
  SemaRef.Diag(PointOfInstantiation, diag::note_template_recursion_depth)
    << SemaRef.getLangOptions().InstantiationDepth;
  return true;
}

/// \brief Prints the current instantiation stack through a series of
/// notes.
void Sema::PrintInstantiationStack() {
  for (llvm::SmallVector<ActiveTemplateInstantiation, 16>::reverse_iterator
         Active = ActiveTemplateInstantiations.rbegin(),
         ActiveEnd = ActiveTemplateInstantiations.rend();
       Active != ActiveEnd;
       ++Active) {
    switch (Active->Kind) {
    case ActiveTemplateInstantiation::TemplateInstantiation: {
      unsigned DiagID = diag::note_template_member_class_here;
      CXXRecordDecl *Record = (CXXRecordDecl *)Active->Entity;
      if (isa<ClassTemplateSpecializationDecl>(Record))
        DiagID = diag::note_template_class_instantiation_here;
      Diags.Report(FullSourceLoc(Active->PointOfInstantiation, SourceMgr), 
                   DiagID)
        << Context.getTypeDeclType(Record)
        << Active->InstantiationRange;
      break;
    }

    case ActiveTemplateInstantiation::DefaultTemplateArgumentInstantiation: {
      TemplateDecl *Template = cast<TemplateDecl>((Decl *)Active->Entity);
      std::string TemplateArgsStr
        = TemplateSpecializationType::PrintTemplateArgumentList(
                                                      Active->TemplateArgs, 
                                                      Active->NumTemplateArgs);
      Diags.Report(FullSourceLoc(Active->PointOfInstantiation, SourceMgr),
                   diag::note_default_arg_instantiation_here)
        << (Template->getNameAsString() + TemplateArgsStr)
        << Active->InstantiationRange;
      break;
    }
    }
  }
}

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
TemplateTypeInstantiator::InstantiateLValueReferenceType(
    const LValueReferenceType *T, unsigned Quals) const {
  QualType ReferentType = Instantiate(T->getPointeeType());
  if (ReferentType.isNull())
    return QualType();

  return SemaRef.BuildReferenceType(ReferentType, true, Quals, Loc, Entity);
}

QualType
TemplateTypeInstantiator::InstantiateRValueReferenceType(
    const RValueReferenceType *T, unsigned Quals) const {
  QualType ReferentType = Instantiate(T->getPointeeType());
  if (ReferentType.isNull())
    return QualType();

  return SemaRef.BuildReferenceType(ReferentType, false, Quals, Loc, Entity);
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
  Expr *ArraySize = T->getSizeExpr();
  assert(ArraySize->isValueDependent() && 
         "dependent sized array types must have value dependent size expr");
  
  // Instantiate the element type if needed
  QualType ElementType = T->getElementType();
  if (ElementType->isDependentType()) {
    ElementType = Instantiate(ElementType);
    if (ElementType.isNull())
      return QualType();
  }
  
  // Instantiate the size expression
  Sema::OwningExprResult InstantiatedArraySize = 
    SemaRef.InstantiateExpr(ArraySize, TemplateArgs, NumTemplateArgs);
  if (InstantiatedArraySize.isInvalid())
    return QualType();
  
  return SemaRef.BuildArrayType(ElementType, T->getSizeModifier(),
                                InstantiatedArraySize.takeAs<Expr>(),
                                T->getIndexTypeQualifier(), Loc, Entity);
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
InstantiateTemplateSpecializationType(
                                  const TemplateSpecializationType *T,
                                  unsigned Quals) const {
  llvm::SmallVector<TemplateArgument, 16> InstantiatedTemplateArgs;
  InstantiatedTemplateArgs.reserve(T->getNumArgs());
  for (TemplateSpecializationType::iterator Arg = T->begin(), ArgEnd = T->end();
       Arg != ArgEnd; ++Arg) {
    switch (Arg->getKind()) {
    case TemplateArgument::Type: {
      QualType T = SemaRef.InstantiateType(Arg->getAsType(), 
                                           TemplateArgs, NumTemplateArgs,
                                           Arg->getLocation(),
                                           DeclarationName());
      if (T.isNull())
        return QualType();

      InstantiatedTemplateArgs.push_back(
                                TemplateArgument(Arg->getLocation(), T));
      break;
    }

    case TemplateArgument::Declaration:
    case TemplateArgument::Integral:
      InstantiatedTemplateArgs.push_back(*Arg);
      break;

    case TemplateArgument::Expression:
      Sema::OwningExprResult E 
        = SemaRef.InstantiateExpr(Arg->getAsExpr(), TemplateArgs,
                                  NumTemplateArgs);
      if (E.isInvalid())
        return QualType();
      InstantiatedTemplateArgs.push_back(E.takeAs<Expr>());
      break;
    }
  }

  // FIXME: We're missing the locations of the template name, '<', and
  // '>'.

  TemplateName Name = SemaRef.InstantiateTemplateName(T->getTemplateName(),
                                                      Loc, 
                                                      TemplateArgs,
                                                      NumTemplateArgs);

  return SemaRef.CheckTemplateIdType(Name, Loc, SourceLocation(),
                                     &InstantiatedTemplateArgs[0],
                                     InstantiatedTemplateArgs.size(),
                                     SourceLocation());
}

QualType 
TemplateTypeInstantiator::
InstantiateQualifiedNameType(const QualifiedNameType *T, 
                             unsigned Quals) const {
  // When we instantiated a qualified name type, there's no point in
  // keeping the qualification around in the instantiated result. So,
  // just instantiate the named type.
  return (*this)(T->getNamedType());
}

QualType 
TemplateTypeInstantiator::
InstantiateTypenameType(const TypenameType *T, unsigned Quals) const {
  if (const TemplateSpecializationType *TemplateId = T->getTemplateId()) {
    // When the typename type refers to a template-id, the template-id
    // is dependent and has enough information to instantiate the
    // result of the typename type. Since we don't care about keeping
    // the spelling of the typename type in template instantiations,
    // we just instantiate the template-id.
    return InstantiateTemplateSpecializationType(TemplateId, Quals);
  }

  NestedNameSpecifier *NNS 
    = SemaRef.InstantiateNestedNameSpecifier(T->getQualifier(), 
                                             SourceRange(Loc),
                                             TemplateArgs, NumTemplateArgs);
  if (!NNS)
    return QualType();

  return SemaRef.CheckTypenameType(NNS, *T->getIdentifier(), SourceRange(Loc));
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
  assert(!ActiveTemplateInstantiations.empty() &&
         "Cannot perform an instantiation without some context on the "
         "instantiation stack");

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
Sema::InstantiateBaseSpecifiers(CXXRecordDecl *Instantiation,
                                CXXRecordDecl *Pattern,
                                const TemplateArgument *TemplateArgs,
                                unsigned NumTemplateArgs) {
  bool Invalid = false;
  llvm::SmallVector<CXXBaseSpecifier*, 8> InstantiatedBases;
  for (ClassTemplateSpecializationDecl::base_class_iterator 
         Base = Pattern->bases_begin(), BaseEnd = Pattern->bases_end();
       Base != BaseEnd; ++Base) {
    if (!Base->getType()->isDependentType()) {
      // FIXME: Allocate via ASTContext
      InstantiatedBases.push_back(new CXXBaseSpecifier(*Base));
      continue;
    }

    QualType BaseType = InstantiateType(Base->getType(), 
                                        TemplateArgs, NumTemplateArgs,
                                        Base->getSourceRange().getBegin(),
                                        DeclarationName());
    if (BaseType.isNull()) {
      Invalid = true;
      continue;
    }

    if (CXXBaseSpecifier *InstantiatedBase
          = CheckBaseSpecifier(Instantiation,
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

  if (!Invalid &&
      AttachBaseSpecifiers(Instantiation, &InstantiatedBases[0],
                           InstantiatedBases.size()))
    Invalid = true;

  return Invalid;
}

/// \brief Instantiate the definition of a class from a given pattern.
///
/// \param PointOfInstantiation The point of instantiation within the
/// source code.
///
/// \param Instantiation is the declaration whose definition is being
/// instantiated. This will be either a class template specialization
/// or a member class of a class template specialization.
///
/// \param Pattern is the pattern from which the instantiation
/// occurs. This will be either the declaration of a class template or
/// the declaration of a member class of a class template.
///
/// \param TemplateArgs The template arguments to be substituted into
/// the pattern.
///
/// \param NumTemplateArgs The number of templates arguments in
/// TemplateArgs.
///
/// \returns true if an error occurred, false otherwise.
bool
Sema::InstantiateClass(SourceLocation PointOfInstantiation,
                       CXXRecordDecl *Instantiation, CXXRecordDecl *Pattern,
                       const TemplateArgument *TemplateArgs,
                       unsigned NumTemplateArgs) {
  bool Invalid = false;
  
  CXXRecordDecl *PatternDef 
    = cast_or_null<CXXRecordDecl>(Pattern->getDefinition(Context));
  if (!PatternDef) {
    if (Pattern == Instantiation->getInstantiatedFromMemberClass()) {
      Diag(PointOfInstantiation,
           diag::err_implicit_instantiate_member_undefined)
        << Context.getTypeDeclType(Instantiation);
      Diag(Pattern->getLocation(), diag::note_member_of_template_here);
    } else {
      Diag(PointOfInstantiation, 
           diag::err_template_implicit_instantiate_undefined)
        << Context.getTypeDeclType(Instantiation);
      Diag(Pattern->getLocation(), diag::note_template_decl_here);
    }
    return true;
  }
  Pattern = PatternDef;

  InstantiatingTemplate Inst(*this, PointOfInstantiation, Instantiation);
  if (Inst)
    return true;

  // Enter the scope of this instantiation. We don't use
  // PushDeclContext because we don't have a scope.
  DeclContext *PreviousContext = CurContext;
  CurContext = Instantiation;

  // Start the definition of this instantiation.
  Instantiation->startDefinition();

  // Instantiate the base class specifiers.
  if (InstantiateBaseSpecifiers(Instantiation, Pattern, TemplateArgs,
                                NumTemplateArgs))
    Invalid = true;

  llvm::SmallVector<DeclPtrTy, 32> Fields;
  for (RecordDecl::decl_iterator Member = Pattern->decls_begin(Context),
         MemberEnd = Pattern->decls_end(Context); 
       Member != MemberEnd; ++Member) {
    Decl *NewMember = InstantiateDecl(*Member, Instantiation,
                                      TemplateArgs, NumTemplateArgs);
    if (NewMember) {
      if (NewMember->isInvalidDecl())
        Invalid = true;
      else if (FieldDecl *Field = dyn_cast<FieldDecl>(NewMember))
        Fields.push_back(DeclPtrTy::make(Field));
    } else {
      // FIXME: Eventually, a NULL return will mean that one of the
      // instantiations was a semantic disaster, and we'll want to set
      // Invalid = true. For now, we expect to skip some members that
      // we can't yet handle.
    }
  }

  // Finish checking fields.
  ActOnFields(0, Instantiation->getLocation(), DeclPtrTy::make(Instantiation),
              &Fields[0], Fields.size(), SourceLocation(), SourceLocation(),
              0);

  // Add any implicitly-declared members that we might need.
  AddImplicitlyDeclaredMembersToClass(Instantiation);

  // Exit the scope of this instantiation.
  CurContext = PreviousContext;

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

  CXXRecordDecl *Pattern = Template->getTemplatedDecl();

  // Note that this is an instantiation.  
  ClassTemplateSpec->setSpecializationKind(
                        ExplicitInstantiation? TSK_ExplicitInstantiation 
                                             : TSK_ImplicitInstantiation);

  return InstantiateClass(ClassTemplateSpec->getLocation(),
                          ClassTemplateSpec, Pattern,
                          ClassTemplateSpec->getTemplateArgs(),
                          ClassTemplateSpec->getNumTemplateArgs());
}

/// \brief Instantiate a nested-name-specifier.
NestedNameSpecifier *
Sema::InstantiateNestedNameSpecifier(NestedNameSpecifier *NNS,
                                     SourceRange Range,
                                     const TemplateArgument *TemplateArgs,
                                     unsigned NumTemplateArgs) {
  // Instantiate the prefix of this nested name specifier.
  NestedNameSpecifier *Prefix = NNS->getPrefix();
  if (Prefix) {
    Prefix = InstantiateNestedNameSpecifier(Prefix, Range, TemplateArgs,
                                            NumTemplateArgs);
    if (!Prefix)
      return 0;
  }

  switch (NNS->getKind()) {
  case NestedNameSpecifier::Identifier: {
    assert(Prefix && 
           "Can't have an identifier nested-name-specifier with no prefix");
    CXXScopeSpec SS;
    // FIXME: The source location information is all wrong.
    SS.setRange(Range);
    SS.setScopeRep(Prefix);
    return static_cast<NestedNameSpecifier *>(
                                 ActOnCXXNestedNameSpecifier(0, SS,
                                                             Range.getEnd(),
                                                             Range.getEnd(),
                                                   *NNS->getAsIdentifier()));
    break;
  }

  case NestedNameSpecifier::Namespace:
  case NestedNameSpecifier::Global:
    return NNS;
    
  case NestedNameSpecifier::TypeSpecWithTemplate:
  case NestedNameSpecifier::TypeSpec: {
    QualType T = QualType(NNS->getAsType(), 0);
    if (!T->isDependentType())
      return NNS;

    T = InstantiateType(T, TemplateArgs, NumTemplateArgs, Range.getBegin(),
                        DeclarationName());
    if (T.isNull())
      return 0;

    if (T->isRecordType() ||
        (getLangOptions().CPlusPlus0x && T->isEnumeralType())) {
      assert(T.getCVRQualifiers() == 0 && "Can't get cv-qualifiers here");
      return NestedNameSpecifier::Create(Context, Prefix, 
                 NNS->getKind() == NestedNameSpecifier::TypeSpecWithTemplate,
                                         T.getTypePtr());
    }

    Diag(Range.getBegin(), diag::err_nested_name_spec_non_tag) << T;
    return 0;
  }
  }

  // Required to silence a GCC warning
  return 0;
}

TemplateName
Sema::InstantiateTemplateName(TemplateName Name, SourceLocation Loc,
                              const TemplateArgument *TemplateArgs,
                              unsigned NumTemplateArgs) {
  if (TemplateTemplateParmDecl *TTP 
        = dyn_cast_or_null<TemplateTemplateParmDecl>(
                                                 Name.getAsTemplateDecl())) {
    assert(TTP->getDepth() == 0 && 
           "Cannot reduce depth of a template template parameter");
    assert(TTP->getPosition() < NumTemplateArgs && "Wrong # of template args");
    assert(TemplateArgs[TTP->getPosition()].getAsDecl() &&
           "Wrong kind of template template argument");
    ClassTemplateDecl *ClassTemplate 
      = dyn_cast<ClassTemplateDecl>(
                               TemplateArgs[TTP->getPosition()].getAsDecl());
    assert(ClassTemplate && "Expected a class template");
    if (QualifiedTemplateName *QTN = Name.getAsQualifiedTemplateName()) {
      NestedNameSpecifier *NNS 
        = InstantiateNestedNameSpecifier(QTN->getQualifier(),
                                         /*FIXME=*/SourceRange(Loc),
                                         TemplateArgs, NumTemplateArgs);
      if (NNS)
        return Context.getQualifiedTemplateName(NNS, 
                                                QTN->hasTemplateKeyword(),
                                                ClassTemplate);
    }

    return TemplateName(ClassTemplate);
  } else if (DependentTemplateName *DTN = Name.getAsDependentTemplateName()) {
    NestedNameSpecifier *NNS 
      = InstantiateNestedNameSpecifier(DTN->getQualifier(),
                                       /*FIXME=*/SourceRange(Loc),
                                       TemplateArgs, NumTemplateArgs);
    
    if (!NNS) // FIXME: Not the best recovery strategy.
      return Name;
    
    if (NNS->isDependent())
      return Context.getDependentTemplateName(NNS, DTN->getName());

    // Somewhat redundant with ActOnDependentTemplateName.
    CXXScopeSpec SS;
    SS.setRange(SourceRange(Loc));
    SS.setScopeRep(NNS);
    TemplateTy Template;
    TemplateNameKind TNK = isTemplateName(*DTN->getName(), 0, Template, &SS);
    if (TNK == TNK_Non_template) {
      Diag(Loc, diag::err_template_kw_refers_to_non_template)
        << DTN->getName();
      return Name;
    } else if (TNK == TNK_Function_template) {
      Diag(Loc, diag::err_template_kw_refers_to_non_template)
        << DTN->getName();
      return Name;
    }
    
    return Template.getAsVal<TemplateName>();
  }

  

  // FIXME: Even if we're referring to a Decl that isn't a template
  // template parameter, we may need to instantiate the outer contexts
  // of that Decl. However, this won't be needed until we implement
  // member templates.
  return Name;
}
