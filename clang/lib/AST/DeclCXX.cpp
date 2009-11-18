//===--- DeclCXX.cpp - C++ Declaration AST Node Implementation ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the C++ related Decl classes.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/Basic/IdentifierTable.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
using namespace clang;

//===----------------------------------------------------------------------===//
// Decl Allocation/Deallocation Method Implementations
//===----------------------------------------------------------------------===//

CXXRecordDecl::CXXRecordDecl(Kind K, TagKind TK, DeclContext *DC,
                             SourceLocation L, IdentifierInfo *Id,
                             CXXRecordDecl *PrevDecl,
                             SourceLocation TKL)
  : RecordDecl(K, TK, DC, L, Id, PrevDecl, TKL),
    UserDeclaredConstructor(false), UserDeclaredCopyConstructor(false),
    UserDeclaredCopyAssignment(false), UserDeclaredDestructor(false),
    Aggregate(true), PlainOldData(true), Empty(true), Polymorphic(false),
    Abstract(false), HasTrivialConstructor(true),
    HasTrivialCopyConstructor(true), HasTrivialCopyAssignment(true),
    HasTrivialDestructor(true), ComputedVisibleConversions(false),
    Bases(0), NumBases(0), VBases(0), NumVBases(0),
    Conversions(DC, DeclarationName()),
    VisibleConversions(DC, DeclarationName()),
    TemplateOrInstantiation() { }

CXXRecordDecl *CXXRecordDecl::Create(ASTContext &C, TagKind TK, DeclContext *DC,
                                     SourceLocation L, IdentifierInfo *Id,
                                     SourceLocation TKL,
                                     CXXRecordDecl* PrevDecl,
                                     bool DelayTypeCreation) {
  CXXRecordDecl* R = new (C) CXXRecordDecl(CXXRecord, TK, DC, L, Id,
                                           PrevDecl, TKL);

  // FIXME: DelayTypeCreation seems like such a hack
  if (!DelayTypeCreation)
    C.getTypeDeclType(R, PrevDecl);
  return R;
}

CXXRecordDecl::~CXXRecordDecl() {
}

void CXXRecordDecl::Destroy(ASTContext &C) {
  C.Deallocate(Bases);
  C.Deallocate(VBases);
  this->RecordDecl::Destroy(C);
}

void
CXXRecordDecl::setBases(ASTContext &C,
                        CXXBaseSpecifier const * const *Bases,
                        unsigned NumBases) {
  // C++ [dcl.init.aggr]p1:
  //   An aggregate is an array or a class (clause 9) with [...]
  //   no base classes [...].
  Aggregate = false;

  if (this->Bases)
    C.Deallocate(this->Bases);

  int vbaseCount = 0;
  llvm::SmallVector<const CXXBaseSpecifier*, 8> UniqueVbases;
  bool hasDirectVirtualBase = false;

  this->Bases = new(C) CXXBaseSpecifier [NumBases];
  this->NumBases = NumBases;
  for (unsigned i = 0; i < NumBases; ++i) {
    this->Bases[i] = *Bases[i];
    // Keep track of inherited vbases for this base class.
    const CXXBaseSpecifier *Base = Bases[i];
    QualType BaseType = Base->getType();
    // Skip template types.
    // FIXME. This means that this list must be rebuilt during template
    // instantiation.
    if (BaseType->isDependentType())
      continue;
    CXXRecordDecl *BaseClassDecl
      = cast<CXXRecordDecl>(BaseType->getAs<RecordType>()->getDecl());
    if (Base->isVirtual())
      hasDirectVirtualBase = true;
    for (CXXRecordDecl::base_class_iterator VBase =
          BaseClassDecl->vbases_begin(),
         E = BaseClassDecl->vbases_end(); VBase != E; ++VBase) {
      // Add this vbase to the array of vbases for current class if it is
      // not already in the list.
      // FIXME. Note that we do a linear search as number of such classes are
      // very few.
      int i;
      for (i = 0; i < vbaseCount; ++i)
        if (UniqueVbases[i]->getType() == VBase->getType())
          break;
      if (i == vbaseCount) {
        UniqueVbases.push_back(VBase);
        ++vbaseCount;
      }
    }
  }
  if (hasDirectVirtualBase) {
    // Iterate one more time through the direct bases and add the virtual
    // base to the list of vritual bases for current class.
    for (unsigned i = 0; i < NumBases; ++i) {
      const CXXBaseSpecifier *VBase = Bases[i];
      if (!VBase->isVirtual())
        continue;
      int j;
      for (j = 0; j < vbaseCount; ++j)
        if (UniqueVbases[j]->getType() == VBase->getType())
          break;
      if (j == vbaseCount) {
        UniqueVbases.push_back(VBase);
        ++vbaseCount;
      }
    }
  }
  if (vbaseCount > 0) {
    // build AST for inhireted, direct or indirect, virtual bases.
    this->VBases = new (C) CXXBaseSpecifier [vbaseCount];
    this->NumVBases = vbaseCount;
    for (int i = 0; i < vbaseCount; i++) {
      QualType QT = UniqueVbases[i]->getType();
      CXXRecordDecl *VBaseClassDecl
        = cast<CXXRecordDecl>(QT->getAs<RecordType>()->getDecl());
      this->VBases[i] =
        CXXBaseSpecifier(VBaseClassDecl->getSourceRange(), true,
                         VBaseClassDecl->getTagKind() == RecordDecl::TK_class,
                         UniqueVbases[i]->getAccessSpecifier(), QT);
    }
  }
}

bool CXXRecordDecl::hasConstCopyConstructor(ASTContext &Context) const {
  return getCopyConstructor(Context, Qualifiers::Const) != 0;
}

CXXConstructorDecl *CXXRecordDecl::getCopyConstructor(ASTContext &Context,
                                                      unsigned TypeQuals) const{
  QualType ClassType
    = Context.getTypeDeclType(const_cast<CXXRecordDecl*>(this));
  DeclarationName ConstructorName
    = Context.DeclarationNames.getCXXConstructorName(
                                          Context.getCanonicalType(ClassType));
  unsigned FoundTQs;
  DeclContext::lookup_const_iterator Con, ConEnd;
  for (llvm::tie(Con, ConEnd) = this->lookup(ConstructorName);
       Con != ConEnd; ++Con) {
    // C++ [class.copy]p2:
    //   A non-template constructor for class X is a copy constructor if [...]
    if (isa<FunctionTemplateDecl>(*Con))
      continue;

    if (cast<CXXConstructorDecl>(*Con)->isCopyConstructor(Context,
                                                          FoundTQs)) {
      if (((TypeQuals & Qualifiers::Const) == (FoundTQs & Qualifiers::Const)) ||
          (!(TypeQuals & Qualifiers::Const) && (FoundTQs & Qualifiers::Const)))
        return cast<CXXConstructorDecl>(*Con);

    }
  }
  return 0;
}

bool CXXRecordDecl::hasConstCopyAssignment(ASTContext &Context,
                                           const CXXMethodDecl *& MD) const {
  QualType ClassType = Context.getCanonicalType(Context.getTypeDeclType(
    const_cast<CXXRecordDecl*>(this)));
  DeclarationName OpName =Context.DeclarationNames.getCXXOperatorName(OO_Equal);

  DeclContext::lookup_const_iterator Op, OpEnd;
  for (llvm::tie(Op, OpEnd) = this->lookup(OpName);
       Op != OpEnd; ++Op) {
    // C++ [class.copy]p9:
    //   A user-declared copy assignment operator is a non-static non-template
    //   member function of class X with exactly one parameter of type X, X&,
    //   const X&, volatile X& or const volatile X&.
    const CXXMethodDecl* Method = dyn_cast<CXXMethodDecl>(*Op);
    if (!Method)
      continue;

    if (Method->isStatic())
      continue;
    if (Method->getPrimaryTemplate())
      continue;
    const FunctionProtoType *FnType =
      Method->getType()->getAs<FunctionProtoType>();
    assert(FnType && "Overloaded operator has no prototype.");
    // Don't assert on this; an invalid decl might have been left in the AST.
    if (FnType->getNumArgs() != 1 || FnType->isVariadic())
      continue;
    bool AcceptsConst = true;
    QualType ArgType = FnType->getArgType(0);
    if (const LValueReferenceType *Ref = ArgType->getAs<LValueReferenceType>()) {
      ArgType = Ref->getPointeeType();
      // Is it a non-const lvalue reference?
      if (!ArgType.isConstQualified())
        AcceptsConst = false;
    }
    if (!Context.hasSameUnqualifiedType(ArgType, ClassType))
      continue;
    MD = Method;
    // We have a single argument of type cv X or cv X&, i.e. we've found the
    // copy assignment operator. Return whether it accepts const arguments.
    return AcceptsConst;
  }
  assert(isInvalidDecl() &&
         "No copy assignment operator declared in valid code.");
  return false;
}

void
CXXRecordDecl::addedConstructor(ASTContext &Context,
                                CXXConstructorDecl *ConDecl) {
  assert(!ConDecl->isImplicit() && "addedConstructor - not for implicit decl");
  // Note that we have a user-declared constructor.
  UserDeclaredConstructor = true;

  // C++ [dcl.init.aggr]p1:
  //   An aggregate is an array or a class (clause 9) with no
  //   user-declared constructors (12.1) [...].
  Aggregate = false;

  // C++ [class]p4:
  //   A POD-struct is an aggregate class [...]
  PlainOldData = false;

  // C++ [class.ctor]p5:
  //   A constructor is trivial if it is an implicitly-declared default
  //   constructor.
  // FIXME: C++0x: don't do this for "= default" default constructors.
  HasTrivialConstructor = false;

  // Note when we have a user-declared copy constructor, which will
  // suppress the implicit declaration of a copy constructor.
  if (ConDecl->isCopyConstructor(Context)) {
    UserDeclaredCopyConstructor = true;

    // C++ [class.copy]p6:
    //   A copy constructor is trivial if it is implicitly declared.
    // FIXME: C++0x: don't do this for "= default" copy constructors.
    HasTrivialCopyConstructor = false;
  }
}

void CXXRecordDecl::addedAssignmentOperator(ASTContext &Context,
                                            CXXMethodDecl *OpDecl) {
  // We're interested specifically in copy assignment operators.
  const FunctionProtoType *FnType = OpDecl->getType()->getAs<FunctionProtoType>();
  assert(FnType && "Overloaded operator has no proto function type.");
  assert(FnType->getNumArgs() == 1 && !FnType->isVariadic());
  
  // Copy assignment operators must be non-templates.
  if (OpDecl->getPrimaryTemplate() || OpDecl->getDescribedFunctionTemplate())
    return;
  
  QualType ArgType = FnType->getArgType(0);
  if (const LValueReferenceType *Ref = ArgType->getAs<LValueReferenceType>())
    ArgType = Ref->getPointeeType();

  ArgType = ArgType.getUnqualifiedType();
  QualType ClassType = Context.getCanonicalType(Context.getTypeDeclType(
    const_cast<CXXRecordDecl*>(this)));

  if (!Context.hasSameUnqualifiedType(ClassType, ArgType))
    return;

  // This is a copy assignment operator.
  // Note on the decl that it is a copy assignment operator.
  OpDecl->setCopyAssignment(true);

  // Suppress the implicit declaration of a copy constructor.
  UserDeclaredCopyAssignment = true;

  // C++ [class.copy]p11:
  //   A copy assignment operator is trivial if it is implicitly declared.
  // FIXME: C++0x: don't do this for "= default" copy operators.
  HasTrivialCopyAssignment = false;

  // C++ [class]p4:
  //   A POD-struct is an aggregate class that [...] has no user-defined copy
  //   assignment operator [...].
  PlainOldData = false;
}

void
CXXRecordDecl::collectConversionFunctions(
                        llvm::SmallPtrSet<CanQualType, 8>& ConversionsTypeSet) 
{
  OverloadedFunctionDecl *TopConversions = getConversionFunctions();
  for (OverloadedFunctionDecl::function_iterator
       TFunc = TopConversions->function_begin(),
       TFuncEnd = TopConversions->function_end();
       TFunc != TFuncEnd; ++TFunc) {
    NamedDecl *TopConv = TFunc->get();
    CanQualType TConvType;
    if (FunctionTemplateDecl *TConversionTemplate =
        dyn_cast<FunctionTemplateDecl>(TopConv))
      TConvType = 
        getASTContext().getCanonicalType(
                    TConversionTemplate->getTemplatedDecl()->getResultType());
    else 
      TConvType = 
        getASTContext().getCanonicalType(
                      cast<CXXConversionDecl>(TopConv)->getConversionType());
    ConversionsTypeSet.insert(TConvType);
  }  
}

/// getNestedVisibleConversionFunctions - imports unique conversion 
/// functions from base classes into the visible conversion function
/// list of the class 'RD'. This is a private helper method.
/// TopConversionsTypeSet is the set of conversion functions of the class
/// we are interested in. HiddenConversionTypes is set of conversion functions
/// of the immediate derived class which  hides the conversion functions found 
/// in current class.
void
CXXRecordDecl::getNestedVisibleConversionFunctions(CXXRecordDecl *RD,
                const llvm::SmallPtrSet<CanQualType, 8> &TopConversionsTypeSet,                               
                const llvm::SmallPtrSet<CanQualType, 8> &HiddenConversionTypes) 
{
  bool inTopClass = (RD == this);
  QualType ClassType = getASTContext().getTypeDeclType(this);
  if (const RecordType *Record = ClassType->getAs<RecordType>()) {
    OverloadedFunctionDecl *Conversions
      = cast<CXXRecordDecl>(Record->getDecl())->getConversionFunctions();
    
    for (OverloadedFunctionDecl::function_iterator
         Func = Conversions->function_begin(),
         FuncEnd = Conversions->function_end();
         Func != FuncEnd; ++Func) {
      NamedDecl *Conv = Func->get();
      // Only those conversions not exact match of conversions in current
      // class are candidateconversion routines.
      CanQualType ConvType;
      if (FunctionTemplateDecl *ConversionTemplate = 
            dyn_cast<FunctionTemplateDecl>(Conv))
        ConvType = 
          getASTContext().getCanonicalType(
                      ConversionTemplate->getTemplatedDecl()->getResultType());
      else
        ConvType = 
          getASTContext().getCanonicalType(
                          cast<CXXConversionDecl>(Conv)->getConversionType());
      // We only add conversion functions found in the base class if they
      // are not hidden by those found in HiddenConversionTypes which are
      // the conversion functions in its derived class.
      if (inTopClass || 
          (!TopConversionsTypeSet.count(ConvType) && 
           !HiddenConversionTypes.count(ConvType)) ) {
        if (FunctionTemplateDecl *ConversionTemplate =
              dyn_cast<FunctionTemplateDecl>(Conv))
          RD->addVisibleConversionFunction(ConversionTemplate);
        else
          RD->addVisibleConversionFunction(cast<CXXConversionDecl>(Conv));
      }
    }
  }

  if (getNumBases() == 0 && getNumVBases() == 0)
    return;

  llvm::SmallPtrSet<CanQualType, 8> ConversionFunctions;
  if (!inTopClass)
    collectConversionFunctions(ConversionFunctions);

  for (CXXRecordDecl::base_class_iterator VBase = vbases_begin(),
       E = vbases_end(); VBase != E; ++VBase) {
    if (const RecordType *RT = VBase->getType()->getAs<RecordType>()) {
      CXXRecordDecl *VBaseClassDecl
        = cast<CXXRecordDecl>(RT->getDecl());
      VBaseClassDecl->getNestedVisibleConversionFunctions(RD,
                    TopConversionsTypeSet,
                    (inTopClass ? TopConversionsTypeSet : ConversionFunctions));
    }
  }
  for (CXXRecordDecl::base_class_iterator Base = bases_begin(),
       E = bases_end(); Base != E; ++Base) {
    if (Base->isVirtual())
      continue;
    if (const RecordType *RT = Base->getType()->getAs<RecordType>()) {
      CXXRecordDecl *BaseClassDecl
        = cast<CXXRecordDecl>(RT->getDecl());

      BaseClassDecl->getNestedVisibleConversionFunctions(RD,
                    TopConversionsTypeSet,
                    (inTopClass ? TopConversionsTypeSet : ConversionFunctions));
    }
  }
}

/// getVisibleConversionFunctions - get all conversion functions visible
/// in current class; including conversion function templates.
OverloadedFunctionDecl *
CXXRecordDecl::getVisibleConversionFunctions() {
  // If root class, all conversions are visible.
  if (bases_begin() == bases_end())
    return &Conversions;
  // If visible conversion list is already evaluated, return it.
  if (ComputedVisibleConversions)
    return &VisibleConversions;
  llvm::SmallPtrSet<CanQualType, 8> TopConversionsTypeSet;
  collectConversionFunctions(TopConversionsTypeSet);
  getNestedVisibleConversionFunctions(this, TopConversionsTypeSet,
                                      TopConversionsTypeSet);
  ComputedVisibleConversions = true;
  return &VisibleConversions;
}

void CXXRecordDecl::addVisibleConversionFunction(
                                          CXXConversionDecl *ConvDecl) {
  assert(!ConvDecl->getDescribedFunctionTemplate() &&
         "Conversion function templates should cast to FunctionTemplateDecl.");
  VisibleConversions.addOverload(ConvDecl);
}

void CXXRecordDecl::addVisibleConversionFunction(
                                          FunctionTemplateDecl *ConvDecl) {
  assert(isa<CXXConversionDecl>(ConvDecl->getTemplatedDecl()) &&
         "Function template is not a conversion function template");
  VisibleConversions.addOverload(ConvDecl);
}

void CXXRecordDecl::addConversionFunction(CXXConversionDecl *ConvDecl) {
  assert(!ConvDecl->getDescribedFunctionTemplate() &&
         "Conversion function templates should cast to FunctionTemplateDecl.");
  Conversions.addOverload(ConvDecl);
}

void CXXRecordDecl::addConversionFunction(FunctionTemplateDecl *ConvDecl) {
  assert(isa<CXXConversionDecl>(ConvDecl->getTemplatedDecl()) &&
         "Function template is not a conversion function template");
  Conversions.addOverload(ConvDecl);
}

CXXRecordDecl *CXXRecordDecl::getInstantiatedFromMemberClass() const {
  if (MemberSpecializationInfo *MSInfo = getMemberSpecializationInfo())
    return cast<CXXRecordDecl>(MSInfo->getInstantiatedFrom());
  
  return 0;
}

MemberSpecializationInfo *CXXRecordDecl::getMemberSpecializationInfo() const {
  return TemplateOrInstantiation.dyn_cast<MemberSpecializationInfo *>();
}

void 
CXXRecordDecl::setInstantiationOfMemberClass(CXXRecordDecl *RD,
                                             TemplateSpecializationKind TSK) {
  assert(TemplateOrInstantiation.isNull() && 
         "Previous template or instantiation?");
  assert(!isa<ClassTemplateSpecializationDecl>(this));
  TemplateOrInstantiation 
    = new (getASTContext()) MemberSpecializationInfo(RD, TSK);
}

TemplateSpecializationKind CXXRecordDecl::getTemplateSpecializationKind() {
  if (ClassTemplateSpecializationDecl *Spec
        = dyn_cast<ClassTemplateSpecializationDecl>(this))
    return Spec->getSpecializationKind();
  
  if (MemberSpecializationInfo *MSInfo = getMemberSpecializationInfo())
    return MSInfo->getTemplateSpecializationKind();
  
  return TSK_Undeclared;
}

void 
CXXRecordDecl::setTemplateSpecializationKind(TemplateSpecializationKind TSK) {
  if (ClassTemplateSpecializationDecl *Spec
      = dyn_cast<ClassTemplateSpecializationDecl>(this)) {
    Spec->setSpecializationKind(TSK);
    return;
  }
  
  if (MemberSpecializationInfo *MSInfo = getMemberSpecializationInfo()) {
    MSInfo->setTemplateSpecializationKind(TSK);
    return;
  }
  
  assert(false && "Not a class template or member class specialization");
}

CXXConstructorDecl *
CXXRecordDecl::getDefaultConstructor(ASTContext &Context) {
  QualType ClassType = Context.getTypeDeclType(this);
  DeclarationName ConstructorName
    = Context.DeclarationNames.getCXXConstructorName(
                      Context.getCanonicalType(ClassType.getUnqualifiedType()));

  DeclContext::lookup_const_iterator Con, ConEnd;
  for (llvm::tie(Con, ConEnd) = lookup(ConstructorName);
       Con != ConEnd; ++Con) {
    // FIXME: In C++0x, a constructor template can be a default constructor.
    if (isa<FunctionTemplateDecl>(*Con))
      continue;

    CXXConstructorDecl *Constructor = cast<CXXConstructorDecl>(*Con);
    if (Constructor->isDefaultConstructor())
      return Constructor;
  }
  return 0;
}

const CXXDestructorDecl *
CXXRecordDecl::getDestructor(ASTContext &Context) {
  QualType ClassType = Context.getTypeDeclType(this);

  DeclarationName Name
    = Context.DeclarationNames.getCXXDestructorName(
                                          Context.getCanonicalType(ClassType));

  DeclContext::lookup_iterator I, E;
  llvm::tie(I, E) = lookup(Name);
  assert(I != E && "Did not find a destructor!");

  const CXXDestructorDecl *Dtor = cast<CXXDestructorDecl>(*I);
  assert(++I == E && "Found more than one destructor!");

  return Dtor;
}

CXXMethodDecl *
CXXMethodDecl::Create(ASTContext &C, CXXRecordDecl *RD,
                      SourceLocation L, DeclarationName N,
                      QualType T, DeclaratorInfo *DInfo,
                      bool isStatic, bool isInline) {
  return new (C) CXXMethodDecl(CXXMethod, RD, L, N, T, DInfo,
                               isStatic, isInline);
}

bool CXXMethodDecl::isUsualDeallocationFunction() const {
  if (getOverloadedOperator() != OO_Delete &&
      getOverloadedOperator() != OO_Array_Delete)
    return false;
  
  // C++ [basic.stc.dynamic.deallocation]p2:
  //   If a class T has a member deallocation function named operator delete 
  //   with exactly one parameter, then that function is a usual (non-placement)
  //   deallocation function. [...]
  if (getNumParams() == 1)
    return true;
  
  // C++ [basic.stc.dynamic.deallocation]p2:
  //   [...] If class T does not declare such an operator delete but does 
  //   declare a member deallocation function named operator delete with 
  //   exactly two parameters, the second of which has type std::size_t (18.1),
  //   then this function is a usual deallocation function.
  ASTContext &Context = getASTContext();
  if (getNumParams() != 2 ||
      !Context.hasSameType(getParamDecl(1)->getType(), Context.getSizeType()))
    return false;
                 
  // This function is a usual deallocation function if there are no 
  // single-parameter deallocation functions of the same kind.
  for (DeclContext::lookup_const_result R = getDeclContext()->lookup(getDeclName());
       R.first != R.second; ++R.first) {
    if (const FunctionDecl *FD = dyn_cast<FunctionDecl>(*R.first))
      if (FD->getNumParams() == 1)
        return false;
  }
  
  return true;
}

typedef llvm::DenseMap<const CXXMethodDecl*,
                       std::vector<const CXXMethodDecl *> *>
                       OverriddenMethodsMapTy;

// FIXME: We hate static data.  This doesn't survive PCH saving/loading, and
// the vtable building code uses it at CG time.
static OverriddenMethodsMapTy *OverriddenMethods = 0;

void CXXMethodDecl::addOverriddenMethod(const CXXMethodDecl *MD) {
  // FIXME: The CXXMethodDecl dtor needs to remove and free the entry.

  if (!OverriddenMethods)
    OverriddenMethods = new OverriddenMethodsMapTy();

  std::vector<const CXXMethodDecl *> *&Methods = (*OverriddenMethods)[this];
  if (!Methods)
    Methods = new std::vector<const CXXMethodDecl *>;

  Methods->push_back(MD);
}

CXXMethodDecl::method_iterator CXXMethodDecl::begin_overridden_methods() const {
  if (!OverriddenMethods)
    return 0;

  OverriddenMethodsMapTy::iterator it = OverriddenMethods->find(this);
  if (it == OverriddenMethods->end() || it->second->empty())
    return 0;

  return &(*it->second)[0];
}

CXXMethodDecl::method_iterator CXXMethodDecl::end_overridden_methods() const {
  if (!OverriddenMethods)
    return 0;

  OverriddenMethodsMapTy::iterator it = OverriddenMethods->find(this);
  if (it == OverriddenMethods->end() || it->second->empty())
    return 0;

  return &(*it->second)[0] + it->second->size();
}

QualType CXXMethodDecl::getThisType(ASTContext &C) const {
  // C++ 9.3.2p1: The type of this in a member function of a class X is X*.
  // If the member function is declared const, the type of this is const X*,
  // if the member function is declared volatile, the type of this is
  // volatile X*, and if the member function is declared const volatile,
  // the type of this is const volatile X*.

  assert(isInstance() && "No 'this' for static methods!");

  QualType ClassTy;
  if (ClassTemplateDecl *TD = getParent()->getDescribedClassTemplate())
    ClassTy = TD->getInjectedClassNameType(C);
  else
    ClassTy = C.getTagDeclType(getParent());
  ClassTy = C.getQualifiedType(ClassTy,
                               Qualifiers::fromCVRMask(getTypeQualifiers()));
  return C.getPointerType(ClassTy);
}

CXXBaseOrMemberInitializer::
CXXBaseOrMemberInitializer(QualType BaseType, Expr **Args, unsigned NumArgs,
                           CXXConstructorDecl *C,
                           SourceLocation L, SourceLocation R)
  : Args(0), NumArgs(0), CtorOrAnonUnion(), IdLoc(L), RParenLoc(R) {
  BaseOrMember = reinterpret_cast<uintptr_t>(BaseType.getTypePtr());
  assert((BaseOrMember & 0x01) == 0 && "Invalid base class type pointer");
  BaseOrMember |= 0x01;

  if (NumArgs > 0) {
    this->NumArgs = NumArgs;
    // FIXME. Allocation via Context
    this->Args = new Stmt*[NumArgs];
    for (unsigned Idx = 0; Idx < NumArgs; ++Idx)
      this->Args[Idx] = Args[Idx];
  }
  CtorOrAnonUnion = C;
}

CXXBaseOrMemberInitializer::
CXXBaseOrMemberInitializer(FieldDecl *Member, Expr **Args, unsigned NumArgs,
                           CXXConstructorDecl *C,
                           SourceLocation L, SourceLocation R)
  : Args(0), NumArgs(0), CtorOrAnonUnion(), IdLoc(L), RParenLoc(R) {
  BaseOrMember = reinterpret_cast<uintptr_t>(Member);
  assert((BaseOrMember & 0x01) == 0 && "Invalid member pointer");

  if (NumArgs > 0) {
    this->NumArgs = NumArgs;
    this->Args = new Stmt*[NumArgs];
    for (unsigned Idx = 0; Idx < NumArgs; ++Idx)
      this->Args[Idx] = Args[Idx];
  }
  CtorOrAnonUnion = C;
}

CXXBaseOrMemberInitializer::~CXXBaseOrMemberInitializer() {
  delete [] Args;
}

CXXConstructorDecl *
CXXConstructorDecl::Create(ASTContext &C, CXXRecordDecl *RD,
                           SourceLocation L, DeclarationName N,
                           QualType T, DeclaratorInfo *DInfo,
                           bool isExplicit,
                           bool isInline, bool isImplicitlyDeclared) {
  assert(N.getNameKind() == DeclarationName::CXXConstructorName &&
         "Name must refer to a constructor");
  return new (C) CXXConstructorDecl(RD, L, N, T, DInfo, isExplicit, isInline,
                                      isImplicitlyDeclared);
}

bool CXXConstructorDecl::isDefaultConstructor() const {
  // C++ [class.ctor]p5:
  //   A default constructor for a class X is a constructor of class
  //   X that can be called without an argument.
  return (getNumParams() == 0) ||
         (getNumParams() > 0 && getParamDecl(0)->hasDefaultArg());
}

bool
CXXConstructorDecl::isCopyConstructor(ASTContext &Context,
                                      unsigned &TypeQuals) const {
  // C++ [class.copy]p2:
  //   A non-template constructor for class X is a copy constructor
  //   if its first parameter is of type X&, const X&, volatile X& or
  //   const volatile X&, and either there are no other parameters
  //   or else all other parameters have default arguments (8.3.6).
  if ((getNumParams() < 1) ||
      (getNumParams() > 1 && !getParamDecl(1)->hasDefaultArg()) ||
      (getPrimaryTemplate() != 0) ||
      (getDescribedFunctionTemplate() != 0))
    return false;

  const ParmVarDecl *Param = getParamDecl(0);

  // Do we have a reference type? Rvalue references don't count.
  const LValueReferenceType *ParamRefType =
    Param->getType()->getAs<LValueReferenceType>();
  if (!ParamRefType)
    return false;

  // Is it a reference to our class type?
  CanQualType PointeeType
    = Context.getCanonicalType(ParamRefType->getPointeeType());
  CanQualType ClassTy 
    = Context.getCanonicalType(Context.getTagDeclType(getParent()));
  if (PointeeType.getUnqualifiedType() != ClassTy)
    return false;

  // FIXME: other qualifiers?

  // We have a copy constructor.
  TypeQuals = PointeeType.getCVRQualifiers();
  return true;
}

bool CXXConstructorDecl::isConvertingConstructor(bool AllowExplicit) const {
  // C++ [class.conv.ctor]p1:
  //   A constructor declared without the function-specifier explicit
  //   that can be called with a single parameter specifies a
  //   conversion from the type of its first parameter to the type of
  //   its class. Such a constructor is called a converting
  //   constructor.
  if (isExplicit() && !AllowExplicit)
    return false;

  return (getNumParams() == 0 &&
          getType()->getAs<FunctionProtoType>()->isVariadic()) ||
         (getNumParams() == 1) ||
         (getNumParams() > 1 && getParamDecl(1)->hasDefaultArg());
}

bool CXXConstructorDecl::isCopyConstructorLikeSpecialization() const {
  if ((getNumParams() < 1) ||
      (getNumParams() > 1 && !getParamDecl(1)->hasDefaultArg()) ||
      (getPrimaryTemplate() == 0) ||
      (getDescribedFunctionTemplate() != 0))
    return false;

  const ParmVarDecl *Param = getParamDecl(0);

  ASTContext &Context = getASTContext();
  CanQualType ParamType = Context.getCanonicalType(Param->getType());
  
  // Strip off the lvalue reference, if any.
  if (CanQual<LValueReferenceType> ParamRefType
                                    = ParamType->getAs<LValueReferenceType>())
    ParamType = ParamRefType->getPointeeType();

  
  // Is it the same as our our class type?
  CanQualType ClassTy 
    = Context.getCanonicalType(Context.getTagDeclType(getParent()));
  if (ParamType.getUnqualifiedType() != ClassTy)
    return false;
  
  return true;  
}

CXXDestructorDecl *
CXXDestructorDecl::Create(ASTContext &C, CXXRecordDecl *RD,
                          SourceLocation L, DeclarationName N,
                          QualType T, bool isInline,
                          bool isImplicitlyDeclared) {
  assert(N.getNameKind() == DeclarationName::CXXDestructorName &&
         "Name must refer to a destructor");
  return new (C) CXXDestructorDecl(RD, L, N, T, isInline,
                                   isImplicitlyDeclared);
}

void
CXXConstructorDecl::Destroy(ASTContext& C) {
  C.Deallocate(BaseOrMemberInitializers);
  CXXMethodDecl::Destroy(C);
}

CXXConversionDecl *
CXXConversionDecl::Create(ASTContext &C, CXXRecordDecl *RD,
                          SourceLocation L, DeclarationName N,
                          QualType T, DeclaratorInfo *DInfo,
                          bool isInline, bool isExplicit) {
  assert(N.getNameKind() == DeclarationName::CXXConversionFunctionName &&
         "Name must refer to a conversion function");
  return new (C) CXXConversionDecl(RD, L, N, T, DInfo, isInline, isExplicit);
}

OverloadedFunctionDecl *
OverloadedFunctionDecl::Create(ASTContext &C, DeclContext *DC,
                               DeclarationName N) {
  return new (C) OverloadedFunctionDecl(DC, N);
}

OverloadIterator::OverloadIterator(NamedDecl *ND) : D(0) {
  if (!ND)
    return;

  if (isa<FunctionDecl>(ND) || isa<FunctionTemplateDecl>(ND))
    D = ND;
  else if (OverloadedFunctionDecl *Ovl = dyn_cast<OverloadedFunctionDecl>(ND)) {
    if (Ovl->size() != 0) {
      D = ND;
      Iter = Ovl->function_begin();
    }
  }
}

void OverloadedFunctionDecl::addOverload(AnyFunctionDecl F) {
  Functions.push_back(F);
  this->setLocation(F.get()->getLocation());
}

OverloadIterator::reference OverloadIterator::operator*() const {
  if (FunctionDecl *FD = dyn_cast<FunctionDecl>(D))
    return FD;

  if (FunctionTemplateDecl *FTD = dyn_cast<FunctionTemplateDecl>(D))
    return FTD;

  assert(isa<OverloadedFunctionDecl>(D));
  return *Iter;
}

OverloadIterator &OverloadIterator::operator++() {
  if (isa<FunctionDecl>(D) || isa<FunctionTemplateDecl>(D)) {
    D = 0;
    return *this;
  }

  if (++Iter == cast<OverloadedFunctionDecl>(D)->function_end())
    D = 0;

  return *this;
}

bool OverloadIterator::Equals(const OverloadIterator &Other) const {
  if (!D || !Other.D)
    return D == Other.D;

  if (D != Other.D)
    return false;

  return !isa<OverloadedFunctionDecl>(D) || Iter == Other.Iter;
}

FriendDecl *FriendDecl::Create(ASTContext &C, DeclContext *DC,
                               SourceLocation L,
                               FriendUnion Friend,
                               SourceLocation FriendL) {
#ifndef NDEBUG
  if (Friend.is<NamedDecl*>()) {
    NamedDecl *D = Friend.get<NamedDecl*>();
    assert(isa<FunctionDecl>(D) ||
           isa<CXXRecordDecl>(D) ||
           isa<FunctionTemplateDecl>(D) ||
           isa<ClassTemplateDecl>(D));
    assert(D->getFriendObjectKind());
  }
#endif

  return new (C) FriendDecl(DC, L, Friend, FriendL);
}

LinkageSpecDecl *LinkageSpecDecl::Create(ASTContext &C,
                                         DeclContext *DC,
                                         SourceLocation L,
                                         LanguageIDs Lang, bool Braces) {
  return new (C) LinkageSpecDecl(DC, L, Lang, Braces);
}

UsingDirectiveDecl *UsingDirectiveDecl::Create(ASTContext &C, DeclContext *DC,
                                               SourceLocation L,
                                               SourceLocation NamespaceLoc,
                                               SourceRange QualifierRange,
                                               NestedNameSpecifier *Qualifier,
                                               SourceLocation IdentLoc,
                                               NamespaceDecl *Used,
                                               DeclContext *CommonAncestor) {
  return new (C) UsingDirectiveDecl(DC, L, NamespaceLoc, QualifierRange,
                                    Qualifier, IdentLoc, Used, CommonAncestor);
}

NamespaceAliasDecl *NamespaceAliasDecl::Create(ASTContext &C, DeclContext *DC,
                                               SourceLocation L,
                                               SourceLocation AliasLoc,
                                               IdentifierInfo *Alias,
                                               SourceRange QualifierRange,
                                               NestedNameSpecifier *Qualifier,
                                               SourceLocation IdentLoc,
                                               NamedDecl *Namespace) {
  return new (C) NamespaceAliasDecl(DC, L, AliasLoc, Alias, QualifierRange,
                                    Qualifier, IdentLoc, Namespace);
}

UsingDecl *UsingDecl::Create(ASTContext &C, DeclContext *DC,
      SourceLocation L, SourceRange NNR, SourceLocation UL,
      NestedNameSpecifier* TargetNNS, DeclarationName Name,
      bool IsTypeNameArg) {
  return new (C) UsingDecl(DC, L, NNR, UL, TargetNNS, Name, IsTypeNameArg);
}

UnresolvedUsingValueDecl *
UnresolvedUsingValueDecl::Create(ASTContext &C, DeclContext *DC,
                                 SourceLocation UsingLoc,
                                 SourceRange TargetNNR,
                                 NestedNameSpecifier *TargetNNS,
                                 SourceLocation TargetNameLoc,
                                 DeclarationName TargetName) {
  return new (C) UnresolvedUsingValueDecl(DC, C.DependentTy, UsingLoc,
                                          TargetNNR, TargetNNS,
                                          TargetNameLoc, TargetName);
}

UnresolvedUsingTypenameDecl *
UnresolvedUsingTypenameDecl::Create(ASTContext &C, DeclContext *DC,
                                    SourceLocation UsingLoc,
                                    SourceLocation TypenameLoc,
                                    SourceRange TargetNNR,
                                    NestedNameSpecifier *TargetNNS,
                                    SourceLocation TargetNameLoc,
                                    DeclarationName TargetName) {
  return new (C) UnresolvedUsingTypenameDecl(DC, UsingLoc, TypenameLoc,
                                             TargetNNR, TargetNNS,
                                             TargetNameLoc,
                                             TargetName.getAsIdentifierInfo());
}

StaticAssertDecl *StaticAssertDecl::Create(ASTContext &C, DeclContext *DC,
                                           SourceLocation L, Expr *AssertExpr,
                                           StringLiteral *Message) {
  return new (C) StaticAssertDecl(DC, L, AssertExpr, Message);
}

void StaticAssertDecl::Destroy(ASTContext& C) {
  AssertExpr->Destroy(C);
  Message->Destroy(C);
  this->~StaticAssertDecl();
  C.Deallocate((void *)this);
}

StaticAssertDecl::~StaticAssertDecl() {
}

static const char *getAccessName(AccessSpecifier AS) {
  switch (AS) {
    default:
    case AS_none:
      assert("Invalid access specifier!");
      return 0;
    case AS_public:
      return "public";
    case AS_private:
      return "private";
    case AS_protected:
      return "protected";
  }
}

const DiagnosticBuilder &clang::operator<<(const DiagnosticBuilder &DB,
                                           AccessSpecifier AS) {
  return DB << getAccessName(AS);
}


