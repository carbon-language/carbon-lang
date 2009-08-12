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
    Aggregate(true), PlainOldData(true), Polymorphic(false), Abstract(false),
    HasTrivialConstructor(true), HasTrivialCopyConstructor(true),
    HasTrivialCopyAssignment(true), HasTrivialDestructor(true),
    Bases(0), NumBases(0), VBases(0), NumVBases(0),
    Conversions(DC, DeclarationName()),
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
  return getCopyConstructor(Context, QualType::Const) != 0;
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
    if (cast<CXXConstructorDecl>(*Con)->isCopyConstructor(Context, 
                                                          FoundTQs)) {
      if (((TypeQuals & QualType::Const) == (FoundTQs & QualType::Const)) ||
          (!(TypeQuals & QualType::Const) && (FoundTQs & QualType::Const)))
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
    const CXXMethodDecl* Method = cast<CXXMethodDecl>(*Op);
    if (Method->isStatic())
      continue;
    // TODO: Skip templates? Or is this implicitly done due to parameter types?
    const FunctionProtoType *FnType =
      Method->getType()->getAsFunctionProtoType();
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
    if (Context.getCanonicalType(ArgType).getUnqualifiedType() != ClassType)
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
  const FunctionProtoType *FnType = OpDecl->getType()->getAsFunctionProtoType();
  assert(FnType && "Overloaded operator has no proto function type.");
  assert(FnType->getNumArgs() == 1 && !FnType->isVariadic());
  QualType ArgType = FnType->getArgType(0);
  if (const LValueReferenceType *Ref = ArgType->getAs<LValueReferenceType>())
    ArgType = Ref->getPointeeType();

  ArgType = ArgType.getUnqualifiedType();
  QualType ClassType = Context.getCanonicalType(Context.getTypeDeclType(
    const_cast<CXXRecordDecl*>(this)));

  if (ClassType != Context.getCanonicalType(ArgType))
    return;

  // This is a copy assignment operator.
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

void CXXRecordDecl::addConversionFunction(ASTContext &Context, 
                                          CXXConversionDecl *ConvDecl) {
  Conversions.addOverload(ConvDecl);
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
                      QualType T, bool isStatic, bool isInline) {
  return new (C) CXXMethodDecl(CXXMethod, RD, L, N, T, isStatic, isInline);
}


typedef llvm::DenseMap<const CXXMethodDecl*, 
                       std::vector<const CXXMethodDecl *> *> 
                       OverriddenMethodsMapTy;

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
  ClassTy = ClassTy.getWithAdditionalQualifiers(getTypeQualifiers());
  return C.getPointerType(ClassTy);
}

CXXBaseOrMemberInitializer::
CXXBaseOrMemberInitializer(QualType BaseType, Expr **Args, unsigned NumArgs,
                           CXXConstructorDecl *C,
                           SourceLocation L) 
  : Args(0), NumArgs(0), IdLoc(L) {
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
  CtorToCall = C;
}

CXXBaseOrMemberInitializer::
CXXBaseOrMemberInitializer(FieldDecl *Member, Expr **Args, unsigned NumArgs,
                           CXXConstructorDecl *C,
                           SourceLocation L)
  : Args(0), NumArgs(0), IdLoc(L) {
  BaseOrMember = reinterpret_cast<uintptr_t>(Member);
  assert((BaseOrMember & 0x01) == 0 && "Invalid member pointer");  

  if (NumArgs > 0) {
    this->NumArgs = NumArgs;
    this->Args = new Stmt*[NumArgs];
    for (unsigned Idx = 0; Idx < NumArgs; ++Idx)
      this->Args[Idx] = Args[Idx];
  }
  CtorToCall = C;
}

CXXBaseOrMemberInitializer::~CXXBaseOrMemberInitializer() {
  delete [] Args;
}

CXXConstructorDecl *
CXXConstructorDecl::Create(ASTContext &C, CXXRecordDecl *RD,
                           SourceLocation L, DeclarationName N,
                           QualType T, bool isExplicit,
                           bool isInline, bool isImplicitlyDeclared) {
  assert(N.getNameKind() == DeclarationName::CXXConstructorName &&
         "Name must refer to a constructor");
  return new (C) CXXConstructorDecl(RD, L, N, T, isExplicit, isInline,
                                      isImplicitlyDeclared);
}

bool CXXConstructorDecl::isDefaultConstructor() const {
  // C++ [class.ctor]p5:
  //   A default constructor for a class X is a constructor of class
  //   X that can be called without an argument.
  return (getNumParams() == 0) ||
         (getNumParams() > 0 && getParamDecl(0)->getDefaultArg() != 0);
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
      (getNumParams() > 1 && !getParamDecl(1)->hasDefaultArg()))
    return false;

  const ParmVarDecl *Param = getParamDecl(0);

  // Do we have a reference type? Rvalue references don't count.
  const LValueReferenceType *ParamRefType =
    Param->getType()->getAs<LValueReferenceType>();
  if (!ParamRefType)
    return false;

  // Is it a reference to our class type?
  QualType PointeeType
    = Context.getCanonicalType(ParamRefType->getPointeeType());
  QualType ClassTy = Context.getTagDeclType(getParent());
  if (PointeeType.getUnqualifiedType() != ClassTy)
    return false;

  // We have a copy constructor.
  TypeQuals = PointeeType.getCVRQualifiers();
  return true;
}

bool CXXConstructorDecl::isConvertingConstructor() const {
  // C++ [class.conv.ctor]p1:
  //   A constructor declared without the function-specifier explicit
  //   that can be called with a single parameter specifies a
  //   conversion from the type of its first parameter to the type of
  //   its class. Such a constructor is called a converting
  //   constructor.
  if (isExplicit())
    return false;

  return (getNumParams() == 0 && 
          getType()->getAsFunctionProtoType()->isVariadic()) ||
         (getNumParams() == 1) ||
         (getNumParams() > 1 && getParamDecl(1)->hasDefaultArg());
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
CXXDestructorDecl::Destroy(ASTContext& C) {
  C.Deallocate(BaseOrMemberDestructions);
  CXXMethodDecl::Destroy(C);
}

void
CXXDestructorDecl::computeBaseOrMembersToDestroy(ASTContext &C) {
  CXXRecordDecl *ClassDecl = cast<CXXRecordDecl>(getDeclContext());
  llvm::SmallVector<uintptr_t, 32> AllToDestruct;
  
  for (CXXRecordDecl::base_class_iterator VBase = ClassDecl->vbases_begin(),
       E = ClassDecl->vbases_end(); VBase != E; ++VBase) {
    // Skip over virtual bases which have trivial destructors.
    CXXRecordDecl *BaseClassDecl
      = cast<CXXRecordDecl>(VBase->getType()->getAs<RecordType>()->getDecl());
    if (BaseClassDecl->hasTrivialDestructor())
      continue;
    uintptr_t Member = 
      reinterpret_cast<uintptr_t>(VBase->getType().getTypePtr()) | VBASE;
    AllToDestruct.push_back(Member);
  }
  for (CXXRecordDecl::base_class_iterator Base =
       ClassDecl->bases_begin(),
       E = ClassDecl->bases_end(); Base != E; ++Base) {
    if (Base->isVirtual())
      continue;
    // Skip over virtual bases which have trivial destructors.
    CXXRecordDecl *BaseClassDecl
      = cast<CXXRecordDecl>(Base->getType()->getAs<RecordType>()->getDecl());
    if (BaseClassDecl->hasTrivialDestructor())
      continue;
    
    uintptr_t Member = 
      reinterpret_cast<uintptr_t>(Base->getType().getTypePtr()) | DRCTNONVBASE;
    AllToDestruct.push_back(Member);
  }
  
  // non-static data members.
  for (CXXRecordDecl::field_iterator Field = ClassDecl->field_begin(),
       E = ClassDecl->field_end(); Field != E; ++Field) {
    QualType FieldType = C.getBaseElementType((*Field)->getType());
    
    if (const RecordType* RT = FieldType->getAs<RecordType>()) {
      // Skip over virtual bases which have trivial destructors.
      CXXRecordDecl *BaseClassDecl = cast<CXXRecordDecl>(RT->getDecl());
      if (BaseClassDecl->hasTrivialDestructor())
        continue;
      uintptr_t Member = reinterpret_cast<uintptr_t>(*Field);
      AllToDestruct.push_back(Member);
    }
  }
  
  unsigned NumDestructions = AllToDestruct.size();
  if (NumDestructions > 0) {
    NumBaseOrMemberDestructions = NumDestructions;
    BaseOrMemberDestructions = new (C) uintptr_t [NumDestructions];
    // Insert in reverse order.
    for (int Idx = NumDestructions-1, i=0 ; Idx >= 0; --Idx)
      BaseOrMemberDestructions[i++] = AllToDestruct[Idx];
  }
}

void
CXXConstructorDecl::setBaseOrMemberInitializers(
                                ASTContext &C,
                                CXXBaseOrMemberInitializer **Initializers,
                                unsigned NumInitializers,
                                llvm::SmallVectorImpl<CXXBaseSpecifier *>& Bases,          
                                llvm::SmallVectorImpl<FieldDecl *>&Fields) {
  // We need to build the initializer AST according to order of construction
  // and not what user specified in the Initializers list.
  CXXRecordDecl *ClassDecl = cast<CXXRecordDecl>(getDeclContext());
  llvm::SmallVector<CXXBaseOrMemberInitializer*, 32> AllToInit;
  llvm::DenseMap<const void *, CXXBaseOrMemberInitializer*> AllBaseFields;
  
  for (unsigned i = 0; i < NumInitializers; i++) {
    CXXBaseOrMemberInitializer *Member = Initializers[i];
    if (Member->isBaseInitializer())
      AllBaseFields[Member->getBaseClass()->getAs<RecordType>()] = Member;
    else
      AllBaseFields[Member->getMember()] = Member;
  }
    
  // Push virtual bases before others.
  for (CXXRecordDecl::base_class_iterator VBase =
       ClassDecl->vbases_begin(),
       E = ClassDecl->vbases_end(); VBase != E; ++VBase) {
    if (CXXBaseOrMemberInitializer *Value = 
        AllBaseFields.lookup(VBase->getType()->getAs<RecordType>()))
      AllToInit.push_back(Value);
    else {
      CXXRecordDecl *VBaseDecl = 
        cast<CXXRecordDecl>(VBase->getType()->getAs<RecordType>()->getDecl());
      assert(VBaseDecl && "setBaseOrMemberInitializers - VBaseDecl null");
      if (!VBaseDecl->getDefaultConstructor(C) && 
          !VBase->getType()->isDependentType())
        Bases.push_back(VBase);
      CXXBaseOrMemberInitializer *Member = 
        new (C) CXXBaseOrMemberInitializer(VBase->getType(), 0, 0,
                                           VBaseDecl->getDefaultConstructor(C),
                                           SourceLocation());
      AllToInit.push_back(Member);
    }
  }
  
  for (CXXRecordDecl::base_class_iterator Base =
       ClassDecl->bases_begin(),
       E = ClassDecl->bases_end(); Base != E; ++Base) {
    // Virtuals are in the virtual base list and already constructed.
    if (Base->isVirtual())
      continue;
    if (CXXBaseOrMemberInitializer *Value = 
        AllBaseFields.lookup(Base->getType()->getAs<RecordType>()))
      AllToInit.push_back(Value);
    else {
      CXXRecordDecl *BaseDecl = 
        cast<CXXRecordDecl>(Base->getType()->getAs<RecordType>()->getDecl());
      assert(BaseDecl && "setBaseOrMemberInitializers - BaseDecl null");
      if (!BaseDecl->getDefaultConstructor(C) && 
          !Base->getType()->isDependentType())
        Bases.push_back(Base);
      CXXBaseOrMemberInitializer *Member = 
      new (C) CXXBaseOrMemberInitializer(Base->getType(), 0, 0,
                                         BaseDecl->getDefaultConstructor(C),
                                         SourceLocation());
      AllToInit.push_back(Member);
    }
  }
  
  // non-static data members.
  for (CXXRecordDecl::field_iterator Field = ClassDecl->field_begin(),
       E = ClassDecl->field_end(); Field != E; ++Field) {
    if ((*Field)->isAnonymousStructOrUnion()) {
      if (const RecordType *FieldClassType = 
            Field->getType()->getAs<RecordType>()) {
        CXXRecordDecl *FieldClassDecl
          = cast<CXXRecordDecl>(FieldClassType->getDecl());
        for(RecordDecl::field_iterator FA = FieldClassDecl->field_begin(),
            EA = FieldClassDecl->field_end(); FA != EA; FA++) {
          if (CXXBaseOrMemberInitializer *Value = AllBaseFields.lookup(*FA)) {
            // 'Member' is the anonymous union field and 'AnonUnionMember' is
            // set to the anonymous union data member used in the initializer
            // list.
            Value->setMember(*Field);
            Value->setAnonUnionMember(*FA);
            AllToInit.push_back(Value);
            break;
          }
        }
      }
      continue;
    }
    if (CXXBaseOrMemberInitializer *Value = AllBaseFields.lookup(*Field)) {
      AllToInit.push_back(Value);
      continue;
    }

    QualType FT = C.getBaseElementType((*Field)->getType());
    if (const RecordType* RT = FT->getAs<RecordType>()) {
      CXXConstructorDecl *Ctor =
        cast<CXXRecordDecl>(RT->getDecl())->getDefaultConstructor(C);
      if (!Ctor && !FT->isDependentType())
        Fields.push_back(*Field);
      CXXBaseOrMemberInitializer *Member = 
        new (C) CXXBaseOrMemberInitializer((*Field), 0, 0,
                                           Ctor,
                                           SourceLocation());
      AllToInit.push_back(Member);
    } 
  }

  NumInitializers = AllToInit.size();
  if (NumInitializers > 0) {
    NumBaseOrMemberInitializers = NumInitializers;
    BaseOrMemberInitializers = 
      new (C) CXXBaseOrMemberInitializer*[NumInitializers]; 
    for (unsigned Idx = 0; Idx < NumInitializers; ++Idx)
      BaseOrMemberInitializers[Idx] = AllToInit[Idx];
  }
}

void
CXXConstructorDecl::Destroy(ASTContext& C) {
  C.Deallocate(BaseOrMemberInitializers);
  CXXMethodDecl::Destroy(C);
}

CXXConversionDecl *
CXXConversionDecl::Create(ASTContext &C, CXXRecordDecl *RD,
                          SourceLocation L, DeclarationName N,
                          QualType T, bool isInline, bool isExplicit) {
  assert(N.getNameKind() == DeclarationName::CXXConversionFunctionName &&
         "Name must refer to a conversion function");
  return new (C) CXXConversionDecl(RD, L, N, T, isInline, isExplicit);
}

OverloadedFunctionDecl *
OverloadedFunctionDecl::Create(ASTContext &C, DeclContext *DC,
                               DeclarationName N) {
  return new (C) OverloadedFunctionDecl(DC, N);
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

FriendFunctionDecl *FriendFunctionDecl::Create(ASTContext &C,
                                               DeclContext *DC,
                                               SourceLocation L,
                                               DeclarationName N, QualType T,
                                               bool isInline,
                                               SourceLocation FriendL) {
  return new (C) FriendFunctionDecl(DC, L, N, T, isInline, FriendL);
}

FriendClassDecl *FriendClassDecl::Create(ASTContext &C, DeclContext *DC,
                                         SourceLocation L, QualType T,
                                         SourceLocation FriendL) {
  return new (C) FriendClassDecl(DC, L, T, FriendL);
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
      SourceLocation L, SourceRange NNR, SourceLocation TargetNL,
      SourceLocation UL, NamedDecl* Target,
      NestedNameSpecifier* TargetNNS, bool IsTypeNameArg) {
  return new (C) UsingDecl(DC, L, NNR, TargetNL, UL, Target,
      TargetNNS, IsTypeNameArg);
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


