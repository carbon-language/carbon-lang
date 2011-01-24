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
#include "clang/AST/ASTMutationListener.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/Expr.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Basic/IdentifierTable.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
using namespace clang;

//===----------------------------------------------------------------------===//
// Decl Allocation/Deallocation Method Implementations
//===----------------------------------------------------------------------===//

CXXRecordDecl::DefinitionData::DefinitionData(CXXRecordDecl *D)
  : UserDeclaredConstructor(false), UserDeclaredCopyConstructor(false),
    UserDeclaredCopyAssignment(false), UserDeclaredDestructor(false),
    Aggregate(true), PlainOldData(true), Empty(true), Polymorphic(false),
    Abstract(false), HasTrivialConstructor(true),
    HasTrivialCopyConstructor(true), HasTrivialCopyAssignment(true),
    HasTrivialDestructor(true), ComputedVisibleConversions(false),
    DeclaredDefaultConstructor(false), DeclaredCopyConstructor(false), 
    DeclaredCopyAssignment(false), DeclaredDestructor(false),
    NumBases(0), NumVBases(0), Bases(), VBases(),
  Definition(D), FirstFriend(0) {
}

CXXRecordDecl::CXXRecordDecl(Kind K, TagKind TK, DeclContext *DC,
                             SourceLocation L, IdentifierInfo *Id,
                             CXXRecordDecl *PrevDecl,
                             SourceLocation TKL)
  : RecordDecl(K, TK, DC, L, Id, PrevDecl, TKL),
    DefinitionData(PrevDecl ? PrevDecl->DefinitionData : 0),
    TemplateOrInstantiation() { }

CXXRecordDecl *CXXRecordDecl::Create(const ASTContext &C, TagKind TK,
                                     DeclContext *DC, SourceLocation L,
                                     IdentifierInfo *Id, SourceLocation TKL,
                                     CXXRecordDecl* PrevDecl,
                                     bool DelayTypeCreation) {
  CXXRecordDecl* R = new (C) CXXRecordDecl(CXXRecord, TK, DC, L, Id,
                                           PrevDecl, TKL);

  // FIXME: DelayTypeCreation seems like such a hack
  if (!DelayTypeCreation)
    C.getTypeDeclType(R, PrevDecl);
  return R;
}

CXXRecordDecl *CXXRecordDecl::Create(const ASTContext &C, EmptyShell Empty) {
  return new (C) CXXRecordDecl(CXXRecord, TTK_Struct, 0, SourceLocation(), 0, 0,
                               SourceLocation());
}

void
CXXRecordDecl::setBases(CXXBaseSpecifier const * const *Bases,
                        unsigned NumBases) {
  ASTContext &C = getASTContext();
  
  // C++ [dcl.init.aggr]p1:
  //   An aggregate is an array or a class (clause 9) with [...]
  //   no base classes [...].
  data().Aggregate = false;

  if (!data().Bases.isOffset() && data().NumBases > 0)
    C.Deallocate(data().getBases());

  // The set of seen virtual base types.
  llvm::SmallPtrSet<CanQualType, 8> SeenVBaseTypes;
  
  // The virtual bases of this class.
  llvm::SmallVector<const CXXBaseSpecifier *, 8> VBases;

  data().Bases = new(C) CXXBaseSpecifier [NumBases];
  data().NumBases = NumBases;
  for (unsigned i = 0; i < NumBases; ++i) {
    data().getBases()[i] = *Bases[i];
    // Keep track of inherited vbases for this base class.
    const CXXBaseSpecifier *Base = Bases[i];
    QualType BaseType = Base->getType();
    // Skip dependent types; we can't do any checking on them now.
    if (BaseType->isDependentType())
      continue;
    CXXRecordDecl *BaseClassDecl
      = cast<CXXRecordDecl>(BaseType->getAs<RecordType>()->getDecl());

    // C++ [dcl.init.aggr]p1:
    //   An aggregate is [...] a class with [...] no base classes [...].
    data().Aggregate = false;    
    
    // C++ [class]p4:
    //   A POD-struct is an aggregate class...
    data().PlainOldData = false;
    
    // A class with a non-empty base class is not empty.
    // FIXME: Standard ref?
    if (!BaseClassDecl->isEmpty())
      data().Empty = false;
    
    // C++ [class.virtual]p1:
    //   A class that declares or inherits a virtual function is called a 
    //   polymorphic class.
    if (BaseClassDecl->isPolymorphic())
      data().Polymorphic = true;
    
    // Now go through all virtual bases of this base and add them.
    for (CXXRecordDecl::base_class_iterator VBase =
          BaseClassDecl->vbases_begin(),
         E = BaseClassDecl->vbases_end(); VBase != E; ++VBase) {
      // Add this base if it's not already in the list.
      if (SeenVBaseTypes.insert(C.getCanonicalType(VBase->getType())))
        VBases.push_back(VBase);
    }

    if (Base->isVirtual()) {
      // Add this base if it's not already in the list.
      if (SeenVBaseTypes.insert(C.getCanonicalType(BaseType)))
          VBases.push_back(Base);
      
      // C++0x [meta.unary.prop] is_empty:
      //    T is a class type, but not a union type, with ... no virtual base
      //    classes
      data().Empty = false;
      
      // C++ [class.ctor]p5:
      //   A constructor is trivial if its class has no virtual base classes.
      data().HasTrivialConstructor = false;
      
      // C++ [class.copy]p6:
      //   A copy constructor is trivial if its class has no virtual base 
      //   classes.
      data().HasTrivialCopyConstructor = false;
      
      // C++ [class.copy]p11:
      //   A copy assignment operator is trivial if its class has no virtual
      //   base classes.
      data().HasTrivialCopyAssignment = false;
    } else {
      // C++ [class.ctor]p5:
      //   A constructor is trivial if all the direct base classes of its
      //   class have trivial constructors.
      if (!BaseClassDecl->hasTrivialConstructor())
        data().HasTrivialConstructor = false;
      
      // C++ [class.copy]p6:
      //   A copy constructor is trivial if all the direct base classes of its
      //   class have trivial copy constructors.
      if (!BaseClassDecl->hasTrivialCopyConstructor())
        data().HasTrivialCopyConstructor = false;
      
      // C++ [class.copy]p11:
      //   A copy assignment operator is trivial if all the direct base classes
      //   of its class have trivial copy assignment operators.
      if (!BaseClassDecl->hasTrivialCopyAssignment())
        data().HasTrivialCopyAssignment = false;
    }
    
    // C++ [class.ctor]p3:
    //   A destructor is trivial if all the direct base classes of its class
    //   have trivial destructors.
    if (!BaseClassDecl->hasTrivialDestructor())
      data().HasTrivialDestructor = false;
  }
  
  if (VBases.empty())
    return;

  // Create base specifier for any direct or indirect virtual bases.
  data().VBases = new (C) CXXBaseSpecifier[VBases.size()];
  data().NumVBases = VBases.size();
  for (int I = 0, E = VBases.size(); I != E; ++I) {
    TypeSourceInfo *VBaseTypeInfo = VBases[I]->getTypeSourceInfo();

    // Skip dependent types; we can't do any checking on them now.
    if (VBaseTypeInfo->getType()->isDependentType())
      continue;

    CXXRecordDecl *VBaseClassDecl = cast<CXXRecordDecl>(
      VBaseTypeInfo->getType()->getAs<RecordType>()->getDecl());

    data().getVBases()[I] =
      CXXBaseSpecifier(VBaseClassDecl->getSourceRange(), true,
                       VBaseClassDecl->getTagKind() == TTK_Class,
                       VBases[I]->getAccessSpecifier(), VBaseTypeInfo,
                       SourceLocation());
  }
}

/// Callback function for CXXRecordDecl::forallBases that acknowledges
/// that it saw a base class.
static bool SawBase(const CXXRecordDecl *, void *) {
  return true;
}

bool CXXRecordDecl::hasAnyDependentBases() const {
  if (!isDependentContext())
    return false;

  return !forallBases(SawBase, 0);
}

bool CXXRecordDecl::hasConstCopyConstructor(const ASTContext &Context) const {
  return getCopyConstructor(Context, Qualifiers::Const) != 0;
}

/// \brief Perform a simplistic form of overload resolution that only considers
/// cv-qualifiers on a single parameter, and return the best overload candidate
/// (if there is one).
static CXXMethodDecl *
GetBestOverloadCandidateSimple(
  const llvm::SmallVectorImpl<std::pair<CXXMethodDecl *, Qualifiers> > &Cands) {
  if (Cands.empty())
    return 0;
  if (Cands.size() == 1)
    return Cands[0].first;
  
  unsigned Best = 0, N = Cands.size();
  for (unsigned I = 1; I != N; ++I)
    if (Cands[Best].second.isSupersetOf(Cands[I].second))
      Best = I;
  
  for (unsigned I = 1; I != N; ++I)
    if (Cands[Best].second.isSupersetOf(Cands[I].second))
      return 0;
  
  return Cands[Best].first;
}

CXXConstructorDecl *CXXRecordDecl::getCopyConstructor(const ASTContext &Context,
                                                      unsigned TypeQuals) const{
  QualType ClassType
    = Context.getTypeDeclType(const_cast<CXXRecordDecl*>(this));
  DeclarationName ConstructorName
    = Context.DeclarationNames.getCXXConstructorName(
                                          Context.getCanonicalType(ClassType));
  unsigned FoundTQs;
  llvm::SmallVector<std::pair<CXXMethodDecl *, Qualifiers>, 4> Found;
  DeclContext::lookup_const_iterator Con, ConEnd;
  for (llvm::tie(Con, ConEnd) = this->lookup(ConstructorName);
       Con != ConEnd; ++Con) {
    // C++ [class.copy]p2:
    //   A non-template constructor for class X is a copy constructor if [...]
    if (isa<FunctionTemplateDecl>(*Con))
      continue;

    CXXConstructorDecl *Constructor = cast<CXXConstructorDecl>(*Con);
    if (Constructor->isCopyConstructor(FoundTQs)) {
      if (((TypeQuals & Qualifiers::Const) == (FoundTQs & Qualifiers::Const)) ||
          (!(TypeQuals & Qualifiers::Const) && (FoundTQs & Qualifiers::Const)))
        Found.push_back(std::make_pair(
                                 const_cast<CXXConstructorDecl *>(Constructor), 
                                       Qualifiers::fromCVRMask(FoundTQs)));
    }
  }
  
  return cast_or_null<CXXConstructorDecl>(
                                        GetBestOverloadCandidateSimple(Found));
}

CXXMethodDecl *CXXRecordDecl::getCopyAssignmentOperator(bool ArgIsConst) const {
  ASTContext &Context = getASTContext();
  QualType Class = Context.getTypeDeclType(const_cast<CXXRecordDecl *>(this));
  DeclarationName Name = Context.DeclarationNames.getCXXOperatorName(OO_Equal);
  
  llvm::SmallVector<std::pair<CXXMethodDecl *, Qualifiers>, 4> Found;
  DeclContext::lookup_const_iterator Op, OpEnd;
  for (llvm::tie(Op, OpEnd) = this->lookup(Name); Op != OpEnd; ++Op) {
    // C++ [class.copy]p9:
    //   A user-declared copy assignment operator is a non-static non-template
    //   member function of class X with exactly one parameter of type X, X&,
    //   const X&, volatile X& or const volatile X&.
    const CXXMethodDecl* Method = dyn_cast<CXXMethodDecl>(*Op);
    if (!Method || Method->isStatic() || Method->getPrimaryTemplate())
      continue;
    
    const FunctionProtoType *FnType 
      = Method->getType()->getAs<FunctionProtoType>();
    assert(FnType && "Overloaded operator has no prototype.");
    // Don't assert on this; an invalid decl might have been left in the AST.
    if (FnType->getNumArgs() != 1 || FnType->isVariadic())
      continue;
    
    QualType ArgType = FnType->getArgType(0);
    Qualifiers Quals;
    if (const LValueReferenceType *Ref = ArgType->getAs<LValueReferenceType>()) {
      ArgType = Ref->getPointeeType();
      // If we have a const argument and we have a reference to a non-const,
      // this function does not match.
      if (ArgIsConst && !ArgType.isConstQualified())
        continue;
      
      Quals = ArgType.getQualifiers();
    } else {
      // By-value copy-assignment operators are treated like const X&
      // copy-assignment operators.
      Quals = Qualifiers::fromCVRMask(Qualifiers::Const);
    }
    
    if (!Context.hasSameUnqualifiedType(ArgType, Class))
      continue;

    // Save this copy-assignment operator. It might be "the one".
    Found.push_back(std::make_pair(const_cast<CXXMethodDecl *>(Method), Quals));
  }
  
  // Use a simplistic form of overload resolution to find the candidate.
  return GetBestOverloadCandidateSimple(Found);
}

void CXXRecordDecl::markedVirtualFunctionPure() {
  // C++ [class.abstract]p2: 
  //   A class is abstract if it has at least one pure virtual function.
  data().Abstract = true;
}

void CXXRecordDecl::addedMember(Decl *D) {
  // Ignore friends and invalid declarations.
  if (D->getFriendObjectKind() || D->isInvalidDecl())
    return;
  
  FunctionTemplateDecl *FunTmpl = dyn_cast<FunctionTemplateDecl>(D);
  if (FunTmpl)
    D = FunTmpl->getTemplatedDecl();
  
  if (CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(D)) {
    if (Method->isVirtual()) {
      // C++ [dcl.init.aggr]p1:
      //   An aggregate is an array or a class with [...] no virtual functions.
      data().Aggregate = false;
      
      // C++ [class]p4:
      //   A POD-struct is an aggregate class...
      data().PlainOldData = false;
      
      // Virtual functions make the class non-empty.
      // FIXME: Standard ref?
      data().Empty = false;

      // C++ [class.virtual]p1:
      //   A class that declares or inherits a virtual function is called a 
      //   polymorphic class.
      data().Polymorphic = true;
      
      // None of the special member functions are trivial.
      data().HasTrivialConstructor = false;
      data().HasTrivialCopyConstructor = false;
      data().HasTrivialCopyAssignment = false;
      // FIXME: Destructor?
    }
  }
  
  if (D->isImplicit()) {
    // Notify that an implicit member was added after the definition
    // was completed.
    if (!isBeingDefined())
      if (ASTMutationListener *L = getASTMutationListener())
        L->AddedCXXImplicitMember(data().Definition, D);

    if (CXXConstructorDecl *Constructor = dyn_cast<CXXConstructorDecl>(D)) {
      // If this is the implicit default constructor, note that we have now
      // declared it.
      if (Constructor->isDefaultConstructor())
        data().DeclaredDefaultConstructor = true;
      // If this is the implicit copy constructor, note that we have now
      // declared it.
      else if (Constructor->isCopyConstructor())
        data().DeclaredCopyConstructor = true;
      return;
    } 

    if (isa<CXXDestructorDecl>(D)) {
      data().DeclaredDestructor = true;
      return;
    } 

    if (CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(D)) {
      // If this is the implicit copy constructor, note that we have now
      // declared it.
      // FIXME: Move constructors
      if (Method->getOverloadedOperator() == OO_Equal)
        data().DeclaredCopyAssignment = true;
      return;
    }

    // Any other implicit declarations are handled like normal declarations.
  }
  
  // Handle (user-declared) constructors.
  if (CXXConstructorDecl *Constructor = dyn_cast<CXXConstructorDecl>(D)) {
    // Note that we have a user-declared constructor.
    data().UserDeclaredConstructor = true;

    // Note that we have no need of an implicitly-declared default constructor.
    data().DeclaredDefaultConstructor = true;
    
    // C++ [dcl.init.aggr]p1:
    //   An aggregate is an array or a class (clause 9) with no
    //   user-declared constructors (12.1) [...].
    data().Aggregate = false;

    // C++ [class]p4:
    //   A POD-struct is an aggregate class [...]
    data().PlainOldData = false;

    // C++ [class.ctor]p5:
    //   A constructor is trivial if it is an implicitly-declared default
    //   constructor.
    // FIXME: C++0x: don't do this for "= default" default constructors.
    data().HasTrivialConstructor = false;

    // Note when we have a user-declared copy constructor, which will
    // suppress the implicit declaration of a copy constructor.
    if (!FunTmpl && Constructor->isCopyConstructor()) {
      data().UserDeclaredCopyConstructor = true;
      data().DeclaredCopyConstructor = true;
      
      // C++ [class.copy]p6:
      //   A copy constructor is trivial if it is implicitly declared.
      // FIXME: C++0x: don't do this for "= default" copy constructors.
      data().HasTrivialCopyConstructor = false;
    }
    
    return;
  }

  // Handle (user-declared) destructors.
  if (isa<CXXDestructorDecl>(D)) {
    data().DeclaredDestructor = true;
    data().UserDeclaredDestructor = true;
    
    // C++ [class]p4: 
    //   A POD-struct is an aggregate class that has [...] no user-defined 
    //   destructor.
    data().PlainOldData = false;
    
    // C++ [class.dtor]p3: 
    //   A destructor is trivial if it is an implicitly-declared destructor and
    //   [...].
    //
    // FIXME: C++0x: don't do this for "= default" destructors
    data().HasTrivialDestructor = false;
    
    return;
  }
  
  // Handle (user-declared) member functions.
  if (CXXMethodDecl *Method = dyn_cast<CXXMethodDecl>(D)) {
    if (Method->getOverloadedOperator() == OO_Equal) {
      // We're interested specifically in copy assignment operators.
      const FunctionProtoType *FnType 
        = Method->getType()->getAs<FunctionProtoType>();
      assert(FnType && "Overloaded operator has no proto function type.");
      assert(FnType->getNumArgs() == 1 && !FnType->isVariadic());
      
      // Copy assignment operators must be non-templates.
      if (Method->getPrimaryTemplate() || FunTmpl)
        return;
      
      ASTContext &Context = getASTContext();
      QualType ArgType = FnType->getArgType(0);
      if (const LValueReferenceType *Ref =ArgType->getAs<LValueReferenceType>())
        ArgType = Ref->getPointeeType();
      
      ArgType = ArgType.getUnqualifiedType();
      QualType ClassType = Context.getCanonicalType(Context.getTypeDeclType(
                                             const_cast<CXXRecordDecl*>(this)));
      
      if (!Context.hasSameUnqualifiedType(ClassType, ArgType))
        return;
      
      // This is a copy assignment operator.
      // FIXME: Move assignment operators.
      
      // Suppress the implicit declaration of a copy constructor.
      data().UserDeclaredCopyAssignment = true;
      data().DeclaredCopyAssignment = true;
      
      // C++ [class.copy]p11:
      //   A copy assignment operator is trivial if it is implicitly declared.
      // FIXME: C++0x: don't do this for "= default" copy operators.
      data().HasTrivialCopyAssignment = false;
      
      // C++ [class]p4:
      //   A POD-struct is an aggregate class that [...] has no user-defined copy
      //   assignment operator [...].
      data().PlainOldData = false;
    }
    
    // Keep the list of conversion functions up-to-date.
    if (CXXConversionDecl *Conversion = dyn_cast<CXXConversionDecl>(D)) {
      // We don't record specializations.
      if (Conversion->getPrimaryTemplate())
        return;
      
      // FIXME: We intentionally don't use the decl's access here because it
      // hasn't been set yet.  That's really just a misdesign in Sema.

      if (FunTmpl) {
        if (FunTmpl->getPreviousDeclaration())
          data().Conversions.replace(FunTmpl->getPreviousDeclaration(),
                                     FunTmpl);
        else
          data().Conversions.addDecl(FunTmpl);
      } else {
        if (Conversion->getPreviousDeclaration())
          data().Conversions.replace(Conversion->getPreviousDeclaration(),
                                     Conversion);
        else
          data().Conversions.addDecl(Conversion);        
      }
    }
    
    return;
  }
  
  // Handle non-static data members.
  if (FieldDecl *Field = dyn_cast<FieldDecl>(D)) {
    // C++ [dcl.init.aggr]p1:
    //   An aggregate is an array or a class (clause 9) with [...] no
    //   private or protected non-static data members (clause 11).
    //
    // A POD must be an aggregate.    
    if (D->getAccess() == AS_private || D->getAccess() == AS_protected) {
      data().Aggregate = false;
      data().PlainOldData = false;
    }
    
    // C++ [class]p9:
    //   A POD struct is a class that is both a trivial class and a 
    //   standard-layout class, and has no non-static data members of type 
    //   non-POD struct, non-POD union (or array of such types).
    ASTContext &Context = getASTContext();
    QualType T = Context.getBaseElementType(Field->getType());
    if (!T->isPODType())
      data().PlainOldData = false;
    if (T->isReferenceType())
      data().HasTrivialConstructor = false;
    
    if (const RecordType *RecordTy = T->getAs<RecordType>()) {
      CXXRecordDecl* FieldRec = cast<CXXRecordDecl>(RecordTy->getDecl());
      if (FieldRec->getDefinition()) {
        if (!FieldRec->hasTrivialConstructor())
          data().HasTrivialConstructor = false;
        if (!FieldRec->hasTrivialCopyConstructor())
          data().HasTrivialCopyConstructor = false;
        if (!FieldRec->hasTrivialCopyAssignment())
          data().HasTrivialCopyAssignment = false;
        if (!FieldRec->hasTrivialDestructor())
          data().HasTrivialDestructor = false;
      }
    }
    
    // If this is not a zero-length bit-field, then the class is not empty.
    if (data().Empty) {
      if (!Field->getBitWidth())
        data().Empty = false;
      else if (!Field->getBitWidth()->isTypeDependent() &&
               !Field->getBitWidth()->isValueDependent()) {
        llvm::APSInt Bits;
        if (Field->getBitWidth()->isIntegerConstantExpr(Bits, Context))
          if (!!Bits)
            data().Empty = false;
      } 
    }
  }
  
  // Handle using declarations of conversion functions.
  if (UsingShadowDecl *Shadow = dyn_cast<UsingShadowDecl>(D))
    if (Shadow->getDeclName().getNameKind()
          == DeclarationName::CXXConversionFunctionName)
      data().Conversions.addDecl(Shadow, Shadow->getAccess());
}

static CanQualType GetConversionType(ASTContext &Context, NamedDecl *Conv) {
  QualType T;
  if (isa<UsingShadowDecl>(Conv))
    Conv = cast<UsingShadowDecl>(Conv)->getTargetDecl();
  if (FunctionTemplateDecl *ConvTemp = dyn_cast<FunctionTemplateDecl>(Conv))
    T = ConvTemp->getTemplatedDecl()->getResultType();
  else 
    T = cast<CXXConversionDecl>(Conv)->getConversionType();
  return Context.getCanonicalType(T);
}

/// Collect the visible conversions of a base class.
///
/// \param Base a base class of the class we're considering
/// \param InVirtual whether this base class is a virtual base (or a base
///   of a virtual base)
/// \param Access the access along the inheritance path to this base
/// \param ParentHiddenTypes the conversions provided by the inheritors
///   of this base
/// \param Output the set to which to add conversions from non-virtual bases
/// \param VOutput the set to which to add conversions from virtual bases
/// \param HiddenVBaseCs the set of conversions which were hidden in a
///   virtual base along some inheritance path
static void CollectVisibleConversions(ASTContext &Context,
                                      CXXRecordDecl *Record,
                                      bool InVirtual,
                                      AccessSpecifier Access,
                  const llvm::SmallPtrSet<CanQualType, 8> &ParentHiddenTypes,
                                      UnresolvedSetImpl &Output,
                                      UnresolvedSetImpl &VOutput,
                           llvm::SmallPtrSet<NamedDecl*, 8> &HiddenVBaseCs) {
  // The set of types which have conversions in this class or its
  // subclasses.  As an optimization, we don't copy the derived set
  // unless it might change.
  const llvm::SmallPtrSet<CanQualType, 8> *HiddenTypes = &ParentHiddenTypes;
  llvm::SmallPtrSet<CanQualType, 8> HiddenTypesBuffer;

  // Collect the direct conversions and figure out which conversions
  // will be hidden in the subclasses.
  UnresolvedSetImpl &Cs = *Record->getConversionFunctions();
  if (!Cs.empty()) {
    HiddenTypesBuffer = ParentHiddenTypes;
    HiddenTypes = &HiddenTypesBuffer;

    for (UnresolvedSetIterator I = Cs.begin(), E = Cs.end(); I != E; ++I) {
      bool Hidden =
        !HiddenTypesBuffer.insert(GetConversionType(Context, I.getDecl()));

      // If this conversion is hidden and we're in a virtual base,
      // remember that it's hidden along some inheritance path.
      if (Hidden && InVirtual)
        HiddenVBaseCs.insert(cast<NamedDecl>(I.getDecl()->getCanonicalDecl()));

      // If this conversion isn't hidden, add it to the appropriate output.
      else if (!Hidden) {
        AccessSpecifier IAccess
          = CXXRecordDecl::MergeAccess(Access, I.getAccess());

        if (InVirtual)
          VOutput.addDecl(I.getDecl(), IAccess);
        else
          Output.addDecl(I.getDecl(), IAccess);
      }
    }
  }

  // Collect information recursively from any base classes.
  for (CXXRecordDecl::base_class_iterator
         I = Record->bases_begin(), E = Record->bases_end(); I != E; ++I) {
    const RecordType *RT = I->getType()->getAs<RecordType>();
    if (!RT) continue;

    AccessSpecifier BaseAccess
      = CXXRecordDecl::MergeAccess(Access, I->getAccessSpecifier());
    bool BaseInVirtual = InVirtual || I->isVirtual();

    CXXRecordDecl *Base = cast<CXXRecordDecl>(RT->getDecl());
    CollectVisibleConversions(Context, Base, BaseInVirtual, BaseAccess,
                              *HiddenTypes, Output, VOutput, HiddenVBaseCs);
  }
}

/// Collect the visible conversions of a class.
///
/// This would be extremely straightforward if it weren't for virtual
/// bases.  It might be worth special-casing that, really.
static void CollectVisibleConversions(ASTContext &Context,
                                      CXXRecordDecl *Record,
                                      UnresolvedSetImpl &Output) {
  // The collection of all conversions in virtual bases that we've
  // found.  These will be added to the output as long as they don't
  // appear in the hidden-conversions set.
  UnresolvedSet<8> VBaseCs;
  
  // The set of conversions in virtual bases that we've determined to
  // be hidden.
  llvm::SmallPtrSet<NamedDecl*, 8> HiddenVBaseCs;

  // The set of types hidden by classes derived from this one.
  llvm::SmallPtrSet<CanQualType, 8> HiddenTypes;

  // Go ahead and collect the direct conversions and add them to the
  // hidden-types set.
  UnresolvedSetImpl &Cs = *Record->getConversionFunctions();
  Output.append(Cs.begin(), Cs.end());
  for (UnresolvedSetIterator I = Cs.begin(), E = Cs.end(); I != E; ++I)
    HiddenTypes.insert(GetConversionType(Context, I.getDecl()));

  // Recursively collect conversions from base classes.
  for (CXXRecordDecl::base_class_iterator
         I = Record->bases_begin(), E = Record->bases_end(); I != E; ++I) {
    const RecordType *RT = I->getType()->getAs<RecordType>();
    if (!RT) continue;

    CollectVisibleConversions(Context, cast<CXXRecordDecl>(RT->getDecl()),
                              I->isVirtual(), I->getAccessSpecifier(),
                              HiddenTypes, Output, VBaseCs, HiddenVBaseCs);
  }

  // Add any unhidden conversions provided by virtual bases.
  for (UnresolvedSetIterator I = VBaseCs.begin(), E = VBaseCs.end();
         I != E; ++I) {
    if (!HiddenVBaseCs.count(cast<NamedDecl>(I.getDecl()->getCanonicalDecl())))
      Output.addDecl(I.getDecl(), I.getAccess());
  }
}

/// getVisibleConversionFunctions - get all conversion functions visible
/// in current class; including conversion function templates.
const UnresolvedSetImpl *CXXRecordDecl::getVisibleConversionFunctions() {
  // If root class, all conversions are visible.
  if (bases_begin() == bases_end())
    return &data().Conversions;
  // If visible conversion list is already evaluated, return it.
  if (data().ComputedVisibleConversions)
    return &data().VisibleConversions;
  CollectVisibleConversions(getASTContext(), this, data().VisibleConversions);
  data().ComputedVisibleConversions = true;
  return &data().VisibleConversions;
}

void CXXRecordDecl::removeConversion(const NamedDecl *ConvDecl) {
  // This operation is O(N) but extremely rare.  Sema only uses it to
  // remove UsingShadowDecls in a class that were followed by a direct
  // declaration, e.g.:
  //   class A : B {
  //     using B::operator int;
  //     operator int();
  //   };
  // This is uncommon by itself and even more uncommon in conjunction
  // with sufficiently large numbers of directly-declared conversions
  // that asymptotic behavior matters.

  UnresolvedSetImpl &Convs = *getConversionFunctions();
  for (unsigned I = 0, E = Convs.size(); I != E; ++I) {
    if (Convs[I].getDecl() == ConvDecl) {
      Convs.erase(I);
      assert(std::find(Convs.begin(), Convs.end(), ConvDecl) == Convs.end()
             && "conversion was found multiple times in unresolved set");
      return;
    }
  }

  llvm_unreachable("conversion not found in set!");
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

TemplateSpecializationKind CXXRecordDecl::getTemplateSpecializationKind() const{
  if (const ClassTemplateSpecializationDecl *Spec
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

CXXDestructorDecl *CXXRecordDecl::getDestructor() const {
  ASTContext &Context = getASTContext();
  QualType ClassType = Context.getTypeDeclType(this);

  DeclarationName Name
    = Context.DeclarationNames.getCXXDestructorName(
                                          Context.getCanonicalType(ClassType));

  DeclContext::lookup_const_iterator I, E;
  llvm::tie(I, E) = lookup(Name);
  if (I == E)
    return 0;

  CXXDestructorDecl *Dtor = cast<CXXDestructorDecl>(*I);
  assert(++I == E && "Found more than one destructor!");

  return Dtor;
}

void CXXRecordDecl::completeDefinition() {
  completeDefinition(0);
}

void CXXRecordDecl::completeDefinition(CXXFinalOverriderMap *FinalOverriders) {
  RecordDecl::completeDefinition();
  
  // If the class may be abstract (but hasn't been marked as such), check for
  // any pure final overriders.
  if (mayBeAbstract()) {
    CXXFinalOverriderMap MyFinalOverriders;
    if (!FinalOverriders) {
      getFinalOverriders(MyFinalOverriders);
      FinalOverriders = &MyFinalOverriders;
    }
    
    bool Done = false;
    for (CXXFinalOverriderMap::iterator M = FinalOverriders->begin(), 
                                     MEnd = FinalOverriders->end();
         M != MEnd && !Done; ++M) {
      for (OverridingMethods::iterator SO = M->second.begin(), 
                                    SOEnd = M->second.end();
           SO != SOEnd && !Done; ++SO) {
        assert(SO->second.size() > 0 && 
               "All virtual functions have overridding virtual functions");
        
        // C++ [class.abstract]p4:
        //   A class is abstract if it contains or inherits at least one
        //   pure virtual function for which the final overrider is pure
        //   virtual.
        if (SO->second.front().Method->isPure()) {
          data().Abstract = true;
          Done = true;
          break;
        }
      }
    }
  }
  
  // Set access bits correctly on the directly-declared conversions.
  for (UnresolvedSetIterator I = data().Conversions.begin(), 
                             E = data().Conversions.end(); 
       I != E; ++I)
    data().Conversions.setAccess(I, (*I)->getAccess());
}

bool CXXRecordDecl::mayBeAbstract() const {
  if (data().Abstract || isInvalidDecl() || !data().Polymorphic ||
      isDependentContext())
    return false;
  
  for (CXXRecordDecl::base_class_const_iterator B = bases_begin(),
                                             BEnd = bases_end();
       B != BEnd; ++B) {
    CXXRecordDecl *BaseDecl 
      = cast<CXXRecordDecl>(B->getType()->getAs<RecordType>()->getDecl());
    if (BaseDecl->isAbstract())
      return true;
  }
  
  return false;
}

CXXMethodDecl *
CXXMethodDecl::Create(ASTContext &C, CXXRecordDecl *RD,
                      const DeclarationNameInfo &NameInfo,
                      QualType T, TypeSourceInfo *TInfo,
                      bool isStatic, StorageClass SCAsWritten, bool isInline) {
  return new (C) CXXMethodDecl(CXXMethod, RD, NameInfo, T, TInfo,
                               isStatic, SCAsWritten, isInline);
}

bool CXXMethodDecl::isUsualDeallocationFunction() const {
  if (getOverloadedOperator() != OO_Delete &&
      getOverloadedOperator() != OO_Array_Delete)
    return false;

  // C++ [basic.stc.dynamic.deallocation]p2:
  //   A template instance is never a usual deallocation function,
  //   regardless of its signature.
  if (getPrimaryTemplate())
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
      !Context.hasSameUnqualifiedType(getParamDecl(1)->getType(),
                                      Context.getSizeType()))
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

bool CXXMethodDecl::isCopyAssignmentOperator() const {
  // C++0x [class.copy]p19:
  //  A user-declared copy assignment operator X::operator= is a non-static 
  //  non-template member function of class X with exactly one parameter of 
  //  type X, X&, const X&, volatile X& or const volatile X&.
  if (/*operator=*/getOverloadedOperator() != OO_Equal ||
      /*non-static*/ isStatic() || 
      /*non-template*/getPrimaryTemplate() || getDescribedFunctionTemplate() ||
      /*exactly one parameter*/getNumParams() != 1)
    return false;
      
  QualType ParamType = getParamDecl(0)->getType();
  if (const LValueReferenceType *Ref = ParamType->getAs<LValueReferenceType>())
    ParamType = Ref->getPointeeType();
  
  ASTContext &Context = getASTContext();
  QualType ClassType
    = Context.getCanonicalType(Context.getTypeDeclType(getParent()));
  return Context.hasSameUnqualifiedType(ClassType, ParamType);
}

void CXXMethodDecl::addOverriddenMethod(const CXXMethodDecl *MD) {
  assert(MD->isCanonicalDecl() && "Method is not canonical!");
  assert(!MD->getParent()->isDependentContext() &&
         "Can't add an overridden method to a class template!");

  getASTContext().addOverriddenMethod(this, MD);
}

CXXMethodDecl::method_iterator CXXMethodDecl::begin_overridden_methods() const {
  return getASTContext().overridden_methods_begin(this);
}

CXXMethodDecl::method_iterator CXXMethodDecl::end_overridden_methods() const {
  return getASTContext().overridden_methods_end(this);
}

unsigned CXXMethodDecl::size_overridden_methods() const {
  return getASTContext().overridden_methods_size(this);
}

QualType CXXMethodDecl::getThisType(ASTContext &C) const {
  // C++ 9.3.2p1: The type of this in a member function of a class X is X*.
  // If the member function is declared const, the type of this is const X*,
  // if the member function is declared volatile, the type of this is
  // volatile X*, and if the member function is declared const volatile,
  // the type of this is const volatile X*.

  assert(isInstance() && "No 'this' for static methods!");

  QualType ClassTy = C.getTypeDeclType(getParent());
  ClassTy = C.getQualifiedType(ClassTy,
                               Qualifiers::fromCVRMask(getTypeQualifiers()));
  return C.getPointerType(ClassTy);
}

bool CXXMethodDecl::hasInlineBody() const {
  // If this function is a template instantiation, look at the template from 
  // which it was instantiated.
  const FunctionDecl *CheckFn = getTemplateInstantiationPattern();
  if (!CheckFn)
    CheckFn = this;
  
  const FunctionDecl *fn;
  return CheckFn->hasBody(fn) && !fn->isOutOfLine();
}

CXXCtorInitializer::CXXCtorInitializer(ASTContext &Context,
                                       TypeSourceInfo *TInfo, bool IsVirtual,
                                       SourceLocation L, Expr *Init,
                                       SourceLocation R,
                                       SourceLocation EllipsisLoc)
  : Initializee(TInfo), MemberOrEllipsisLocation(EllipsisLoc), Init(Init), 
    LParenLoc(L), RParenLoc(R), IsVirtual(IsVirtual), IsWritten(false),
    SourceOrderOrNumArrayIndices(0)
{
}

CXXCtorInitializer::CXXCtorInitializer(ASTContext &Context,
                                       FieldDecl *Member,
                                       SourceLocation MemberLoc,
                                       SourceLocation L, Expr *Init,
                                       SourceLocation R)
  : Initializee(Member), MemberOrEllipsisLocation(MemberLoc), Init(Init),
    LParenLoc(L), RParenLoc(R), IsVirtual(false),
    IsWritten(false), SourceOrderOrNumArrayIndices(0)
{
}

CXXCtorInitializer::CXXCtorInitializer(ASTContext &Context,
                                       IndirectFieldDecl *Member,
                                       SourceLocation MemberLoc,
                                       SourceLocation L, Expr *Init,
                                       SourceLocation R)
  : Initializee(Member), MemberOrEllipsisLocation(MemberLoc), Init(Init),
    LParenLoc(L), RParenLoc(R), IsVirtual(false),
    IsWritten(false), SourceOrderOrNumArrayIndices(0)
{
}

CXXCtorInitializer::CXXCtorInitializer(ASTContext &Context,
                                       FieldDecl *Member,
                                       SourceLocation MemberLoc,
                                       SourceLocation L, Expr *Init,
                                       SourceLocation R,
                                       VarDecl **Indices,
                                       unsigned NumIndices)
  : Initializee(Member), MemberOrEllipsisLocation(MemberLoc), Init(Init), 
    LParenLoc(L), RParenLoc(R), IsVirtual(false),
    IsWritten(false), SourceOrderOrNumArrayIndices(NumIndices)
{
  VarDecl **MyIndices = reinterpret_cast<VarDecl **> (this + 1);
  memcpy(MyIndices, Indices, NumIndices * sizeof(VarDecl *));
}

CXXCtorInitializer *CXXCtorInitializer::Create(ASTContext &Context,
                                               FieldDecl *Member, 
                                               SourceLocation MemberLoc,
                                               SourceLocation L, Expr *Init,
                                               SourceLocation R,
                                               VarDecl **Indices,
                                               unsigned NumIndices) {
  void *Mem = Context.Allocate(sizeof(CXXCtorInitializer) +
                               sizeof(VarDecl *) * NumIndices,
                               llvm::alignOf<CXXCtorInitializer>());
  return new (Mem) CXXCtorInitializer(Context, Member, MemberLoc, L, Init, R,
                                      Indices, NumIndices);
}

TypeLoc CXXCtorInitializer::getBaseClassLoc() const {
  if (isBaseInitializer())
    return Initializee.get<TypeSourceInfo*>()->getTypeLoc();
  else
    return TypeLoc();
}

const Type *CXXCtorInitializer::getBaseClass() const {
  if (isBaseInitializer())
    return Initializee.get<TypeSourceInfo*>()->getType().getTypePtr();
  else
    return 0;
}

SourceLocation CXXCtorInitializer::getSourceLocation() const {
  if (isAnyMemberInitializer())
    return getMemberLocation();
  
  return getBaseClassLoc().getLocalSourceRange().getBegin();
}

SourceRange CXXCtorInitializer::getSourceRange() const {
  return SourceRange(getSourceLocation(), getRParenLoc());
}

CXXConstructorDecl *
CXXConstructorDecl::Create(ASTContext &C, EmptyShell Empty) {
  return new (C) CXXConstructorDecl(0, DeclarationNameInfo(),
                                    QualType(), 0, false, false, false);
}

CXXConstructorDecl *
CXXConstructorDecl::Create(ASTContext &C, CXXRecordDecl *RD,
                           const DeclarationNameInfo &NameInfo,
                           QualType T, TypeSourceInfo *TInfo,
                           bool isExplicit,
                           bool isInline,
                           bool isImplicitlyDeclared) {
  assert(NameInfo.getName().getNameKind()
         == DeclarationName::CXXConstructorName &&
         "Name must refer to a constructor");
  return new (C) CXXConstructorDecl(RD, NameInfo, T, TInfo, isExplicit,
                                    isInline, isImplicitlyDeclared);
}

bool CXXConstructorDecl::isDefaultConstructor() const {
  // C++ [class.ctor]p5:
  //   A default constructor for a class X is a constructor of class
  //   X that can be called without an argument.
  return (getNumParams() == 0) ||
         (getNumParams() > 0 && getParamDecl(0)->hasDefaultArg());
}

bool
CXXConstructorDecl::isCopyConstructor(unsigned &TypeQuals) const {
  return isCopyOrMoveConstructor(TypeQuals) &&
         getParamDecl(0)->getType()->isLValueReferenceType();
}

bool CXXConstructorDecl::isMoveConstructor(unsigned &TypeQuals) const {
  return isCopyOrMoveConstructor(TypeQuals) &&
    getParamDecl(0)->getType()->isRValueReferenceType();
}

/// \brief Determine whether this is a copy or move constructor.
bool CXXConstructorDecl::isCopyOrMoveConstructor(unsigned &TypeQuals) const {
  // C++ [class.copy]p2:
  //   A non-template constructor for class X is a copy constructor
  //   if its first parameter is of type X&, const X&, volatile X& or
  //   const volatile X&, and either there are no other parameters
  //   or else all other parameters have default arguments (8.3.6).
  // C++0x [class.copy]p3:
  //   A non-template constructor for class X is a move constructor if its
  //   first parameter is of type X&&, const X&&, volatile X&&, or 
  //   const volatile X&&, and either there are no other parameters or else 
  //   all other parameters have default arguments.
  if ((getNumParams() < 1) ||
      (getNumParams() > 1 && !getParamDecl(1)->hasDefaultArg()) ||
      (getPrimaryTemplate() != 0) ||
      (getDescribedFunctionTemplate() != 0))
    return false;
  
  const ParmVarDecl *Param = getParamDecl(0);
  
  // Do we have a reference type? 
  const ReferenceType *ParamRefType = Param->getType()->getAs<ReferenceType>();
  if (!ParamRefType)
    return false;
  
  // Is it a reference to our class type?
  ASTContext &Context = getASTContext();
  
  CanQualType PointeeType
    = Context.getCanonicalType(ParamRefType->getPointeeType());
  CanQualType ClassTy 
    = Context.getCanonicalType(Context.getTagDeclType(getParent()));
  if (PointeeType.getUnqualifiedType() != ClassTy)
    return false;
  
  // FIXME: other qualifiers?
  
  // We have a copy or move constructor.
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

bool CXXConstructorDecl::isSpecializationCopyingObject() const {
  if ((getNumParams() < 1) ||
      (getNumParams() > 1 && !getParamDecl(1)->hasDefaultArg()) ||
      (getPrimaryTemplate() == 0) ||
      (getDescribedFunctionTemplate() != 0))
    return false;

  const ParmVarDecl *Param = getParamDecl(0);

  ASTContext &Context = getASTContext();
  CanQualType ParamType = Context.getCanonicalType(Param->getType());
  
  // Is it the same as our our class type?
  CanQualType ClassTy 
    = Context.getCanonicalType(Context.getTagDeclType(getParent()));
  if (ParamType.getUnqualifiedType() != ClassTy)
    return false;
  
  return true;  
}

CXXDestructorDecl *
CXXDestructorDecl::Create(ASTContext &C, EmptyShell Empty) {
  return new (C) CXXDestructorDecl(0, DeclarationNameInfo(),
                                   QualType(), 0, false, false);
}

CXXDestructorDecl *
CXXDestructorDecl::Create(ASTContext &C, CXXRecordDecl *RD,
                          const DeclarationNameInfo &NameInfo,
                          QualType T, TypeSourceInfo *TInfo,
                          bool isInline,
                          bool isImplicitlyDeclared) {
  assert(NameInfo.getName().getNameKind()
         == DeclarationName::CXXDestructorName &&
         "Name must refer to a destructor");
  return new (C) CXXDestructorDecl(RD, NameInfo, T, TInfo, isInline,
                                   isImplicitlyDeclared);
}

CXXConversionDecl *
CXXConversionDecl::Create(ASTContext &C, EmptyShell Empty) {
  return new (C) CXXConversionDecl(0, DeclarationNameInfo(),
                                   QualType(), 0, false, false);
}

CXXConversionDecl *
CXXConversionDecl::Create(ASTContext &C, CXXRecordDecl *RD,
                          const DeclarationNameInfo &NameInfo,
                          QualType T, TypeSourceInfo *TInfo,
                          bool isInline, bool isExplicit) {
  assert(NameInfo.getName().getNameKind()
         == DeclarationName::CXXConversionFunctionName &&
         "Name must refer to a conversion function");
  return new (C) CXXConversionDecl(RD, NameInfo, T, TInfo,
                                   isInline, isExplicit);
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
                                               NamedDecl *Used,
                                               DeclContext *CommonAncestor) {
  if (NamespaceDecl *NS = dyn_cast_or_null<NamespaceDecl>(Used))
    Used = NS->getOriginalNamespace();
  return new (C) UsingDirectiveDecl(DC, L, NamespaceLoc, QualifierRange,
                                    Qualifier, IdentLoc, Used, CommonAncestor);
}

NamespaceDecl *UsingDirectiveDecl::getNominatedNamespace() {
  if (NamespaceAliasDecl *NA =
        dyn_cast_or_null<NamespaceAliasDecl>(NominatedNamespace))
    return NA->getNamespace();
  return cast_or_null<NamespaceDecl>(NominatedNamespace);
}

NamespaceAliasDecl *NamespaceAliasDecl::Create(ASTContext &C, DeclContext *DC,
                                               SourceLocation UsingLoc,
                                               SourceLocation AliasLoc,
                                               IdentifierInfo *Alias,
                                               SourceRange QualifierRange,
                                               NestedNameSpecifier *Qualifier,
                                               SourceLocation IdentLoc,
                                               NamedDecl *Namespace) {
  if (NamespaceDecl *NS = dyn_cast_or_null<NamespaceDecl>(Namespace))
    Namespace = NS->getOriginalNamespace();
  return new (C) NamespaceAliasDecl(DC, UsingLoc, AliasLoc, Alias, QualifierRange,
                                    Qualifier, IdentLoc, Namespace);
}

UsingDecl *UsingShadowDecl::getUsingDecl() const {
  const UsingShadowDecl *Shadow = this;
  while (const UsingShadowDecl *NextShadow =
         dyn_cast<UsingShadowDecl>(Shadow->UsingOrNextShadow))
    Shadow = NextShadow;
  return cast<UsingDecl>(Shadow->UsingOrNextShadow);
}

void UsingDecl::addShadowDecl(UsingShadowDecl *S) {
  assert(std::find(shadow_begin(), shadow_end(), S) == shadow_end() &&
         "declaration already in set");
  assert(S->getUsingDecl() == this);

  if (FirstUsingShadow)
    S->UsingOrNextShadow = FirstUsingShadow;
  FirstUsingShadow = S;
}

void UsingDecl::removeShadowDecl(UsingShadowDecl *S) {
  assert(std::find(shadow_begin(), shadow_end(), S) != shadow_end() &&
         "declaration not in set");
  assert(S->getUsingDecl() == this);

  // Remove S from the shadow decl chain. This is O(n) but hopefully rare.

  if (FirstUsingShadow == S) {
    FirstUsingShadow = dyn_cast<UsingShadowDecl>(S->UsingOrNextShadow);
    S->UsingOrNextShadow = this;
    return;
  }

  UsingShadowDecl *Prev = FirstUsingShadow;
  while (Prev->UsingOrNextShadow != S)
    Prev = cast<UsingShadowDecl>(Prev->UsingOrNextShadow);
  Prev->UsingOrNextShadow = S->UsingOrNextShadow;
  S->UsingOrNextShadow = this;
}

UsingDecl *UsingDecl::Create(ASTContext &C, DeclContext *DC,
                             SourceRange NNR, SourceLocation UL,
                             NestedNameSpecifier* TargetNNS,
                             const DeclarationNameInfo &NameInfo,
                             bool IsTypeNameArg) {
  return new (C) UsingDecl(DC, NNR, UL, TargetNNS, NameInfo, IsTypeNameArg);
}

UnresolvedUsingValueDecl *
UnresolvedUsingValueDecl::Create(ASTContext &C, DeclContext *DC,
                                 SourceLocation UsingLoc,
                                 SourceRange TargetNNR,
                                 NestedNameSpecifier *TargetNNS,
                                 const DeclarationNameInfo &NameInfo) {
  return new (C) UnresolvedUsingValueDecl(DC, C.DependentTy, UsingLoc,
                                          TargetNNR, TargetNNS, NameInfo);
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
