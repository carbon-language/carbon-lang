//===--- CGVtable.cpp - Emit LLVM Code for C++ vtables --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This contains code dealing with C++ code generation of virtual tables.
//
//===----------------------------------------------------------------------===//

#include "CodeGenModule.h"
#include "CodeGenFunction.h"

#include "clang/AST/RecordLayout.h"

using namespace clang;
using namespace CodeGen;

class VtableBuilder {
public:
  /// Index_t - Vtable index type.
  typedef uint64_t Index_t;
private:
  std::vector<llvm::Constant *> &methods;
  std::vector<llvm::Constant *> submethods;
  llvm::Type *Ptr8Ty;
  /// Class - The most derived class that this vtable is being built for.
  const CXXRecordDecl *Class;
  /// BLayout - Layout for the most derived class that this vtable is being
  /// built for.
  const ASTRecordLayout &BLayout;
  llvm::SmallSet<const CXXRecordDecl *, 32> IndirectPrimary;
  llvm::SmallSet<const CXXRecordDecl *, 32> SeenVBase;
  llvm::Constant *rtti;
  llvm::LLVMContext &VMContext;
  CodeGenModule &CGM;  // Per-module state.
  /// Index - Maps a method decl into a vtable index.  Useful for virtual
  /// dispatch codegen.
  llvm::DenseMap<const CXXMethodDecl *, Index_t> Index;
  llvm::DenseMap<const CXXMethodDecl *, Index_t> VCall;
  llvm::DenseMap<const CXXMethodDecl *, Index_t> VCallOffset;
  // This is the offset to the nearest virtual base
  llvm::DenseMap<const CXXMethodDecl *, Index_t> NonVirtualOffset;
  llvm::DenseMap<const CXXRecordDecl *, Index_t> VBIndex;

  typedef llvm::DenseMap<const CXXMethodDecl *, int> Pures_t;
  Pures_t Pures;
  typedef std::pair<Index_t, Index_t>  CallOffset;
  typedef llvm::DenseMap<const CXXMethodDecl *, CallOffset> Thunks_t;
  Thunks_t Thunks;
  typedef llvm::DenseMap<const CXXMethodDecl *,
                         std::pair<std::pair<CallOffset, CallOffset>,
                                   CanQualType> > CovariantThunks_t;
  CovariantThunks_t CovariantThunks;
  std::vector<Index_t> VCalls;

  typedef std::pair<const CXXRecordDecl *, uint64_t> CtorVtable_t;
  // CtorVtable - Used to hold the AddressPoints (offsets) into the built vtable
  // for use in computing the initializers for the VTT.
  llvm::DenseMap<CtorVtable_t, int64_t> &AddressPoints;

  typedef CXXRecordDecl::method_iterator method_iter;
  // FIXME: Linkage should follow vtable
  const bool Extern;
  const uint32_t LLVMPointerWidth;
  Index_t extra;
  typedef std::vector<std::pair<const CXXRecordDecl *, int64_t> > Path_t;
  llvm::Constant *cxa_pure;
public:
  VtableBuilder(std::vector<llvm::Constant *> &meth,
                const CXXRecordDecl *c,
                CodeGenModule &cgm)
    : methods(meth), Class(c), BLayout(cgm.getContext().getASTRecordLayout(c)),
      rtti(cgm.GenerateRtti(c)), VMContext(cgm.getModule().getContext()),
      CGM(cgm), AddressPoints(*new llvm::DenseMap<CtorVtable_t, int64_t>),
      Extern(true),
      LLVMPointerWidth(cgm.getContext().Target.getPointerWidth(0)) {
    Ptr8Ty = llvm::PointerType::get(llvm::Type::getInt8Ty(VMContext), 0);

    // Calculate pointer for ___cxa_pure_virtual.
    const llvm::FunctionType *FTy;
    std::vector<const llvm::Type*> ArgTys;
    const llvm::Type *ResultType = llvm::Type::getVoidTy(VMContext);
    FTy = llvm::FunctionType::get(ResultType, ArgTys, false);
    cxa_pure = wrap(CGM.CreateRuntimeFunction(FTy, "__cxa_pure_virtual"));
  }

  llvm::DenseMap<const CXXMethodDecl *, Index_t> &getIndex() { return Index; }
  llvm::DenseMap<const CXXRecordDecl *, Index_t> &getVBIndex()
    { return VBIndex; }

  llvm::DenseMap<CtorVtable_t, int64_t> *getAddressPoints()
    { return &AddressPoints; }

  llvm::Constant *wrap(Index_t i) {
    llvm::Constant *m;
    m = llvm::ConstantInt::get(llvm::Type::getInt64Ty(VMContext), i);
    return llvm::ConstantExpr::getIntToPtr(m, Ptr8Ty);
  }

  llvm::Constant *wrap(llvm::Constant *m) {
    return llvm::ConstantExpr::getBitCast(m, Ptr8Ty);
  }

#define D1(x)
//#define D1(X) do { if (getenv("DEBUG")) { X; } } while (0)

  void GenerateVBaseOffsets(const CXXRecordDecl *RD, uint64_t Offset,
                            bool updateVBIndex, Index_t current_vbindex) {
    for (CXXRecordDecl::base_class_const_iterator i = RD->bases_begin(),
           e = RD->bases_end(); i != e; ++i) {
      const CXXRecordDecl *Base =
        cast<CXXRecordDecl>(i->getType()->getAs<RecordType>()->getDecl());
      Index_t next_vbindex = current_vbindex;
      if (i->isVirtual() && !SeenVBase.count(Base)) {
        SeenVBase.insert(Base);
        if (updateVBIndex) {
          next_vbindex = (ssize_t)(-(VCalls.size()*LLVMPointerWidth/8)
                                   - 3*LLVMPointerWidth/8);
          VBIndex[Base] = next_vbindex;
        }
        int64_t BaseOffset = -(Offset/8) + BLayout.getVBaseClassOffset(Base)/8;
        VCalls.push_back((0?700:0) + BaseOffset);
        D1(printf("  vbase for %s at %d delta %d most derived %s\n",
                  Base->getNameAsCString(),
                  (int)-VCalls.size()-3, (int)BaseOffset,
                  Class->getNameAsCString()));
      }
      // We also record offsets for non-virtual bases to closest enclosing
      // virtual base.  We do this so that we don't have to search
      // for the nearst virtual base class when generating thunks.
      if (updateVBIndex && VBIndex.count(Base) == 0)
        VBIndex[Base] = next_vbindex;
      GenerateVBaseOffsets(Base, Offset, updateVBIndex, next_vbindex);
    }
  }

  void StartNewTable() {
    SeenVBase.clear();
  }

  Index_t VBlookup(CXXRecordDecl *D, CXXRecordDecl *B);

  Index_t getNVOffset_1(const CXXRecordDecl *D, const CXXRecordDecl *B,
    Index_t Offset = 0) {

    if (B == D)
      return Offset;

    const ASTRecordLayout &Layout = CGM.getContext().getASTRecordLayout(D);
    for (CXXRecordDecl::base_class_const_iterator i = D->bases_begin(),
           e = D->bases_end(); i != e; ++i) {
      const CXXRecordDecl *Base =
        cast<CXXRecordDecl>(i->getType()->getAs<RecordType>()->getDecl());
      int64_t BaseOffset = 0;
      if (!i->isVirtual())
        BaseOffset = Offset + Layout.getBaseClassOffset(Base);
      int64_t o = getNVOffset_1(Base, B, BaseOffset);
      if (o >= 0)
        return o;
    }

    return -1;
  }

  /// getNVOffset - Returns the non-virtual offset for the given (B) base of the
  /// derived class D.
  Index_t getNVOffset(QualType qB, QualType qD) {
    qD = qD->getPointeeType();
    qB = qB->getPointeeType();
    CXXRecordDecl *D = cast<CXXRecordDecl>(qD->getAs<RecordType>()->getDecl());
    CXXRecordDecl *B = cast<CXXRecordDecl>(qB->getAs<RecordType>()->getDecl());
    int64_t o = getNVOffset_1(D, B);
    if (o >= 0)
      return o;

    assert(false && "FIXME: non-virtual base not found");
    return 0;
  }

  /// getVbaseOffset - Returns the index into the vtable for the virtual base
  /// offset for the given (B) virtual base of the derived class D.
  Index_t getVbaseOffset(QualType qB, QualType qD) {
    qD = qD->getPointeeType();
    qB = qB->getPointeeType();
    CXXRecordDecl *D = cast<CXXRecordDecl>(qD->getAs<RecordType>()->getDecl());
    CXXRecordDecl *B = cast<CXXRecordDecl>(qB->getAs<RecordType>()->getDecl());
    if (D != Class)
      return VBlookup(D, B);
    llvm::DenseMap<const CXXRecordDecl *, Index_t>::iterator i;
    i = VBIndex.find(B);
    if (i != VBIndex.end())
      return i->second;

    assert(false && "FIXME: Base not found");
    return 0;
  }

  bool OverrideMethod(const CXXMethodDecl *MD, llvm::Constant *m,
                      bool MorallyVirtual, Index_t OverrideOffset,
                      Index_t Offset, int64_t CurrentVBaseOffset) {
    const bool isPure = MD->isPure();
    typedef CXXMethodDecl::method_iterator meth_iter;
    // FIXME: Should OverrideOffset's be Offset?

    // FIXME: Don't like the nested loops.  For very large inheritance
    // heirarchies we could have a table on the side with the final overridder
    // and just replace each instance of an overridden method once.  Would be
    // nice to measure the cost/benefit on real code.

    for (meth_iter mi = MD->begin_overridden_methods(),
           e = MD->end_overridden_methods();
         mi != e; ++mi) {
      const CXXMethodDecl *OMD = *mi;
      llvm::Constant *om;
      om = CGM.GetAddrOfFunction(OMD, Ptr8Ty);
      om = llvm::ConstantExpr::getBitCast(om, Ptr8Ty);

      for (Index_t i = 0, e = submethods.size();
           i != e; ++i) {
        // FIXME: begin_overridden_methods might be too lax, covariance */
        if (submethods[i] != om)
          continue;
        QualType nc_oret = OMD->getType()->getAs<FunctionType>()->getResultType();
        CanQualType oret = CGM.getContext().getCanonicalType(nc_oret);
        QualType nc_ret = MD->getType()->getAs<FunctionType>()->getResultType();
        CanQualType ret = CGM.getContext().getCanonicalType(nc_ret);
        CallOffset ReturnOffset = std::make_pair(0, 0);
        if (oret != ret) {
          // FIXME: calculate offsets for covariance
          if (CovariantThunks.count(OMD)) {
            oret = CovariantThunks[OMD].second;
            CovariantThunks.erase(OMD);
          }
          // FIXME: Double check oret
          Index_t nv = getNVOffset(oret, ret)/8;
          ReturnOffset = std::make_pair(nv, getVbaseOffset(oret, ret));
        }
        Index[MD] = i;
        submethods[i] = m;
        if (isPure)
          Pures[MD] = 1;
        Pures.erase(OMD);
        Thunks.erase(OMD);
        if (MorallyVirtual || VCall.count(OMD)) {
          Index_t &idx = VCall[OMD];
          if (idx == 0) {
            NonVirtualOffset[MD] = -OverrideOffset/8 + CurrentVBaseOffset/8;
            VCallOffset[MD] = OverrideOffset/8;
            idx = VCalls.size()+1;
            VCalls.push_back(0);
            D1(printf("  vcall for %s at %d with delta %d most derived %s\n",
                      MD->getNameAsCString(), (int)-idx-3, (int)VCalls[idx-1],
                      Class->getNameAsCString()));
          } else {
            NonVirtualOffset[MD] = NonVirtualOffset[OMD];
            VCallOffset[MD] = VCallOffset[OMD];
            VCalls[idx-1] = -VCallOffset[OMD] + OverrideOffset/8;
            D1(printf("  vcall patch for %s at %d with delta %d most derived %s\n",
                      MD->getNameAsCString(), (int)-idx-3, (int)VCalls[idx-1],
                      Class->getNameAsCString()));
          }
          VCall[MD] = idx;
          int64_t O = NonVirtualOffset[MD];
          int v = -((idx+extra+2)*LLVMPointerWidth/8);
          // Optimize out virtual adjustments of 0.
          if (VCalls[idx-1] == 0)
            v = 0;
          CallOffset ThisOffset = std::make_pair(O, v);
          // FIXME: Do we always have to build a covariant thunk to save oret,
          // which is the containing virtual base class?
          if (ReturnOffset.first || ReturnOffset.second)
            CovariantThunks[MD] = std::make_pair(std::make_pair(ThisOffset,
                                                                ReturnOffset),
                                                 oret);
          else if (!isPure && (ThisOffset.first || ThisOffset.second))
            Thunks[MD] = ThisOffset;
          return true;
        }

        // FIXME: finish off
        int64_t O = VCallOffset[OMD] - OverrideOffset/8;

        if (O || ReturnOffset.first || ReturnOffset.second) {
          CallOffset ThisOffset = std::make_pair(O, 0);
          
          if (ReturnOffset.first || ReturnOffset.second)
            CovariantThunks[MD] = std::make_pair(std::make_pair(ThisOffset,
                                                                ReturnOffset),
                                                 oret);
          else if (!isPure)
            Thunks[MD] = ThisOffset;
        }
        return true;
      }
    }

    return false;
  }

  void InstallThunks() {
    for (Thunks_t::iterator i = Thunks.begin(), e = Thunks.end();
         i != e; ++i) {
      const CXXMethodDecl *MD = i->first;
      assert(!MD->isPure() && "Trying to thunk a pure");
      Index_t idx = Index[MD];
      Index_t nv_O = i->second.first;
      Index_t v_O = i->second.second;
      submethods[idx] = CGM.BuildThunk(MD, Extern, nv_O, v_O);
    }
    Thunks.clear();
    for (CovariantThunks_t::iterator i = CovariantThunks.begin(),
           e = CovariantThunks.end();
         i != e; ++i) {
      const CXXMethodDecl *MD = i->first;
      if (MD->isPure())
        continue;
      Index_t idx = Index[MD];
      Index_t nv_t = i->second.first.first.first;
      Index_t v_t = i->second.first.first.second;
      Index_t nv_r = i->second.first.second.first;
      Index_t v_r = i->second.first.second.second;
      submethods[idx] = CGM.BuildCovariantThunk(MD, Extern, nv_t, v_t, nv_r,
                                                v_r);
    }
    CovariantThunks.clear();
    for (Pures_t::iterator i = Pures.begin(), e = Pures.end();
         i != e; ++i) {
      const CXXMethodDecl *MD = i->first;
      Index_t idx = Index[MD];
      submethods[idx] = cxa_pure;
    }
    Pures.clear();
  }

  llvm::Constant *WrapAddrOf(const CXXMethodDecl *MD) {
    if (const CXXDestructorDecl *Dtor = dyn_cast<CXXDestructorDecl>(MD))
      return wrap(CGM.GetAddrOfCXXDestructor(Dtor, Dtor_Complete));

    const FunctionProtoType *FPT = MD->getType()->getAs<FunctionProtoType>();
    const llvm::Type *Ty =
      CGM.getTypes().GetFunctionType(CGM.getTypes().getFunctionInfo(MD),
                                     FPT->isVariadic());

    return wrap(CGM.GetAddrOfFunction(MD, Ty));
  }

  void OverrideMethods(Path_t *Path, bool MorallyVirtual, int64_t Offset,
                       int64_t CurrentVBaseOffset) {
    for (Path_t::reverse_iterator i = Path->rbegin(),
           e = Path->rend(); i != e; ++i) {
      const CXXRecordDecl *RD = i->first;
      int64_t OverrideOffset = i->second;
      for (method_iter mi = RD->method_begin(), me = RD->method_end(); mi != me;
           ++mi) {
        if (!mi->isVirtual())
          continue;

        const CXXMethodDecl *MD = *mi;
        llvm::Constant *m = WrapAddrOf(MD);
        OverrideMethod(MD, m, MorallyVirtual, OverrideOffset, Offset,
                       CurrentVBaseOffset);
      }
    }
  }

  void AddMethod(const CXXMethodDecl *MD, bool MorallyVirtual, Index_t Offset,
                 bool ForVirtualBase, int64_t CurrentVBaseOffset) {
    llvm::Constant *m = WrapAddrOf(MD);

    // If we can find a previously allocated slot for this, reuse it.
    if (OverrideMethod(MD, m, MorallyVirtual, Offset, Offset,
                       CurrentVBaseOffset))
      return;

    // else allocate a new slot.
    Index[MD] = submethods.size();
    submethods.push_back(m);
    D1(printf("  vfn for %s at %d\n", MD->getNameAsCString(), (int)Index[MD]));
    if (MD->isPure())
      Pures[MD] = 1;
    if (MorallyVirtual) {
      VCallOffset[MD] = Offset/8;
      Index_t &idx = VCall[MD];
      // Allocate the first one, after that, we reuse the previous one.
      if (idx == 0) {
        NonVirtualOffset[MD] = CurrentVBaseOffset/8 - Offset/8;
        idx = VCalls.size()+1;
        VCalls.push_back(0);
        D1(printf("  vcall for %s at %d with delta %d\n",
                  MD->getNameAsCString(), (int)-VCalls.size()-3, 0));
      }
    }
  }

  void AddMethods(const CXXRecordDecl *RD, bool MorallyVirtual,
                  Index_t Offset, bool RDisVirtualBase,
                  int64_t CurrentVBaseOffset) {
    for (method_iter mi = RD->method_begin(), me = RD->method_end(); mi != me;
         ++mi)
      if (mi->isVirtual())
        AddMethod(*mi, MorallyVirtual, Offset, RDisVirtualBase,
                  CurrentVBaseOffset);
  }

  void NonVirtualBases(const CXXRecordDecl *RD, const ASTRecordLayout &Layout,
                       const CXXRecordDecl *PrimaryBase,
                       bool PrimaryBaseWasVirtual, bool MorallyVirtual,
                       int64_t Offset, int64_t CurrentVBaseOffset,
                       Path_t *Path) {
    Path->push_back(std::make_pair(RD, Offset));
    for (CXXRecordDecl::base_class_const_iterator i = RD->bases_begin(),
           e = RD->bases_end(); i != e; ++i) {
      if (i->isVirtual())
        continue;
      const CXXRecordDecl *Base =
        cast<CXXRecordDecl>(i->getType()->getAs<RecordType>()->getDecl());
      if (Base != PrimaryBase || PrimaryBaseWasVirtual) {
        uint64_t o = Offset + Layout.getBaseClassOffset(Base);
        StartNewTable();
        GenerateVtableForBase(Base, MorallyVirtual, o, false,
                              CurrentVBaseOffset, Path);
      }
    }
    Path->pop_back();
  }

// #define D(X) do { X; } while (0)
#define D(X)

  void insertVCalls(int InsertionPoint) {
    llvm::Constant *e = 0;
    D1(printf("============= combining vbase/vcall\n"));
    D(VCalls.insert(VCalls.begin(), 673));
    D(VCalls.push_back(672));
    methods.insert(methods.begin() + InsertionPoint, VCalls.size(), e);
    // The vcalls come first...
    for (std::vector<Index_t>::reverse_iterator i = VCalls.rbegin(),
           e = VCalls.rend();
         i != e; ++i)
      methods[InsertionPoint++] = wrap((0?600:0) + *i);
    VCalls.clear();
    VCall.clear();
  }

  Index_t end(const CXXRecordDecl *RD, const ASTRecordLayout &Layout,
              const CXXRecordDecl *PrimaryBase, bool PrimaryBaseWasVirtual,
              bool MorallyVirtual, int64_t Offset, bool ForVirtualBase,
              int64_t CurrentVBaseOffset,
              Path_t *Path) {
    bool alloc = false;
    if (Path == 0) {
      alloc = true;
      Path = new Path_t;
    }

    StartNewTable();
    extra = 0;
    bool DeferVCalls = MorallyVirtual || ForVirtualBase;
    int VCallInsertionPoint = methods.size();
    if (!DeferVCalls) {
      insertVCalls(VCallInsertionPoint);
    } else
      // FIXME: just for extra, or for all uses of VCalls.size post this?
      extra = -VCalls.size();

    methods.push_back(wrap(-(Offset/8)));
    methods.push_back(rtti);
    Index_t AddressPoint = methods.size();

    InstallThunks();
    D1(printf("============= combining methods\n"));
    methods.insert(methods.end(), submethods.begin(), submethods.end());
    submethods.clear();

    // and then the non-virtual bases.
    NonVirtualBases(RD, Layout, PrimaryBase, PrimaryBaseWasVirtual,
                    MorallyVirtual, Offset, CurrentVBaseOffset, Path);

    if (ForVirtualBase) {
      // FIXME: We're adding to VCalls in callers, we need to do the overrides
      // in the inner part, so that we know the complete set of vcalls during
      // the build and don't have to insert into methods.  Saving out the
      // AddressPoint here, would need to be fixed, if we didn't do that.  Also
      // retroactively adding vcalls for overrides later wind up in the wrong
      // place, the vcall slot has to be alloted during the walk of the base
      // when the function is first introduces.
      AddressPoint += VCalls.size();
      insertVCalls(VCallInsertionPoint);
    }
    
    if (MorallyVirtual) {
      D1(printf("XXX address point for %s in %s at offset %d is %d\n",
                RD->getNameAsCString(), Class->getNameAsCString(),
                (int)Offset, (int)AddressPoint));
      AddressPoints[std::make_pair(RD, Offset)] = AddressPoint;
      
    }

    if (alloc) {
      delete Path;
    }
    return AddressPoint;
  }

  void Primaries(const CXXRecordDecl *RD, bool MorallyVirtual, int64_t Offset,
                 bool updateVBIndex, Index_t current_vbindex,
                 bool RDisVirtualBase, int64_t CurrentVBaseOffset) {
    if (!RD->isDynamicClass())
      return;

    const ASTRecordLayout &Layout = CGM.getContext().getASTRecordLayout(RD);
    const CXXRecordDecl *PrimaryBase = Layout.getPrimaryBase();
    const bool PrimaryBaseWasVirtual = Layout.getPrimaryBaseWasVirtual();

    // vtables are composed from the chain of primaries.
    if (PrimaryBase) {
      D1(printf(" doing primaries for %s most derived %s\n",
                RD->getNameAsCString(), Class->getNameAsCString()));
      
      int BaseCurrentVBaseOffset = CurrentVBaseOffset;
      if (PrimaryBaseWasVirtual)
        BaseCurrentVBaseOffset = BLayout.getVBaseClassOffset(PrimaryBase);
        
      if (!PrimaryBaseWasVirtual)
        Primaries(PrimaryBase, PrimaryBaseWasVirtual|MorallyVirtual, Offset,
                  updateVBIndex, current_vbindex, PrimaryBaseWasVirtual,
                  BaseCurrentVBaseOffset);
    }

    D1(printf(" doing vcall entries for %s most derived %s\n",
              RD->getNameAsCString(), Class->getNameAsCString()));

    // And add the virtuals for the class to the primary vtable.
    AddMethods(RD, MorallyVirtual, Offset, RDisVirtualBase, CurrentVBaseOffset);
  }

  void VBPrimaries(const CXXRecordDecl *RD, bool MorallyVirtual, int64_t Offset,
                   bool updateVBIndex, Index_t current_vbindex,
                   bool RDisVirtualBase, int64_t CurrentVBaseOffset,
                   bool bottom=false) {
    if (!RD->isDynamicClass())
      return;

    const ASTRecordLayout &Layout = CGM.getContext().getASTRecordLayout(RD);
    const CXXRecordDecl *PrimaryBase = Layout.getPrimaryBase();
    const bool PrimaryBaseWasVirtual = Layout.getPrimaryBaseWasVirtual();

    // vtables are composed from the chain of primaries.
    if (PrimaryBase) {
      int BaseCurrentVBaseOffset = CurrentVBaseOffset;
      if (PrimaryBaseWasVirtual) {
        IndirectPrimary.insert(PrimaryBase);
        BaseCurrentVBaseOffset = BLayout.getVBaseClassOffset(PrimaryBase);
      }

      D1(printf(" doing primaries for %s most derived %s\n",
                RD->getNameAsCString(), Class->getNameAsCString()));
      
      VBPrimaries(PrimaryBase, PrimaryBaseWasVirtual|MorallyVirtual, Offset,
                  updateVBIndex, current_vbindex, PrimaryBaseWasVirtual,
                  BaseCurrentVBaseOffset);
    }

    D1(printf(" doing vbase entries for %s most derived %s\n",
              RD->getNameAsCString(), Class->getNameAsCString()));
    GenerateVBaseOffsets(RD, Offset, updateVBIndex, current_vbindex);

    if (RDisVirtualBase || bottom) {
      Primaries(RD, MorallyVirtual, Offset, updateVBIndex, current_vbindex,
                RDisVirtualBase, CurrentVBaseOffset);
    }
  }

  int64_t GenerateVtableForBase(const CXXRecordDecl *RD,
                                bool MorallyVirtual = false, int64_t Offset = 0,
                                bool ForVirtualBase = false,
                                int CurrentVBaseOffset = 0,
                                Path_t *Path = 0) {
    if (!RD->isDynamicClass())
      return 0;

    const ASTRecordLayout &Layout = CGM.getContext().getASTRecordLayout(RD);
    const CXXRecordDecl *PrimaryBase = Layout.getPrimaryBase();
    const bool PrimaryBaseWasVirtual = Layout.getPrimaryBaseWasVirtual();

    extra = 0;
    D1(printf("building entries for base %s most derived %s\n",
              RD->getNameAsCString(), Class->getNameAsCString()));

    if (ForVirtualBase)
      extra = VCalls.size();

    VBPrimaries(RD, MorallyVirtual, Offset, !ForVirtualBase, 0, ForVirtualBase,
                CurrentVBaseOffset, true);

    if (Path)
      OverrideMethods(Path, MorallyVirtual, Offset, CurrentVBaseOffset);

    return end(RD, Layout, PrimaryBase, PrimaryBaseWasVirtual, MorallyVirtual,
               Offset, ForVirtualBase, CurrentVBaseOffset, Path);
  }

  void GenerateVtableForVBases(const CXXRecordDecl *RD,
                               int64_t Offset = 0,
                               Path_t *Path = 0) {
    bool alloc = false;
    if (Path == 0) {
      alloc = true;
      Path = new Path_t;
    }
    // FIXME: We also need to override using all paths to a virtual base,
    // right now, we just process the first path
    Path->push_back(std::make_pair(RD, Offset));
    for (CXXRecordDecl::base_class_const_iterator i = RD->bases_begin(),
           e = RD->bases_end(); i != e; ++i) {
      const CXXRecordDecl *Base =
        cast<CXXRecordDecl>(i->getType()->getAs<RecordType>()->getDecl());
      if (i->isVirtual() && !IndirectPrimary.count(Base)) {
        // Mark it so we don't output it twice.
        IndirectPrimary.insert(Base);
        StartNewTable();
        VCall.clear();
        int64_t BaseOffset = BLayout.getVBaseClassOffset(Base);
        int64_t CurrentVBaseOffset = BaseOffset;
        D1(printf("vtable %s virtual base %s\n",
                  Class->getNameAsCString(), Base->getNameAsCString()));
        GenerateVtableForBase(Base, true, BaseOffset, true, CurrentVBaseOffset,
                              Path);
      }
      int64_t BaseOffset;
      if (i->isVirtual())
        BaseOffset = BLayout.getVBaseClassOffset(Base);
      else {
        const ASTRecordLayout &Layout = CGM.getContext().getASTRecordLayout(RD);
        BaseOffset = Offset + Layout.getBaseClassOffset(Base);
      }
        
      if (Base->getNumVBases()) {
        GenerateVtableForVBases(Base, BaseOffset, Path);
      }
    }
    Path->pop_back();
    if (alloc)
      delete Path;
  }
};


VtableBuilder::Index_t VtableBuilder::VBlookup(CXXRecordDecl *D,
                                               CXXRecordDecl *B) {
  return CGM.getVtableInfo().getVirtualBaseOffsetIndex(D, B);
}

int64_t CGVtableInfo::getMethodVtableIndex(const CXXMethodDecl *MD) {
  MD = MD->getCanonicalDecl();

  MethodVtableIndicesTy::iterator I = MethodVtableIndices.find(MD);
  if (I != MethodVtableIndices.end())
    return I->second;
  
  const CXXRecordDecl *RD = MD->getParent();
  
  std::vector<llvm::Constant *> methods;
  // FIXME: This seems expensive.  Can we do a partial job to get
  // just this data.
  VtableBuilder b(methods, RD, CGM);
  D1(printf("vtable %s\n", RD->getNameAsCString()));
  b.GenerateVtableForBase(RD);
  b.GenerateVtableForVBases(RD);
  
  MethodVtableIndices.insert(b.getIndex().begin(),
                             b.getIndex().end());
  
  I = MethodVtableIndices.find(MD);
  assert(I != MethodVtableIndices.end() && "Did not find index!");
  return I->second;
}

int64_t CGVtableInfo::getVirtualBaseOffsetIndex(const CXXRecordDecl *RD, 
                                                const CXXRecordDecl *VBase) {
  ClassPairTy ClassPair(RD, VBase);
  
  VirtualBaseClassIndiciesTy::iterator I = 
    VirtualBaseClassIndicies.find(ClassPair);
  if (I != VirtualBaseClassIndicies.end())
    return I->second;
  
  std::vector<llvm::Constant *> methods;
  // FIXME: This seems expensive.  Can we do a partial job to get
  // just this data.
  VtableBuilder b(methods, RD, CGM);
  D1(printf("vtable %s\n", RD->getNameAsCString()));
  b.GenerateVtableForBase(RD);
  b.GenerateVtableForVBases(RD);
  
  for (llvm::DenseMap<const CXXRecordDecl *, uint64_t>::iterator I =
       b.getVBIndex().begin(), E = b.getVBIndex().end(); I != E; ++I) {
    // Insert all types.
    ClassPairTy ClassPair(RD, I->first);
    
    VirtualBaseClassIndicies.insert(std::make_pair(ClassPair, I->second));
  }
  
  I = VirtualBaseClassIndicies.find(ClassPair);
  assert(I != VirtualBaseClassIndicies.end() && "Did not find index!");
  
  return I->second;
}

llvm::Constant *CodeGenModule::GenerateVtable(const CXXRecordDecl *LayoutClass,
                                              const CXXRecordDecl *RD,
                                              uint64_t Offset) {
  llvm::SmallString<256> OutName;
  llvm::raw_svector_ostream Out(OutName);
  if (LayoutClass != RD)
    mangleCXXCtorVtable(getMangleContext(), LayoutClass, Offset/8, RD, Out);
  else
    mangleCXXVtable(getMangleContext(), RD, Out);

  llvm::GlobalVariable::LinkageTypes linktype;
  linktype = llvm::GlobalValue::LinkOnceODRLinkage;
  std::vector<llvm::Constant *> methods;
  llvm::Type *Ptr8Ty=llvm::PointerType::get(llvm::Type::getInt8Ty(VMContext),0);
  int64_t AddressPoint;

  VtableBuilder b(methods, RD, *this);

  D1(printf("vtable %s\n", RD->getNameAsCString()));
  // First comes the vtables for all the non-virtual bases...
  AddressPoint = b.GenerateVtableForBase(RD);

  // then the vtables for all the virtual bases.
  b.GenerateVtableForVBases(RD);

  if (LayoutClass == RD)
    AddressPoints[RD] = b.getAddressPoints();

  llvm::Constant *C;
  llvm::ArrayType *type = llvm::ArrayType::get(Ptr8Ty, methods.size());
  C = llvm::ConstantArray::get(type, methods);
  llvm::Constant *vtable = new llvm::GlobalVariable(getModule(), type,
                                                    true, linktype, C,
                                                    Out.str());
  vtable = llvm::ConstantExpr::getBitCast(vtable, Ptr8Ty);
  llvm::Constant *AddressPointC;
  uint32_t LLVMPointerWidth = getContext().Target.getPointerWidth(0);
  AddressPointC = llvm::ConstantInt::get(llvm::Type::getInt64Ty(VMContext),
                                         AddressPoint*LLVMPointerWidth/8);
  vtable = llvm::ConstantExpr::getInBoundsGetElementPtr(vtable, &AddressPointC,
                                                        1);

  return vtable;
}

class VTTBuilder {
  /// Inits - The list of values built for the VTT.
  std::vector<llvm::Constant *> &Inits;
  /// Class - The most derived class that this vtable is being built for.
  const CXXRecordDecl *Class;
  CodeGenModule &CGM;  // Per-module state.
  llvm::SmallSet<const CXXRecordDecl *, 32> SeenVBase;
  /// BLayout - Layout for the most derived class that this vtable is being
  /// built for.
  const ASTRecordLayout &BLayout;
  // vtbl - A pointer to the vtable for Class.
  llvm::Constant *ClassVtbl;
  llvm::LLVMContext &VMContext;

  /// BuildVtablePtr - Build up a referene to the given secondary vtable
  llvm::Constant *BuildVtablePtr(llvm::Constant *vtbl, const CXXRecordDecl *RD,
                                 uint64_t Offset) {
    int64_t AddressPoint;
    AddressPoint = (*CGM.AddressPoints[Class])[std::make_pair(RD, Offset)];
    D1(printf("XXX address point for %s in %s at offset %d was %d\n",
              RD->getNameAsCString(), Class->getNameAsCString(),
              (int)Offset, (int)AddressPoint));
    uint32_t LLVMPointerWidth = CGM.getContext().Target.getPointerWidth(0);
    llvm::Constant *init;
    init = llvm::ConstantInt::get(llvm::Type::getInt64Ty(VMContext),
                                  AddressPoint*LLVMPointerWidth/8);
    init = llvm::ConstantExpr::getInBoundsGetElementPtr(vtbl, &init, 1);
    return init;
  }

  /// Secondary - Add the secondary vtable pointers to Inits.  Offset is the
  /// current offset in bits to the object we're working on.
  void Secondary(const CXXRecordDecl *RD, llvm::Constant *vtbl,
                 uint64_t Offset=0, bool MorallyVirtual=false) {
    if (RD->getNumVBases() == 0 && ! MorallyVirtual)
      return;

    for (CXXRecordDecl::base_class_const_iterator i = RD->bases_begin(),
           e = RD->bases_end(); i != e; ++i) {
      const CXXRecordDecl *Base =
        cast<CXXRecordDecl>(i->getType()->getAs<RecordType>()->getDecl());
      const ASTRecordLayout &Layout = CGM.getContext().getASTRecordLayout(RD);
      const CXXRecordDecl *PrimaryBase = Layout.getPrimaryBase();
      const bool PrimaryBaseWasVirtual = Layout.getPrimaryBaseWasVirtual();
      bool NonVirtualPrimaryBase;
      NonVirtualPrimaryBase = !PrimaryBaseWasVirtual && Base == PrimaryBase;
      bool BaseMorallyVirtual = MorallyVirtual | i->isVirtual();
      uint64_t BaseOffset;
      if (!i->isVirtual()) {
        const ASTRecordLayout &Layout = CGM.getContext().getASTRecordLayout(RD);
        BaseOffset = Offset + Layout.getBaseClassOffset(Base);
      } else
        BaseOffset = BLayout.getVBaseClassOffset(Base);
      if ((Base->getNumVBases() || BaseMorallyVirtual)
          && !NonVirtualPrimaryBase) {
        // FIXME: Slightly too many of these for __ZTT8test8_B2
        llvm::Constant *init;
        if (MorallyVirtual)
          init = BuildVtablePtr(vtbl, RD, Offset);
        else
          init = CGM.getVtableInfo().getCtorVtable(Class, Base, BaseOffset);
        Inits.push_back(init);
        // vtbl = dyn_cast<llvm::Constant>(init->getOperand(0));
      }
      Secondary(Base, vtbl, BaseOffset, BaseMorallyVirtual);
    }
  }

  /// BuiltVTT - Add the VTT to Inits.  Offset is the offset in bits to the
  /// currnet object we're working on.
  void BuildVTT(const CXXRecordDecl *RD, uint64_t Offset, bool MorallyVirtual) {
    if (RD->getNumVBases() == 0 && !MorallyVirtual)
      return;

    llvm::Constant *init;
    // First comes the primary virtual table pointer...
    if (MorallyVirtual)
      init = BuildVtablePtr(ClassVtbl, RD, Offset);
    else
      init = CGM.getVtableInfo().getCtorVtable(Class, RD, Offset);
    llvm::Constant *vtbl = dyn_cast<llvm::Constant>(init->getOperand(0));
    Inits.push_back(init);

    // then the secondary VTTs....
    SecondaryVTTs(RD, Offset, MorallyVirtual);

    // and last the secondary vtable pointers.
    Secondary(RD, vtbl, MorallyVirtual, Offset);
  }

  /// SecondaryVTTs - Add the secondary VTTs to Inits.  The secondary VTTs are
  /// built from each direct non-virtual proper base that requires a VTT in
  /// declaration order.
  void SecondaryVTTs(const CXXRecordDecl *RD, uint64_t Offset=0,
                     bool MorallyVirtual=false) {
    for (CXXRecordDecl::base_class_const_iterator i = RD->bases_begin(),
           e = RD->bases_end(); i != e; ++i) {
      const CXXRecordDecl *Base =
        cast<CXXRecordDecl>(i->getType()->getAs<RecordType>()->getDecl());
      if (i->isVirtual())
        continue;
      const ASTRecordLayout &Layout = CGM.getContext().getASTRecordLayout(RD);
      uint64_t BaseOffset = Offset + Layout.getBaseClassOffset(Base);
      BuildVTT(Base, BaseOffset, MorallyVirtual);
    }
  }

  /// VirtualVTTs - Add the VTT for each proper virtual base in inheritance
  /// graph preorder.
  void VirtualVTTs(const CXXRecordDecl *RD) {
    for (CXXRecordDecl::base_class_const_iterator i = RD->bases_begin(),
           e = RD->bases_end(); i != e; ++i) {
      const CXXRecordDecl *Base =
        cast<CXXRecordDecl>(i->getType()->getAs<RecordType>()->getDecl());
      if (i->isVirtual() && !SeenVBase.count(Base)) {
        SeenVBase.insert(Base);
        uint64_t BaseOffset = BLayout.getVBaseClassOffset(Base);
        BuildVTT(Base, BaseOffset, true);
      }
      VirtualVTTs(Base);
    }
  }
public:
  VTTBuilder(std::vector<llvm::Constant *> &inits, const CXXRecordDecl *c,
             CodeGenModule &cgm)
    : Inits(inits), Class(c), CGM(cgm),
      BLayout(cgm.getContext().getASTRecordLayout(c)),
      VMContext(cgm.getModule().getContext()) {
    
    // First comes the primary virtual table pointer for the complete class...
    ClassVtbl = CGM.getVtableInfo().getVtable(Class);
    Inits.push_back(ClassVtbl);
    ClassVtbl = dyn_cast<llvm::Constant>(ClassVtbl->getOperand(0));
    
    // then the secondary VTTs...
    SecondaryVTTs(Class);

    // then the secondary vtable pointers...
    Secondary(Class, ClassVtbl);

    // and last, the virtual VTTs.
    VirtualVTTs(Class);
  }
};

llvm::Constant *CodeGenModule::GenerateVTT(const CXXRecordDecl *RD) {
  // Only classes that have virtual bases need a VTT.
  if (RD->getNumVBases() == 0)
    return 0;

  llvm::SmallString<256> OutName;
  llvm::raw_svector_ostream Out(OutName);
  mangleCXXVTT(getMangleContext(), RD, Out);

  llvm::GlobalVariable::LinkageTypes linktype;
  linktype = llvm::GlobalValue::LinkOnceODRLinkage;
  std::vector<llvm::Constant *> inits;
  llvm::Type *Ptr8Ty=llvm::PointerType::get(llvm::Type::getInt8Ty(VMContext),0);

  D1(printf("vtt %s\n", RD->getNameAsCString()));

  VTTBuilder b(inits, RD, *this);

  llvm::Constant *C;
  llvm::ArrayType *type = llvm::ArrayType::get(Ptr8Ty, inits.size());
  C = llvm::ConstantArray::get(type, inits);
  llvm::Constant *vtt = new llvm::GlobalVariable(getModule(), type, true,
                                                 linktype, C, Out.str());
  vtt = llvm::ConstantExpr::getBitCast(vtt, Ptr8Ty);
  return vtt;
}

llvm::Constant *CGVtableInfo::getVtable(const CXXRecordDecl *RD) {
  llvm::Constant *&vtbl = Vtables[RD];
  if (vtbl)
    return vtbl;
  vtbl = CGM.GenerateVtable(RD, RD);
  CGM.GenerateVTT(RD);
  return vtbl;
}

llvm::Constant *CGVtableInfo::getCtorVtable(const CXXRecordDecl *LayoutClass,
                                            const CXXRecordDecl *RD,
                                            uint64_t Offset) {
  return CGM.GenerateVtable(LayoutClass, RD, Offset);
}
