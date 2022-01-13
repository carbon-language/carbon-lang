//===----- CGOpenMPRuntime.cpp - Interface to OpenMP Runtimes -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides a class for OpenMP runtime code generation.
//
//===----------------------------------------------------------------------===//

#include "CGOpenMPRuntime.h"
#include "CGCXXABI.h"
#include "CGCleanup.h"
#include "CGRecordLayout.h"
#include "CodeGenFunction.h"
#include "TargetInfo.h"
#include "clang/AST/APValue.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/OpenMPClause.h"
#include "clang/AST/StmtOpenMP.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Basic/BitmaskEnum.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/OpenMPKinds.h"
#include "clang/Basic/SourceManager.h"
#include "clang/CodeGen/ConstantInitBuilder.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/AtomicOrdering.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <numeric>

using namespace clang;
using namespace CodeGen;
using namespace llvm::omp;

namespace {
/// Base class for handling code generation inside OpenMP regions.
class CGOpenMPRegionInfo : public CodeGenFunction::CGCapturedStmtInfo {
public:
  /// Kinds of OpenMP regions used in codegen.
  enum CGOpenMPRegionKind {
    /// Region with outlined function for standalone 'parallel'
    /// directive.
    ParallelOutlinedRegion,
    /// Region with outlined function for standalone 'task' directive.
    TaskOutlinedRegion,
    /// Region for constructs that do not require function outlining,
    /// like 'for', 'sections', 'atomic' etc. directives.
    InlinedRegion,
    /// Region with outlined function for standalone 'target' directive.
    TargetRegion,
  };

  CGOpenMPRegionInfo(const CapturedStmt &CS,
                     const CGOpenMPRegionKind RegionKind,
                     const RegionCodeGenTy &CodeGen, OpenMPDirectiveKind Kind,
                     bool HasCancel)
      : CGCapturedStmtInfo(CS, CR_OpenMP), RegionKind(RegionKind),
        CodeGen(CodeGen), Kind(Kind), HasCancel(HasCancel) {}

  CGOpenMPRegionInfo(const CGOpenMPRegionKind RegionKind,
                     const RegionCodeGenTy &CodeGen, OpenMPDirectiveKind Kind,
                     bool HasCancel)
      : CGCapturedStmtInfo(CR_OpenMP), RegionKind(RegionKind), CodeGen(CodeGen),
        Kind(Kind), HasCancel(HasCancel) {}

  /// Get a variable or parameter for storing global thread id
  /// inside OpenMP construct.
  virtual const VarDecl *getThreadIDVariable() const = 0;

  /// Emit the captured statement body.
  void EmitBody(CodeGenFunction &CGF, const Stmt *S) override;

  /// Get an LValue for the current ThreadID variable.
  /// \return LValue for thread id variable. This LValue always has type int32*.
  virtual LValue getThreadIDVariableLValue(CodeGenFunction &CGF);

  virtual void emitUntiedSwitch(CodeGenFunction & /*CGF*/) {}

  CGOpenMPRegionKind getRegionKind() const { return RegionKind; }

  OpenMPDirectiveKind getDirectiveKind() const { return Kind; }

  bool hasCancel() const { return HasCancel; }

  static bool classof(const CGCapturedStmtInfo *Info) {
    return Info->getKind() == CR_OpenMP;
  }

  ~CGOpenMPRegionInfo() override = default;

protected:
  CGOpenMPRegionKind RegionKind;
  RegionCodeGenTy CodeGen;
  OpenMPDirectiveKind Kind;
  bool HasCancel;
};

/// API for captured statement code generation in OpenMP constructs.
class CGOpenMPOutlinedRegionInfo final : public CGOpenMPRegionInfo {
public:
  CGOpenMPOutlinedRegionInfo(const CapturedStmt &CS, const VarDecl *ThreadIDVar,
                             const RegionCodeGenTy &CodeGen,
                             OpenMPDirectiveKind Kind, bool HasCancel,
                             StringRef HelperName)
      : CGOpenMPRegionInfo(CS, ParallelOutlinedRegion, CodeGen, Kind,
                           HasCancel),
        ThreadIDVar(ThreadIDVar), HelperName(HelperName) {
    assert(ThreadIDVar != nullptr && "No ThreadID in OpenMP region.");
  }

  /// Get a variable or parameter for storing global thread id
  /// inside OpenMP construct.
  const VarDecl *getThreadIDVariable() const override { return ThreadIDVar; }

  /// Get the name of the capture helper.
  StringRef getHelperName() const override { return HelperName; }

  static bool classof(const CGCapturedStmtInfo *Info) {
    return CGOpenMPRegionInfo::classof(Info) &&
           cast<CGOpenMPRegionInfo>(Info)->getRegionKind() ==
               ParallelOutlinedRegion;
  }

private:
  /// A variable or parameter storing global thread id for OpenMP
  /// constructs.
  const VarDecl *ThreadIDVar;
  StringRef HelperName;
};

/// API for captured statement code generation in OpenMP constructs.
class CGOpenMPTaskOutlinedRegionInfo final : public CGOpenMPRegionInfo {
public:
  class UntiedTaskActionTy final : public PrePostActionTy {
    bool Untied;
    const VarDecl *PartIDVar;
    const RegionCodeGenTy UntiedCodeGen;
    llvm::SwitchInst *UntiedSwitch = nullptr;

  public:
    UntiedTaskActionTy(bool Tied, const VarDecl *PartIDVar,
                       const RegionCodeGenTy &UntiedCodeGen)
        : Untied(!Tied), PartIDVar(PartIDVar), UntiedCodeGen(UntiedCodeGen) {}
    void Enter(CodeGenFunction &CGF) override {
      if (Untied) {
        // Emit task switching point.
        LValue PartIdLVal = CGF.EmitLoadOfPointerLValue(
            CGF.GetAddrOfLocalVar(PartIDVar),
            PartIDVar->getType()->castAs<PointerType>());
        llvm::Value *Res =
            CGF.EmitLoadOfScalar(PartIdLVal, PartIDVar->getLocation());
        llvm::BasicBlock *DoneBB = CGF.createBasicBlock(".untied.done.");
        UntiedSwitch = CGF.Builder.CreateSwitch(Res, DoneBB);
        CGF.EmitBlock(DoneBB);
        CGF.EmitBranchThroughCleanup(CGF.ReturnBlock);
        CGF.EmitBlock(CGF.createBasicBlock(".untied.jmp."));
        UntiedSwitch->addCase(CGF.Builder.getInt32(0),
                              CGF.Builder.GetInsertBlock());
        emitUntiedSwitch(CGF);
      }
    }
    void emitUntiedSwitch(CodeGenFunction &CGF) const {
      if (Untied) {
        LValue PartIdLVal = CGF.EmitLoadOfPointerLValue(
            CGF.GetAddrOfLocalVar(PartIDVar),
            PartIDVar->getType()->castAs<PointerType>());
        CGF.EmitStoreOfScalar(CGF.Builder.getInt32(UntiedSwitch->getNumCases()),
                              PartIdLVal);
        UntiedCodeGen(CGF);
        CodeGenFunction::JumpDest CurPoint =
            CGF.getJumpDestInCurrentScope(".untied.next.");
        CGF.EmitBranch(CGF.ReturnBlock.getBlock());
        CGF.EmitBlock(CGF.createBasicBlock(".untied.jmp."));
        UntiedSwitch->addCase(CGF.Builder.getInt32(UntiedSwitch->getNumCases()),
                              CGF.Builder.GetInsertBlock());
        CGF.EmitBranchThroughCleanup(CurPoint);
        CGF.EmitBlock(CurPoint.getBlock());
      }
    }
    unsigned getNumberOfParts() const { return UntiedSwitch->getNumCases(); }
  };
  CGOpenMPTaskOutlinedRegionInfo(const CapturedStmt &CS,
                                 const VarDecl *ThreadIDVar,
                                 const RegionCodeGenTy &CodeGen,
                                 OpenMPDirectiveKind Kind, bool HasCancel,
                                 const UntiedTaskActionTy &Action)
      : CGOpenMPRegionInfo(CS, TaskOutlinedRegion, CodeGen, Kind, HasCancel),
        ThreadIDVar(ThreadIDVar), Action(Action) {
    assert(ThreadIDVar != nullptr && "No ThreadID in OpenMP region.");
  }

  /// Get a variable or parameter for storing global thread id
  /// inside OpenMP construct.
  const VarDecl *getThreadIDVariable() const override { return ThreadIDVar; }

  /// Get an LValue for the current ThreadID variable.
  LValue getThreadIDVariableLValue(CodeGenFunction &CGF) override;

  /// Get the name of the capture helper.
  StringRef getHelperName() const override { return ".omp_outlined."; }

  void emitUntiedSwitch(CodeGenFunction &CGF) override {
    Action.emitUntiedSwitch(CGF);
  }

  static bool classof(const CGCapturedStmtInfo *Info) {
    return CGOpenMPRegionInfo::classof(Info) &&
           cast<CGOpenMPRegionInfo>(Info)->getRegionKind() ==
               TaskOutlinedRegion;
  }

private:
  /// A variable or parameter storing global thread id for OpenMP
  /// constructs.
  const VarDecl *ThreadIDVar;
  /// Action for emitting code for untied tasks.
  const UntiedTaskActionTy &Action;
};

/// API for inlined captured statement code generation in OpenMP
/// constructs.
class CGOpenMPInlinedRegionInfo : public CGOpenMPRegionInfo {
public:
  CGOpenMPInlinedRegionInfo(CodeGenFunction::CGCapturedStmtInfo *OldCSI,
                            const RegionCodeGenTy &CodeGen,
                            OpenMPDirectiveKind Kind, bool HasCancel)
      : CGOpenMPRegionInfo(InlinedRegion, CodeGen, Kind, HasCancel),
        OldCSI(OldCSI),
        OuterRegionInfo(dyn_cast_or_null<CGOpenMPRegionInfo>(OldCSI)) {}

  // Retrieve the value of the context parameter.
  llvm::Value *getContextValue() const override {
    if (OuterRegionInfo)
      return OuterRegionInfo->getContextValue();
    llvm_unreachable("No context value for inlined OpenMP region");
  }

  void setContextValue(llvm::Value *V) override {
    if (OuterRegionInfo) {
      OuterRegionInfo->setContextValue(V);
      return;
    }
    llvm_unreachable("No context value for inlined OpenMP region");
  }

  /// Lookup the captured field decl for a variable.
  const FieldDecl *lookup(const VarDecl *VD) const override {
    if (OuterRegionInfo)
      return OuterRegionInfo->lookup(VD);
    // If there is no outer outlined region,no need to lookup in a list of
    // captured variables, we can use the original one.
    return nullptr;
  }

  FieldDecl *getThisFieldDecl() const override {
    if (OuterRegionInfo)
      return OuterRegionInfo->getThisFieldDecl();
    return nullptr;
  }

  /// Get a variable or parameter for storing global thread id
  /// inside OpenMP construct.
  const VarDecl *getThreadIDVariable() const override {
    if (OuterRegionInfo)
      return OuterRegionInfo->getThreadIDVariable();
    return nullptr;
  }

  /// Get an LValue for the current ThreadID variable.
  LValue getThreadIDVariableLValue(CodeGenFunction &CGF) override {
    if (OuterRegionInfo)
      return OuterRegionInfo->getThreadIDVariableLValue(CGF);
    llvm_unreachable("No LValue for inlined OpenMP construct");
  }

  /// Get the name of the capture helper.
  StringRef getHelperName() const override {
    if (auto *OuterRegionInfo = getOldCSI())
      return OuterRegionInfo->getHelperName();
    llvm_unreachable("No helper name for inlined OpenMP construct");
  }

  void emitUntiedSwitch(CodeGenFunction &CGF) override {
    if (OuterRegionInfo)
      OuterRegionInfo->emitUntiedSwitch(CGF);
  }

  CodeGenFunction::CGCapturedStmtInfo *getOldCSI() const { return OldCSI; }

  static bool classof(const CGCapturedStmtInfo *Info) {
    return CGOpenMPRegionInfo::classof(Info) &&
           cast<CGOpenMPRegionInfo>(Info)->getRegionKind() == InlinedRegion;
  }

  ~CGOpenMPInlinedRegionInfo() override = default;

private:
  /// CodeGen info about outer OpenMP region.
  CodeGenFunction::CGCapturedStmtInfo *OldCSI;
  CGOpenMPRegionInfo *OuterRegionInfo;
};

/// API for captured statement code generation in OpenMP target
/// constructs. For this captures, implicit parameters are used instead of the
/// captured fields. The name of the target region has to be unique in a given
/// application so it is provided by the client, because only the client has
/// the information to generate that.
class CGOpenMPTargetRegionInfo final : public CGOpenMPRegionInfo {
public:
  CGOpenMPTargetRegionInfo(const CapturedStmt &CS,
                           const RegionCodeGenTy &CodeGen, StringRef HelperName)
      : CGOpenMPRegionInfo(CS, TargetRegion, CodeGen, OMPD_target,
                           /*HasCancel=*/false),
        HelperName(HelperName) {}

  /// This is unused for target regions because each starts executing
  /// with a single thread.
  const VarDecl *getThreadIDVariable() const override { return nullptr; }

  /// Get the name of the capture helper.
  StringRef getHelperName() const override { return HelperName; }

  static bool classof(const CGCapturedStmtInfo *Info) {
    return CGOpenMPRegionInfo::classof(Info) &&
           cast<CGOpenMPRegionInfo>(Info)->getRegionKind() == TargetRegion;
  }

private:
  StringRef HelperName;
};

static void EmptyCodeGen(CodeGenFunction &, PrePostActionTy &) {
  llvm_unreachable("No codegen for expressions");
}
/// API for generation of expressions captured in a innermost OpenMP
/// region.
class CGOpenMPInnerExprInfo final : public CGOpenMPInlinedRegionInfo {
public:
  CGOpenMPInnerExprInfo(CodeGenFunction &CGF, const CapturedStmt &CS)
      : CGOpenMPInlinedRegionInfo(CGF.CapturedStmtInfo, EmptyCodeGen,
                                  OMPD_unknown,
                                  /*HasCancel=*/false),
        PrivScope(CGF) {
    // Make sure the globals captured in the provided statement are local by
    // using the privatization logic. We assume the same variable is not
    // captured more than once.
    for (const auto &C : CS.captures()) {
      if (!C.capturesVariable() && !C.capturesVariableByCopy())
        continue;

      const VarDecl *VD = C.getCapturedVar();
      if (VD->isLocalVarDeclOrParm())
        continue;

      DeclRefExpr DRE(CGF.getContext(), const_cast<VarDecl *>(VD),
                      /*RefersToEnclosingVariableOrCapture=*/false,
                      VD->getType().getNonReferenceType(), VK_LValue,
                      C.getLocation());
      PrivScope.addPrivate(
          VD, [&CGF, &DRE]() { return CGF.EmitLValue(&DRE).getAddress(CGF); });
    }
    (void)PrivScope.Privatize();
  }

  /// Lookup the captured field decl for a variable.
  const FieldDecl *lookup(const VarDecl *VD) const override {
    if (const FieldDecl *FD = CGOpenMPInlinedRegionInfo::lookup(VD))
      return FD;
    return nullptr;
  }

  /// Emit the captured statement body.
  void EmitBody(CodeGenFunction &CGF, const Stmt *S) override {
    llvm_unreachable("No body for expressions");
  }

  /// Get a variable or parameter for storing global thread id
  /// inside OpenMP construct.
  const VarDecl *getThreadIDVariable() const override {
    llvm_unreachable("No thread id for expressions");
  }

  /// Get the name of the capture helper.
  StringRef getHelperName() const override {
    llvm_unreachable("No helper name for expressions");
  }

  static bool classof(const CGCapturedStmtInfo *Info) { return false; }

private:
  /// Private scope to capture global variables.
  CodeGenFunction::OMPPrivateScope PrivScope;
};

/// RAII for emitting code of OpenMP constructs.
class InlinedOpenMPRegionRAII {
  CodeGenFunction &CGF;
  llvm::DenseMap<const VarDecl *, FieldDecl *> LambdaCaptureFields;
  FieldDecl *LambdaThisCaptureField = nullptr;
  const CodeGen::CGBlockInfo *BlockInfo = nullptr;
  bool NoInheritance = false;

public:
  /// Constructs region for combined constructs.
  /// \param CodeGen Code generation sequence for combined directives. Includes
  /// a list of functions used for code generation of implicitly inlined
  /// regions.
  InlinedOpenMPRegionRAII(CodeGenFunction &CGF, const RegionCodeGenTy &CodeGen,
                          OpenMPDirectiveKind Kind, bool HasCancel,
                          bool NoInheritance = true)
      : CGF(CGF), NoInheritance(NoInheritance) {
    // Start emission for the construct.
    CGF.CapturedStmtInfo = new CGOpenMPInlinedRegionInfo(
        CGF.CapturedStmtInfo, CodeGen, Kind, HasCancel);
    if (NoInheritance) {
      std::swap(CGF.LambdaCaptureFields, LambdaCaptureFields);
      LambdaThisCaptureField = CGF.LambdaThisCaptureField;
      CGF.LambdaThisCaptureField = nullptr;
      BlockInfo = CGF.BlockInfo;
      CGF.BlockInfo = nullptr;
    }
  }

  ~InlinedOpenMPRegionRAII() {
    // Restore original CapturedStmtInfo only if we're done with code emission.
    auto *OldCSI =
        cast<CGOpenMPInlinedRegionInfo>(CGF.CapturedStmtInfo)->getOldCSI();
    delete CGF.CapturedStmtInfo;
    CGF.CapturedStmtInfo = OldCSI;
    if (NoInheritance) {
      std::swap(CGF.LambdaCaptureFields, LambdaCaptureFields);
      CGF.LambdaThisCaptureField = LambdaThisCaptureField;
      CGF.BlockInfo = BlockInfo;
    }
  }
};

/// Values for bit flags used in the ident_t to describe the fields.
/// All enumeric elements are named and described in accordance with the code
/// from https://github.com/llvm/llvm-project/blob/main/openmp/runtime/src/kmp.h
enum OpenMPLocationFlags : unsigned {
  /// Use trampoline for internal microtask.
  OMP_IDENT_IMD = 0x01,
  /// Use c-style ident structure.
  OMP_IDENT_KMPC = 0x02,
  /// Atomic reduction option for kmpc_reduce.
  OMP_ATOMIC_REDUCE = 0x10,
  /// Explicit 'barrier' directive.
  OMP_IDENT_BARRIER_EXPL = 0x20,
  /// Implicit barrier in code.
  OMP_IDENT_BARRIER_IMPL = 0x40,
  /// Implicit barrier in 'for' directive.
  OMP_IDENT_BARRIER_IMPL_FOR = 0x40,
  /// Implicit barrier in 'sections' directive.
  OMP_IDENT_BARRIER_IMPL_SECTIONS = 0xC0,
  /// Implicit barrier in 'single' directive.
  OMP_IDENT_BARRIER_IMPL_SINGLE = 0x140,
  /// Call of __kmp_for_static_init for static loop.
  OMP_IDENT_WORK_LOOP = 0x200,
  /// Call of __kmp_for_static_init for sections.
  OMP_IDENT_WORK_SECTIONS = 0x400,
  /// Call of __kmp_for_static_init for distribute.
  OMP_IDENT_WORK_DISTRIBUTE = 0x800,
  LLVM_MARK_AS_BITMASK_ENUM(/*LargestValue=*/OMP_IDENT_WORK_DISTRIBUTE)
};

namespace {
LLVM_ENABLE_BITMASK_ENUMS_IN_NAMESPACE();
/// Values for bit flags for marking which requires clauses have been used.
enum OpenMPOffloadingRequiresDirFlags : int64_t {
  /// flag undefined.
  OMP_REQ_UNDEFINED               = 0x000,
  /// no requires clause present.
  OMP_REQ_NONE                    = 0x001,
  /// reverse_offload clause.
  OMP_REQ_REVERSE_OFFLOAD         = 0x002,
  /// unified_address clause.
  OMP_REQ_UNIFIED_ADDRESS         = 0x004,
  /// unified_shared_memory clause.
  OMP_REQ_UNIFIED_SHARED_MEMORY   = 0x008,
  /// dynamic_allocators clause.
  OMP_REQ_DYNAMIC_ALLOCATORS      = 0x010,
  LLVM_MARK_AS_BITMASK_ENUM(/*LargestValue=*/OMP_REQ_DYNAMIC_ALLOCATORS)
};

enum OpenMPOffloadingReservedDeviceIDs {
  /// Device ID if the device was not defined, runtime should get it
  /// from environment variables in the spec.
  OMP_DEVICEID_UNDEF = -1,
};
} // anonymous namespace

/// Describes ident structure that describes a source location.
/// All descriptions are taken from
/// https://github.com/llvm/llvm-project/blob/main/openmp/runtime/src/kmp.h
/// Original structure:
/// typedef struct ident {
///    kmp_int32 reserved_1;   /**<  might be used in Fortran;
///                                  see above  */
///    kmp_int32 flags;        /**<  also f.flags; KMP_IDENT_xxx flags;
///                                  KMP_IDENT_KMPC identifies this union
///                                  member  */
///    kmp_int32 reserved_2;   /**<  not really used in Fortran any more;
///                                  see above */
///#if USE_ITT_BUILD
///                            /*  but currently used for storing
///                                region-specific ITT */
///                            /*  contextual information. */
///#endif /* USE_ITT_BUILD */
///    kmp_int32 reserved_3;   /**< source[4] in Fortran, do not use for
///                                 C++  */
///    char const *psource;    /**< String describing the source location.
///                            The string is composed of semi-colon separated
//                             fields which describe the source file,
///                            the function and a pair of line numbers that
///                            delimit the construct.
///                             */
/// } ident_t;
enum IdentFieldIndex {
  /// might be used in Fortran
  IdentField_Reserved_1,
  /// OMP_IDENT_xxx flags; OMP_IDENT_KMPC identifies this union member.
  IdentField_Flags,
  /// Not really used in Fortran any more
  IdentField_Reserved_2,
  /// Source[4] in Fortran, do not use for C++
  IdentField_Reserved_3,
  /// String describing the source location. The string is composed of
  /// semi-colon separated fields which describe the source file, the function
  /// and a pair of line numbers that delimit the construct.
  IdentField_PSource
};

/// Schedule types for 'omp for' loops (these enumerators are taken from
/// the enum sched_type in kmp.h).
enum OpenMPSchedType {
  /// Lower bound for default (unordered) versions.
  OMP_sch_lower = 32,
  OMP_sch_static_chunked = 33,
  OMP_sch_static = 34,
  OMP_sch_dynamic_chunked = 35,
  OMP_sch_guided_chunked = 36,
  OMP_sch_runtime = 37,
  OMP_sch_auto = 38,
  /// static with chunk adjustment (e.g., simd)
  OMP_sch_static_balanced_chunked = 45,
  /// Lower bound for 'ordered' versions.
  OMP_ord_lower = 64,
  OMP_ord_static_chunked = 65,
  OMP_ord_static = 66,
  OMP_ord_dynamic_chunked = 67,
  OMP_ord_guided_chunked = 68,
  OMP_ord_runtime = 69,
  OMP_ord_auto = 70,
  OMP_sch_default = OMP_sch_static,
  /// dist_schedule types
  OMP_dist_sch_static_chunked = 91,
  OMP_dist_sch_static = 92,
  /// Support for OpenMP 4.5 monotonic and nonmonotonic schedule modifiers.
  /// Set if the monotonic schedule modifier was present.
  OMP_sch_modifier_monotonic = (1 << 29),
  /// Set if the nonmonotonic schedule modifier was present.
  OMP_sch_modifier_nonmonotonic = (1 << 30),
};

/// A basic class for pre|post-action for advanced codegen sequence for OpenMP
/// region.
class CleanupTy final : public EHScopeStack::Cleanup {
  PrePostActionTy *Action;

public:
  explicit CleanupTy(PrePostActionTy *Action) : Action(Action) {}
  void Emit(CodeGenFunction &CGF, Flags /*flags*/) override {
    if (!CGF.HaveInsertPoint())
      return;
    Action->Exit(CGF);
  }
};

} // anonymous namespace

void RegionCodeGenTy::operator()(CodeGenFunction &CGF) const {
  CodeGenFunction::RunCleanupsScope Scope(CGF);
  if (PrePostAction) {
    CGF.EHStack.pushCleanup<CleanupTy>(NormalAndEHCleanup, PrePostAction);
    Callback(CodeGen, CGF, *PrePostAction);
  } else {
    PrePostActionTy Action;
    Callback(CodeGen, CGF, Action);
  }
}

/// Check if the combiner is a call to UDR combiner and if it is so return the
/// UDR decl used for reduction.
static const OMPDeclareReductionDecl *
getReductionInit(const Expr *ReductionOp) {
  if (const auto *CE = dyn_cast<CallExpr>(ReductionOp))
    if (const auto *OVE = dyn_cast<OpaqueValueExpr>(CE->getCallee()))
      if (const auto *DRE =
              dyn_cast<DeclRefExpr>(OVE->getSourceExpr()->IgnoreImpCasts()))
        if (const auto *DRD = dyn_cast<OMPDeclareReductionDecl>(DRE->getDecl()))
          return DRD;
  return nullptr;
}

static void emitInitWithReductionInitializer(CodeGenFunction &CGF,
                                             const OMPDeclareReductionDecl *DRD,
                                             const Expr *InitOp,
                                             Address Private, Address Original,
                                             QualType Ty) {
  if (DRD->getInitializer()) {
    std::pair<llvm::Function *, llvm::Function *> Reduction =
        CGF.CGM.getOpenMPRuntime().getUserDefinedReduction(DRD);
    const auto *CE = cast<CallExpr>(InitOp);
    const auto *OVE = cast<OpaqueValueExpr>(CE->getCallee());
    const Expr *LHS = CE->getArg(/*Arg=*/0)->IgnoreParenImpCasts();
    const Expr *RHS = CE->getArg(/*Arg=*/1)->IgnoreParenImpCasts();
    const auto *LHSDRE =
        cast<DeclRefExpr>(cast<UnaryOperator>(LHS)->getSubExpr());
    const auto *RHSDRE =
        cast<DeclRefExpr>(cast<UnaryOperator>(RHS)->getSubExpr());
    CodeGenFunction::OMPPrivateScope PrivateScope(CGF);
    PrivateScope.addPrivate(cast<VarDecl>(LHSDRE->getDecl()),
                            [=]() { return Private; });
    PrivateScope.addPrivate(cast<VarDecl>(RHSDRE->getDecl()),
                            [=]() { return Original; });
    (void)PrivateScope.Privatize();
    RValue Func = RValue::get(Reduction.second);
    CodeGenFunction::OpaqueValueMapping Map(CGF, OVE, Func);
    CGF.EmitIgnoredExpr(InitOp);
  } else {
    llvm::Constant *Init = CGF.CGM.EmitNullConstant(Ty);
    std::string Name = CGF.CGM.getOpenMPRuntime().getName({"init"});
    auto *GV = new llvm::GlobalVariable(
        CGF.CGM.getModule(), Init->getType(), /*isConstant=*/true,
        llvm::GlobalValue::PrivateLinkage, Init, Name);
    LValue LV = CGF.MakeNaturalAlignAddrLValue(GV, Ty);
    RValue InitRVal;
    switch (CGF.getEvaluationKind(Ty)) {
    case TEK_Scalar:
      InitRVal = CGF.EmitLoadOfLValue(LV, DRD->getLocation());
      break;
    case TEK_Complex:
      InitRVal =
          RValue::getComplex(CGF.EmitLoadOfComplex(LV, DRD->getLocation()));
      break;
    case TEK_Aggregate: {
      OpaqueValueExpr OVE(DRD->getLocation(), Ty, VK_LValue);
      CodeGenFunction::OpaqueValueMapping OpaqueMap(CGF, &OVE, LV);
      CGF.EmitAnyExprToMem(&OVE, Private, Ty.getQualifiers(),
                           /*IsInitializer=*/false);
      return;
    }
    }
    OpaqueValueExpr OVE(DRD->getLocation(), Ty, VK_PRValue);
    CodeGenFunction::OpaqueValueMapping OpaqueMap(CGF, &OVE, InitRVal);
    CGF.EmitAnyExprToMem(&OVE, Private, Ty.getQualifiers(),
                         /*IsInitializer=*/false);
  }
}

/// Emit initialization of arrays of complex types.
/// \param DestAddr Address of the array.
/// \param Type Type of array.
/// \param Init Initial expression of array.
/// \param SrcAddr Address of the original array.
static void EmitOMPAggregateInit(CodeGenFunction &CGF, Address DestAddr,
                                 QualType Type, bool EmitDeclareReductionInit,
                                 const Expr *Init,
                                 const OMPDeclareReductionDecl *DRD,
                                 Address SrcAddr = Address::invalid()) {
  // Perform element-by-element initialization.
  QualType ElementTy;

  // Drill down to the base element type on both arrays.
  const ArrayType *ArrayTy = Type->getAsArrayTypeUnsafe();
  llvm::Value *NumElements = CGF.emitArrayLength(ArrayTy, ElementTy, DestAddr);
  if (DRD)
    SrcAddr =
        CGF.Builder.CreateElementBitCast(SrcAddr, DestAddr.getElementType());

  llvm::Value *SrcBegin = nullptr;
  if (DRD)
    SrcBegin = SrcAddr.getPointer();
  llvm::Value *DestBegin = DestAddr.getPointer();
  // Cast from pointer to array type to pointer to single element.
  llvm::Value *DestEnd =
      CGF.Builder.CreateGEP(DestAddr.getElementType(), DestBegin, NumElements);
  // The basic structure here is a while-do loop.
  llvm::BasicBlock *BodyBB = CGF.createBasicBlock("omp.arrayinit.body");
  llvm::BasicBlock *DoneBB = CGF.createBasicBlock("omp.arrayinit.done");
  llvm::Value *IsEmpty =
      CGF.Builder.CreateICmpEQ(DestBegin, DestEnd, "omp.arrayinit.isempty");
  CGF.Builder.CreateCondBr(IsEmpty, DoneBB, BodyBB);

  // Enter the loop body, making that address the current address.
  llvm::BasicBlock *EntryBB = CGF.Builder.GetInsertBlock();
  CGF.EmitBlock(BodyBB);

  CharUnits ElementSize = CGF.getContext().getTypeSizeInChars(ElementTy);

  llvm::PHINode *SrcElementPHI = nullptr;
  Address SrcElementCurrent = Address::invalid();
  if (DRD) {
    SrcElementPHI = CGF.Builder.CreatePHI(SrcBegin->getType(), 2,
                                          "omp.arraycpy.srcElementPast");
    SrcElementPHI->addIncoming(SrcBegin, EntryBB);
    SrcElementCurrent =
        Address(SrcElementPHI,
                SrcAddr.getAlignment().alignmentOfArrayElement(ElementSize));
  }
  llvm::PHINode *DestElementPHI = CGF.Builder.CreatePHI(
      DestBegin->getType(), 2, "omp.arraycpy.destElementPast");
  DestElementPHI->addIncoming(DestBegin, EntryBB);
  Address DestElementCurrent =
      Address(DestElementPHI,
              DestAddr.getAlignment().alignmentOfArrayElement(ElementSize));

  // Emit copy.
  {
    CodeGenFunction::RunCleanupsScope InitScope(CGF);
    if (EmitDeclareReductionInit) {
      emitInitWithReductionInitializer(CGF, DRD, Init, DestElementCurrent,
                                       SrcElementCurrent, ElementTy);
    } else
      CGF.EmitAnyExprToMem(Init, DestElementCurrent, ElementTy.getQualifiers(),
                           /*IsInitializer=*/false);
  }

  if (DRD) {
    // Shift the address forward by one element.
    llvm::Value *SrcElementNext = CGF.Builder.CreateConstGEP1_32(
        SrcAddr.getElementType(), SrcElementPHI, /*Idx0=*/1,
        "omp.arraycpy.dest.element");
    SrcElementPHI->addIncoming(SrcElementNext, CGF.Builder.GetInsertBlock());
  }

  // Shift the address forward by one element.
  llvm::Value *DestElementNext = CGF.Builder.CreateConstGEP1_32(
      DestAddr.getElementType(), DestElementPHI, /*Idx0=*/1,
      "omp.arraycpy.dest.element");
  // Check whether we've reached the end.
  llvm::Value *Done =
      CGF.Builder.CreateICmpEQ(DestElementNext, DestEnd, "omp.arraycpy.done");
  CGF.Builder.CreateCondBr(Done, DoneBB, BodyBB);
  DestElementPHI->addIncoming(DestElementNext, CGF.Builder.GetInsertBlock());

  // Done.
  CGF.EmitBlock(DoneBB, /*IsFinished=*/true);
}

LValue ReductionCodeGen::emitSharedLValue(CodeGenFunction &CGF, const Expr *E) {
  return CGF.EmitOMPSharedLValue(E);
}

LValue ReductionCodeGen::emitSharedLValueUB(CodeGenFunction &CGF,
                                            const Expr *E) {
  if (const auto *OASE = dyn_cast<OMPArraySectionExpr>(E))
    return CGF.EmitOMPArraySectionExpr(OASE, /*IsLowerBound=*/false);
  return LValue();
}

void ReductionCodeGen::emitAggregateInitialization(
    CodeGenFunction &CGF, unsigned N, Address PrivateAddr, Address SharedAddr,
    const OMPDeclareReductionDecl *DRD) {
  // Emit VarDecl with copy init for arrays.
  // Get the address of the original variable captured in current
  // captured region.
  const auto *PrivateVD =
      cast<VarDecl>(cast<DeclRefExpr>(ClausesData[N].Private)->getDecl());
  bool EmitDeclareReductionInit =
      DRD && (DRD->getInitializer() || !PrivateVD->hasInit());
  EmitOMPAggregateInit(CGF, PrivateAddr, PrivateVD->getType(),
                       EmitDeclareReductionInit,
                       EmitDeclareReductionInit ? ClausesData[N].ReductionOp
                                                : PrivateVD->getInit(),
                       DRD, SharedAddr);
}

ReductionCodeGen::ReductionCodeGen(ArrayRef<const Expr *> Shareds,
                                   ArrayRef<const Expr *> Origs,
                                   ArrayRef<const Expr *> Privates,
                                   ArrayRef<const Expr *> ReductionOps) {
  ClausesData.reserve(Shareds.size());
  SharedAddresses.reserve(Shareds.size());
  Sizes.reserve(Shareds.size());
  BaseDecls.reserve(Shareds.size());
  const auto *IOrig = Origs.begin();
  const auto *IPriv = Privates.begin();
  const auto *IRed = ReductionOps.begin();
  for (const Expr *Ref : Shareds) {
    ClausesData.emplace_back(Ref, *IOrig, *IPriv, *IRed);
    std::advance(IOrig, 1);
    std::advance(IPriv, 1);
    std::advance(IRed, 1);
  }
}

void ReductionCodeGen::emitSharedOrigLValue(CodeGenFunction &CGF, unsigned N) {
  assert(SharedAddresses.size() == N && OrigAddresses.size() == N &&
         "Number of generated lvalues must be exactly N.");
  LValue First = emitSharedLValue(CGF, ClausesData[N].Shared);
  LValue Second = emitSharedLValueUB(CGF, ClausesData[N].Shared);
  SharedAddresses.emplace_back(First, Second);
  if (ClausesData[N].Shared == ClausesData[N].Ref) {
    OrigAddresses.emplace_back(First, Second);
  } else {
    LValue First = emitSharedLValue(CGF, ClausesData[N].Ref);
    LValue Second = emitSharedLValueUB(CGF, ClausesData[N].Ref);
    OrigAddresses.emplace_back(First, Second);
  }
}

void ReductionCodeGen::emitAggregateType(CodeGenFunction &CGF, unsigned N) {
  const auto *PrivateVD =
      cast<VarDecl>(cast<DeclRefExpr>(ClausesData[N].Private)->getDecl());
  QualType PrivateType = PrivateVD->getType();
  bool AsArraySection = isa<OMPArraySectionExpr>(ClausesData[N].Ref);
  if (!PrivateType->isVariablyModifiedType()) {
    Sizes.emplace_back(
        CGF.getTypeSize(OrigAddresses[N].first.getType().getNonReferenceType()),
        nullptr);
    return;
  }
  llvm::Value *Size;
  llvm::Value *SizeInChars;
  auto *ElemType =
      cast<llvm::PointerType>(OrigAddresses[N].first.getPointer(CGF)->getType())
          ->getElementType();
  auto *ElemSizeOf = llvm::ConstantExpr::getSizeOf(ElemType);
  if (AsArraySection) {
    Size = CGF.Builder.CreatePtrDiff(OrigAddresses[N].second.getPointer(CGF),
                                     OrigAddresses[N].first.getPointer(CGF));
    Size = CGF.Builder.CreateNUWAdd(
        Size, llvm::ConstantInt::get(Size->getType(), /*V=*/1));
    SizeInChars = CGF.Builder.CreateNUWMul(Size, ElemSizeOf);
  } else {
    SizeInChars =
        CGF.getTypeSize(OrigAddresses[N].first.getType().getNonReferenceType());
    Size = CGF.Builder.CreateExactUDiv(SizeInChars, ElemSizeOf);
  }
  Sizes.emplace_back(SizeInChars, Size);
  CodeGenFunction::OpaqueValueMapping OpaqueMap(
      CGF,
      cast<OpaqueValueExpr>(
          CGF.getContext().getAsVariableArrayType(PrivateType)->getSizeExpr()),
      RValue::get(Size));
  CGF.EmitVariablyModifiedType(PrivateType);
}

void ReductionCodeGen::emitAggregateType(CodeGenFunction &CGF, unsigned N,
                                         llvm::Value *Size) {
  const auto *PrivateVD =
      cast<VarDecl>(cast<DeclRefExpr>(ClausesData[N].Private)->getDecl());
  QualType PrivateType = PrivateVD->getType();
  if (!PrivateType->isVariablyModifiedType()) {
    assert(!Size && !Sizes[N].second &&
           "Size should be nullptr for non-variably modified reduction "
           "items.");
    return;
  }
  CodeGenFunction::OpaqueValueMapping OpaqueMap(
      CGF,
      cast<OpaqueValueExpr>(
          CGF.getContext().getAsVariableArrayType(PrivateType)->getSizeExpr()),
      RValue::get(Size));
  CGF.EmitVariablyModifiedType(PrivateType);
}

void ReductionCodeGen::emitInitialization(
    CodeGenFunction &CGF, unsigned N, Address PrivateAddr, Address SharedAddr,
    llvm::function_ref<bool(CodeGenFunction &)> DefaultInit) {
  assert(SharedAddresses.size() > N && "No variable was generated");
  const auto *PrivateVD =
      cast<VarDecl>(cast<DeclRefExpr>(ClausesData[N].Private)->getDecl());
  const OMPDeclareReductionDecl *DRD =
      getReductionInit(ClausesData[N].ReductionOp);
  QualType PrivateType = PrivateVD->getType();
  PrivateAddr = CGF.Builder.CreateElementBitCast(
      PrivateAddr, CGF.ConvertTypeForMem(PrivateType));
  if (CGF.getContext().getAsArrayType(PrivateVD->getType())) {
    if (DRD && DRD->getInitializer())
      (void)DefaultInit(CGF);
    emitAggregateInitialization(CGF, N, PrivateAddr, SharedAddr, DRD);
  } else if (DRD && (DRD->getInitializer() || !PrivateVD->hasInit())) {
    (void)DefaultInit(CGF);
    QualType SharedType = SharedAddresses[N].first.getType();
    emitInitWithReductionInitializer(CGF, DRD, ClausesData[N].ReductionOp,
                                     PrivateAddr, SharedAddr, SharedType);
  } else if (!DefaultInit(CGF) && PrivateVD->hasInit() &&
             !CGF.isTrivialInitializer(PrivateVD->getInit())) {
    CGF.EmitAnyExprToMem(PrivateVD->getInit(), PrivateAddr,
                         PrivateVD->getType().getQualifiers(),
                         /*IsInitializer=*/false);
  }
}

bool ReductionCodeGen::needCleanups(unsigned N) {
  const auto *PrivateVD =
      cast<VarDecl>(cast<DeclRefExpr>(ClausesData[N].Private)->getDecl());
  QualType PrivateType = PrivateVD->getType();
  QualType::DestructionKind DTorKind = PrivateType.isDestructedType();
  return DTorKind != QualType::DK_none;
}

void ReductionCodeGen::emitCleanups(CodeGenFunction &CGF, unsigned N,
                                    Address PrivateAddr) {
  const auto *PrivateVD =
      cast<VarDecl>(cast<DeclRefExpr>(ClausesData[N].Private)->getDecl());
  QualType PrivateType = PrivateVD->getType();
  QualType::DestructionKind DTorKind = PrivateType.isDestructedType();
  if (needCleanups(N)) {
    PrivateAddr = CGF.Builder.CreateElementBitCast(
        PrivateAddr, CGF.ConvertTypeForMem(PrivateType));
    CGF.pushDestroy(DTorKind, PrivateAddr, PrivateType);
  }
}

static LValue loadToBegin(CodeGenFunction &CGF, QualType BaseTy, QualType ElTy,
                          LValue BaseLV) {
  BaseTy = BaseTy.getNonReferenceType();
  while ((BaseTy->isPointerType() || BaseTy->isReferenceType()) &&
         !CGF.getContext().hasSameType(BaseTy, ElTy)) {
    if (const auto *PtrTy = BaseTy->getAs<PointerType>()) {
      BaseLV = CGF.EmitLoadOfPointerLValue(BaseLV.getAddress(CGF), PtrTy);
    } else {
      LValue RefLVal = CGF.MakeAddrLValue(BaseLV.getAddress(CGF), BaseTy);
      BaseLV = CGF.EmitLoadOfReferenceLValue(RefLVal);
    }
    BaseTy = BaseTy->getPointeeType();
  }
  return CGF.MakeAddrLValue(
      CGF.Builder.CreateElementBitCast(BaseLV.getAddress(CGF),
                                       CGF.ConvertTypeForMem(ElTy)),
      BaseLV.getType(), BaseLV.getBaseInfo(),
      CGF.CGM.getTBAAInfoForSubobject(BaseLV, BaseLV.getType()));
}

static Address castToBase(CodeGenFunction &CGF, QualType BaseTy, QualType ElTy,
                          llvm::Type *BaseLVType, CharUnits BaseLVAlignment,
                          llvm::Value *Addr) {
  Address Tmp = Address::invalid();
  Address TopTmp = Address::invalid();
  Address MostTopTmp = Address::invalid();
  BaseTy = BaseTy.getNonReferenceType();
  while ((BaseTy->isPointerType() || BaseTy->isReferenceType()) &&
         !CGF.getContext().hasSameType(BaseTy, ElTy)) {
    Tmp = CGF.CreateMemTemp(BaseTy);
    if (TopTmp.isValid())
      CGF.Builder.CreateStore(Tmp.getPointer(), TopTmp);
    else
      MostTopTmp = Tmp;
    TopTmp = Tmp;
    BaseTy = BaseTy->getPointeeType();
  }
  llvm::Type *Ty = BaseLVType;
  if (Tmp.isValid())
    Ty = Tmp.getElementType();
  Addr = CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(Addr, Ty);
  if (Tmp.isValid()) {
    CGF.Builder.CreateStore(Addr, Tmp);
    return MostTopTmp;
  }
  return Address(Addr, BaseLVAlignment);
}

static const VarDecl *getBaseDecl(const Expr *Ref, const DeclRefExpr *&DE) {
  const VarDecl *OrigVD = nullptr;
  if (const auto *OASE = dyn_cast<OMPArraySectionExpr>(Ref)) {
    const Expr *Base = OASE->getBase()->IgnoreParenImpCasts();
    while (const auto *TempOASE = dyn_cast<OMPArraySectionExpr>(Base))
      Base = TempOASE->getBase()->IgnoreParenImpCasts();
    while (const auto *TempASE = dyn_cast<ArraySubscriptExpr>(Base))
      Base = TempASE->getBase()->IgnoreParenImpCasts();
    DE = cast<DeclRefExpr>(Base);
    OrigVD = cast<VarDecl>(DE->getDecl());
  } else if (const auto *ASE = dyn_cast<ArraySubscriptExpr>(Ref)) {
    const Expr *Base = ASE->getBase()->IgnoreParenImpCasts();
    while (const auto *TempASE = dyn_cast<ArraySubscriptExpr>(Base))
      Base = TempASE->getBase()->IgnoreParenImpCasts();
    DE = cast<DeclRefExpr>(Base);
    OrigVD = cast<VarDecl>(DE->getDecl());
  }
  return OrigVD;
}

Address ReductionCodeGen::adjustPrivateAddress(CodeGenFunction &CGF, unsigned N,
                                               Address PrivateAddr) {
  const DeclRefExpr *DE;
  if (const VarDecl *OrigVD = ::getBaseDecl(ClausesData[N].Ref, DE)) {
    BaseDecls.emplace_back(OrigVD);
    LValue OriginalBaseLValue = CGF.EmitLValue(DE);
    LValue BaseLValue =
        loadToBegin(CGF, OrigVD->getType(), SharedAddresses[N].first.getType(),
                    OriginalBaseLValue);
    Address SharedAddr = SharedAddresses[N].first.getAddress(CGF);
    llvm::Value *Adjustment = CGF.Builder.CreatePtrDiff(
        BaseLValue.getPointer(CGF), SharedAddr.getPointer());
    llvm::Value *PrivatePointer =
        CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
            PrivateAddr.getPointer(), SharedAddr.getType());
    llvm::Value *Ptr = CGF.Builder.CreateGEP(
        SharedAddr.getElementType(), PrivatePointer, Adjustment);
    return castToBase(CGF, OrigVD->getType(),
                      SharedAddresses[N].first.getType(),
                      OriginalBaseLValue.getAddress(CGF).getType(),
                      OriginalBaseLValue.getAlignment(), Ptr);
  }
  BaseDecls.emplace_back(
      cast<VarDecl>(cast<DeclRefExpr>(ClausesData[N].Ref)->getDecl()));
  return PrivateAddr;
}

bool ReductionCodeGen::usesReductionInitializer(unsigned N) const {
  const OMPDeclareReductionDecl *DRD =
      getReductionInit(ClausesData[N].ReductionOp);
  return DRD && DRD->getInitializer();
}

LValue CGOpenMPRegionInfo::getThreadIDVariableLValue(CodeGenFunction &CGF) {
  return CGF.EmitLoadOfPointerLValue(
      CGF.GetAddrOfLocalVar(getThreadIDVariable()),
      getThreadIDVariable()->getType()->castAs<PointerType>());
}

void CGOpenMPRegionInfo::EmitBody(CodeGenFunction &CGF, const Stmt *S) {
  if (!CGF.HaveInsertPoint())
    return;
  // 1.2.2 OpenMP Language Terminology
  // Structured block - An executable statement with a single entry at the
  // top and a single exit at the bottom.
  // The point of exit cannot be a branch out of the structured block.
  // longjmp() and throw() must not violate the entry/exit criteria.
  CGF.EHStack.pushTerminate();
  if (S)
    CGF.incrementProfileCounter(S);
  CodeGen(CGF);
  CGF.EHStack.popTerminate();
}

LValue CGOpenMPTaskOutlinedRegionInfo::getThreadIDVariableLValue(
    CodeGenFunction &CGF) {
  return CGF.MakeAddrLValue(CGF.GetAddrOfLocalVar(getThreadIDVariable()),
                            getThreadIDVariable()->getType(),
                            AlignmentSource::Decl);
}

static FieldDecl *addFieldToRecordDecl(ASTContext &C, DeclContext *DC,
                                       QualType FieldTy) {
  auto *Field = FieldDecl::Create(
      C, DC, SourceLocation(), SourceLocation(), /*Id=*/nullptr, FieldTy,
      C.getTrivialTypeSourceInfo(FieldTy, SourceLocation()),
      /*BW=*/nullptr, /*Mutable=*/false, /*InitStyle=*/ICIS_NoInit);
  Field->setAccess(AS_public);
  DC->addDecl(Field);
  return Field;
}

CGOpenMPRuntime::CGOpenMPRuntime(CodeGenModule &CGM, StringRef FirstSeparator,
                                 StringRef Separator)
    : CGM(CGM), FirstSeparator(FirstSeparator), Separator(Separator),
      OMPBuilder(CGM.getModule()), OffloadEntriesInfoManager(CGM) {
  KmpCriticalNameTy = llvm::ArrayType::get(CGM.Int32Ty, /*NumElements*/ 8);

  // Initialize Types used in OpenMPIRBuilder from OMPKinds.def
  OMPBuilder.initialize();
  loadOffloadInfoMetadata();
}

void CGOpenMPRuntime::clear() {
  InternalVars.clear();
  // Clean non-target variable declarations possibly used only in debug info.
  for (const auto &Data : EmittedNonTargetVariables) {
    if (!Data.getValue().pointsToAliveValue())
      continue;
    auto *GV = dyn_cast<llvm::GlobalVariable>(Data.getValue());
    if (!GV)
      continue;
    if (!GV->isDeclaration() || GV->getNumUses() > 0)
      continue;
    GV->eraseFromParent();
  }
}

std::string CGOpenMPRuntime::getName(ArrayRef<StringRef> Parts) const {
  SmallString<128> Buffer;
  llvm::raw_svector_ostream OS(Buffer);
  StringRef Sep = FirstSeparator;
  for (StringRef Part : Parts) {
    OS << Sep << Part;
    Sep = Separator;
  }
  return std::string(OS.str());
}

static llvm::Function *
emitCombinerOrInitializer(CodeGenModule &CGM, QualType Ty,
                          const Expr *CombinerInitializer, const VarDecl *In,
                          const VarDecl *Out, bool IsCombiner) {
  // void .omp_combiner.(Ty *in, Ty *out);
  ASTContext &C = CGM.getContext();
  QualType PtrTy = C.getPointerType(Ty).withRestrict();
  FunctionArgList Args;
  ImplicitParamDecl OmpOutParm(C, /*DC=*/nullptr, Out->getLocation(),
                               /*Id=*/nullptr, PtrTy, ImplicitParamDecl::Other);
  ImplicitParamDecl OmpInParm(C, /*DC=*/nullptr, In->getLocation(),
                              /*Id=*/nullptr, PtrTy, ImplicitParamDecl::Other);
  Args.push_back(&OmpOutParm);
  Args.push_back(&OmpInParm);
  const CGFunctionInfo &FnInfo =
      CGM.getTypes().arrangeBuiltinFunctionDeclaration(C.VoidTy, Args);
  llvm::FunctionType *FnTy = CGM.getTypes().GetFunctionType(FnInfo);
  std::string Name = CGM.getOpenMPRuntime().getName(
      {IsCombiner ? "omp_combiner" : "omp_initializer", ""});
  auto *Fn = llvm::Function::Create(FnTy, llvm::GlobalValue::InternalLinkage,
                                    Name, &CGM.getModule());
  CGM.SetInternalFunctionAttributes(GlobalDecl(), Fn, FnInfo);
  if (CGM.getLangOpts().Optimize) {
    Fn->removeFnAttr(llvm::Attribute::NoInline);
    Fn->removeFnAttr(llvm::Attribute::OptimizeNone);
    Fn->addFnAttr(llvm::Attribute::AlwaysInline);
  }
  CodeGenFunction CGF(CGM);
  // Map "T omp_in;" variable to "*omp_in_parm" value in all expressions.
  // Map "T omp_out;" variable to "*omp_out_parm" value in all expressions.
  CGF.StartFunction(GlobalDecl(), C.VoidTy, Fn, FnInfo, Args, In->getLocation(),
                    Out->getLocation());
  CodeGenFunction::OMPPrivateScope Scope(CGF);
  Address AddrIn = CGF.GetAddrOfLocalVar(&OmpInParm);
  Scope.addPrivate(In, [&CGF, AddrIn, PtrTy]() {
    return CGF.EmitLoadOfPointerLValue(AddrIn, PtrTy->castAs<PointerType>())
        .getAddress(CGF);
  });
  Address AddrOut = CGF.GetAddrOfLocalVar(&OmpOutParm);
  Scope.addPrivate(Out, [&CGF, AddrOut, PtrTy]() {
    return CGF.EmitLoadOfPointerLValue(AddrOut, PtrTy->castAs<PointerType>())
        .getAddress(CGF);
  });
  (void)Scope.Privatize();
  if (!IsCombiner && Out->hasInit() &&
      !CGF.isTrivialInitializer(Out->getInit())) {
    CGF.EmitAnyExprToMem(Out->getInit(), CGF.GetAddrOfLocalVar(Out),
                         Out->getType().getQualifiers(),
                         /*IsInitializer=*/true);
  }
  if (CombinerInitializer)
    CGF.EmitIgnoredExpr(CombinerInitializer);
  Scope.ForceCleanup();
  CGF.FinishFunction();
  return Fn;
}

void CGOpenMPRuntime::emitUserDefinedReduction(
    CodeGenFunction *CGF, const OMPDeclareReductionDecl *D) {
  if (UDRMap.count(D) > 0)
    return;
  llvm::Function *Combiner = emitCombinerOrInitializer(
      CGM, D->getType(), D->getCombiner(),
      cast<VarDecl>(cast<DeclRefExpr>(D->getCombinerIn())->getDecl()),
      cast<VarDecl>(cast<DeclRefExpr>(D->getCombinerOut())->getDecl()),
      /*IsCombiner=*/true);
  llvm::Function *Initializer = nullptr;
  if (const Expr *Init = D->getInitializer()) {
    Initializer = emitCombinerOrInitializer(
        CGM, D->getType(),
        D->getInitializerKind() == OMPDeclareReductionDecl::CallInit ? Init
                                                                     : nullptr,
        cast<VarDecl>(cast<DeclRefExpr>(D->getInitOrig())->getDecl()),
        cast<VarDecl>(cast<DeclRefExpr>(D->getInitPriv())->getDecl()),
        /*IsCombiner=*/false);
  }
  UDRMap.try_emplace(D, Combiner, Initializer);
  if (CGF) {
    auto &Decls = FunctionUDRMap.FindAndConstruct(CGF->CurFn);
    Decls.second.push_back(D);
  }
}

std::pair<llvm::Function *, llvm::Function *>
CGOpenMPRuntime::getUserDefinedReduction(const OMPDeclareReductionDecl *D) {
  auto I = UDRMap.find(D);
  if (I != UDRMap.end())
    return I->second;
  emitUserDefinedReduction(/*CGF=*/nullptr, D);
  return UDRMap.lookup(D);
}

namespace {
// Temporary RAII solution to perform a push/pop stack event on the OpenMP IR
// Builder if one is present.
struct PushAndPopStackRAII {
  PushAndPopStackRAII(llvm::OpenMPIRBuilder *OMPBuilder, CodeGenFunction &CGF,
                      bool HasCancel, llvm::omp::Directive Kind)
      : OMPBuilder(OMPBuilder) {
    if (!OMPBuilder)
      return;

    // The following callback is the crucial part of clangs cleanup process.
    //
    // NOTE:
    // Once the OpenMPIRBuilder is used to create parallel regions (and
    // similar), the cancellation destination (Dest below) is determined via
    // IP. That means if we have variables to finalize we split the block at IP,
    // use the new block (=BB) as destination to build a JumpDest (via
    // getJumpDestInCurrentScope(BB)) which then is fed to
    // EmitBranchThroughCleanup. Furthermore, there will not be the need
    // to push & pop an FinalizationInfo object.
    // The FiniCB will still be needed but at the point where the
    // OpenMPIRBuilder is asked to construct a parallel (or similar) construct.
    auto FiniCB = [&CGF](llvm::OpenMPIRBuilder::InsertPointTy IP) {
      assert(IP.getBlock()->end() == IP.getPoint() &&
             "Clang CG should cause non-terminated block!");
      CGBuilderTy::InsertPointGuard IPG(CGF.Builder);
      CGF.Builder.restoreIP(IP);
      CodeGenFunction::JumpDest Dest =
          CGF.getOMPCancelDestination(OMPD_parallel);
      CGF.EmitBranchThroughCleanup(Dest);
    };

    // TODO: Remove this once we emit parallel regions through the
    //       OpenMPIRBuilder as it can do this setup internally.
    llvm::OpenMPIRBuilder::FinalizationInfo FI({FiniCB, Kind, HasCancel});
    OMPBuilder->pushFinalizationCB(std::move(FI));
  }
  ~PushAndPopStackRAII() {
    if (OMPBuilder)
      OMPBuilder->popFinalizationCB();
  }
  llvm::OpenMPIRBuilder *OMPBuilder;
};
} // namespace

static llvm::Function *emitParallelOrTeamsOutlinedFunction(
    CodeGenModule &CGM, const OMPExecutableDirective &D, const CapturedStmt *CS,
    const VarDecl *ThreadIDVar, OpenMPDirectiveKind InnermostKind,
    const StringRef OutlinedHelperName, const RegionCodeGenTy &CodeGen) {
  assert(ThreadIDVar->getType()->isPointerType() &&
         "thread id variable must be of type kmp_int32 *");
  CodeGenFunction CGF(CGM, true);
  bool HasCancel = false;
  if (const auto *OPD = dyn_cast<OMPParallelDirective>(&D))
    HasCancel = OPD->hasCancel();
  else if (const auto *OPD = dyn_cast<OMPTargetParallelDirective>(&D))
    HasCancel = OPD->hasCancel();
  else if (const auto *OPSD = dyn_cast<OMPParallelSectionsDirective>(&D))
    HasCancel = OPSD->hasCancel();
  else if (const auto *OPFD = dyn_cast<OMPParallelForDirective>(&D))
    HasCancel = OPFD->hasCancel();
  else if (const auto *OPFD = dyn_cast<OMPTargetParallelForDirective>(&D))
    HasCancel = OPFD->hasCancel();
  else if (const auto *OPFD = dyn_cast<OMPDistributeParallelForDirective>(&D))
    HasCancel = OPFD->hasCancel();
  else if (const auto *OPFD =
               dyn_cast<OMPTeamsDistributeParallelForDirective>(&D))
    HasCancel = OPFD->hasCancel();
  else if (const auto *OPFD =
               dyn_cast<OMPTargetTeamsDistributeParallelForDirective>(&D))
    HasCancel = OPFD->hasCancel();

  // TODO: Temporarily inform the OpenMPIRBuilder, if any, about the new
  //       parallel region to make cancellation barriers work properly.
  llvm::OpenMPIRBuilder &OMPBuilder = CGM.getOpenMPRuntime().getOMPBuilder();
  PushAndPopStackRAII PSR(&OMPBuilder, CGF, HasCancel, InnermostKind);
  CGOpenMPOutlinedRegionInfo CGInfo(*CS, ThreadIDVar, CodeGen, InnermostKind,
                                    HasCancel, OutlinedHelperName);
  CodeGenFunction::CGCapturedStmtRAII CapInfoRAII(CGF, &CGInfo);
  return CGF.GenerateOpenMPCapturedStmtFunction(*CS, D.getBeginLoc());
}

llvm::Function *CGOpenMPRuntime::emitParallelOutlinedFunction(
    const OMPExecutableDirective &D, const VarDecl *ThreadIDVar,
    OpenMPDirectiveKind InnermostKind, const RegionCodeGenTy &CodeGen) {
  const CapturedStmt *CS = D.getCapturedStmt(OMPD_parallel);
  return emitParallelOrTeamsOutlinedFunction(
      CGM, D, CS, ThreadIDVar, InnermostKind, getOutlinedHelperName(), CodeGen);
}

llvm::Function *CGOpenMPRuntime::emitTeamsOutlinedFunction(
    const OMPExecutableDirective &D, const VarDecl *ThreadIDVar,
    OpenMPDirectiveKind InnermostKind, const RegionCodeGenTy &CodeGen) {
  const CapturedStmt *CS = D.getCapturedStmt(OMPD_teams);
  return emitParallelOrTeamsOutlinedFunction(
      CGM, D, CS, ThreadIDVar, InnermostKind, getOutlinedHelperName(), CodeGen);
}

llvm::Function *CGOpenMPRuntime::emitTaskOutlinedFunction(
    const OMPExecutableDirective &D, const VarDecl *ThreadIDVar,
    const VarDecl *PartIDVar, const VarDecl *TaskTVar,
    OpenMPDirectiveKind InnermostKind, const RegionCodeGenTy &CodeGen,
    bool Tied, unsigned &NumberOfParts) {
  auto &&UntiedCodeGen = [this, &D, TaskTVar](CodeGenFunction &CGF,
                                              PrePostActionTy &) {
    llvm::Value *ThreadID = getThreadID(CGF, D.getBeginLoc());
    llvm::Value *UpLoc = emitUpdateLocation(CGF, D.getBeginLoc());
    llvm::Value *TaskArgs[] = {
        UpLoc, ThreadID,
        CGF.EmitLoadOfPointerLValue(CGF.GetAddrOfLocalVar(TaskTVar),
                                    TaskTVar->getType()->castAs<PointerType>())
            .getPointer(CGF)};
    CGF.EmitRuntimeCall(OMPBuilder.getOrCreateRuntimeFunction(
                            CGM.getModule(), OMPRTL___kmpc_omp_task),
                        TaskArgs);
  };
  CGOpenMPTaskOutlinedRegionInfo::UntiedTaskActionTy Action(Tied, PartIDVar,
                                                            UntiedCodeGen);
  CodeGen.setAction(Action);
  assert(!ThreadIDVar->getType()->isPointerType() &&
         "thread id variable must be of type kmp_int32 for tasks");
  const OpenMPDirectiveKind Region =
      isOpenMPTaskLoopDirective(D.getDirectiveKind()) ? OMPD_taskloop
                                                      : OMPD_task;
  const CapturedStmt *CS = D.getCapturedStmt(Region);
  bool HasCancel = false;
  if (const auto *TD = dyn_cast<OMPTaskDirective>(&D))
    HasCancel = TD->hasCancel();
  else if (const auto *TD = dyn_cast<OMPTaskLoopDirective>(&D))
    HasCancel = TD->hasCancel();
  else if (const auto *TD = dyn_cast<OMPMasterTaskLoopDirective>(&D))
    HasCancel = TD->hasCancel();
  else if (const auto *TD = dyn_cast<OMPParallelMasterTaskLoopDirective>(&D))
    HasCancel = TD->hasCancel();

  CodeGenFunction CGF(CGM, true);
  CGOpenMPTaskOutlinedRegionInfo CGInfo(*CS, ThreadIDVar, CodeGen,
                                        InnermostKind, HasCancel, Action);
  CodeGenFunction::CGCapturedStmtRAII CapInfoRAII(CGF, &CGInfo);
  llvm::Function *Res = CGF.GenerateCapturedStmtFunction(*CS);
  if (!Tied)
    NumberOfParts = Action.getNumberOfParts();
  return Res;
}

static void buildStructValue(ConstantStructBuilder &Fields, CodeGenModule &CGM,
                             const RecordDecl *RD, const CGRecordLayout &RL,
                             ArrayRef<llvm::Constant *> Data) {
  llvm::StructType *StructTy = RL.getLLVMType();
  unsigned PrevIdx = 0;
  ConstantInitBuilder CIBuilder(CGM);
  auto DI = Data.begin();
  for (const FieldDecl *FD : RD->fields()) {
    unsigned Idx = RL.getLLVMFieldNo(FD);
    // Fill the alignment.
    for (unsigned I = PrevIdx; I < Idx; ++I)
      Fields.add(llvm::Constant::getNullValue(StructTy->getElementType(I)));
    PrevIdx = Idx + 1;
    Fields.add(*DI);
    ++DI;
  }
}

template <class... As>
static llvm::GlobalVariable *
createGlobalStruct(CodeGenModule &CGM, QualType Ty, bool IsConstant,
                   ArrayRef<llvm::Constant *> Data, const Twine &Name,
                   As &&... Args) {
  const auto *RD = cast<RecordDecl>(Ty->getAsTagDecl());
  const CGRecordLayout &RL = CGM.getTypes().getCGRecordLayout(RD);
  ConstantInitBuilder CIBuilder(CGM);
  ConstantStructBuilder Fields = CIBuilder.beginStruct(RL.getLLVMType());
  buildStructValue(Fields, CGM, RD, RL, Data);
  return Fields.finishAndCreateGlobal(
      Name, CGM.getContext().getAlignOfGlobalVarInChars(Ty), IsConstant,
      std::forward<As>(Args)...);
}

template <typename T>
static void
createConstantGlobalStructAndAddToParent(CodeGenModule &CGM, QualType Ty,
                                         ArrayRef<llvm::Constant *> Data,
                                         T &Parent) {
  const auto *RD = cast<RecordDecl>(Ty->getAsTagDecl());
  const CGRecordLayout &RL = CGM.getTypes().getCGRecordLayout(RD);
  ConstantStructBuilder Fields = Parent.beginStruct(RL.getLLVMType());
  buildStructValue(Fields, CGM, RD, RL, Data);
  Fields.finishAndAddTo(Parent);
}

void CGOpenMPRuntime::setLocThreadIdInsertPt(CodeGenFunction &CGF,
                                             bool AtCurrentPoint) {
  auto &Elem = OpenMPLocThreadIDMap.FindAndConstruct(CGF.CurFn);
  assert(!Elem.second.ServiceInsertPt && "Insert point is set already.");

  llvm::Value *Undef = llvm::UndefValue::get(CGF.Int32Ty);
  if (AtCurrentPoint) {
    Elem.second.ServiceInsertPt = new llvm::BitCastInst(
        Undef, CGF.Int32Ty, "svcpt", CGF.Builder.GetInsertBlock());
  } else {
    Elem.second.ServiceInsertPt =
        new llvm::BitCastInst(Undef, CGF.Int32Ty, "svcpt");
    Elem.second.ServiceInsertPt->insertAfter(CGF.AllocaInsertPt);
  }
}

void CGOpenMPRuntime::clearLocThreadIdInsertPt(CodeGenFunction &CGF) {
  auto &Elem = OpenMPLocThreadIDMap.FindAndConstruct(CGF.CurFn);
  if (Elem.second.ServiceInsertPt) {
    llvm::Instruction *Ptr = Elem.second.ServiceInsertPt;
    Elem.second.ServiceInsertPt = nullptr;
    Ptr->eraseFromParent();
  }
}

static StringRef getIdentStringFromSourceLocation(CodeGenFunction &CGF,
                                                  SourceLocation Loc,
                                                  SmallString<128> &Buffer) {
  llvm::raw_svector_ostream OS(Buffer);
  // Build debug location
  PresumedLoc PLoc = CGF.getContext().getSourceManager().getPresumedLoc(Loc);
  OS << ";" << PLoc.getFilename() << ";";
  if (const auto *FD = dyn_cast_or_null<FunctionDecl>(CGF.CurFuncDecl))
    OS << FD->getQualifiedNameAsString();
  OS << ";" << PLoc.getLine() << ";" << PLoc.getColumn() << ";;";
  return OS.str();
}

llvm::Value *CGOpenMPRuntime::emitUpdateLocation(CodeGenFunction &CGF,
                                                 SourceLocation Loc,
                                                 unsigned Flags) {
  uint32_t SrcLocStrSize;
  llvm::Constant *SrcLocStr;
  if (CGM.getCodeGenOpts().getDebugInfo() == codegenoptions::NoDebugInfo ||
      Loc.isInvalid()) {
    SrcLocStr = OMPBuilder.getOrCreateDefaultSrcLocStr(SrcLocStrSize);
  } else {
    std::string FunctionName;
    if (const auto *FD = dyn_cast_or_null<FunctionDecl>(CGF.CurFuncDecl))
      FunctionName = FD->getQualifiedNameAsString();
    PresumedLoc PLoc = CGF.getContext().getSourceManager().getPresumedLoc(Loc);
    const char *FileName = PLoc.getFilename();
    unsigned Line = PLoc.getLine();
    unsigned Column = PLoc.getColumn();
    SrcLocStr = OMPBuilder.getOrCreateSrcLocStr(FunctionName, FileName, Line,
                                                Column, SrcLocStrSize);
  }
  unsigned Reserved2Flags = getDefaultLocationReserved2Flags();
  return OMPBuilder.getOrCreateIdent(
      SrcLocStr, SrcLocStrSize, llvm::omp::IdentFlag(Flags), Reserved2Flags);
}

llvm::Value *CGOpenMPRuntime::getThreadID(CodeGenFunction &CGF,
                                          SourceLocation Loc) {
  assert(CGF.CurFn && "No function in current CodeGenFunction.");
  // If the OpenMPIRBuilder is used we need to use it for all thread id calls as
  // the clang invariants used below might be broken.
  if (CGM.getLangOpts().OpenMPIRBuilder) {
    SmallString<128> Buffer;
    OMPBuilder.updateToLocation(CGF.Builder.saveIP());
    uint32_t SrcLocStrSize;
    auto *SrcLocStr = OMPBuilder.getOrCreateSrcLocStr(
        getIdentStringFromSourceLocation(CGF, Loc, Buffer), SrcLocStrSize);
    return OMPBuilder.getOrCreateThreadID(
        OMPBuilder.getOrCreateIdent(SrcLocStr, SrcLocStrSize));
  }

  llvm::Value *ThreadID = nullptr;
  // Check whether we've already cached a load of the thread id in this
  // function.
  auto I = OpenMPLocThreadIDMap.find(CGF.CurFn);
  if (I != OpenMPLocThreadIDMap.end()) {
    ThreadID = I->second.ThreadID;
    if (ThreadID != nullptr)
      return ThreadID;
  }
  // If exceptions are enabled, do not use parameter to avoid possible crash.
  if (auto *OMPRegionInfo =
          dyn_cast_or_null<CGOpenMPRegionInfo>(CGF.CapturedStmtInfo)) {
    if (OMPRegionInfo->getThreadIDVariable()) {
      // Check if this an outlined function with thread id passed as argument.
      LValue LVal = OMPRegionInfo->getThreadIDVariableLValue(CGF);
      llvm::BasicBlock *TopBlock = CGF.AllocaInsertPt->getParent();
      if (!CGF.EHStack.requiresLandingPad() || !CGF.getLangOpts().Exceptions ||
          !CGF.getLangOpts().CXXExceptions ||
          CGF.Builder.GetInsertBlock() == TopBlock ||
          !isa<llvm::Instruction>(LVal.getPointer(CGF)) ||
          cast<llvm::Instruction>(LVal.getPointer(CGF))->getParent() ==
              TopBlock ||
          cast<llvm::Instruction>(LVal.getPointer(CGF))->getParent() ==
              CGF.Builder.GetInsertBlock()) {
        ThreadID = CGF.EmitLoadOfScalar(LVal, Loc);
        // If value loaded in entry block, cache it and use it everywhere in
        // function.
        if (CGF.Builder.GetInsertBlock() == TopBlock) {
          auto &Elem = OpenMPLocThreadIDMap.FindAndConstruct(CGF.CurFn);
          Elem.second.ThreadID = ThreadID;
        }
        return ThreadID;
      }
    }
  }

  // This is not an outlined function region - need to call __kmpc_int32
  // kmpc_global_thread_num(ident_t *loc).
  // Generate thread id value and cache this value for use across the
  // function.
  auto &Elem = OpenMPLocThreadIDMap.FindAndConstruct(CGF.CurFn);
  if (!Elem.second.ServiceInsertPt)
    setLocThreadIdInsertPt(CGF);
  CGBuilderTy::InsertPointGuard IPG(CGF.Builder);
  CGF.Builder.SetInsertPoint(Elem.second.ServiceInsertPt);
  llvm::CallInst *Call = CGF.Builder.CreateCall(
      OMPBuilder.getOrCreateRuntimeFunction(CGM.getModule(),
                                            OMPRTL___kmpc_global_thread_num),
      emitUpdateLocation(CGF, Loc));
  Call->setCallingConv(CGF.getRuntimeCC());
  Elem.second.ThreadID = Call;
  return Call;
}

void CGOpenMPRuntime::functionFinished(CodeGenFunction &CGF) {
  assert(CGF.CurFn && "No function in current CodeGenFunction.");
  if (OpenMPLocThreadIDMap.count(CGF.CurFn)) {
    clearLocThreadIdInsertPt(CGF);
    OpenMPLocThreadIDMap.erase(CGF.CurFn);
  }
  if (FunctionUDRMap.count(CGF.CurFn) > 0) {
    for(const auto *D : FunctionUDRMap[CGF.CurFn])
      UDRMap.erase(D);
    FunctionUDRMap.erase(CGF.CurFn);
  }
  auto I = FunctionUDMMap.find(CGF.CurFn);
  if (I != FunctionUDMMap.end()) {
    for(const auto *D : I->second)
      UDMMap.erase(D);
    FunctionUDMMap.erase(I);
  }
  LastprivateConditionalToTypes.erase(CGF.CurFn);
  FunctionToUntiedTaskStackMap.erase(CGF.CurFn);
}

llvm::Type *CGOpenMPRuntime::getIdentTyPointerTy() {
  return OMPBuilder.IdentPtr;
}

llvm::Type *CGOpenMPRuntime::getKmpc_MicroPointerTy() {
  if (!Kmpc_MicroTy) {
    // Build void (*kmpc_micro)(kmp_int32 *global_tid, kmp_int32 *bound_tid,...)
    llvm::Type *MicroParams[] = {llvm::PointerType::getUnqual(CGM.Int32Ty),
                                 llvm::PointerType::getUnqual(CGM.Int32Ty)};
    Kmpc_MicroTy = llvm::FunctionType::get(CGM.VoidTy, MicroParams, true);
  }
  return llvm::PointerType::getUnqual(Kmpc_MicroTy);
}

llvm::FunctionCallee
CGOpenMPRuntime::createForStaticInitFunction(unsigned IVSize, bool IVSigned,
                                             bool IsGPUDistribute) {
  assert((IVSize == 32 || IVSize == 64) &&
         "IV size is not compatible with the omp runtime");
  StringRef Name;
  if (IsGPUDistribute)
    Name = IVSize == 32 ? (IVSigned ? "__kmpc_distribute_static_init_4"
                                    : "__kmpc_distribute_static_init_4u")
                        : (IVSigned ? "__kmpc_distribute_static_init_8"
                                    : "__kmpc_distribute_static_init_8u");
  else
    Name = IVSize == 32 ? (IVSigned ? "__kmpc_for_static_init_4"
                                    : "__kmpc_for_static_init_4u")
                        : (IVSigned ? "__kmpc_for_static_init_8"
                                    : "__kmpc_for_static_init_8u");

  llvm::Type *ITy = IVSize == 32 ? CGM.Int32Ty : CGM.Int64Ty;
  auto *PtrTy = llvm::PointerType::getUnqual(ITy);
  llvm::Type *TypeParams[] = {
    getIdentTyPointerTy(),                     // loc
    CGM.Int32Ty,                               // tid
    CGM.Int32Ty,                               // schedtype
    llvm::PointerType::getUnqual(CGM.Int32Ty), // p_lastiter
    PtrTy,                                     // p_lower
    PtrTy,                                     // p_upper
    PtrTy,                                     // p_stride
    ITy,                                       // incr
    ITy                                        // chunk
  };
  auto *FnTy =
      llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
  return CGM.CreateRuntimeFunction(FnTy, Name);
}

llvm::FunctionCallee
CGOpenMPRuntime::createDispatchInitFunction(unsigned IVSize, bool IVSigned) {
  assert((IVSize == 32 || IVSize == 64) &&
         "IV size is not compatible with the omp runtime");
  StringRef Name =
      IVSize == 32
          ? (IVSigned ? "__kmpc_dispatch_init_4" : "__kmpc_dispatch_init_4u")
          : (IVSigned ? "__kmpc_dispatch_init_8" : "__kmpc_dispatch_init_8u");
  llvm::Type *ITy = IVSize == 32 ? CGM.Int32Ty : CGM.Int64Ty;
  llvm::Type *TypeParams[] = { getIdentTyPointerTy(), // loc
                               CGM.Int32Ty,           // tid
                               CGM.Int32Ty,           // schedtype
                               ITy,                   // lower
                               ITy,                   // upper
                               ITy,                   // stride
                               ITy                    // chunk
  };
  auto *FnTy =
      llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg*/ false);
  return CGM.CreateRuntimeFunction(FnTy, Name);
}

llvm::FunctionCallee
CGOpenMPRuntime::createDispatchFiniFunction(unsigned IVSize, bool IVSigned) {
  assert((IVSize == 32 || IVSize == 64) &&
         "IV size is not compatible with the omp runtime");
  StringRef Name =
      IVSize == 32
          ? (IVSigned ? "__kmpc_dispatch_fini_4" : "__kmpc_dispatch_fini_4u")
          : (IVSigned ? "__kmpc_dispatch_fini_8" : "__kmpc_dispatch_fini_8u");
  llvm::Type *TypeParams[] = {
      getIdentTyPointerTy(), // loc
      CGM.Int32Ty,           // tid
  };
  auto *FnTy =
      llvm::FunctionType::get(CGM.VoidTy, TypeParams, /*isVarArg=*/false);
  return CGM.CreateRuntimeFunction(FnTy, Name);
}

llvm::FunctionCallee
CGOpenMPRuntime::createDispatchNextFunction(unsigned IVSize, bool IVSigned) {
  assert((IVSize == 32 || IVSize == 64) &&
         "IV size is not compatible with the omp runtime");
  StringRef Name =
      IVSize == 32
          ? (IVSigned ? "__kmpc_dispatch_next_4" : "__kmpc_dispatch_next_4u")
          : (IVSigned ? "__kmpc_dispatch_next_8" : "__kmpc_dispatch_next_8u");
  llvm::Type *ITy = IVSize == 32 ? CGM.Int32Ty : CGM.Int64Ty;
  auto *PtrTy = llvm::PointerType::getUnqual(ITy);
  llvm::Type *TypeParams[] = {
    getIdentTyPointerTy(),                     // loc
    CGM.Int32Ty,                               // tid
    llvm::PointerType::getUnqual(CGM.Int32Ty), // p_lastiter
    PtrTy,                                     // p_lower
    PtrTy,                                     // p_upper
    PtrTy                                      // p_stride
  };
  auto *FnTy =
      llvm::FunctionType::get(CGM.Int32Ty, TypeParams, /*isVarArg*/ false);
  return CGM.CreateRuntimeFunction(FnTy, Name);
}

/// Obtain information that uniquely identifies a target entry. This
/// consists of the file and device IDs as well as line number associated with
/// the relevant entry source location.
static void getTargetEntryUniqueInfo(ASTContext &C, SourceLocation Loc,
                                     unsigned &DeviceID, unsigned &FileID,
                                     unsigned &LineNum) {
  SourceManager &SM = C.getSourceManager();

  // The loc should be always valid and have a file ID (the user cannot use
  // #pragma directives in macros)

  assert(Loc.isValid() && "Source location is expected to be always valid.");

  PresumedLoc PLoc = SM.getPresumedLoc(Loc);
  assert(PLoc.isValid() && "Source location is expected to be always valid.");

  llvm::sys::fs::UniqueID ID;
  if (auto EC = llvm::sys::fs::getUniqueID(PLoc.getFilename(), ID)) {
    PLoc = SM.getPresumedLoc(Loc, /*UseLineDirectives=*/false);
    assert(PLoc.isValid() && "Source location is expected to be always valid.");
    if (auto EC = llvm::sys::fs::getUniqueID(PLoc.getFilename(), ID))
      SM.getDiagnostics().Report(diag::err_cannot_open_file)
          << PLoc.getFilename() << EC.message();
  }

  DeviceID = ID.getDevice();
  FileID = ID.getFile();
  LineNum = PLoc.getLine();
}

Address CGOpenMPRuntime::getAddrOfDeclareTargetVar(const VarDecl *VD) {
  if (CGM.getLangOpts().OpenMPSimd)
    return Address::invalid();
  llvm::Optional<OMPDeclareTargetDeclAttr::MapTypeTy> Res =
      OMPDeclareTargetDeclAttr::isDeclareTargetDeclaration(VD);
  if (Res && (*Res == OMPDeclareTargetDeclAttr::MT_Link ||
              (*Res == OMPDeclareTargetDeclAttr::MT_To &&
               HasRequiresUnifiedSharedMemory))) {
    SmallString<64> PtrName;
    {
      llvm::raw_svector_ostream OS(PtrName);
      OS << CGM.getMangledName(GlobalDecl(VD));
      if (!VD->isExternallyVisible()) {
        unsigned DeviceID, FileID, Line;
        getTargetEntryUniqueInfo(CGM.getContext(),
                                 VD->getCanonicalDecl()->getBeginLoc(),
                                 DeviceID, FileID, Line);
        OS << llvm::format("_%x", FileID);
      }
      OS << "_decl_tgt_ref_ptr";
    }
    llvm::Value *Ptr = CGM.getModule().getNamedValue(PtrName);
    if (!Ptr) {
      QualType PtrTy = CGM.getContext().getPointerType(VD->getType());
      Ptr = getOrCreateInternalVariable(CGM.getTypes().ConvertTypeForMem(PtrTy),
                                        PtrName);

      auto *GV = cast<llvm::GlobalVariable>(Ptr);
      GV->setLinkage(llvm::GlobalValue::WeakAnyLinkage);

      if (!CGM.getLangOpts().OpenMPIsDevice)
        GV->setInitializer(CGM.GetAddrOfGlobal(VD));
      registerTargetGlobalVariable(VD, cast<llvm::Constant>(Ptr));
    }
    return Address(Ptr, CGM.getContext().getDeclAlign(VD));
  }
  return Address::invalid();
}

llvm::Constant *
CGOpenMPRuntime::getOrCreateThreadPrivateCache(const VarDecl *VD) {
  assert(!CGM.getLangOpts().OpenMPUseTLS ||
         !CGM.getContext().getTargetInfo().isTLSSupported());
  // Lookup the entry, lazily creating it if necessary.
  std::string Suffix = getName({"cache", ""});
  return getOrCreateInternalVariable(
      CGM.Int8PtrPtrTy, Twine(CGM.getMangledName(VD)).concat(Suffix));
}

Address CGOpenMPRuntime::getAddrOfThreadPrivate(CodeGenFunction &CGF,
                                                const VarDecl *VD,
                                                Address VDAddr,
                                                SourceLocation Loc) {
  if (CGM.getLangOpts().OpenMPUseTLS &&
      CGM.getContext().getTargetInfo().isTLSSupported())
    return VDAddr;

  llvm::Type *VarTy = VDAddr.getElementType();
  llvm::Value *Args[] = {emitUpdateLocation(CGF, Loc), getThreadID(CGF, Loc),
                         CGF.Builder.CreatePointerCast(VDAddr.getPointer(),
                                                       CGM.Int8PtrTy),
                         CGM.getSize(CGM.GetTargetTypeStoreSize(VarTy)),
                         getOrCreateThreadPrivateCache(VD)};
  return Address(CGF.EmitRuntimeCall(
                     OMPBuilder.getOrCreateRuntimeFunction(
                         CGM.getModule(), OMPRTL___kmpc_threadprivate_cached),
                     Args),
                 VDAddr.getAlignment());
}

void CGOpenMPRuntime::emitThreadPrivateVarInit(
    CodeGenFunction &CGF, Address VDAddr, llvm::Value *Ctor,
    llvm::Value *CopyCtor, llvm::Value *Dtor, SourceLocation Loc) {
  // Call kmp_int32 __kmpc_global_thread_num(&loc) to init OpenMP runtime
  // library.
  llvm::Value *OMPLoc = emitUpdateLocation(CGF, Loc);
  CGF.EmitRuntimeCall(OMPBuilder.getOrCreateRuntimeFunction(
                          CGM.getModule(), OMPRTL___kmpc_global_thread_num),
                      OMPLoc);
  // Call __kmpc_threadprivate_register(&loc, &var, ctor, cctor/*NULL*/, dtor)
  // to register constructor/destructor for variable.
  llvm::Value *Args[] = {
      OMPLoc, CGF.Builder.CreatePointerCast(VDAddr.getPointer(), CGM.VoidPtrTy),
      Ctor, CopyCtor, Dtor};
  CGF.EmitRuntimeCall(
      OMPBuilder.getOrCreateRuntimeFunction(
          CGM.getModule(), OMPRTL___kmpc_threadprivate_register),
      Args);
}

llvm::Function *CGOpenMPRuntime::emitThreadPrivateVarDefinition(
    const VarDecl *VD, Address VDAddr, SourceLocation Loc,
    bool PerformInit, CodeGenFunction *CGF) {
  if (CGM.getLangOpts().OpenMPUseTLS &&
      CGM.getContext().getTargetInfo().isTLSSupported())
    return nullptr;

  VD = VD->getDefinition(CGM.getContext());
  if (VD && ThreadPrivateWithDefinition.insert(CGM.getMangledName(VD)).second) {
    QualType ASTTy = VD->getType();

    llvm::Value *Ctor = nullptr, *CopyCtor = nullptr, *Dtor = nullptr;
    const Expr *Init = VD->getAnyInitializer();
    if (CGM.getLangOpts().CPlusPlus && PerformInit) {
      // Generate function that re-emits the declaration's initializer into the
      // threadprivate copy of the variable VD
      CodeGenFunction CtorCGF(CGM);
      FunctionArgList Args;
      ImplicitParamDecl Dst(CGM.getContext(), /*DC=*/nullptr, Loc,
                            /*Id=*/nullptr, CGM.getContext().VoidPtrTy,
                            ImplicitParamDecl::Other);
      Args.push_back(&Dst);

      const auto &FI = CGM.getTypes().arrangeBuiltinFunctionDeclaration(
          CGM.getContext().VoidPtrTy, Args);
      llvm::FunctionType *FTy = CGM.getTypes().GetFunctionType(FI);
      std::string Name = getName({"__kmpc_global_ctor_", ""});
      llvm::Function *Fn =
          CGM.CreateGlobalInitOrCleanUpFunction(FTy, Name, FI, Loc);
      CtorCGF.StartFunction(GlobalDecl(), CGM.getContext().VoidPtrTy, Fn, FI,
                            Args, Loc, Loc);
      llvm::Value *ArgVal = CtorCGF.EmitLoadOfScalar(
          CtorCGF.GetAddrOfLocalVar(&Dst), /*Volatile=*/false,
          CGM.getContext().VoidPtrTy, Dst.getLocation());
      Address Arg = Address(ArgVal, VDAddr.getAlignment());
      Arg = CtorCGF.Builder.CreateElementBitCast(
          Arg, CtorCGF.ConvertTypeForMem(ASTTy));
      CtorCGF.EmitAnyExprToMem(Init, Arg, Init->getType().getQualifiers(),
                               /*IsInitializer=*/true);
      ArgVal = CtorCGF.EmitLoadOfScalar(
          CtorCGF.GetAddrOfLocalVar(&Dst), /*Volatile=*/false,
          CGM.getContext().VoidPtrTy, Dst.getLocation());
      CtorCGF.Builder.CreateStore(ArgVal, CtorCGF.ReturnValue);
      CtorCGF.FinishFunction();
      Ctor = Fn;
    }
    if (VD->getType().isDestructedType() != QualType::DK_none) {
      // Generate function that emits destructor call for the threadprivate copy
      // of the variable VD
      CodeGenFunction DtorCGF(CGM);
      FunctionArgList Args;
      ImplicitParamDecl Dst(CGM.getContext(), /*DC=*/nullptr, Loc,
                            /*Id=*/nullptr, CGM.getContext().VoidPtrTy,
                            ImplicitParamDecl::Other);
      Args.push_back(&Dst);

      const auto &FI = CGM.getTypes().arrangeBuiltinFunctionDeclaration(
          CGM.getContext().VoidTy, Args);
      llvm::FunctionType *FTy = CGM.getTypes().GetFunctionType(FI);
      std::string Name = getName({"__kmpc_global_dtor_", ""});
      llvm::Function *Fn =
          CGM.CreateGlobalInitOrCleanUpFunction(FTy, Name, FI, Loc);
      auto NL = ApplyDebugLocation::CreateEmpty(DtorCGF);
      DtorCGF.StartFunction(GlobalDecl(), CGM.getContext().VoidTy, Fn, FI, Args,
                            Loc, Loc);
      // Create a scope with an artificial location for the body of this function.
      auto AL = ApplyDebugLocation::CreateArtificial(DtorCGF);
      llvm::Value *ArgVal = DtorCGF.EmitLoadOfScalar(
          DtorCGF.GetAddrOfLocalVar(&Dst),
          /*Volatile=*/false, CGM.getContext().VoidPtrTy, Dst.getLocation());
      DtorCGF.emitDestroy(Address(ArgVal, VDAddr.getAlignment()), ASTTy,
                          DtorCGF.getDestroyer(ASTTy.isDestructedType()),
                          DtorCGF.needsEHCleanup(ASTTy.isDestructedType()));
      DtorCGF.FinishFunction();
      Dtor = Fn;
    }
    // Do not emit init function if it is not required.
    if (!Ctor && !Dtor)
      return nullptr;

    llvm::Type *CopyCtorTyArgs[] = {CGM.VoidPtrTy, CGM.VoidPtrTy};
    auto *CopyCtorTy = llvm::FunctionType::get(CGM.VoidPtrTy, CopyCtorTyArgs,
                                               /*isVarArg=*/false)
                           ->getPointerTo();
    // Copying constructor for the threadprivate variable.
    // Must be NULL - reserved by runtime, but currently it requires that this
    // parameter is always NULL. Otherwise it fires assertion.
    CopyCtor = llvm::Constant::getNullValue(CopyCtorTy);
    if (Ctor == nullptr) {
      auto *CtorTy = llvm::FunctionType::get(CGM.VoidPtrTy, CGM.VoidPtrTy,
                                             /*isVarArg=*/false)
                         ->getPointerTo();
      Ctor = llvm::Constant::getNullValue(CtorTy);
    }
    if (Dtor == nullptr) {
      auto *DtorTy = llvm::FunctionType::get(CGM.VoidTy, CGM.VoidPtrTy,
                                             /*isVarArg=*/false)
                         ->getPointerTo();
      Dtor = llvm::Constant::getNullValue(DtorTy);
    }
    if (!CGF) {
      auto *InitFunctionTy =
          llvm::FunctionType::get(CGM.VoidTy, /*isVarArg*/ false);
      std::string Name = getName({"__omp_threadprivate_init_", ""});
      llvm::Function *InitFunction = CGM.CreateGlobalInitOrCleanUpFunction(
          InitFunctionTy, Name, CGM.getTypes().arrangeNullaryFunction());
      CodeGenFunction InitCGF(CGM);
      FunctionArgList ArgList;
      InitCGF.StartFunction(GlobalDecl(), CGM.getContext().VoidTy, InitFunction,
                            CGM.getTypes().arrangeNullaryFunction(), ArgList,
                            Loc, Loc);
      emitThreadPrivateVarInit(InitCGF, VDAddr, Ctor, CopyCtor, Dtor, Loc);
      InitCGF.FinishFunction();
      return InitFunction;
    }
    emitThreadPrivateVarInit(*CGF, VDAddr, Ctor, CopyCtor, Dtor, Loc);
  }
  return nullptr;
}

bool CGOpenMPRuntime::emitDeclareTargetVarDefinition(const VarDecl *VD,
                                                     llvm::GlobalVariable *Addr,
                                                     bool PerformInit) {
  if (CGM.getLangOpts().OMPTargetTriples.empty() &&
      !CGM.getLangOpts().OpenMPIsDevice)
    return false;
  Optional<OMPDeclareTargetDeclAttr::MapTypeTy> Res =
      OMPDeclareTargetDeclAttr::isDeclareTargetDeclaration(VD);
  if (!Res || *Res == OMPDeclareTargetDeclAttr::MT_Link ||
      (*Res == OMPDeclareTargetDeclAttr::MT_To &&
       HasRequiresUnifiedSharedMemory))
    return CGM.getLangOpts().OpenMPIsDevice;
  VD = VD->getDefinition(CGM.getContext());
  assert(VD && "Unknown VarDecl");

  if (!DeclareTargetWithDefinition.insert(CGM.getMangledName(VD)).second)
    return CGM.getLangOpts().OpenMPIsDevice;

  QualType ASTTy = VD->getType();
  SourceLocation Loc = VD->getCanonicalDecl()->getBeginLoc();

  // Produce the unique prefix to identify the new target regions. We use
  // the source location of the variable declaration which we know to not
  // conflict with any target region.
  unsigned DeviceID;
  unsigned FileID;
  unsigned Line;
  getTargetEntryUniqueInfo(CGM.getContext(), Loc, DeviceID, FileID, Line);
  SmallString<128> Buffer, Out;
  {
    llvm::raw_svector_ostream OS(Buffer);
    OS << "__omp_offloading_" << llvm::format("_%x", DeviceID)
       << llvm::format("_%x_", FileID) << VD->getName() << "_l" << Line;
  }

  const Expr *Init = VD->getAnyInitializer();
  if (CGM.getLangOpts().CPlusPlus && PerformInit) {
    llvm::Constant *Ctor;
    llvm::Constant *ID;
    if (CGM.getLangOpts().OpenMPIsDevice) {
      // Generate function that re-emits the declaration's initializer into
      // the threadprivate copy of the variable VD
      CodeGenFunction CtorCGF(CGM);

      const CGFunctionInfo &FI = CGM.getTypes().arrangeNullaryFunction();
      llvm::FunctionType *FTy = CGM.getTypes().GetFunctionType(FI);
      llvm::Function *Fn = CGM.CreateGlobalInitOrCleanUpFunction(
          FTy, Twine(Buffer, "_ctor"), FI, Loc);
      auto NL = ApplyDebugLocation::CreateEmpty(CtorCGF);
      CtorCGF.StartFunction(GlobalDecl(), CGM.getContext().VoidTy, Fn, FI,
                            FunctionArgList(), Loc, Loc);
      auto AL = ApplyDebugLocation::CreateArtificial(CtorCGF);
      CtorCGF.EmitAnyExprToMem(Init,
                               Address(Addr, CGM.getContext().getDeclAlign(VD)),
                               Init->getType().getQualifiers(),
                               /*IsInitializer=*/true);
      CtorCGF.FinishFunction();
      Ctor = Fn;
      ID = llvm::ConstantExpr::getBitCast(Fn, CGM.Int8PtrTy);
      CGM.addUsedGlobal(cast<llvm::GlobalValue>(Ctor));
    } else {
      Ctor = new llvm::GlobalVariable(
          CGM.getModule(), CGM.Int8Ty, /*isConstant=*/true,
          llvm::GlobalValue::PrivateLinkage,
          llvm::Constant::getNullValue(CGM.Int8Ty), Twine(Buffer, "_ctor"));
      ID = Ctor;
    }

    // Register the information for the entry associated with the constructor.
    Out.clear();
    OffloadEntriesInfoManager.registerTargetRegionEntryInfo(
        DeviceID, FileID, Twine(Buffer, "_ctor").toStringRef(Out), Line, Ctor,
        ID, OffloadEntriesInfoManagerTy::OMPTargetRegionEntryCtor);
  }
  if (VD->getType().isDestructedType() != QualType::DK_none) {
    llvm::Constant *Dtor;
    llvm::Constant *ID;
    if (CGM.getLangOpts().OpenMPIsDevice) {
      // Generate function that emits destructor call for the threadprivate
      // copy of the variable VD
      CodeGenFunction DtorCGF(CGM);

      const CGFunctionInfo &FI = CGM.getTypes().arrangeNullaryFunction();
      llvm::FunctionType *FTy = CGM.getTypes().GetFunctionType(FI);
      llvm::Function *Fn = CGM.CreateGlobalInitOrCleanUpFunction(
          FTy, Twine(Buffer, "_dtor"), FI, Loc);
      auto NL = ApplyDebugLocation::CreateEmpty(DtorCGF);
      DtorCGF.StartFunction(GlobalDecl(), CGM.getContext().VoidTy, Fn, FI,
                            FunctionArgList(), Loc, Loc);
      // Create a scope with an artificial location for the body of this
      // function.
      auto AL = ApplyDebugLocation::CreateArtificial(DtorCGF);
      DtorCGF.emitDestroy(Address(Addr, CGM.getContext().getDeclAlign(VD)),
                          ASTTy, DtorCGF.getDestroyer(ASTTy.isDestructedType()),
                          DtorCGF.needsEHCleanup(ASTTy.isDestructedType()));
      DtorCGF.FinishFunction();
      Dtor = Fn;
      ID = llvm::ConstantExpr::getBitCast(Fn, CGM.Int8PtrTy);
      CGM.addUsedGlobal(cast<llvm::GlobalValue>(Dtor));
    } else {
      Dtor = new llvm::GlobalVariable(
          CGM.getModule(), CGM.Int8Ty, /*isConstant=*/true,
          llvm::GlobalValue::PrivateLinkage,
          llvm::Constant::getNullValue(CGM.Int8Ty), Twine(Buffer, "_dtor"));
      ID = Dtor;
    }
    // Register the information for the entry associated with the destructor.
    Out.clear();
    OffloadEntriesInfoManager.registerTargetRegionEntryInfo(
        DeviceID, FileID, Twine(Buffer, "_dtor").toStringRef(Out), Line, Dtor,
        ID, OffloadEntriesInfoManagerTy::OMPTargetRegionEntryDtor);
  }
  return CGM.getLangOpts().OpenMPIsDevice;
}

Address CGOpenMPRuntime::getAddrOfArtificialThreadPrivate(CodeGenFunction &CGF,
                                                          QualType VarType,
                                                          StringRef Name) {
  std::string Suffix = getName({"artificial", ""});
  llvm::Type *VarLVType = CGF.ConvertTypeForMem(VarType);
  llvm::GlobalVariable *GAddr =
      getOrCreateInternalVariable(VarLVType, Twine(Name).concat(Suffix));
  if (CGM.getLangOpts().OpenMP && CGM.getLangOpts().OpenMPUseTLS &&
      CGM.getTarget().isTLSSupported()) {
    GAddr->setThreadLocal(/*Val=*/true);
    return Address(GAddr, GAddr->getValueType(),
                   CGM.getContext().getTypeAlignInChars(VarType));
  }
  std::string CacheSuffix = getName({"cache", ""});
  llvm::Value *Args[] = {
      emitUpdateLocation(CGF, SourceLocation()),
      getThreadID(CGF, SourceLocation()),
      CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(GAddr, CGM.VoidPtrTy),
      CGF.Builder.CreateIntCast(CGF.getTypeSize(VarType), CGM.SizeTy,
                                /*isSigned=*/false),
      getOrCreateInternalVariable(
          CGM.VoidPtrPtrTy, Twine(Name).concat(Suffix).concat(CacheSuffix))};
  return Address(
      CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
          CGF.EmitRuntimeCall(
              OMPBuilder.getOrCreateRuntimeFunction(
                  CGM.getModule(), OMPRTL___kmpc_threadprivate_cached),
              Args),
          VarLVType->getPointerTo(/*AddrSpace=*/0)),
      CGM.getContext().getTypeAlignInChars(VarType));
}

void CGOpenMPRuntime::emitIfClause(CodeGenFunction &CGF, const Expr *Cond,
                                   const RegionCodeGenTy &ThenGen,
                                   const RegionCodeGenTy &ElseGen) {
  CodeGenFunction::LexicalScope ConditionScope(CGF, Cond->getSourceRange());

  // If the condition constant folds and can be elided, try to avoid emitting
  // the condition and the dead arm of the if/else.
  bool CondConstant;
  if (CGF.ConstantFoldsToSimpleInteger(Cond, CondConstant)) {
    if (CondConstant)
      ThenGen(CGF);
    else
      ElseGen(CGF);
    return;
  }

  // Otherwise, the condition did not fold, or we couldn't elide it.  Just
  // emit the conditional branch.
  llvm::BasicBlock *ThenBlock = CGF.createBasicBlock("omp_if.then");
  llvm::BasicBlock *ElseBlock = CGF.createBasicBlock("omp_if.else");
  llvm::BasicBlock *ContBlock = CGF.createBasicBlock("omp_if.end");
  CGF.EmitBranchOnBoolExpr(Cond, ThenBlock, ElseBlock, /*TrueCount=*/0);

  // Emit the 'then' code.
  CGF.EmitBlock(ThenBlock);
  ThenGen(CGF);
  CGF.EmitBranch(ContBlock);
  // Emit the 'else' code if present.
  // There is no need to emit line number for unconditional branch.
  (void)ApplyDebugLocation::CreateEmpty(CGF);
  CGF.EmitBlock(ElseBlock);
  ElseGen(CGF);
  // There is no need to emit line number for unconditional branch.
  (void)ApplyDebugLocation::CreateEmpty(CGF);
  CGF.EmitBranch(ContBlock);
  // Emit the continuation block for code after the if.
  CGF.EmitBlock(ContBlock, /*IsFinished=*/true);
}

void CGOpenMPRuntime::emitParallelCall(CodeGenFunction &CGF, SourceLocation Loc,
                                       llvm::Function *OutlinedFn,
                                       ArrayRef<llvm::Value *> CapturedVars,
                                       const Expr *IfCond,
                                       llvm::Value *NumThreads) {
  if (!CGF.HaveInsertPoint())
    return;
  llvm::Value *RTLoc = emitUpdateLocation(CGF, Loc);
  auto &M = CGM.getModule();
  auto &&ThenGen = [&M, OutlinedFn, CapturedVars, RTLoc,
                    this](CodeGenFunction &CGF, PrePostActionTy &) {
    // Build call __kmpc_fork_call(loc, n, microtask, var1, .., varn);
    CGOpenMPRuntime &RT = CGF.CGM.getOpenMPRuntime();
    llvm::Value *Args[] = {
        RTLoc,
        CGF.Builder.getInt32(CapturedVars.size()), // Number of captured vars
        CGF.Builder.CreateBitCast(OutlinedFn, RT.getKmpc_MicroPointerTy())};
    llvm::SmallVector<llvm::Value *, 16> RealArgs;
    RealArgs.append(std::begin(Args), std::end(Args));
    RealArgs.append(CapturedVars.begin(), CapturedVars.end());

    llvm::FunctionCallee RTLFn =
        OMPBuilder.getOrCreateRuntimeFunction(M, OMPRTL___kmpc_fork_call);
    CGF.EmitRuntimeCall(RTLFn, RealArgs);
  };
  auto &&ElseGen = [&M, OutlinedFn, CapturedVars, RTLoc, Loc,
                    this](CodeGenFunction &CGF, PrePostActionTy &) {
    CGOpenMPRuntime &RT = CGF.CGM.getOpenMPRuntime();
    llvm::Value *ThreadID = RT.getThreadID(CGF, Loc);
    // Build calls:
    // __kmpc_serialized_parallel(&Loc, GTid);
    llvm::Value *Args[] = {RTLoc, ThreadID};
    CGF.EmitRuntimeCall(OMPBuilder.getOrCreateRuntimeFunction(
                            M, OMPRTL___kmpc_serialized_parallel),
                        Args);

    // OutlinedFn(&GTid, &zero_bound, CapturedStruct);
    Address ThreadIDAddr = RT.emitThreadIDAddress(CGF, Loc);
    Address ZeroAddrBound =
        CGF.CreateDefaultAlignTempAlloca(CGF.Int32Ty,
                                         /*Name=*/".bound.zero.addr");
    CGF.Builder.CreateStore(CGF.Builder.getInt32(/*C*/ 0), ZeroAddrBound);
    llvm::SmallVector<llvm::Value *, 16> OutlinedFnArgs;
    // ThreadId for serialized parallels is 0.
    OutlinedFnArgs.push_back(ThreadIDAddr.getPointer());
    OutlinedFnArgs.push_back(ZeroAddrBound.getPointer());
    OutlinedFnArgs.append(CapturedVars.begin(), CapturedVars.end());

    // Ensure we do not inline the function. This is trivially true for the ones
    // passed to __kmpc_fork_call but the ones called in serialized regions
    // could be inlined. This is not a perfect but it is closer to the invariant
    // we want, namely, every data environment starts with a new function.
    // TODO: We should pass the if condition to the runtime function and do the
    //       handling there. Much cleaner code.
    OutlinedFn->removeFnAttr(llvm::Attribute::AlwaysInline);
    OutlinedFn->addFnAttr(llvm::Attribute::NoInline);
    RT.emitOutlinedFunctionCall(CGF, Loc, OutlinedFn, OutlinedFnArgs);

    // __kmpc_end_serialized_parallel(&Loc, GTid);
    llvm::Value *EndArgs[] = {RT.emitUpdateLocation(CGF, Loc), ThreadID};
    CGF.EmitRuntimeCall(OMPBuilder.getOrCreateRuntimeFunction(
                            M, OMPRTL___kmpc_end_serialized_parallel),
                        EndArgs);
  };
  if (IfCond) {
    emitIfClause(CGF, IfCond, ThenGen, ElseGen);
  } else {
    RegionCodeGenTy ThenRCG(ThenGen);
    ThenRCG(CGF);
  }
}

// If we're inside an (outlined) parallel region, use the region info's
// thread-ID variable (it is passed in a first argument of the outlined function
// as "kmp_int32 *gtid"). Otherwise, if we're not inside parallel region, but in
// regular serial code region, get thread ID by calling kmp_int32
// kmpc_global_thread_num(ident_t *loc), stash this thread ID in a temporary and
// return the address of that temp.
Address CGOpenMPRuntime::emitThreadIDAddress(CodeGenFunction &CGF,
                                             SourceLocation Loc) {
  if (auto *OMPRegionInfo =
          dyn_cast_or_null<CGOpenMPRegionInfo>(CGF.CapturedStmtInfo))
    if (OMPRegionInfo->getThreadIDVariable())
      return OMPRegionInfo->getThreadIDVariableLValue(CGF).getAddress(CGF);

  llvm::Value *ThreadID = getThreadID(CGF, Loc);
  QualType Int32Ty =
      CGF.getContext().getIntTypeForBitwidth(/*DestWidth*/ 32, /*Signed*/ true);
  Address ThreadIDTemp = CGF.CreateMemTemp(Int32Ty, /*Name*/ ".threadid_temp.");
  CGF.EmitStoreOfScalar(ThreadID,
                        CGF.MakeAddrLValue(ThreadIDTemp, Int32Ty));

  return ThreadIDTemp;
}

llvm::GlobalVariable *CGOpenMPRuntime::getOrCreateInternalVariable(
    llvm::Type *Ty, const llvm::Twine &Name, unsigned AddressSpace) {
  SmallString<256> Buffer;
  llvm::raw_svector_ostream Out(Buffer);
  Out << Name;
  StringRef RuntimeName = Out.str();
  auto &Elem = *InternalVars.try_emplace(RuntimeName, nullptr).first;
  if (Elem.second) {
    assert(Elem.second->getType()->isOpaqueOrPointeeTypeMatches(Ty) &&
           "OMP internal variable has different type than requested");
    return &*Elem.second;
  }

  return Elem.second = new llvm::GlobalVariable(
             CGM.getModule(), Ty, /*IsConstant*/ false,
             llvm::GlobalValue::CommonLinkage, llvm::Constant::getNullValue(Ty),
             Elem.first(), /*InsertBefore=*/nullptr,
             llvm::GlobalValue::NotThreadLocal, AddressSpace);
}

llvm::Value *CGOpenMPRuntime::getCriticalRegionLock(StringRef CriticalName) {
  std::string Prefix = Twine("gomp_critical_user_", CriticalName).str();
  std::string Name = getName({Prefix, "var"});
  return getOrCreateInternalVariable(KmpCriticalNameTy, Name);
}

namespace {
/// Common pre(post)-action for different OpenMP constructs.
class CommonActionTy final : public PrePostActionTy {
  llvm::FunctionCallee EnterCallee;
  ArrayRef<llvm::Value *> EnterArgs;
  llvm::FunctionCallee ExitCallee;
  ArrayRef<llvm::Value *> ExitArgs;
  bool Conditional;
  llvm::BasicBlock *ContBlock = nullptr;

public:
  CommonActionTy(llvm::FunctionCallee EnterCallee,
                 ArrayRef<llvm::Value *> EnterArgs,
                 llvm::FunctionCallee ExitCallee,
                 ArrayRef<llvm::Value *> ExitArgs, bool Conditional = false)
      : EnterCallee(EnterCallee), EnterArgs(EnterArgs), ExitCallee(ExitCallee),
        ExitArgs(ExitArgs), Conditional(Conditional) {}
  void Enter(CodeGenFunction &CGF) override {
    llvm::Value *EnterRes = CGF.EmitRuntimeCall(EnterCallee, EnterArgs);
    if (Conditional) {
      llvm::Value *CallBool = CGF.Builder.CreateIsNotNull(EnterRes);
      auto *ThenBlock = CGF.createBasicBlock("omp_if.then");
      ContBlock = CGF.createBasicBlock("omp_if.end");
      // Generate the branch (If-stmt)
      CGF.Builder.CreateCondBr(CallBool, ThenBlock, ContBlock);
      CGF.EmitBlock(ThenBlock);
    }
  }
  void Done(CodeGenFunction &CGF) {
    // Emit the rest of blocks/branches
    CGF.EmitBranch(ContBlock);
    CGF.EmitBlock(ContBlock, true);
  }
  void Exit(CodeGenFunction &CGF) override {
    CGF.EmitRuntimeCall(ExitCallee, ExitArgs);
  }
};
} // anonymous namespace

void CGOpenMPRuntime::emitCriticalRegion(CodeGenFunction &CGF,
                                         StringRef CriticalName,
                                         const RegionCodeGenTy &CriticalOpGen,
                                         SourceLocation Loc, const Expr *Hint) {
  // __kmpc_critical[_with_hint](ident_t *, gtid, Lock[, hint]);
  // CriticalOpGen();
  // __kmpc_end_critical(ident_t *, gtid, Lock);
  // Prepare arguments and build a call to __kmpc_critical
  if (!CGF.HaveInsertPoint())
    return;
  llvm::Value *Args[] = {emitUpdateLocation(CGF, Loc), getThreadID(CGF, Loc),
                         getCriticalRegionLock(CriticalName)};
  llvm::SmallVector<llvm::Value *, 4> EnterArgs(std::begin(Args),
                                                std::end(Args));
  if (Hint) {
    EnterArgs.push_back(CGF.Builder.CreateIntCast(
        CGF.EmitScalarExpr(Hint), CGM.Int32Ty, /*isSigned=*/false));
  }
  CommonActionTy Action(
      OMPBuilder.getOrCreateRuntimeFunction(
          CGM.getModule(),
          Hint ? OMPRTL___kmpc_critical_with_hint : OMPRTL___kmpc_critical),
      EnterArgs,
      OMPBuilder.getOrCreateRuntimeFunction(CGM.getModule(),
                                            OMPRTL___kmpc_end_critical),
      Args);
  CriticalOpGen.setAction(Action);
  emitInlinedDirective(CGF, OMPD_critical, CriticalOpGen);
}

void CGOpenMPRuntime::emitMasterRegion(CodeGenFunction &CGF,
                                       const RegionCodeGenTy &MasterOpGen,
                                       SourceLocation Loc) {
  if (!CGF.HaveInsertPoint())
    return;
  // if(__kmpc_master(ident_t *, gtid)) {
  //   MasterOpGen();
  //   __kmpc_end_master(ident_t *, gtid);
  // }
  // Prepare arguments and build a call to __kmpc_master
  llvm::Value *Args[] = {emitUpdateLocation(CGF, Loc), getThreadID(CGF, Loc)};
  CommonActionTy Action(OMPBuilder.getOrCreateRuntimeFunction(
                            CGM.getModule(), OMPRTL___kmpc_master),
                        Args,
                        OMPBuilder.getOrCreateRuntimeFunction(
                            CGM.getModule(), OMPRTL___kmpc_end_master),
                        Args,
                        /*Conditional=*/true);
  MasterOpGen.setAction(Action);
  emitInlinedDirective(CGF, OMPD_master, MasterOpGen);
  Action.Done(CGF);
}

void CGOpenMPRuntime::emitMaskedRegion(CodeGenFunction &CGF,
                                       const RegionCodeGenTy &MaskedOpGen,
                                       SourceLocation Loc, const Expr *Filter) {
  if (!CGF.HaveInsertPoint())
    return;
  // if(__kmpc_masked(ident_t *, gtid, filter)) {
  //   MaskedOpGen();
  //   __kmpc_end_masked(iden_t *, gtid);
  // }
  // Prepare arguments and build a call to __kmpc_masked
  llvm::Value *FilterVal = Filter
                               ? CGF.EmitScalarExpr(Filter, CGF.Int32Ty)
                               : llvm::ConstantInt::get(CGM.Int32Ty, /*V=*/0);
  llvm::Value *Args[] = {emitUpdateLocation(CGF, Loc), getThreadID(CGF, Loc),
                         FilterVal};
  llvm::Value *ArgsEnd[] = {emitUpdateLocation(CGF, Loc),
                            getThreadID(CGF, Loc)};
  CommonActionTy Action(OMPBuilder.getOrCreateRuntimeFunction(
                            CGM.getModule(), OMPRTL___kmpc_masked),
                        Args,
                        OMPBuilder.getOrCreateRuntimeFunction(
                            CGM.getModule(), OMPRTL___kmpc_end_masked),
                        ArgsEnd,
                        /*Conditional=*/true);
  MaskedOpGen.setAction(Action);
  emitInlinedDirective(CGF, OMPD_masked, MaskedOpGen);
  Action.Done(CGF);
}

void CGOpenMPRuntime::emitTaskyieldCall(CodeGenFunction &CGF,
                                        SourceLocation Loc) {
  if (!CGF.HaveInsertPoint())
    return;
  if (CGF.CGM.getLangOpts().OpenMPIRBuilder) {
    OMPBuilder.createTaskyield(CGF.Builder);
  } else {
    // Build call __kmpc_omp_taskyield(loc, thread_id, 0);
    llvm::Value *Args[] = {
        emitUpdateLocation(CGF, Loc), getThreadID(CGF, Loc),
        llvm::ConstantInt::get(CGM.IntTy, /*V=*/0, /*isSigned=*/true)};
    CGF.EmitRuntimeCall(OMPBuilder.getOrCreateRuntimeFunction(
                            CGM.getModule(), OMPRTL___kmpc_omp_taskyield),
                        Args);
  }

  if (auto *Region = dyn_cast_or_null<CGOpenMPRegionInfo>(CGF.CapturedStmtInfo))
    Region->emitUntiedSwitch(CGF);
}

void CGOpenMPRuntime::emitTaskgroupRegion(CodeGenFunction &CGF,
                                          const RegionCodeGenTy &TaskgroupOpGen,
                                          SourceLocation Loc) {
  if (!CGF.HaveInsertPoint())
    return;
  // __kmpc_taskgroup(ident_t *, gtid);
  // TaskgroupOpGen();
  // __kmpc_end_taskgroup(ident_t *, gtid);
  // Prepare arguments and build a call to __kmpc_taskgroup
  llvm::Value *Args[] = {emitUpdateLocation(CGF, Loc), getThreadID(CGF, Loc)};
  CommonActionTy Action(OMPBuilder.getOrCreateRuntimeFunction(
                            CGM.getModule(), OMPRTL___kmpc_taskgroup),
                        Args,
                        OMPBuilder.getOrCreateRuntimeFunction(
                            CGM.getModule(), OMPRTL___kmpc_end_taskgroup),
                        Args);
  TaskgroupOpGen.setAction(Action);
  emitInlinedDirective(CGF, OMPD_taskgroup, TaskgroupOpGen);
}

/// Given an array of pointers to variables, project the address of a
/// given variable.
static Address emitAddrOfVarFromArray(CodeGenFunction &CGF, Address Array,
                                      unsigned Index, const VarDecl *Var) {
  // Pull out the pointer to the variable.
  Address PtrAddr = CGF.Builder.CreateConstArrayGEP(Array, Index);
  llvm::Value *Ptr = CGF.Builder.CreateLoad(PtrAddr);

  Address Addr = Address(Ptr, CGF.getContext().getDeclAlign(Var));
  Addr = CGF.Builder.CreateElementBitCast(
      Addr, CGF.ConvertTypeForMem(Var->getType()));
  return Addr;
}

static llvm::Value *emitCopyprivateCopyFunction(
    CodeGenModule &CGM, llvm::Type *ArgsType,
    ArrayRef<const Expr *> CopyprivateVars, ArrayRef<const Expr *> DestExprs,
    ArrayRef<const Expr *> SrcExprs, ArrayRef<const Expr *> AssignmentOps,
    SourceLocation Loc) {
  ASTContext &C = CGM.getContext();
  // void copy_func(void *LHSArg, void *RHSArg);
  FunctionArgList Args;
  ImplicitParamDecl LHSArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr, C.VoidPtrTy,
                           ImplicitParamDecl::Other);
  ImplicitParamDecl RHSArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr, C.VoidPtrTy,
                           ImplicitParamDecl::Other);
  Args.push_back(&LHSArg);
  Args.push_back(&RHSArg);
  const auto &CGFI =
      CGM.getTypes().arrangeBuiltinFunctionDeclaration(C.VoidTy, Args);
  std::string Name =
      CGM.getOpenMPRuntime().getName({"omp", "copyprivate", "copy_func"});
  auto *Fn = llvm::Function::Create(CGM.getTypes().GetFunctionType(CGFI),
                                    llvm::GlobalValue::InternalLinkage, Name,
                                    &CGM.getModule());
  CGM.SetInternalFunctionAttributes(GlobalDecl(), Fn, CGFI);
  Fn->setDoesNotRecurse();
  CodeGenFunction CGF(CGM);
  CGF.StartFunction(GlobalDecl(), C.VoidTy, Fn, CGFI, Args, Loc, Loc);
  // Dest = (void*[n])(LHSArg);
  // Src = (void*[n])(RHSArg);
  Address LHS(CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
      CGF.Builder.CreateLoad(CGF.GetAddrOfLocalVar(&LHSArg)),
      ArgsType), CGF.getPointerAlign());
  Address RHS(CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
      CGF.Builder.CreateLoad(CGF.GetAddrOfLocalVar(&RHSArg)),
      ArgsType), CGF.getPointerAlign());
  // *(Type0*)Dst[0] = *(Type0*)Src[0];
  // *(Type1*)Dst[1] = *(Type1*)Src[1];
  // ...
  // *(Typen*)Dst[n] = *(Typen*)Src[n];
  for (unsigned I = 0, E = AssignmentOps.size(); I < E; ++I) {
    const auto *DestVar =
        cast<VarDecl>(cast<DeclRefExpr>(DestExprs[I])->getDecl());
    Address DestAddr = emitAddrOfVarFromArray(CGF, LHS, I, DestVar);

    const auto *SrcVar =
        cast<VarDecl>(cast<DeclRefExpr>(SrcExprs[I])->getDecl());
    Address SrcAddr = emitAddrOfVarFromArray(CGF, RHS, I, SrcVar);

    const auto *VD = cast<DeclRefExpr>(CopyprivateVars[I])->getDecl();
    QualType Type = VD->getType();
    CGF.EmitOMPCopy(Type, DestAddr, SrcAddr, DestVar, SrcVar, AssignmentOps[I]);
  }
  CGF.FinishFunction();
  return Fn;
}

void CGOpenMPRuntime::emitSingleRegion(CodeGenFunction &CGF,
                                       const RegionCodeGenTy &SingleOpGen,
                                       SourceLocation Loc,
                                       ArrayRef<const Expr *> CopyprivateVars,
                                       ArrayRef<const Expr *> SrcExprs,
                                       ArrayRef<const Expr *> DstExprs,
                                       ArrayRef<const Expr *> AssignmentOps) {
  if (!CGF.HaveInsertPoint())
    return;
  assert(CopyprivateVars.size() == SrcExprs.size() &&
         CopyprivateVars.size() == DstExprs.size() &&
         CopyprivateVars.size() == AssignmentOps.size());
  ASTContext &C = CGM.getContext();
  // int32 did_it = 0;
  // if(__kmpc_single(ident_t *, gtid)) {
  //   SingleOpGen();
  //   __kmpc_end_single(ident_t *, gtid);
  //   did_it = 1;
  // }
  // call __kmpc_copyprivate(ident_t *, gtid, <buf_size>, <copyprivate list>,
  // <copy_func>, did_it);

  Address DidIt = Address::invalid();
  if (!CopyprivateVars.empty()) {
    // int32 did_it = 0;
    QualType KmpInt32Ty =
        C.getIntTypeForBitwidth(/*DestWidth=*/32, /*Signed=*/1);
    DidIt = CGF.CreateMemTemp(KmpInt32Ty, ".omp.copyprivate.did_it");
    CGF.Builder.CreateStore(CGF.Builder.getInt32(0), DidIt);
  }
  // Prepare arguments and build a call to __kmpc_single
  llvm::Value *Args[] = {emitUpdateLocation(CGF, Loc), getThreadID(CGF, Loc)};
  CommonActionTy Action(OMPBuilder.getOrCreateRuntimeFunction(
                            CGM.getModule(), OMPRTL___kmpc_single),
                        Args,
                        OMPBuilder.getOrCreateRuntimeFunction(
                            CGM.getModule(), OMPRTL___kmpc_end_single),
                        Args,
                        /*Conditional=*/true);
  SingleOpGen.setAction(Action);
  emitInlinedDirective(CGF, OMPD_single, SingleOpGen);
  if (DidIt.isValid()) {
    // did_it = 1;
    CGF.Builder.CreateStore(CGF.Builder.getInt32(1), DidIt);
  }
  Action.Done(CGF);
  // call __kmpc_copyprivate(ident_t *, gtid, <buf_size>, <copyprivate list>,
  // <copy_func>, did_it);
  if (DidIt.isValid()) {
    llvm::APInt ArraySize(/*unsigned int numBits=*/32, CopyprivateVars.size());
    QualType CopyprivateArrayTy = C.getConstantArrayType(
        C.VoidPtrTy, ArraySize, nullptr, ArrayType::Normal,
        /*IndexTypeQuals=*/0);
    // Create a list of all private variables for copyprivate.
    Address CopyprivateList =
        CGF.CreateMemTemp(CopyprivateArrayTy, ".omp.copyprivate.cpr_list");
    for (unsigned I = 0, E = CopyprivateVars.size(); I < E; ++I) {
      Address Elem = CGF.Builder.CreateConstArrayGEP(CopyprivateList, I);
      CGF.Builder.CreateStore(
          CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
              CGF.EmitLValue(CopyprivateVars[I]).getPointer(CGF),
              CGF.VoidPtrTy),
          Elem);
    }
    // Build function that copies private values from single region to all other
    // threads in the corresponding parallel region.
    llvm::Value *CpyFn = emitCopyprivateCopyFunction(
        CGM, CGF.ConvertTypeForMem(CopyprivateArrayTy)->getPointerTo(),
        CopyprivateVars, SrcExprs, DstExprs, AssignmentOps, Loc);
    llvm::Value *BufSize = CGF.getTypeSize(CopyprivateArrayTy);
    Address CL =
      CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(CopyprivateList,
                                                      CGF.VoidPtrTy);
    llvm::Value *DidItVal = CGF.Builder.CreateLoad(DidIt);
    llvm::Value *Args[] = {
        emitUpdateLocation(CGF, Loc), // ident_t *<loc>
        getThreadID(CGF, Loc),        // i32 <gtid>
        BufSize,                      // size_t <buf_size>
        CL.getPointer(),              // void *<copyprivate list>
        CpyFn,                        // void (*) (void *, void *) <copy_func>
        DidItVal                      // i32 did_it
    };
    CGF.EmitRuntimeCall(OMPBuilder.getOrCreateRuntimeFunction(
                            CGM.getModule(), OMPRTL___kmpc_copyprivate),
                        Args);
  }
}

void CGOpenMPRuntime::emitOrderedRegion(CodeGenFunction &CGF,
                                        const RegionCodeGenTy &OrderedOpGen,
                                        SourceLocation Loc, bool IsThreads) {
  if (!CGF.HaveInsertPoint())
    return;
  // __kmpc_ordered(ident_t *, gtid);
  // OrderedOpGen();
  // __kmpc_end_ordered(ident_t *, gtid);
  // Prepare arguments and build a call to __kmpc_ordered
  if (IsThreads) {
    llvm::Value *Args[] = {emitUpdateLocation(CGF, Loc), getThreadID(CGF, Loc)};
    CommonActionTy Action(OMPBuilder.getOrCreateRuntimeFunction(
                              CGM.getModule(), OMPRTL___kmpc_ordered),
                          Args,
                          OMPBuilder.getOrCreateRuntimeFunction(
                              CGM.getModule(), OMPRTL___kmpc_end_ordered),
                          Args);
    OrderedOpGen.setAction(Action);
    emitInlinedDirective(CGF, OMPD_ordered, OrderedOpGen);
    return;
  }
  emitInlinedDirective(CGF, OMPD_ordered, OrderedOpGen);
}

unsigned CGOpenMPRuntime::getDefaultFlagsForBarriers(OpenMPDirectiveKind Kind) {
  unsigned Flags;
  if (Kind == OMPD_for)
    Flags = OMP_IDENT_BARRIER_IMPL_FOR;
  else if (Kind == OMPD_sections)
    Flags = OMP_IDENT_BARRIER_IMPL_SECTIONS;
  else if (Kind == OMPD_single)
    Flags = OMP_IDENT_BARRIER_IMPL_SINGLE;
  else if (Kind == OMPD_barrier)
    Flags = OMP_IDENT_BARRIER_EXPL;
  else
    Flags = OMP_IDENT_BARRIER_IMPL;
  return Flags;
}

void CGOpenMPRuntime::getDefaultScheduleAndChunk(
    CodeGenFunction &CGF, const OMPLoopDirective &S,
    OpenMPScheduleClauseKind &ScheduleKind, const Expr *&ChunkExpr) const {
  // Check if the loop directive is actually a doacross loop directive. In this
  // case choose static, 1 schedule.
  if (llvm::any_of(
          S.getClausesOfKind<OMPOrderedClause>(),
          [](const OMPOrderedClause *C) { return C->getNumForLoops(); })) {
    ScheduleKind = OMPC_SCHEDULE_static;
    // Chunk size is 1 in this case.
    llvm::APInt ChunkSize(32, 1);
    ChunkExpr = IntegerLiteral::Create(
        CGF.getContext(), ChunkSize,
        CGF.getContext().getIntTypeForBitwidth(32, /*Signed=*/0),
        SourceLocation());
  }
}

void CGOpenMPRuntime::emitBarrierCall(CodeGenFunction &CGF, SourceLocation Loc,
                                      OpenMPDirectiveKind Kind, bool EmitChecks,
                                      bool ForceSimpleCall) {
  // Check if we should use the OMPBuilder
  auto *OMPRegionInfo =
      dyn_cast_or_null<CGOpenMPRegionInfo>(CGF.CapturedStmtInfo);
  if (CGF.CGM.getLangOpts().OpenMPIRBuilder) {
    CGF.Builder.restoreIP(OMPBuilder.createBarrier(
        CGF.Builder, Kind, ForceSimpleCall, EmitChecks));
    return;
  }

  if (!CGF.HaveInsertPoint())
    return;
  // Build call __kmpc_cancel_barrier(loc, thread_id);
  // Build call __kmpc_barrier(loc, thread_id);
  unsigned Flags = getDefaultFlagsForBarriers(Kind);
  // Build call __kmpc_cancel_barrier(loc, thread_id) or __kmpc_barrier(loc,
  // thread_id);
  llvm::Value *Args[] = {emitUpdateLocation(CGF, Loc, Flags),
                         getThreadID(CGF, Loc)};
  if (OMPRegionInfo) {
    if (!ForceSimpleCall && OMPRegionInfo->hasCancel()) {
      llvm::Value *Result = CGF.EmitRuntimeCall(
          OMPBuilder.getOrCreateRuntimeFunction(CGM.getModule(),
                                                OMPRTL___kmpc_cancel_barrier),
          Args);
      if (EmitChecks) {
        // if (__kmpc_cancel_barrier()) {
        //   exit from construct;
        // }
        llvm::BasicBlock *ExitBB = CGF.createBasicBlock(".cancel.exit");
        llvm::BasicBlock *ContBB = CGF.createBasicBlock(".cancel.continue");
        llvm::Value *Cmp = CGF.Builder.CreateIsNotNull(Result);
        CGF.Builder.CreateCondBr(Cmp, ExitBB, ContBB);
        CGF.EmitBlock(ExitBB);
        //   exit from construct;
        CodeGenFunction::JumpDest CancelDestination =
            CGF.getOMPCancelDestination(OMPRegionInfo->getDirectiveKind());
        CGF.EmitBranchThroughCleanup(CancelDestination);
        CGF.EmitBlock(ContBB, /*IsFinished=*/true);
      }
      return;
    }
  }
  CGF.EmitRuntimeCall(OMPBuilder.getOrCreateRuntimeFunction(
                          CGM.getModule(), OMPRTL___kmpc_barrier),
                      Args);
}

/// Map the OpenMP loop schedule to the runtime enumeration.
static OpenMPSchedType getRuntimeSchedule(OpenMPScheduleClauseKind ScheduleKind,
                                          bool Chunked, bool Ordered) {
  switch (ScheduleKind) {
  case OMPC_SCHEDULE_static:
    return Chunked ? (Ordered ? OMP_ord_static_chunked : OMP_sch_static_chunked)
                   : (Ordered ? OMP_ord_static : OMP_sch_static);
  case OMPC_SCHEDULE_dynamic:
    return Ordered ? OMP_ord_dynamic_chunked : OMP_sch_dynamic_chunked;
  case OMPC_SCHEDULE_guided:
    return Ordered ? OMP_ord_guided_chunked : OMP_sch_guided_chunked;
  case OMPC_SCHEDULE_runtime:
    return Ordered ? OMP_ord_runtime : OMP_sch_runtime;
  case OMPC_SCHEDULE_auto:
    return Ordered ? OMP_ord_auto : OMP_sch_auto;
  case OMPC_SCHEDULE_unknown:
    assert(!Chunked && "chunk was specified but schedule kind not known");
    return Ordered ? OMP_ord_static : OMP_sch_static;
  }
  llvm_unreachable("Unexpected runtime schedule");
}

/// Map the OpenMP distribute schedule to the runtime enumeration.
static OpenMPSchedType
getRuntimeSchedule(OpenMPDistScheduleClauseKind ScheduleKind, bool Chunked) {
  // only static is allowed for dist_schedule
  return Chunked ? OMP_dist_sch_static_chunked : OMP_dist_sch_static;
}

bool CGOpenMPRuntime::isStaticNonchunked(OpenMPScheduleClauseKind ScheduleKind,
                                         bool Chunked) const {
  OpenMPSchedType Schedule =
      getRuntimeSchedule(ScheduleKind, Chunked, /*Ordered=*/false);
  return Schedule == OMP_sch_static;
}

bool CGOpenMPRuntime::isStaticNonchunked(
    OpenMPDistScheduleClauseKind ScheduleKind, bool Chunked) const {
  OpenMPSchedType Schedule = getRuntimeSchedule(ScheduleKind, Chunked);
  return Schedule == OMP_dist_sch_static;
}

bool CGOpenMPRuntime::isStaticChunked(OpenMPScheduleClauseKind ScheduleKind,
                                      bool Chunked) const {
  OpenMPSchedType Schedule =
      getRuntimeSchedule(ScheduleKind, Chunked, /*Ordered=*/false);
  return Schedule == OMP_sch_static_chunked;
}

bool CGOpenMPRuntime::isStaticChunked(
    OpenMPDistScheduleClauseKind ScheduleKind, bool Chunked) const {
  OpenMPSchedType Schedule = getRuntimeSchedule(ScheduleKind, Chunked);
  return Schedule == OMP_dist_sch_static_chunked;
}

bool CGOpenMPRuntime::isDynamic(OpenMPScheduleClauseKind ScheduleKind) const {
  OpenMPSchedType Schedule =
      getRuntimeSchedule(ScheduleKind, /*Chunked=*/false, /*Ordered=*/false);
  assert(Schedule != OMP_sch_static_chunked && "cannot be chunked here");
  return Schedule != OMP_sch_static;
}

static int addMonoNonMonoModifier(CodeGenModule &CGM, OpenMPSchedType Schedule,
                                  OpenMPScheduleClauseModifier M1,
                                  OpenMPScheduleClauseModifier M2) {
  int Modifier = 0;
  switch (M1) {
  case OMPC_SCHEDULE_MODIFIER_monotonic:
    Modifier = OMP_sch_modifier_monotonic;
    break;
  case OMPC_SCHEDULE_MODIFIER_nonmonotonic:
    Modifier = OMP_sch_modifier_nonmonotonic;
    break;
  case OMPC_SCHEDULE_MODIFIER_simd:
    if (Schedule == OMP_sch_static_chunked)
      Schedule = OMP_sch_static_balanced_chunked;
    break;
  case OMPC_SCHEDULE_MODIFIER_last:
  case OMPC_SCHEDULE_MODIFIER_unknown:
    break;
  }
  switch (M2) {
  case OMPC_SCHEDULE_MODIFIER_monotonic:
    Modifier = OMP_sch_modifier_monotonic;
    break;
  case OMPC_SCHEDULE_MODIFIER_nonmonotonic:
    Modifier = OMP_sch_modifier_nonmonotonic;
    break;
  case OMPC_SCHEDULE_MODIFIER_simd:
    if (Schedule == OMP_sch_static_chunked)
      Schedule = OMP_sch_static_balanced_chunked;
    break;
  case OMPC_SCHEDULE_MODIFIER_last:
  case OMPC_SCHEDULE_MODIFIER_unknown:
    break;
  }
  // OpenMP 5.0, 2.9.2 Worksharing-Loop Construct, Desription.
  // If the static schedule kind is specified or if the ordered clause is
  // specified, and if the nonmonotonic modifier is not specified, the effect is
  // as if the monotonic modifier is specified. Otherwise, unless the monotonic
  // modifier is specified, the effect is as if the nonmonotonic modifier is
  // specified.
  if (CGM.getLangOpts().OpenMP >= 50 && Modifier == 0) {
    if (!(Schedule == OMP_sch_static_chunked || Schedule == OMP_sch_static ||
          Schedule == OMP_sch_static_balanced_chunked ||
          Schedule == OMP_ord_static_chunked || Schedule == OMP_ord_static ||
          Schedule == OMP_dist_sch_static_chunked ||
          Schedule == OMP_dist_sch_static))
      Modifier = OMP_sch_modifier_nonmonotonic;
  }
  return Schedule | Modifier;
}

void CGOpenMPRuntime::emitForDispatchInit(
    CodeGenFunction &CGF, SourceLocation Loc,
    const OpenMPScheduleTy &ScheduleKind, unsigned IVSize, bool IVSigned,
    bool Ordered, const DispatchRTInput &DispatchValues) {
  if (!CGF.HaveInsertPoint())
    return;
  OpenMPSchedType Schedule = getRuntimeSchedule(
      ScheduleKind.Schedule, DispatchValues.Chunk != nullptr, Ordered);
  assert(Ordered ||
         (Schedule != OMP_sch_static && Schedule != OMP_sch_static_chunked &&
          Schedule != OMP_ord_static && Schedule != OMP_ord_static_chunked &&
          Schedule != OMP_sch_static_balanced_chunked));
  // Call __kmpc_dispatch_init(
  //          ident_t *loc, kmp_int32 tid, kmp_int32 schedule,
  //          kmp_int[32|64] lower, kmp_int[32|64] upper,
  //          kmp_int[32|64] stride, kmp_int[32|64] chunk);

  // If the Chunk was not specified in the clause - use default value 1.
  llvm::Value *Chunk = DispatchValues.Chunk ? DispatchValues.Chunk
                                            : CGF.Builder.getIntN(IVSize, 1);
  llvm::Value *Args[] = {
      emitUpdateLocation(CGF, Loc),
      getThreadID(CGF, Loc),
      CGF.Builder.getInt32(addMonoNonMonoModifier(
          CGM, Schedule, ScheduleKind.M1, ScheduleKind.M2)), // Schedule type
      DispatchValues.LB,                                     // Lower
      DispatchValues.UB,                                     // Upper
      CGF.Builder.getIntN(IVSize, 1),                        // Stride
      Chunk                                                  // Chunk
  };
  CGF.EmitRuntimeCall(createDispatchInitFunction(IVSize, IVSigned), Args);
}

static void emitForStaticInitCall(
    CodeGenFunction &CGF, llvm::Value *UpdateLocation, llvm::Value *ThreadId,
    llvm::FunctionCallee ForStaticInitFunction, OpenMPSchedType Schedule,
    OpenMPScheduleClauseModifier M1, OpenMPScheduleClauseModifier M2,
    const CGOpenMPRuntime::StaticRTInput &Values) {
  if (!CGF.HaveInsertPoint())
    return;

  assert(!Values.Ordered);
  assert(Schedule == OMP_sch_static || Schedule == OMP_sch_static_chunked ||
         Schedule == OMP_sch_static_balanced_chunked ||
         Schedule == OMP_ord_static || Schedule == OMP_ord_static_chunked ||
         Schedule == OMP_dist_sch_static ||
         Schedule == OMP_dist_sch_static_chunked);

  // Call __kmpc_for_static_init(
  //          ident_t *loc, kmp_int32 tid, kmp_int32 schedtype,
  //          kmp_int32 *p_lastiter, kmp_int[32|64] *p_lower,
  //          kmp_int[32|64] *p_upper, kmp_int[32|64] *p_stride,
  //          kmp_int[32|64] incr, kmp_int[32|64] chunk);
  llvm::Value *Chunk = Values.Chunk;
  if (Chunk == nullptr) {
    assert((Schedule == OMP_sch_static || Schedule == OMP_ord_static ||
            Schedule == OMP_dist_sch_static) &&
           "expected static non-chunked schedule");
    // If the Chunk was not specified in the clause - use default value 1.
    Chunk = CGF.Builder.getIntN(Values.IVSize, 1);
  } else {
    assert((Schedule == OMP_sch_static_chunked ||
            Schedule == OMP_sch_static_balanced_chunked ||
            Schedule == OMP_ord_static_chunked ||
            Schedule == OMP_dist_sch_static_chunked) &&
           "expected static chunked schedule");
  }
  llvm::Value *Args[] = {
      UpdateLocation,
      ThreadId,
      CGF.Builder.getInt32(addMonoNonMonoModifier(CGF.CGM, Schedule, M1,
                                                  M2)), // Schedule type
      Values.IL.getPointer(),                           // &isLastIter
      Values.LB.getPointer(),                           // &LB
      Values.UB.getPointer(),                           // &UB
      Values.ST.getPointer(),                           // &Stride
      CGF.Builder.getIntN(Values.IVSize, 1),            // Incr
      Chunk                                             // Chunk
  };
  CGF.EmitRuntimeCall(ForStaticInitFunction, Args);
}

void CGOpenMPRuntime::emitForStaticInit(CodeGenFunction &CGF,
                                        SourceLocation Loc,
                                        OpenMPDirectiveKind DKind,
                                        const OpenMPScheduleTy &ScheduleKind,
                                        const StaticRTInput &Values) {
  OpenMPSchedType ScheduleNum = getRuntimeSchedule(
      ScheduleKind.Schedule, Values.Chunk != nullptr, Values.Ordered);
  assert(isOpenMPWorksharingDirective(DKind) &&
         "Expected loop-based or sections-based directive.");
  llvm::Value *UpdatedLocation = emitUpdateLocation(CGF, Loc,
                                             isOpenMPLoopDirective(DKind)
                                                 ? OMP_IDENT_WORK_LOOP
                                                 : OMP_IDENT_WORK_SECTIONS);
  llvm::Value *ThreadId = getThreadID(CGF, Loc);
  llvm::FunctionCallee StaticInitFunction =
      createForStaticInitFunction(Values.IVSize, Values.IVSigned, false);
  auto DL = ApplyDebugLocation::CreateDefaultArtificial(CGF, Loc);
  emitForStaticInitCall(CGF, UpdatedLocation, ThreadId, StaticInitFunction,
                        ScheduleNum, ScheduleKind.M1, ScheduleKind.M2, Values);
}

void CGOpenMPRuntime::emitDistributeStaticInit(
    CodeGenFunction &CGF, SourceLocation Loc,
    OpenMPDistScheduleClauseKind SchedKind,
    const CGOpenMPRuntime::StaticRTInput &Values) {
  OpenMPSchedType ScheduleNum =
      getRuntimeSchedule(SchedKind, Values.Chunk != nullptr);
  llvm::Value *UpdatedLocation =
      emitUpdateLocation(CGF, Loc, OMP_IDENT_WORK_DISTRIBUTE);
  llvm::Value *ThreadId = getThreadID(CGF, Loc);
  llvm::FunctionCallee StaticInitFunction;
  bool isGPUDistribute =
      CGM.getLangOpts().OpenMPIsDevice &&
      (CGM.getTriple().isAMDGCN() || CGM.getTriple().isNVPTX());
  StaticInitFunction = createForStaticInitFunction(
      Values.IVSize, Values.IVSigned, isGPUDistribute);

  emitForStaticInitCall(CGF, UpdatedLocation, ThreadId, StaticInitFunction,
                        ScheduleNum, OMPC_SCHEDULE_MODIFIER_unknown,
                        OMPC_SCHEDULE_MODIFIER_unknown, Values);
}

void CGOpenMPRuntime::emitForStaticFinish(CodeGenFunction &CGF,
                                          SourceLocation Loc,
                                          OpenMPDirectiveKind DKind) {
  if (!CGF.HaveInsertPoint())
    return;
  // Call __kmpc_for_static_fini(ident_t *loc, kmp_int32 tid);
  llvm::Value *Args[] = {
      emitUpdateLocation(CGF, Loc,
                         isOpenMPDistributeDirective(DKind)
                             ? OMP_IDENT_WORK_DISTRIBUTE
                             : isOpenMPLoopDirective(DKind)
                                   ? OMP_IDENT_WORK_LOOP
                                   : OMP_IDENT_WORK_SECTIONS),
      getThreadID(CGF, Loc)};
  auto DL = ApplyDebugLocation::CreateDefaultArtificial(CGF, Loc);
  if (isOpenMPDistributeDirective(DKind) && CGM.getLangOpts().OpenMPIsDevice &&
      (CGM.getTriple().isAMDGCN() || CGM.getTriple().isNVPTX()))
    CGF.EmitRuntimeCall(
        OMPBuilder.getOrCreateRuntimeFunction(
            CGM.getModule(), OMPRTL___kmpc_distribute_static_fini),
        Args);
  else
    CGF.EmitRuntimeCall(OMPBuilder.getOrCreateRuntimeFunction(
                            CGM.getModule(), OMPRTL___kmpc_for_static_fini),
                        Args);
}

void CGOpenMPRuntime::emitForOrderedIterationEnd(CodeGenFunction &CGF,
                                                 SourceLocation Loc,
                                                 unsigned IVSize,
                                                 bool IVSigned) {
  if (!CGF.HaveInsertPoint())
    return;
  // Call __kmpc_for_dynamic_fini_(4|8)[u](ident_t *loc, kmp_int32 tid);
  llvm::Value *Args[] = {emitUpdateLocation(CGF, Loc), getThreadID(CGF, Loc)};
  CGF.EmitRuntimeCall(createDispatchFiniFunction(IVSize, IVSigned), Args);
}

llvm::Value *CGOpenMPRuntime::emitForNext(CodeGenFunction &CGF,
                                          SourceLocation Loc, unsigned IVSize,
                                          bool IVSigned, Address IL,
                                          Address LB, Address UB,
                                          Address ST) {
  // Call __kmpc_dispatch_next(
  //          ident_t *loc, kmp_int32 tid, kmp_int32 *p_lastiter,
  //          kmp_int[32|64] *p_lower, kmp_int[32|64] *p_upper,
  //          kmp_int[32|64] *p_stride);
  llvm::Value *Args[] = {
      emitUpdateLocation(CGF, Loc),
      getThreadID(CGF, Loc),
      IL.getPointer(), // &isLastIter
      LB.getPointer(), // &Lower
      UB.getPointer(), // &Upper
      ST.getPointer()  // &Stride
  };
  llvm::Value *Call =
      CGF.EmitRuntimeCall(createDispatchNextFunction(IVSize, IVSigned), Args);
  return CGF.EmitScalarConversion(
      Call, CGF.getContext().getIntTypeForBitwidth(32, /*Signed=*/1),
      CGF.getContext().BoolTy, Loc);
}

void CGOpenMPRuntime::emitNumThreadsClause(CodeGenFunction &CGF,
                                           llvm::Value *NumThreads,
                                           SourceLocation Loc) {
  if (!CGF.HaveInsertPoint())
    return;
  // Build call __kmpc_push_num_threads(&loc, global_tid, num_threads)
  llvm::Value *Args[] = {
      emitUpdateLocation(CGF, Loc), getThreadID(CGF, Loc),
      CGF.Builder.CreateIntCast(NumThreads, CGF.Int32Ty, /*isSigned*/ true)};
  CGF.EmitRuntimeCall(OMPBuilder.getOrCreateRuntimeFunction(
                          CGM.getModule(), OMPRTL___kmpc_push_num_threads),
                      Args);
}

void CGOpenMPRuntime::emitProcBindClause(CodeGenFunction &CGF,
                                         ProcBindKind ProcBind,
                                         SourceLocation Loc) {
  if (!CGF.HaveInsertPoint())
    return;
  assert(ProcBind != OMP_PROC_BIND_unknown && "Unsupported proc_bind value.");
  // Build call __kmpc_push_proc_bind(&loc, global_tid, proc_bind)
  llvm::Value *Args[] = {
      emitUpdateLocation(CGF, Loc), getThreadID(CGF, Loc),
      llvm::ConstantInt::get(CGM.IntTy, unsigned(ProcBind), /*isSigned=*/true)};
  CGF.EmitRuntimeCall(OMPBuilder.getOrCreateRuntimeFunction(
                          CGM.getModule(), OMPRTL___kmpc_push_proc_bind),
                      Args);
}

void CGOpenMPRuntime::emitFlush(CodeGenFunction &CGF, ArrayRef<const Expr *>,
                                SourceLocation Loc, llvm::AtomicOrdering AO) {
  if (CGF.CGM.getLangOpts().OpenMPIRBuilder) {
    OMPBuilder.createFlush(CGF.Builder);
  } else {
    if (!CGF.HaveInsertPoint())
      return;
    // Build call void __kmpc_flush(ident_t *loc)
    CGF.EmitRuntimeCall(OMPBuilder.getOrCreateRuntimeFunction(
                            CGM.getModule(), OMPRTL___kmpc_flush),
                        emitUpdateLocation(CGF, Loc));
  }
}

namespace {
/// Indexes of fields for type kmp_task_t.
enum KmpTaskTFields {
  /// List of shared variables.
  KmpTaskTShareds,
  /// Task routine.
  KmpTaskTRoutine,
  /// Partition id for the untied tasks.
  KmpTaskTPartId,
  /// Function with call of destructors for private variables.
  Data1,
  /// Task priority.
  Data2,
  /// (Taskloops only) Lower bound.
  KmpTaskTLowerBound,
  /// (Taskloops only) Upper bound.
  KmpTaskTUpperBound,
  /// (Taskloops only) Stride.
  KmpTaskTStride,
  /// (Taskloops only) Is last iteration flag.
  KmpTaskTLastIter,
  /// (Taskloops only) Reduction data.
  KmpTaskTReductions,
};
} // anonymous namespace

bool CGOpenMPRuntime::OffloadEntriesInfoManagerTy::empty() const {
  return OffloadEntriesTargetRegion.empty() &&
         OffloadEntriesDeviceGlobalVar.empty();
}

/// Initialize target region entry.
void CGOpenMPRuntime::OffloadEntriesInfoManagerTy::
    initializeTargetRegionEntryInfo(unsigned DeviceID, unsigned FileID,
                                    StringRef ParentName, unsigned LineNum,
                                    unsigned Order) {
  assert(CGM.getLangOpts().OpenMPIsDevice && "Initialization of entries is "
                                             "only required for the device "
                                             "code generation.");
  OffloadEntriesTargetRegion[DeviceID][FileID][ParentName][LineNum] =
      OffloadEntryInfoTargetRegion(Order, /*Addr=*/nullptr, /*ID=*/nullptr,
                                   OMPTargetRegionEntryTargetRegion);
  ++OffloadingEntriesNum;
}

void CGOpenMPRuntime::OffloadEntriesInfoManagerTy::
    registerTargetRegionEntryInfo(unsigned DeviceID, unsigned FileID,
                                  StringRef ParentName, unsigned LineNum,
                                  llvm::Constant *Addr, llvm::Constant *ID,
                                  OMPTargetRegionEntryKind Flags) {
  // If we are emitting code for a target, the entry is already initialized,
  // only has to be registered.
  if (CGM.getLangOpts().OpenMPIsDevice) {
    // This could happen if the device compilation is invoked standalone.
    if (!hasTargetRegionEntryInfo(DeviceID, FileID, ParentName, LineNum))
      return;
    auto &Entry =
        OffloadEntriesTargetRegion[DeviceID][FileID][ParentName][LineNum];
    Entry.setAddress(Addr);
    Entry.setID(ID);
    Entry.setFlags(Flags);
  } else {
    if (Flags ==
            OffloadEntriesInfoManagerTy::OMPTargetRegionEntryTargetRegion &&
        hasTargetRegionEntryInfo(DeviceID, FileID, ParentName, LineNum,
                                 /*IgnoreAddressId*/ true))
      return;
    assert(!hasTargetRegionEntryInfo(DeviceID, FileID, ParentName, LineNum) &&
           "Target region entry already registered!");
    OffloadEntryInfoTargetRegion Entry(OffloadingEntriesNum, Addr, ID, Flags);
    OffloadEntriesTargetRegion[DeviceID][FileID][ParentName][LineNum] = Entry;
    ++OffloadingEntriesNum;
  }
}

bool CGOpenMPRuntime::OffloadEntriesInfoManagerTy::hasTargetRegionEntryInfo(
    unsigned DeviceID, unsigned FileID, StringRef ParentName, unsigned LineNum,
    bool IgnoreAddressId) const {
  auto PerDevice = OffloadEntriesTargetRegion.find(DeviceID);
  if (PerDevice == OffloadEntriesTargetRegion.end())
    return false;
  auto PerFile = PerDevice->second.find(FileID);
  if (PerFile == PerDevice->second.end())
    return false;
  auto PerParentName = PerFile->second.find(ParentName);
  if (PerParentName == PerFile->second.end())
    return false;
  auto PerLine = PerParentName->second.find(LineNum);
  if (PerLine == PerParentName->second.end())
    return false;
  // Fail if this entry is already registered.
  if (!IgnoreAddressId &&
      (PerLine->second.getAddress() || PerLine->second.getID()))
    return false;
  return true;
}

void CGOpenMPRuntime::OffloadEntriesInfoManagerTy::actOnTargetRegionEntriesInfo(
    const OffloadTargetRegionEntryInfoActTy &Action) {
  // Scan all target region entries and perform the provided action.
  for (const auto &D : OffloadEntriesTargetRegion)
    for (const auto &F : D.second)
      for (const auto &P : F.second)
        for (const auto &L : P.second)
          Action(D.first, F.first, P.first(), L.first, L.second);
}

void CGOpenMPRuntime::OffloadEntriesInfoManagerTy::
    initializeDeviceGlobalVarEntryInfo(StringRef Name,
                                       OMPTargetGlobalVarEntryKind Flags,
                                       unsigned Order) {
  assert(CGM.getLangOpts().OpenMPIsDevice && "Initialization of entries is "
                                             "only required for the device "
                                             "code generation.");
  OffloadEntriesDeviceGlobalVar.try_emplace(Name, Order, Flags);
  ++OffloadingEntriesNum;
}

void CGOpenMPRuntime::OffloadEntriesInfoManagerTy::
    registerDeviceGlobalVarEntryInfo(StringRef VarName, llvm::Constant *Addr,
                                     CharUnits VarSize,
                                     OMPTargetGlobalVarEntryKind Flags,
                                     llvm::GlobalValue::LinkageTypes Linkage) {
  if (CGM.getLangOpts().OpenMPIsDevice) {
    // This could happen if the device compilation is invoked standalone.
    if (!hasDeviceGlobalVarEntryInfo(VarName))
      return;
    auto &Entry = OffloadEntriesDeviceGlobalVar[VarName];
    if (Entry.getAddress() && hasDeviceGlobalVarEntryInfo(VarName)) {
      if (Entry.getVarSize().isZero()) {
        Entry.setVarSize(VarSize);
        Entry.setLinkage(Linkage);
      }
      return;
    }
    Entry.setVarSize(VarSize);
    Entry.setLinkage(Linkage);
    Entry.setAddress(Addr);
  } else {
    if (hasDeviceGlobalVarEntryInfo(VarName)) {
      auto &Entry = OffloadEntriesDeviceGlobalVar[VarName];
      assert(Entry.isValid() && Entry.getFlags() == Flags &&
             "Entry not initialized!");
      if (Entry.getVarSize().isZero()) {
        Entry.setVarSize(VarSize);
        Entry.setLinkage(Linkage);
      }
      return;
    }
    OffloadEntriesDeviceGlobalVar.try_emplace(
        VarName, OffloadingEntriesNum, Addr, VarSize, Flags, Linkage);
    ++OffloadingEntriesNum;
  }
}

void CGOpenMPRuntime::OffloadEntriesInfoManagerTy::
    actOnDeviceGlobalVarEntriesInfo(
        const OffloadDeviceGlobalVarEntryInfoActTy &Action) {
  // Scan all target region entries and perform the provided action.
  for (const auto &E : OffloadEntriesDeviceGlobalVar)
    Action(E.getKey(), E.getValue());
}

void CGOpenMPRuntime::createOffloadEntry(
    llvm::Constant *ID, llvm::Constant *Addr, uint64_t Size, int32_t Flags,
    llvm::GlobalValue::LinkageTypes Linkage) {
  StringRef Name = Addr->getName();
  llvm::Module &M = CGM.getModule();
  llvm::LLVMContext &C = M.getContext();

  // Create constant string with the name.
  llvm::Constant *StrPtrInit = llvm::ConstantDataArray::getString(C, Name);

  std::string StringName = getName({"omp_offloading", "entry_name"});
  auto *Str = new llvm::GlobalVariable(
      M, StrPtrInit->getType(), /*isConstant=*/true,
      llvm::GlobalValue::InternalLinkage, StrPtrInit, StringName);
  Str->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Global);

  llvm::Constant *Data[] = {
      llvm::ConstantExpr::getPointerBitCastOrAddrSpaceCast(ID, CGM.VoidPtrTy),
      llvm::ConstantExpr::getPointerBitCastOrAddrSpaceCast(Str, CGM.Int8PtrTy),
      llvm::ConstantInt::get(CGM.SizeTy, Size),
      llvm::ConstantInt::get(CGM.Int32Ty, Flags),
      llvm::ConstantInt::get(CGM.Int32Ty, 0)};
  std::string EntryName = getName({"omp_offloading", "entry", ""});
  llvm::GlobalVariable *Entry = createGlobalStruct(
      CGM, getTgtOffloadEntryQTy(), /*IsConstant=*/true, Data,
      Twine(EntryName).concat(Name), llvm::GlobalValue::WeakAnyLinkage);

  // The entry has to be created in the section the linker expects it to be.
  Entry->setSection("omp_offloading_entries");
}

void CGOpenMPRuntime::createOffloadEntriesAndInfoMetadata() {
  // Emit the offloading entries and metadata so that the device codegen side
  // can easily figure out what to emit. The produced metadata looks like
  // this:
  //
  // !omp_offload.info = !{!1, ...}
  //
  // Right now we only generate metadata for function that contain target
  // regions.

  // If we are in simd mode or there are no entries, we don't need to do
  // anything.
  if (CGM.getLangOpts().OpenMPSimd || OffloadEntriesInfoManager.empty())
    return;

  llvm::Module &M = CGM.getModule();
  llvm::LLVMContext &C = M.getContext();
  SmallVector<std::tuple<const OffloadEntriesInfoManagerTy::OffloadEntryInfo *,
                         SourceLocation, StringRef>,
              16>
      OrderedEntries(OffloadEntriesInfoManager.size());
  llvm::SmallVector<StringRef, 16> ParentFunctions(
      OffloadEntriesInfoManager.size());

  // Auxiliary methods to create metadata values and strings.
  auto &&GetMDInt = [this](unsigned V) {
    return llvm::ConstantAsMetadata::get(
        llvm::ConstantInt::get(CGM.Int32Ty, V));
  };

  auto &&GetMDString = [&C](StringRef V) { return llvm::MDString::get(C, V); };

  // Create the offloading info metadata node.
  llvm::NamedMDNode *MD = M.getOrInsertNamedMetadata("omp_offload.info");

  // Create function that emits metadata for each target region entry;
  auto &&TargetRegionMetadataEmitter =
      [this, &C, MD, &OrderedEntries, &ParentFunctions, &GetMDInt,
       &GetMDString](
          unsigned DeviceID, unsigned FileID, StringRef ParentName,
          unsigned Line,
          const OffloadEntriesInfoManagerTy::OffloadEntryInfoTargetRegion &E) {
        // Generate metadata for target regions. Each entry of this metadata
        // contains:
        // - Entry 0 -> Kind of this type of metadata (0).
        // - Entry 1 -> Device ID of the file where the entry was identified.
        // - Entry 2 -> File ID of the file where the entry was identified.
        // - Entry 3 -> Mangled name of the function where the entry was
        // identified.
        // - Entry 4 -> Line in the file where the entry was identified.
        // - Entry 5 -> Order the entry was created.
        // The first element of the metadata node is the kind.
        llvm::Metadata *Ops[] = {GetMDInt(E.getKind()), GetMDInt(DeviceID),
                                 GetMDInt(FileID),      GetMDString(ParentName),
                                 GetMDInt(Line),        GetMDInt(E.getOrder())};

        SourceLocation Loc;
        for (auto I = CGM.getContext().getSourceManager().fileinfo_begin(),
                  E = CGM.getContext().getSourceManager().fileinfo_end();
             I != E; ++I) {
          if (I->getFirst()->getUniqueID().getDevice() == DeviceID &&
              I->getFirst()->getUniqueID().getFile() == FileID) {
            Loc = CGM.getContext().getSourceManager().translateFileLineCol(
                I->getFirst(), Line, 1);
            break;
          }
        }
        // Save this entry in the right position of the ordered entries array.
        OrderedEntries[E.getOrder()] = std::make_tuple(&E, Loc, ParentName);
        ParentFunctions[E.getOrder()] = ParentName;

        // Add metadata to the named metadata node.
        MD->addOperand(llvm::MDNode::get(C, Ops));
      };

  OffloadEntriesInfoManager.actOnTargetRegionEntriesInfo(
      TargetRegionMetadataEmitter);

  // Create function that emits metadata for each device global variable entry;
  auto &&DeviceGlobalVarMetadataEmitter =
      [&C, &OrderedEntries, &GetMDInt, &GetMDString,
       MD](StringRef MangledName,
           const OffloadEntriesInfoManagerTy::OffloadEntryInfoDeviceGlobalVar
               &E) {
        // Generate metadata for global variables. Each entry of this metadata
        // contains:
        // - Entry 0 -> Kind of this type of metadata (1).
        // - Entry 1 -> Mangled name of the variable.
        // - Entry 2 -> Declare target kind.
        // - Entry 3 -> Order the entry was created.
        // The first element of the metadata node is the kind.
        llvm::Metadata *Ops[] = {
            GetMDInt(E.getKind()), GetMDString(MangledName),
            GetMDInt(E.getFlags()), GetMDInt(E.getOrder())};

        // Save this entry in the right position of the ordered entries array.
        OrderedEntries[E.getOrder()] =
            std::make_tuple(&E, SourceLocation(), MangledName);

        // Add metadata to the named metadata node.
        MD->addOperand(llvm::MDNode::get(C, Ops));
      };

  OffloadEntriesInfoManager.actOnDeviceGlobalVarEntriesInfo(
      DeviceGlobalVarMetadataEmitter);

  for (const auto &E : OrderedEntries) {
    assert(std::get<0>(E) && "All ordered entries must exist!");
    if (const auto *CE =
            dyn_cast<OffloadEntriesInfoManagerTy::OffloadEntryInfoTargetRegion>(
                std::get<0>(E))) {
      if (!CE->getID() || !CE->getAddress()) {
        // Do not blame the entry if the parent funtion is not emitted.
        StringRef FnName = ParentFunctions[CE->getOrder()];
        if (!CGM.GetGlobalValue(FnName))
          continue;
        unsigned DiagID = CGM.getDiags().getCustomDiagID(
            DiagnosticsEngine::Error,
            "Offloading entry for target region in %0 is incorrect: either the "
            "address or the ID is invalid.");
        CGM.getDiags().Report(std::get<1>(E), DiagID) << FnName;
        continue;
      }
      createOffloadEntry(CE->getID(), CE->getAddress(), /*Size=*/0,
                         CE->getFlags(), llvm::GlobalValue::WeakAnyLinkage);
    } else if (const auto *CE = dyn_cast<OffloadEntriesInfoManagerTy::
                                             OffloadEntryInfoDeviceGlobalVar>(
                   std::get<0>(E))) {
      OffloadEntriesInfoManagerTy::OMPTargetGlobalVarEntryKind Flags =
          static_cast<OffloadEntriesInfoManagerTy::OMPTargetGlobalVarEntryKind>(
              CE->getFlags());
      switch (Flags) {
      case OffloadEntriesInfoManagerTy::OMPTargetGlobalVarEntryTo: {
        if (CGM.getLangOpts().OpenMPIsDevice &&
            CGM.getOpenMPRuntime().hasRequiresUnifiedSharedMemory())
          continue;
        if (!CE->getAddress()) {
          unsigned DiagID = CGM.getDiags().getCustomDiagID(
              DiagnosticsEngine::Error, "Offloading entry for declare target "
                                        "variable %0 is incorrect: the "
                                        "address is invalid.");
          CGM.getDiags().Report(std::get<1>(E), DiagID) << std::get<2>(E);
          continue;
        }
        // The vaiable has no definition - no need to add the entry.
        if (CE->getVarSize().isZero())
          continue;
        break;
      }
      case OffloadEntriesInfoManagerTy::OMPTargetGlobalVarEntryLink:
        assert(((CGM.getLangOpts().OpenMPIsDevice && !CE->getAddress()) ||
                (!CGM.getLangOpts().OpenMPIsDevice && CE->getAddress())) &&
               "Declaret target link address is set.");
        if (CGM.getLangOpts().OpenMPIsDevice)
          continue;
        if (!CE->getAddress()) {
          unsigned DiagID = CGM.getDiags().getCustomDiagID(
              DiagnosticsEngine::Error,
              "Offloading entry for declare target variable is incorrect: the "
              "address is invalid.");
          CGM.getDiags().Report(DiagID);
          continue;
        }
        break;
      }
      createOffloadEntry(CE->getAddress(), CE->getAddress(),
                         CE->getVarSize().getQuantity(), Flags,
                         CE->getLinkage());
    } else {
      llvm_unreachable("Unsupported entry kind.");
    }
  }
}

/// Loads all the offload entries information from the host IR
/// metadata.
void CGOpenMPRuntime::loadOffloadInfoMetadata() {
  // If we are in target mode, load the metadata from the host IR. This code has
  // to match the metadaata creation in createOffloadEntriesAndInfoMetadata().

  if (!CGM.getLangOpts().OpenMPIsDevice)
    return;

  if (CGM.getLangOpts().OMPHostIRFile.empty())
    return;

  auto Buf = llvm::MemoryBuffer::getFile(CGM.getLangOpts().OMPHostIRFile);
  if (auto EC = Buf.getError()) {
    CGM.getDiags().Report(diag::err_cannot_open_file)
        << CGM.getLangOpts().OMPHostIRFile << EC.message();
    return;
  }

  llvm::LLVMContext C;
  auto ME = expectedToErrorOrAndEmitErrors(
      C, llvm::parseBitcodeFile(Buf.get()->getMemBufferRef(), C));

  if (auto EC = ME.getError()) {
    unsigned DiagID = CGM.getDiags().getCustomDiagID(
        DiagnosticsEngine::Error, "Unable to parse host IR file '%0':'%1'");
    CGM.getDiags().Report(DiagID)
        << CGM.getLangOpts().OMPHostIRFile << EC.message();
    return;
  }

  llvm::NamedMDNode *MD = ME.get()->getNamedMetadata("omp_offload.info");
  if (!MD)
    return;

  for (llvm::MDNode *MN : MD->operands()) {
    auto &&GetMDInt = [MN](unsigned Idx) {
      auto *V = cast<llvm::ConstantAsMetadata>(MN->getOperand(Idx));
      return cast<llvm::ConstantInt>(V->getValue())->getZExtValue();
    };

    auto &&GetMDString = [MN](unsigned Idx) {
      auto *V = cast<llvm::MDString>(MN->getOperand(Idx));
      return V->getString();
    };

    switch (GetMDInt(0)) {
    default:
      llvm_unreachable("Unexpected metadata!");
      break;
    case OffloadEntriesInfoManagerTy::OffloadEntryInfo::
        OffloadingEntryInfoTargetRegion:
      OffloadEntriesInfoManager.initializeTargetRegionEntryInfo(
          /*DeviceID=*/GetMDInt(1), /*FileID=*/GetMDInt(2),
          /*ParentName=*/GetMDString(3), /*Line=*/GetMDInt(4),
          /*Order=*/GetMDInt(5));
      break;
    case OffloadEntriesInfoManagerTy::OffloadEntryInfo::
        OffloadingEntryInfoDeviceGlobalVar:
      OffloadEntriesInfoManager.initializeDeviceGlobalVarEntryInfo(
          /*MangledName=*/GetMDString(1),
          static_cast<OffloadEntriesInfoManagerTy::OMPTargetGlobalVarEntryKind>(
              /*Flags=*/GetMDInt(2)),
          /*Order=*/GetMDInt(3));
      break;
    }
  }
}

void CGOpenMPRuntime::emitKmpRoutineEntryT(QualType KmpInt32Ty) {
  if (!KmpRoutineEntryPtrTy) {
    // Build typedef kmp_int32 (* kmp_routine_entry_t)(kmp_int32, void *); type.
    ASTContext &C = CGM.getContext();
    QualType KmpRoutineEntryTyArgs[] = {KmpInt32Ty, C.VoidPtrTy};
    FunctionProtoType::ExtProtoInfo EPI;
    KmpRoutineEntryPtrQTy = C.getPointerType(
        C.getFunctionType(KmpInt32Ty, KmpRoutineEntryTyArgs, EPI));
    KmpRoutineEntryPtrTy = CGM.getTypes().ConvertType(KmpRoutineEntryPtrQTy);
  }
}

QualType CGOpenMPRuntime::getTgtOffloadEntryQTy() {
  // Make sure the type of the entry is already created. This is the type we
  // have to create:
  // struct __tgt_offload_entry{
  //   void      *addr;       // Pointer to the offload entry info.
  //                          // (function or global)
  //   char      *name;       // Name of the function or global.
  //   size_t     size;       // Size of the entry info (0 if it a function).
  //   int32_t    flags;      // Flags associated with the entry, e.g. 'link'.
  //   int32_t    reserved;   // Reserved, to use by the runtime library.
  // };
  if (TgtOffloadEntryQTy.isNull()) {
    ASTContext &C = CGM.getContext();
    RecordDecl *RD = C.buildImplicitRecord("__tgt_offload_entry");
    RD->startDefinition();
    addFieldToRecordDecl(C, RD, C.VoidPtrTy);
    addFieldToRecordDecl(C, RD, C.getPointerType(C.CharTy));
    addFieldToRecordDecl(C, RD, C.getSizeType());
    addFieldToRecordDecl(
        C, RD, C.getIntTypeForBitwidth(/*DestWidth=*/32, /*Signed=*/true));
    addFieldToRecordDecl(
        C, RD, C.getIntTypeForBitwidth(/*DestWidth=*/32, /*Signed=*/true));
    RD->completeDefinition();
    RD->addAttr(PackedAttr::CreateImplicit(C));
    TgtOffloadEntryQTy = C.getRecordType(RD);
  }
  return TgtOffloadEntryQTy;
}

namespace {
struct PrivateHelpersTy {
  PrivateHelpersTy(const Expr *OriginalRef, const VarDecl *Original,
                   const VarDecl *PrivateCopy, const VarDecl *PrivateElemInit)
      : OriginalRef(OriginalRef), Original(Original), PrivateCopy(PrivateCopy),
        PrivateElemInit(PrivateElemInit) {}
  PrivateHelpersTy(const VarDecl *Original) : Original(Original) {}
  const Expr *OriginalRef = nullptr;
  const VarDecl *Original = nullptr;
  const VarDecl *PrivateCopy = nullptr;
  const VarDecl *PrivateElemInit = nullptr;
  bool isLocalPrivate() const {
    return !OriginalRef && !PrivateCopy && !PrivateElemInit;
  }
};
typedef std::pair<CharUnits /*Align*/, PrivateHelpersTy> PrivateDataTy;
} // anonymous namespace

static bool isAllocatableDecl(const VarDecl *VD) {
  const VarDecl *CVD = VD->getCanonicalDecl();
  if (!CVD->hasAttr<OMPAllocateDeclAttr>())
    return false;
  const auto *AA = CVD->getAttr<OMPAllocateDeclAttr>();
  // Use the default allocation.
  return !(AA->getAllocatorType() == OMPAllocateDeclAttr::OMPDefaultMemAlloc &&
           !AA->getAllocator());
}

static RecordDecl *
createPrivatesRecordDecl(CodeGenModule &CGM, ArrayRef<PrivateDataTy> Privates) {
  if (!Privates.empty()) {
    ASTContext &C = CGM.getContext();
    // Build struct .kmp_privates_t. {
    //         /*  private vars  */
    //       };
    RecordDecl *RD = C.buildImplicitRecord(".kmp_privates.t");
    RD->startDefinition();
    for (const auto &Pair : Privates) {
      const VarDecl *VD = Pair.second.Original;
      QualType Type = VD->getType().getNonReferenceType();
      // If the private variable is a local variable with lvalue ref type,
      // allocate the pointer instead of the pointee type.
      if (Pair.second.isLocalPrivate()) {
        if (VD->getType()->isLValueReferenceType())
          Type = C.getPointerType(Type);
        if (isAllocatableDecl(VD))
          Type = C.getPointerType(Type);
      }
      FieldDecl *FD = addFieldToRecordDecl(C, RD, Type);
      if (VD->hasAttrs()) {
        for (specific_attr_iterator<AlignedAttr> I(VD->getAttrs().begin()),
             E(VD->getAttrs().end());
             I != E; ++I)
          FD->addAttr(*I);
      }
    }
    RD->completeDefinition();
    return RD;
  }
  return nullptr;
}

static RecordDecl *
createKmpTaskTRecordDecl(CodeGenModule &CGM, OpenMPDirectiveKind Kind,
                         QualType KmpInt32Ty,
                         QualType KmpRoutineEntryPointerQTy) {
  ASTContext &C = CGM.getContext();
  // Build struct kmp_task_t {
  //         void *              shareds;
  //         kmp_routine_entry_t routine;
  //         kmp_int32           part_id;
  //         kmp_cmplrdata_t data1;
  //         kmp_cmplrdata_t data2;
  // For taskloops additional fields:
  //         kmp_uint64          lb;
  //         kmp_uint64          ub;
  //         kmp_int64           st;
  //         kmp_int32           liter;
  //         void *              reductions;
  //       };
  RecordDecl *UD = C.buildImplicitRecord("kmp_cmplrdata_t", TTK_Union);
  UD->startDefinition();
  addFieldToRecordDecl(C, UD, KmpInt32Ty);
  addFieldToRecordDecl(C, UD, KmpRoutineEntryPointerQTy);
  UD->completeDefinition();
  QualType KmpCmplrdataTy = C.getRecordType(UD);
  RecordDecl *RD = C.buildImplicitRecord("kmp_task_t");
  RD->startDefinition();
  addFieldToRecordDecl(C, RD, C.VoidPtrTy);
  addFieldToRecordDecl(C, RD, KmpRoutineEntryPointerQTy);
  addFieldToRecordDecl(C, RD, KmpInt32Ty);
  addFieldToRecordDecl(C, RD, KmpCmplrdataTy);
  addFieldToRecordDecl(C, RD, KmpCmplrdataTy);
  if (isOpenMPTaskLoopDirective(Kind)) {
    QualType KmpUInt64Ty =
        CGM.getContext().getIntTypeForBitwidth(/*DestWidth=*/64, /*Signed=*/0);
    QualType KmpInt64Ty =
        CGM.getContext().getIntTypeForBitwidth(/*DestWidth=*/64, /*Signed=*/1);
    addFieldToRecordDecl(C, RD, KmpUInt64Ty);
    addFieldToRecordDecl(C, RD, KmpUInt64Ty);
    addFieldToRecordDecl(C, RD, KmpInt64Ty);
    addFieldToRecordDecl(C, RD, KmpInt32Ty);
    addFieldToRecordDecl(C, RD, C.VoidPtrTy);
  }
  RD->completeDefinition();
  return RD;
}

static RecordDecl *
createKmpTaskTWithPrivatesRecordDecl(CodeGenModule &CGM, QualType KmpTaskTQTy,
                                     ArrayRef<PrivateDataTy> Privates) {
  ASTContext &C = CGM.getContext();
  // Build struct kmp_task_t_with_privates {
  //         kmp_task_t task_data;
  //         .kmp_privates_t. privates;
  //       };
  RecordDecl *RD = C.buildImplicitRecord("kmp_task_t_with_privates");
  RD->startDefinition();
  addFieldToRecordDecl(C, RD, KmpTaskTQTy);
  if (const RecordDecl *PrivateRD = createPrivatesRecordDecl(CGM, Privates))
    addFieldToRecordDecl(C, RD, C.getRecordType(PrivateRD));
  RD->completeDefinition();
  return RD;
}

/// Emit a proxy function which accepts kmp_task_t as the second
/// argument.
/// \code
/// kmp_int32 .omp_task_entry.(kmp_int32 gtid, kmp_task_t *tt) {
///   TaskFunction(gtid, tt->part_id, &tt->privates, task_privates_map, tt,
///   For taskloops:
///   tt->task_data.lb, tt->task_data.ub, tt->task_data.st, tt->task_data.liter,
///   tt->reductions, tt->shareds);
///   return 0;
/// }
/// \endcode
static llvm::Function *
emitProxyTaskFunction(CodeGenModule &CGM, SourceLocation Loc,
                      OpenMPDirectiveKind Kind, QualType KmpInt32Ty,
                      QualType KmpTaskTWithPrivatesPtrQTy,
                      QualType KmpTaskTWithPrivatesQTy, QualType KmpTaskTQTy,
                      QualType SharedsPtrTy, llvm::Function *TaskFunction,
                      llvm::Value *TaskPrivatesMap) {
  ASTContext &C = CGM.getContext();
  FunctionArgList Args;
  ImplicitParamDecl GtidArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr, KmpInt32Ty,
                            ImplicitParamDecl::Other);
  ImplicitParamDecl TaskTypeArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr,
                                KmpTaskTWithPrivatesPtrQTy.withRestrict(),
                                ImplicitParamDecl::Other);
  Args.push_back(&GtidArg);
  Args.push_back(&TaskTypeArg);
  const auto &TaskEntryFnInfo =
      CGM.getTypes().arrangeBuiltinFunctionDeclaration(KmpInt32Ty, Args);
  llvm::FunctionType *TaskEntryTy =
      CGM.getTypes().GetFunctionType(TaskEntryFnInfo);
  std::string Name = CGM.getOpenMPRuntime().getName({"omp_task_entry", ""});
  auto *TaskEntry = llvm::Function::Create(
      TaskEntryTy, llvm::GlobalValue::InternalLinkage, Name, &CGM.getModule());
  CGM.SetInternalFunctionAttributes(GlobalDecl(), TaskEntry, TaskEntryFnInfo);
  TaskEntry->setDoesNotRecurse();
  CodeGenFunction CGF(CGM);
  CGF.StartFunction(GlobalDecl(), KmpInt32Ty, TaskEntry, TaskEntryFnInfo, Args,
                    Loc, Loc);

  // TaskFunction(gtid, tt->task_data.part_id, &tt->privates, task_privates_map,
  // tt,
  // For taskloops:
  // tt->task_data.lb, tt->task_data.ub, tt->task_data.st, tt->task_data.liter,
  // tt->task_data.shareds);
  llvm::Value *GtidParam = CGF.EmitLoadOfScalar(
      CGF.GetAddrOfLocalVar(&GtidArg), /*Volatile=*/false, KmpInt32Ty, Loc);
  LValue TDBase = CGF.EmitLoadOfPointerLValue(
      CGF.GetAddrOfLocalVar(&TaskTypeArg),
      KmpTaskTWithPrivatesPtrQTy->castAs<PointerType>());
  const auto *KmpTaskTWithPrivatesQTyRD =
      cast<RecordDecl>(KmpTaskTWithPrivatesQTy->getAsTagDecl());
  LValue Base =
      CGF.EmitLValueForField(TDBase, *KmpTaskTWithPrivatesQTyRD->field_begin());
  const auto *KmpTaskTQTyRD = cast<RecordDecl>(KmpTaskTQTy->getAsTagDecl());
  auto PartIdFI = std::next(KmpTaskTQTyRD->field_begin(), KmpTaskTPartId);
  LValue PartIdLVal = CGF.EmitLValueForField(Base, *PartIdFI);
  llvm::Value *PartidParam = PartIdLVal.getPointer(CGF);

  auto SharedsFI = std::next(KmpTaskTQTyRD->field_begin(), KmpTaskTShareds);
  LValue SharedsLVal = CGF.EmitLValueForField(Base, *SharedsFI);
  llvm::Value *SharedsParam = CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
      CGF.EmitLoadOfScalar(SharedsLVal, Loc),
      CGF.ConvertTypeForMem(SharedsPtrTy));

  auto PrivatesFI = std::next(KmpTaskTWithPrivatesQTyRD->field_begin(), 1);
  llvm::Value *PrivatesParam;
  if (PrivatesFI != KmpTaskTWithPrivatesQTyRD->field_end()) {
    LValue PrivatesLVal = CGF.EmitLValueForField(TDBase, *PrivatesFI);
    PrivatesParam = CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
        PrivatesLVal.getPointer(CGF), CGF.VoidPtrTy);
  } else {
    PrivatesParam = llvm::ConstantPointerNull::get(CGF.VoidPtrTy);
  }

  llvm::Value *CommonArgs[] = {GtidParam, PartidParam, PrivatesParam,
                               TaskPrivatesMap,
                               CGF.Builder
                                   .CreatePointerBitCastOrAddrSpaceCast(
                                       TDBase.getAddress(CGF), CGF.VoidPtrTy)
                                   .getPointer()};
  SmallVector<llvm::Value *, 16> CallArgs(std::begin(CommonArgs),
                                          std::end(CommonArgs));
  if (isOpenMPTaskLoopDirective(Kind)) {
    auto LBFI = std::next(KmpTaskTQTyRD->field_begin(), KmpTaskTLowerBound);
    LValue LBLVal = CGF.EmitLValueForField(Base, *LBFI);
    llvm::Value *LBParam = CGF.EmitLoadOfScalar(LBLVal, Loc);
    auto UBFI = std::next(KmpTaskTQTyRD->field_begin(), KmpTaskTUpperBound);
    LValue UBLVal = CGF.EmitLValueForField(Base, *UBFI);
    llvm::Value *UBParam = CGF.EmitLoadOfScalar(UBLVal, Loc);
    auto StFI = std::next(KmpTaskTQTyRD->field_begin(), KmpTaskTStride);
    LValue StLVal = CGF.EmitLValueForField(Base, *StFI);
    llvm::Value *StParam = CGF.EmitLoadOfScalar(StLVal, Loc);
    auto LIFI = std::next(KmpTaskTQTyRD->field_begin(), KmpTaskTLastIter);
    LValue LILVal = CGF.EmitLValueForField(Base, *LIFI);
    llvm::Value *LIParam = CGF.EmitLoadOfScalar(LILVal, Loc);
    auto RFI = std::next(KmpTaskTQTyRD->field_begin(), KmpTaskTReductions);
    LValue RLVal = CGF.EmitLValueForField(Base, *RFI);
    llvm::Value *RParam = CGF.EmitLoadOfScalar(RLVal, Loc);
    CallArgs.push_back(LBParam);
    CallArgs.push_back(UBParam);
    CallArgs.push_back(StParam);
    CallArgs.push_back(LIParam);
    CallArgs.push_back(RParam);
  }
  CallArgs.push_back(SharedsParam);

  CGM.getOpenMPRuntime().emitOutlinedFunctionCall(CGF, Loc, TaskFunction,
                                                  CallArgs);
  CGF.EmitStoreThroughLValue(RValue::get(CGF.Builder.getInt32(/*C=*/0)),
                             CGF.MakeAddrLValue(CGF.ReturnValue, KmpInt32Ty));
  CGF.FinishFunction();
  return TaskEntry;
}

static llvm::Value *emitDestructorsFunction(CodeGenModule &CGM,
                                            SourceLocation Loc,
                                            QualType KmpInt32Ty,
                                            QualType KmpTaskTWithPrivatesPtrQTy,
                                            QualType KmpTaskTWithPrivatesQTy) {
  ASTContext &C = CGM.getContext();
  FunctionArgList Args;
  ImplicitParamDecl GtidArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr, KmpInt32Ty,
                            ImplicitParamDecl::Other);
  ImplicitParamDecl TaskTypeArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr,
                                KmpTaskTWithPrivatesPtrQTy.withRestrict(),
                                ImplicitParamDecl::Other);
  Args.push_back(&GtidArg);
  Args.push_back(&TaskTypeArg);
  const auto &DestructorFnInfo =
      CGM.getTypes().arrangeBuiltinFunctionDeclaration(KmpInt32Ty, Args);
  llvm::FunctionType *DestructorFnTy =
      CGM.getTypes().GetFunctionType(DestructorFnInfo);
  std::string Name =
      CGM.getOpenMPRuntime().getName({"omp_task_destructor", ""});
  auto *DestructorFn =
      llvm::Function::Create(DestructorFnTy, llvm::GlobalValue::InternalLinkage,
                             Name, &CGM.getModule());
  CGM.SetInternalFunctionAttributes(GlobalDecl(), DestructorFn,
                                    DestructorFnInfo);
  DestructorFn->setDoesNotRecurse();
  CodeGenFunction CGF(CGM);
  CGF.StartFunction(GlobalDecl(), KmpInt32Ty, DestructorFn, DestructorFnInfo,
                    Args, Loc, Loc);

  LValue Base = CGF.EmitLoadOfPointerLValue(
      CGF.GetAddrOfLocalVar(&TaskTypeArg),
      KmpTaskTWithPrivatesPtrQTy->castAs<PointerType>());
  const auto *KmpTaskTWithPrivatesQTyRD =
      cast<RecordDecl>(KmpTaskTWithPrivatesQTy->getAsTagDecl());
  auto FI = std::next(KmpTaskTWithPrivatesQTyRD->field_begin());
  Base = CGF.EmitLValueForField(Base, *FI);
  for (const auto *Field :
       cast<RecordDecl>(FI->getType()->getAsTagDecl())->fields()) {
    if (QualType::DestructionKind DtorKind =
            Field->getType().isDestructedType()) {
      LValue FieldLValue = CGF.EmitLValueForField(Base, Field);
      CGF.pushDestroy(DtorKind, FieldLValue.getAddress(CGF), Field->getType());
    }
  }
  CGF.FinishFunction();
  return DestructorFn;
}

/// Emit a privates mapping function for correct handling of private and
/// firstprivate variables.
/// \code
/// void .omp_task_privates_map.(const .privates. *noalias privs, <ty1>
/// **noalias priv1,...,  <tyn> **noalias privn) {
///   *priv1 = &.privates.priv1;
///   ...;
///   *privn = &.privates.privn;
/// }
/// \endcode
static llvm::Value *
emitTaskPrivateMappingFunction(CodeGenModule &CGM, SourceLocation Loc,
                               const OMPTaskDataTy &Data, QualType PrivatesQTy,
                               ArrayRef<PrivateDataTy> Privates) {
  ASTContext &C = CGM.getContext();
  FunctionArgList Args;
  ImplicitParamDecl TaskPrivatesArg(
      C, /*DC=*/nullptr, Loc, /*Id=*/nullptr,
      C.getPointerType(PrivatesQTy).withConst().withRestrict(),
      ImplicitParamDecl::Other);
  Args.push_back(&TaskPrivatesArg);
  llvm::DenseMap<CanonicalDeclPtr<const VarDecl>, unsigned> PrivateVarsPos;
  unsigned Counter = 1;
  for (const Expr *E : Data.PrivateVars) {
    Args.push_back(ImplicitParamDecl::Create(
        C, /*DC=*/nullptr, Loc, /*Id=*/nullptr,
        C.getPointerType(C.getPointerType(E->getType()))
            .withConst()
            .withRestrict(),
        ImplicitParamDecl::Other));
    const auto *VD = cast<VarDecl>(cast<DeclRefExpr>(E)->getDecl());
    PrivateVarsPos[VD] = Counter;
    ++Counter;
  }
  for (const Expr *E : Data.FirstprivateVars) {
    Args.push_back(ImplicitParamDecl::Create(
        C, /*DC=*/nullptr, Loc, /*Id=*/nullptr,
        C.getPointerType(C.getPointerType(E->getType()))
            .withConst()
            .withRestrict(),
        ImplicitParamDecl::Other));
    const auto *VD = cast<VarDecl>(cast<DeclRefExpr>(E)->getDecl());
    PrivateVarsPos[VD] = Counter;
    ++Counter;
  }
  for (const Expr *E : Data.LastprivateVars) {
    Args.push_back(ImplicitParamDecl::Create(
        C, /*DC=*/nullptr, Loc, /*Id=*/nullptr,
        C.getPointerType(C.getPointerType(E->getType()))
            .withConst()
            .withRestrict(),
        ImplicitParamDecl::Other));
    const auto *VD = cast<VarDecl>(cast<DeclRefExpr>(E)->getDecl());
    PrivateVarsPos[VD] = Counter;
    ++Counter;
  }
  for (const VarDecl *VD : Data.PrivateLocals) {
    QualType Ty = VD->getType().getNonReferenceType();
    if (VD->getType()->isLValueReferenceType())
      Ty = C.getPointerType(Ty);
    if (isAllocatableDecl(VD))
      Ty = C.getPointerType(Ty);
    Args.push_back(ImplicitParamDecl::Create(
        C, /*DC=*/nullptr, Loc, /*Id=*/nullptr,
        C.getPointerType(C.getPointerType(Ty)).withConst().withRestrict(),
        ImplicitParamDecl::Other));
    PrivateVarsPos[VD] = Counter;
    ++Counter;
  }
  const auto &TaskPrivatesMapFnInfo =
      CGM.getTypes().arrangeBuiltinFunctionDeclaration(C.VoidTy, Args);
  llvm::FunctionType *TaskPrivatesMapTy =
      CGM.getTypes().GetFunctionType(TaskPrivatesMapFnInfo);
  std::string Name =
      CGM.getOpenMPRuntime().getName({"omp_task_privates_map", ""});
  auto *TaskPrivatesMap = llvm::Function::Create(
      TaskPrivatesMapTy, llvm::GlobalValue::InternalLinkage, Name,
      &CGM.getModule());
  CGM.SetInternalFunctionAttributes(GlobalDecl(), TaskPrivatesMap,
                                    TaskPrivatesMapFnInfo);
  if (CGM.getLangOpts().Optimize) {
    TaskPrivatesMap->removeFnAttr(llvm::Attribute::NoInline);
    TaskPrivatesMap->removeFnAttr(llvm::Attribute::OptimizeNone);
    TaskPrivatesMap->addFnAttr(llvm::Attribute::AlwaysInline);
  }
  CodeGenFunction CGF(CGM);
  CGF.StartFunction(GlobalDecl(), C.VoidTy, TaskPrivatesMap,
                    TaskPrivatesMapFnInfo, Args, Loc, Loc);

  // *privi = &.privates.privi;
  LValue Base = CGF.EmitLoadOfPointerLValue(
      CGF.GetAddrOfLocalVar(&TaskPrivatesArg),
      TaskPrivatesArg.getType()->castAs<PointerType>());
  const auto *PrivatesQTyRD = cast<RecordDecl>(PrivatesQTy->getAsTagDecl());
  Counter = 0;
  for (const FieldDecl *Field : PrivatesQTyRD->fields()) {
    LValue FieldLVal = CGF.EmitLValueForField(Base, Field);
    const VarDecl *VD = Args[PrivateVarsPos[Privates[Counter].second.Original]];
    LValue RefLVal =
        CGF.MakeAddrLValue(CGF.GetAddrOfLocalVar(VD), VD->getType());
    LValue RefLoadLVal = CGF.EmitLoadOfPointerLValue(
        RefLVal.getAddress(CGF), RefLVal.getType()->castAs<PointerType>());
    CGF.EmitStoreOfScalar(FieldLVal.getPointer(CGF), RefLoadLVal);
    ++Counter;
  }
  CGF.FinishFunction();
  return TaskPrivatesMap;
}

/// Emit initialization for private variables in task-based directives.
static void emitPrivatesInit(CodeGenFunction &CGF,
                             const OMPExecutableDirective &D,
                             Address KmpTaskSharedsPtr, LValue TDBase,
                             const RecordDecl *KmpTaskTWithPrivatesQTyRD,
                             QualType SharedsTy, QualType SharedsPtrTy,
                             const OMPTaskDataTy &Data,
                             ArrayRef<PrivateDataTy> Privates, bool ForDup) {
  ASTContext &C = CGF.getContext();
  auto FI = std::next(KmpTaskTWithPrivatesQTyRD->field_begin());
  LValue PrivatesBase = CGF.EmitLValueForField(TDBase, *FI);
  OpenMPDirectiveKind Kind = isOpenMPTaskLoopDirective(D.getDirectiveKind())
                                 ? OMPD_taskloop
                                 : OMPD_task;
  const CapturedStmt &CS = *D.getCapturedStmt(Kind);
  CodeGenFunction::CGCapturedStmtInfo CapturesInfo(CS);
  LValue SrcBase;
  bool IsTargetTask =
      isOpenMPTargetDataManagementDirective(D.getDirectiveKind()) ||
      isOpenMPTargetExecutionDirective(D.getDirectiveKind());
  // For target-based directives skip 4 firstprivate arrays BasePointersArray,
  // PointersArray, SizesArray, and MappersArray. The original variables for
  // these arrays are not captured and we get their addresses explicitly.
  if ((!IsTargetTask && !Data.FirstprivateVars.empty() && ForDup) ||
      (IsTargetTask && KmpTaskSharedsPtr.isValid())) {
    SrcBase = CGF.MakeAddrLValue(
        CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
            KmpTaskSharedsPtr, CGF.ConvertTypeForMem(SharedsPtrTy)),
        SharedsTy);
  }
  FI = cast<RecordDecl>(FI->getType()->getAsTagDecl())->field_begin();
  for (const PrivateDataTy &Pair : Privates) {
    // Do not initialize private locals.
    if (Pair.second.isLocalPrivate()) {
      ++FI;
      continue;
    }
    const VarDecl *VD = Pair.second.PrivateCopy;
    const Expr *Init = VD->getAnyInitializer();
    if (Init && (!ForDup || (isa<CXXConstructExpr>(Init) &&
                             !CGF.isTrivialInitializer(Init)))) {
      LValue PrivateLValue = CGF.EmitLValueForField(PrivatesBase, *FI);
      if (const VarDecl *Elem = Pair.second.PrivateElemInit) {
        const VarDecl *OriginalVD = Pair.second.Original;
        // Check if the variable is the target-based BasePointersArray,
        // PointersArray, SizesArray, or MappersArray.
        LValue SharedRefLValue;
        QualType Type = PrivateLValue.getType();
        const FieldDecl *SharedField = CapturesInfo.lookup(OriginalVD);
        if (IsTargetTask && !SharedField) {
          assert(isa<ImplicitParamDecl>(OriginalVD) &&
                 isa<CapturedDecl>(OriginalVD->getDeclContext()) &&
                 cast<CapturedDecl>(OriginalVD->getDeclContext())
                         ->getNumParams() == 0 &&
                 isa<TranslationUnitDecl>(
                     cast<CapturedDecl>(OriginalVD->getDeclContext())
                         ->getDeclContext()) &&
                 "Expected artificial target data variable.");
          SharedRefLValue =
              CGF.MakeAddrLValue(CGF.GetAddrOfLocalVar(OriginalVD), Type);
        } else if (ForDup) {
          SharedRefLValue = CGF.EmitLValueForField(SrcBase, SharedField);
          SharedRefLValue = CGF.MakeAddrLValue(
              Address(SharedRefLValue.getPointer(CGF),
                      C.getDeclAlign(OriginalVD)),
              SharedRefLValue.getType(), LValueBaseInfo(AlignmentSource::Decl),
              SharedRefLValue.getTBAAInfo());
        } else if (CGF.LambdaCaptureFields.count(
                       Pair.second.Original->getCanonicalDecl()) > 0 ||
                   isa_and_nonnull<BlockDecl>(CGF.CurCodeDecl)) {
          SharedRefLValue = CGF.EmitLValue(Pair.second.OriginalRef);
        } else {
          // Processing for implicitly captured variables.
          InlinedOpenMPRegionRAII Region(
              CGF, [](CodeGenFunction &, PrePostActionTy &) {}, OMPD_unknown,
              /*HasCancel=*/false, /*NoInheritance=*/true);
          SharedRefLValue = CGF.EmitLValue(Pair.second.OriginalRef);
        }
        if (Type->isArrayType()) {
          // Initialize firstprivate array.
          if (!isa<CXXConstructExpr>(Init) || CGF.isTrivialInitializer(Init)) {
            // Perform simple memcpy.
            CGF.EmitAggregateAssign(PrivateLValue, SharedRefLValue, Type);
          } else {
            // Initialize firstprivate array using element-by-element
            // initialization.
            CGF.EmitOMPAggregateAssign(
                PrivateLValue.getAddress(CGF), SharedRefLValue.getAddress(CGF),
                Type,
                [&CGF, Elem, Init, &CapturesInfo](Address DestElement,
                                                  Address SrcElement) {
                  // Clean up any temporaries needed by the initialization.
                  CodeGenFunction::OMPPrivateScope InitScope(CGF);
                  InitScope.addPrivate(
                      Elem, [SrcElement]() -> Address { return SrcElement; });
                  (void)InitScope.Privatize();
                  // Emit initialization for single element.
                  CodeGenFunction::CGCapturedStmtRAII CapInfoRAII(
                      CGF, &CapturesInfo);
                  CGF.EmitAnyExprToMem(Init, DestElement,
                                       Init->getType().getQualifiers(),
                                       /*IsInitializer=*/false);
                });
          }
        } else {
          CodeGenFunction::OMPPrivateScope InitScope(CGF);
          InitScope.addPrivate(Elem, [SharedRefLValue, &CGF]() -> Address {
            return SharedRefLValue.getAddress(CGF);
          });
          (void)InitScope.Privatize();
          CodeGenFunction::CGCapturedStmtRAII CapInfoRAII(CGF, &CapturesInfo);
          CGF.EmitExprAsInit(Init, VD, PrivateLValue,
                             /*capturedByInit=*/false);
        }
      } else {
        CGF.EmitExprAsInit(Init, VD, PrivateLValue, /*capturedByInit=*/false);
      }
    }
    ++FI;
  }
}

/// Check if duplication function is required for taskloops.
static bool checkInitIsRequired(CodeGenFunction &CGF,
                                ArrayRef<PrivateDataTy> Privates) {
  bool InitRequired = false;
  for (const PrivateDataTy &Pair : Privates) {
    if (Pair.second.isLocalPrivate())
      continue;
    const VarDecl *VD = Pair.second.PrivateCopy;
    const Expr *Init = VD->getAnyInitializer();
    InitRequired = InitRequired || (Init && isa<CXXConstructExpr>(Init) &&
                                    !CGF.isTrivialInitializer(Init));
    if (InitRequired)
      break;
  }
  return InitRequired;
}


/// Emit task_dup function (for initialization of
/// private/firstprivate/lastprivate vars and last_iter flag)
/// \code
/// void __task_dup_entry(kmp_task_t *task_dst, const kmp_task_t *task_src, int
/// lastpriv) {
/// // setup lastprivate flag
///    task_dst->last = lastpriv;
/// // could be constructor calls here...
/// }
/// \endcode
static llvm::Value *
emitTaskDupFunction(CodeGenModule &CGM, SourceLocation Loc,
                    const OMPExecutableDirective &D,
                    QualType KmpTaskTWithPrivatesPtrQTy,
                    const RecordDecl *KmpTaskTWithPrivatesQTyRD,
                    const RecordDecl *KmpTaskTQTyRD, QualType SharedsTy,
                    QualType SharedsPtrTy, const OMPTaskDataTy &Data,
                    ArrayRef<PrivateDataTy> Privates, bool WithLastIter) {
  ASTContext &C = CGM.getContext();
  FunctionArgList Args;
  ImplicitParamDecl DstArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr,
                           KmpTaskTWithPrivatesPtrQTy,
                           ImplicitParamDecl::Other);
  ImplicitParamDecl SrcArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr,
                           KmpTaskTWithPrivatesPtrQTy,
                           ImplicitParamDecl::Other);
  ImplicitParamDecl LastprivArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr, C.IntTy,
                                ImplicitParamDecl::Other);
  Args.push_back(&DstArg);
  Args.push_back(&SrcArg);
  Args.push_back(&LastprivArg);
  const auto &TaskDupFnInfo =
      CGM.getTypes().arrangeBuiltinFunctionDeclaration(C.VoidTy, Args);
  llvm::FunctionType *TaskDupTy = CGM.getTypes().GetFunctionType(TaskDupFnInfo);
  std::string Name = CGM.getOpenMPRuntime().getName({"omp_task_dup", ""});
  auto *TaskDup = llvm::Function::Create(
      TaskDupTy, llvm::GlobalValue::InternalLinkage, Name, &CGM.getModule());
  CGM.SetInternalFunctionAttributes(GlobalDecl(), TaskDup, TaskDupFnInfo);
  TaskDup->setDoesNotRecurse();
  CodeGenFunction CGF(CGM);
  CGF.StartFunction(GlobalDecl(), C.VoidTy, TaskDup, TaskDupFnInfo, Args, Loc,
                    Loc);

  LValue TDBase = CGF.EmitLoadOfPointerLValue(
      CGF.GetAddrOfLocalVar(&DstArg),
      KmpTaskTWithPrivatesPtrQTy->castAs<PointerType>());
  // task_dst->liter = lastpriv;
  if (WithLastIter) {
    auto LIFI = std::next(KmpTaskTQTyRD->field_begin(), KmpTaskTLastIter);
    LValue Base = CGF.EmitLValueForField(
        TDBase, *KmpTaskTWithPrivatesQTyRD->field_begin());
    LValue LILVal = CGF.EmitLValueForField(Base, *LIFI);
    llvm::Value *Lastpriv = CGF.EmitLoadOfScalar(
        CGF.GetAddrOfLocalVar(&LastprivArg), /*Volatile=*/false, C.IntTy, Loc);
    CGF.EmitStoreOfScalar(Lastpriv, LILVal);
  }

  // Emit initial values for private copies (if any).
  assert(!Privates.empty());
  Address KmpTaskSharedsPtr = Address::invalid();
  if (!Data.FirstprivateVars.empty()) {
    LValue TDBase = CGF.EmitLoadOfPointerLValue(
        CGF.GetAddrOfLocalVar(&SrcArg),
        KmpTaskTWithPrivatesPtrQTy->castAs<PointerType>());
    LValue Base = CGF.EmitLValueForField(
        TDBase, *KmpTaskTWithPrivatesQTyRD->field_begin());
    KmpTaskSharedsPtr = Address(
        CGF.EmitLoadOfScalar(CGF.EmitLValueForField(
                                 Base, *std::next(KmpTaskTQTyRD->field_begin(),
                                                  KmpTaskTShareds)),
                             Loc),
        CGM.getNaturalTypeAlignment(SharedsTy));
  }
  emitPrivatesInit(CGF, D, KmpTaskSharedsPtr, TDBase, KmpTaskTWithPrivatesQTyRD,
                   SharedsTy, SharedsPtrTy, Data, Privates, /*ForDup=*/true);
  CGF.FinishFunction();
  return TaskDup;
}

/// Checks if destructor function is required to be generated.
/// \return true if cleanups are required, false otherwise.
static bool
checkDestructorsRequired(const RecordDecl *KmpTaskTWithPrivatesQTyRD,
                         ArrayRef<PrivateDataTy> Privates) {
  for (const PrivateDataTy &P : Privates) {
    if (P.second.isLocalPrivate())
      continue;
    QualType Ty = P.second.Original->getType().getNonReferenceType();
    if (Ty.isDestructedType())
      return true;
  }
  return false;
}

namespace {
/// Loop generator for OpenMP iterator expression.
class OMPIteratorGeneratorScope final
    : public CodeGenFunction::OMPPrivateScope {
  CodeGenFunction &CGF;
  const OMPIteratorExpr *E = nullptr;
  SmallVector<CodeGenFunction::JumpDest, 4> ContDests;
  SmallVector<CodeGenFunction::JumpDest, 4> ExitDests;
  OMPIteratorGeneratorScope() = delete;
  OMPIteratorGeneratorScope(OMPIteratorGeneratorScope &) = delete;

public:
  OMPIteratorGeneratorScope(CodeGenFunction &CGF, const OMPIteratorExpr *E)
      : CodeGenFunction::OMPPrivateScope(CGF), CGF(CGF), E(E) {
    if (!E)
      return;
    SmallVector<llvm::Value *, 4> Uppers;
    for (unsigned I = 0, End = E->numOfIterators(); I < End; ++I) {
      Uppers.push_back(CGF.EmitScalarExpr(E->getHelper(I).Upper));
      const auto *VD = cast<VarDecl>(E->getIteratorDecl(I));
      addPrivate(VD, [&CGF, VD]() {
        return CGF.CreateMemTemp(VD->getType(), VD->getName());
      });
      const OMPIteratorHelperData &HelperData = E->getHelper(I);
      addPrivate(HelperData.CounterVD, [&CGF, &HelperData]() {
        return CGF.CreateMemTemp(HelperData.CounterVD->getType(),
                                 "counter.addr");
      });
    }
    Privatize();

    for (unsigned I = 0, End = E->numOfIterators(); I < End; ++I) {
      const OMPIteratorHelperData &HelperData = E->getHelper(I);
      LValue CLVal =
          CGF.MakeAddrLValue(CGF.GetAddrOfLocalVar(HelperData.CounterVD),
                             HelperData.CounterVD->getType());
      // Counter = 0;
      CGF.EmitStoreOfScalar(
          llvm::ConstantInt::get(CLVal.getAddress(CGF).getElementType(), 0),
          CLVal);
      CodeGenFunction::JumpDest &ContDest =
          ContDests.emplace_back(CGF.getJumpDestInCurrentScope("iter.cont"));
      CodeGenFunction::JumpDest &ExitDest =
          ExitDests.emplace_back(CGF.getJumpDestInCurrentScope("iter.exit"));
      // N = <number-of_iterations>;
      llvm::Value *N = Uppers[I];
      // cont:
      // if (Counter < N) goto body; else goto exit;
      CGF.EmitBlock(ContDest.getBlock());
      auto *CVal =
          CGF.EmitLoadOfScalar(CLVal, HelperData.CounterVD->getLocation());
      llvm::Value *Cmp =
          HelperData.CounterVD->getType()->isSignedIntegerOrEnumerationType()
              ? CGF.Builder.CreateICmpSLT(CVal, N)
              : CGF.Builder.CreateICmpULT(CVal, N);
      llvm::BasicBlock *BodyBB = CGF.createBasicBlock("iter.body");
      CGF.Builder.CreateCondBr(Cmp, BodyBB, ExitDest.getBlock());
      // body:
      CGF.EmitBlock(BodyBB);
      // Iteri = Begini + Counter * Stepi;
      CGF.EmitIgnoredExpr(HelperData.Update);
    }
  }
  ~OMPIteratorGeneratorScope() {
    if (!E)
      return;
    for (unsigned I = E->numOfIterators(); I > 0; --I) {
      // Counter = Counter + 1;
      const OMPIteratorHelperData &HelperData = E->getHelper(I - 1);
      CGF.EmitIgnoredExpr(HelperData.CounterUpdate);
      // goto cont;
      CGF.EmitBranchThroughCleanup(ContDests[I - 1]);
      // exit:
      CGF.EmitBlock(ExitDests[I - 1].getBlock(), /*IsFinished=*/I == 1);
    }
  }
};
} // namespace

static std::pair<llvm::Value *, llvm::Value *>
getPointerAndSize(CodeGenFunction &CGF, const Expr *E) {
  const auto *OASE = dyn_cast<OMPArrayShapingExpr>(E);
  llvm::Value *Addr;
  if (OASE) {
    const Expr *Base = OASE->getBase();
    Addr = CGF.EmitScalarExpr(Base);
  } else {
    Addr = CGF.EmitLValue(E).getPointer(CGF);
  }
  llvm::Value *SizeVal;
  QualType Ty = E->getType();
  if (OASE) {
    SizeVal = CGF.getTypeSize(OASE->getBase()->getType()->getPointeeType());
    for (const Expr *SE : OASE->getDimensions()) {
      llvm::Value *Sz = CGF.EmitScalarExpr(SE);
      Sz = CGF.EmitScalarConversion(
          Sz, SE->getType(), CGF.getContext().getSizeType(), SE->getExprLoc());
      SizeVal = CGF.Builder.CreateNUWMul(SizeVal, Sz);
    }
  } else if (const auto *ASE =
                 dyn_cast<OMPArraySectionExpr>(E->IgnoreParenImpCasts())) {
    LValue UpAddrLVal =
        CGF.EmitOMPArraySectionExpr(ASE, /*IsLowerBound=*/false);
    Address UpAddrAddress = UpAddrLVal.getAddress(CGF);
    llvm::Value *UpAddr = CGF.Builder.CreateConstGEP1_32(
        UpAddrAddress.getElementType(), UpAddrAddress.getPointer(), /*Idx0=*/1);
    llvm::Value *LowIntPtr = CGF.Builder.CreatePtrToInt(Addr, CGF.SizeTy);
    llvm::Value *UpIntPtr = CGF.Builder.CreatePtrToInt(UpAddr, CGF.SizeTy);
    SizeVal = CGF.Builder.CreateNUWSub(UpIntPtr, LowIntPtr);
  } else {
    SizeVal = CGF.getTypeSize(Ty);
  }
  return std::make_pair(Addr, SizeVal);
}

/// Builds kmp_depend_info, if it is not built yet, and builds flags type.
static void getKmpAffinityType(ASTContext &C, QualType &KmpTaskAffinityInfoTy) {
  QualType FlagsTy = C.getIntTypeForBitwidth(32, /*Signed=*/false);
  if (KmpTaskAffinityInfoTy.isNull()) {
    RecordDecl *KmpAffinityInfoRD =
        C.buildImplicitRecord("kmp_task_affinity_info_t");
    KmpAffinityInfoRD->startDefinition();
    addFieldToRecordDecl(C, KmpAffinityInfoRD, C.getIntPtrType());
    addFieldToRecordDecl(C, KmpAffinityInfoRD, C.getSizeType());
    addFieldToRecordDecl(C, KmpAffinityInfoRD, FlagsTy);
    KmpAffinityInfoRD->completeDefinition();
    KmpTaskAffinityInfoTy = C.getRecordType(KmpAffinityInfoRD);
  }
}

CGOpenMPRuntime::TaskResultTy
CGOpenMPRuntime::emitTaskInit(CodeGenFunction &CGF, SourceLocation Loc,
                              const OMPExecutableDirective &D,
                              llvm::Function *TaskFunction, QualType SharedsTy,
                              Address Shareds, const OMPTaskDataTy &Data) {
  ASTContext &C = CGM.getContext();
  llvm::SmallVector<PrivateDataTy, 4> Privates;
  // Aggregate privates and sort them by the alignment.
  const auto *I = Data.PrivateCopies.begin();
  for (const Expr *E : Data.PrivateVars) {
    const auto *VD = cast<VarDecl>(cast<DeclRefExpr>(E)->getDecl());
    Privates.emplace_back(
        C.getDeclAlign(VD),
        PrivateHelpersTy(E, VD, cast<VarDecl>(cast<DeclRefExpr>(*I)->getDecl()),
                         /*PrivateElemInit=*/nullptr));
    ++I;
  }
  I = Data.FirstprivateCopies.begin();
  const auto *IElemInitRef = Data.FirstprivateInits.begin();
  for (const Expr *E : Data.FirstprivateVars) {
    const auto *VD = cast<VarDecl>(cast<DeclRefExpr>(E)->getDecl());
    Privates.emplace_back(
        C.getDeclAlign(VD),
        PrivateHelpersTy(
            E, VD, cast<VarDecl>(cast<DeclRefExpr>(*I)->getDecl()),
            cast<VarDecl>(cast<DeclRefExpr>(*IElemInitRef)->getDecl())));
    ++I;
    ++IElemInitRef;
  }
  I = Data.LastprivateCopies.begin();
  for (const Expr *E : Data.LastprivateVars) {
    const auto *VD = cast<VarDecl>(cast<DeclRefExpr>(E)->getDecl());
    Privates.emplace_back(
        C.getDeclAlign(VD),
        PrivateHelpersTy(E, VD, cast<VarDecl>(cast<DeclRefExpr>(*I)->getDecl()),
                         /*PrivateElemInit=*/nullptr));
    ++I;
  }
  for (const VarDecl *VD : Data.PrivateLocals) {
    if (isAllocatableDecl(VD))
      Privates.emplace_back(CGM.getPointerAlign(), PrivateHelpersTy(VD));
    else
      Privates.emplace_back(C.getDeclAlign(VD), PrivateHelpersTy(VD));
  }
  llvm::stable_sort(Privates,
                    [](const PrivateDataTy &L, const PrivateDataTy &R) {
                      return L.first > R.first;
                    });
  QualType KmpInt32Ty = C.getIntTypeForBitwidth(/*DestWidth=*/32, /*Signed=*/1);
  // Build type kmp_routine_entry_t (if not built yet).
  emitKmpRoutineEntryT(KmpInt32Ty);
  // Build type kmp_task_t (if not built yet).
  if (isOpenMPTaskLoopDirective(D.getDirectiveKind())) {
    if (SavedKmpTaskloopTQTy.isNull()) {
      SavedKmpTaskloopTQTy = C.getRecordType(createKmpTaskTRecordDecl(
          CGM, D.getDirectiveKind(), KmpInt32Ty, KmpRoutineEntryPtrQTy));
    }
    KmpTaskTQTy = SavedKmpTaskloopTQTy;
  } else {
    assert((D.getDirectiveKind() == OMPD_task ||
            isOpenMPTargetExecutionDirective(D.getDirectiveKind()) ||
            isOpenMPTargetDataManagementDirective(D.getDirectiveKind())) &&
           "Expected taskloop, task or target directive");
    if (SavedKmpTaskTQTy.isNull()) {
      SavedKmpTaskTQTy = C.getRecordType(createKmpTaskTRecordDecl(
          CGM, D.getDirectiveKind(), KmpInt32Ty, KmpRoutineEntryPtrQTy));
    }
    KmpTaskTQTy = SavedKmpTaskTQTy;
  }
  const auto *KmpTaskTQTyRD = cast<RecordDecl>(KmpTaskTQTy->getAsTagDecl());
  // Build particular struct kmp_task_t for the given task.
  const RecordDecl *KmpTaskTWithPrivatesQTyRD =
      createKmpTaskTWithPrivatesRecordDecl(CGM, KmpTaskTQTy, Privates);
  QualType KmpTaskTWithPrivatesQTy = C.getRecordType(KmpTaskTWithPrivatesQTyRD);
  QualType KmpTaskTWithPrivatesPtrQTy =
      C.getPointerType(KmpTaskTWithPrivatesQTy);
  llvm::Type *KmpTaskTWithPrivatesTy = CGF.ConvertType(KmpTaskTWithPrivatesQTy);
  llvm::Type *KmpTaskTWithPrivatesPtrTy =
      KmpTaskTWithPrivatesTy->getPointerTo();
  llvm::Value *KmpTaskTWithPrivatesTySize =
      CGF.getTypeSize(KmpTaskTWithPrivatesQTy);
  QualType SharedsPtrTy = C.getPointerType(SharedsTy);

  // Emit initial values for private copies (if any).
  llvm::Value *TaskPrivatesMap = nullptr;
  llvm::Type *TaskPrivatesMapTy =
      std::next(TaskFunction->arg_begin(), 3)->getType();
  if (!Privates.empty()) {
    auto FI = std::next(KmpTaskTWithPrivatesQTyRD->field_begin());
    TaskPrivatesMap =
        emitTaskPrivateMappingFunction(CGM, Loc, Data, FI->getType(), Privates);
    TaskPrivatesMap = CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
        TaskPrivatesMap, TaskPrivatesMapTy);
  } else {
    TaskPrivatesMap = llvm::ConstantPointerNull::get(
        cast<llvm::PointerType>(TaskPrivatesMapTy));
  }
  // Build a proxy function kmp_int32 .omp_task_entry.(kmp_int32 gtid,
  // kmp_task_t *tt);
  llvm::Function *TaskEntry = emitProxyTaskFunction(
      CGM, Loc, D.getDirectiveKind(), KmpInt32Ty, KmpTaskTWithPrivatesPtrQTy,
      KmpTaskTWithPrivatesQTy, KmpTaskTQTy, SharedsPtrTy, TaskFunction,
      TaskPrivatesMap);

  // Build call kmp_task_t * __kmpc_omp_task_alloc(ident_t *, kmp_int32 gtid,
  // kmp_int32 flags, size_t sizeof_kmp_task_t, size_t sizeof_shareds,
  // kmp_routine_entry_t *task_entry);
  // Task flags. Format is taken from
  // https://github.com/llvm/llvm-project/blob/main/openmp/runtime/src/kmp.h,
  // description of kmp_tasking_flags struct.
  enum {
    TiedFlag = 0x1,
    FinalFlag = 0x2,
    DestructorsFlag = 0x8,
    PriorityFlag = 0x20,
    DetachableFlag = 0x40,
  };
  unsigned Flags = Data.Tied ? TiedFlag : 0;
  bool NeedsCleanup = false;
  if (!Privates.empty()) {
    NeedsCleanup =
        checkDestructorsRequired(KmpTaskTWithPrivatesQTyRD, Privates);
    if (NeedsCleanup)
      Flags = Flags | DestructorsFlag;
  }
  if (Data.Priority.getInt())
    Flags = Flags | PriorityFlag;
  if (D.hasClausesOfKind<OMPDetachClause>())
    Flags = Flags | DetachableFlag;
  llvm::Value *TaskFlags =
      Data.Final.getPointer()
          ? CGF.Builder.CreateSelect(Data.Final.getPointer(),
                                     CGF.Builder.getInt32(FinalFlag),
                                     CGF.Builder.getInt32(/*C=*/0))
          : CGF.Builder.getInt32(Data.Final.getInt() ? FinalFlag : 0);
  TaskFlags = CGF.Builder.CreateOr(TaskFlags, CGF.Builder.getInt32(Flags));
  llvm::Value *SharedsSize = CGM.getSize(C.getTypeSizeInChars(SharedsTy));
  SmallVector<llvm::Value *, 8> AllocArgs = {emitUpdateLocation(CGF, Loc),
      getThreadID(CGF, Loc), TaskFlags, KmpTaskTWithPrivatesTySize,
      SharedsSize, CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
          TaskEntry, KmpRoutineEntryPtrTy)};
  llvm::Value *NewTask;
  if (D.hasClausesOfKind<OMPNowaitClause>()) {
    // Check if we have any device clause associated with the directive.
    const Expr *Device = nullptr;
    if (auto *C = D.getSingleClause<OMPDeviceClause>())
      Device = C->getDevice();
    // Emit device ID if any otherwise use default value.
    llvm::Value *DeviceID;
    if (Device)
      DeviceID = CGF.Builder.CreateIntCast(CGF.EmitScalarExpr(Device),
                                           CGF.Int64Ty, /*isSigned=*/true);
    else
      DeviceID = CGF.Builder.getInt64(OMP_DEVICEID_UNDEF);
    AllocArgs.push_back(DeviceID);
    NewTask = CGF.EmitRuntimeCall(
        OMPBuilder.getOrCreateRuntimeFunction(
            CGM.getModule(), OMPRTL___kmpc_omp_target_task_alloc),
        AllocArgs);
  } else {
    NewTask =
        CGF.EmitRuntimeCall(OMPBuilder.getOrCreateRuntimeFunction(
                                CGM.getModule(), OMPRTL___kmpc_omp_task_alloc),
                            AllocArgs);
  }
  // Emit detach clause initialization.
  // evt = (typeof(evt))__kmpc_task_allow_completion_event(loc, tid,
  // task_descriptor);
  if (const auto *DC = D.getSingleClause<OMPDetachClause>()) {
    const Expr *Evt = DC->getEventHandler()->IgnoreParenImpCasts();
    LValue EvtLVal = CGF.EmitLValue(Evt);

    // Build kmp_event_t *__kmpc_task_allow_completion_event(ident_t *loc_ref,
    // int gtid, kmp_task_t *task);
    llvm::Value *Loc = emitUpdateLocation(CGF, DC->getBeginLoc());
    llvm::Value *Tid = getThreadID(CGF, DC->getBeginLoc());
    Tid = CGF.Builder.CreateIntCast(Tid, CGF.IntTy, /*isSigned=*/false);
    llvm::Value *EvtVal = CGF.EmitRuntimeCall(
        OMPBuilder.getOrCreateRuntimeFunction(
            CGM.getModule(), OMPRTL___kmpc_task_allow_completion_event),
        {Loc, Tid, NewTask});
    EvtVal = CGF.EmitScalarConversion(EvtVal, C.VoidPtrTy, Evt->getType(),
                                      Evt->getExprLoc());
    CGF.EmitStoreOfScalar(EvtVal, EvtLVal);
  }
  // Process affinity clauses.
  if (D.hasClausesOfKind<OMPAffinityClause>()) {
    // Process list of affinity data.
    ASTContext &C = CGM.getContext();
    Address AffinitiesArray = Address::invalid();
    // Calculate number of elements to form the array of affinity data.
    llvm::Value *NumOfElements = nullptr;
    unsigned NumAffinities = 0;
    for (const auto *C : D.getClausesOfKind<OMPAffinityClause>()) {
      if (const Expr *Modifier = C->getModifier()) {
        const auto *IE = cast<OMPIteratorExpr>(Modifier->IgnoreParenImpCasts());
        for (unsigned I = 0, E = IE->numOfIterators(); I < E; ++I) {
          llvm::Value *Sz = CGF.EmitScalarExpr(IE->getHelper(I).Upper);
          Sz = CGF.Builder.CreateIntCast(Sz, CGF.SizeTy, /*isSigned=*/false);
          NumOfElements =
              NumOfElements ? CGF.Builder.CreateNUWMul(NumOfElements, Sz) : Sz;
        }
      } else {
        NumAffinities += C->varlist_size();
      }
    }
    getKmpAffinityType(CGM.getContext(), KmpTaskAffinityInfoTy);
    // Fields ids in kmp_task_affinity_info record.
    enum RTLAffinityInfoFieldsTy { BaseAddr, Len, Flags };

    QualType KmpTaskAffinityInfoArrayTy;
    if (NumOfElements) {
      NumOfElements = CGF.Builder.CreateNUWAdd(
          llvm::ConstantInt::get(CGF.SizeTy, NumAffinities), NumOfElements);
      auto *OVE = new (C) OpaqueValueExpr(
          Loc,
          C.getIntTypeForBitwidth(C.getTypeSize(C.getSizeType()), /*Signed=*/0),
          VK_PRValue);
      CodeGenFunction::OpaqueValueMapping OpaqueMap(CGF, OVE,
                                                    RValue::get(NumOfElements));
      KmpTaskAffinityInfoArrayTy =
          C.getVariableArrayType(KmpTaskAffinityInfoTy, OVE, ArrayType::Normal,
                                 /*IndexTypeQuals=*/0, SourceRange(Loc, Loc));
      // Properly emit variable-sized array.
      auto *PD = ImplicitParamDecl::Create(C, KmpTaskAffinityInfoArrayTy,
                                           ImplicitParamDecl::Other);
      CGF.EmitVarDecl(*PD);
      AffinitiesArray = CGF.GetAddrOfLocalVar(PD);
      NumOfElements = CGF.Builder.CreateIntCast(NumOfElements, CGF.Int32Ty,
                                                /*isSigned=*/false);
    } else {
      KmpTaskAffinityInfoArrayTy = C.getConstantArrayType(
          KmpTaskAffinityInfoTy,
          llvm::APInt(C.getTypeSize(C.getSizeType()), NumAffinities), nullptr,
          ArrayType::Normal, /*IndexTypeQuals=*/0);
      AffinitiesArray =
          CGF.CreateMemTemp(KmpTaskAffinityInfoArrayTy, ".affs.arr.addr");
      AffinitiesArray = CGF.Builder.CreateConstArrayGEP(AffinitiesArray, 0);
      NumOfElements = llvm::ConstantInt::get(CGM.Int32Ty, NumAffinities,
                                             /*isSigned=*/false);
    }

    const auto *KmpAffinityInfoRD = KmpTaskAffinityInfoTy->getAsRecordDecl();
    // Fill array by elements without iterators.
    unsigned Pos = 0;
    bool HasIterator = false;
    for (const auto *C : D.getClausesOfKind<OMPAffinityClause>()) {
      if (C->getModifier()) {
        HasIterator = true;
        continue;
      }
      for (const Expr *E : C->varlists()) {
        llvm::Value *Addr;
        llvm::Value *Size;
        std::tie(Addr, Size) = getPointerAndSize(CGF, E);
        LValue Base =
            CGF.MakeAddrLValue(CGF.Builder.CreateConstGEP(AffinitiesArray, Pos),
                               KmpTaskAffinityInfoTy);
        // affs[i].base_addr = &<Affinities[i].second>;
        LValue BaseAddrLVal = CGF.EmitLValueForField(
            Base, *std::next(KmpAffinityInfoRD->field_begin(), BaseAddr));
        CGF.EmitStoreOfScalar(CGF.Builder.CreatePtrToInt(Addr, CGF.IntPtrTy),
                              BaseAddrLVal);
        // affs[i].len = sizeof(<Affinities[i].second>);
        LValue LenLVal = CGF.EmitLValueForField(
            Base, *std::next(KmpAffinityInfoRD->field_begin(), Len));
        CGF.EmitStoreOfScalar(Size, LenLVal);
        ++Pos;
      }
    }
    LValue PosLVal;
    if (HasIterator) {
      PosLVal = CGF.MakeAddrLValue(
          CGF.CreateMemTemp(C.getSizeType(), "affs.counter.addr"),
          C.getSizeType());
      CGF.EmitStoreOfScalar(llvm::ConstantInt::get(CGF.SizeTy, Pos), PosLVal);
    }
    // Process elements with iterators.
    for (const auto *C : D.getClausesOfKind<OMPAffinityClause>()) {
      const Expr *Modifier = C->getModifier();
      if (!Modifier)
        continue;
      OMPIteratorGeneratorScope IteratorScope(
          CGF, cast_or_null<OMPIteratorExpr>(Modifier->IgnoreParenImpCasts()));
      for (const Expr *E : C->varlists()) {
        llvm::Value *Addr;
        llvm::Value *Size;
        std::tie(Addr, Size) = getPointerAndSize(CGF, E);
        llvm::Value *Idx = CGF.EmitLoadOfScalar(PosLVal, E->getExprLoc());
        LValue Base = CGF.MakeAddrLValue(
            CGF.Builder.CreateGEP(AffinitiesArray, Idx), KmpTaskAffinityInfoTy);
        // affs[i].base_addr = &<Affinities[i].second>;
        LValue BaseAddrLVal = CGF.EmitLValueForField(
            Base, *std::next(KmpAffinityInfoRD->field_begin(), BaseAddr));
        CGF.EmitStoreOfScalar(CGF.Builder.CreatePtrToInt(Addr, CGF.IntPtrTy),
                              BaseAddrLVal);
        // affs[i].len = sizeof(<Affinities[i].second>);
        LValue LenLVal = CGF.EmitLValueForField(
            Base, *std::next(KmpAffinityInfoRD->field_begin(), Len));
        CGF.EmitStoreOfScalar(Size, LenLVal);
        Idx = CGF.Builder.CreateNUWAdd(
            Idx, llvm::ConstantInt::get(Idx->getType(), 1));
        CGF.EmitStoreOfScalar(Idx, PosLVal);
      }
    }
    // Call to kmp_int32 __kmpc_omp_reg_task_with_affinity(ident_t *loc_ref,
    // kmp_int32 gtid, kmp_task_t *new_task, kmp_int32
    // naffins, kmp_task_affinity_info_t *affin_list);
    llvm::Value *LocRef = emitUpdateLocation(CGF, Loc);
    llvm::Value *GTid = getThreadID(CGF, Loc);
    llvm::Value *AffinListPtr = CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
        AffinitiesArray.getPointer(), CGM.VoidPtrTy);
    // FIXME: Emit the function and ignore its result for now unless the
    // runtime function is properly implemented.
    (void)CGF.EmitRuntimeCall(
        OMPBuilder.getOrCreateRuntimeFunction(
            CGM.getModule(), OMPRTL___kmpc_omp_reg_task_with_affinity),
        {LocRef, GTid, NewTask, NumOfElements, AffinListPtr});
  }
  llvm::Value *NewTaskNewTaskTTy =
      CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
          NewTask, KmpTaskTWithPrivatesPtrTy);
  LValue Base = CGF.MakeNaturalAlignAddrLValue(NewTaskNewTaskTTy,
                                               KmpTaskTWithPrivatesQTy);
  LValue TDBase =
      CGF.EmitLValueForField(Base, *KmpTaskTWithPrivatesQTyRD->field_begin());
  // Fill the data in the resulting kmp_task_t record.
  // Copy shareds if there are any.
  Address KmpTaskSharedsPtr = Address::invalid();
  if (!SharedsTy->getAsStructureType()->getDecl()->field_empty()) {
    KmpTaskSharedsPtr =
        Address(CGF.EmitLoadOfScalar(
                    CGF.EmitLValueForField(
                        TDBase, *std::next(KmpTaskTQTyRD->field_begin(),
                                           KmpTaskTShareds)),
                    Loc),
                CGM.getNaturalTypeAlignment(SharedsTy));
    LValue Dest = CGF.MakeAddrLValue(KmpTaskSharedsPtr, SharedsTy);
    LValue Src = CGF.MakeAddrLValue(Shareds, SharedsTy);
    CGF.EmitAggregateCopy(Dest, Src, SharedsTy, AggValueSlot::DoesNotOverlap);
  }
  // Emit initial values for private copies (if any).
  TaskResultTy Result;
  if (!Privates.empty()) {
    emitPrivatesInit(CGF, D, KmpTaskSharedsPtr, Base, KmpTaskTWithPrivatesQTyRD,
                     SharedsTy, SharedsPtrTy, Data, Privates,
                     /*ForDup=*/false);
    if (isOpenMPTaskLoopDirective(D.getDirectiveKind()) &&
        (!Data.LastprivateVars.empty() || checkInitIsRequired(CGF, Privates))) {
      Result.TaskDupFn = emitTaskDupFunction(
          CGM, Loc, D, KmpTaskTWithPrivatesPtrQTy, KmpTaskTWithPrivatesQTyRD,
          KmpTaskTQTyRD, SharedsTy, SharedsPtrTy, Data, Privates,
          /*WithLastIter=*/!Data.LastprivateVars.empty());
    }
  }
  // Fields of union "kmp_cmplrdata_t" for destructors and priority.
  enum { Priority = 0, Destructors = 1 };
  // Provide pointer to function with destructors for privates.
  auto FI = std::next(KmpTaskTQTyRD->field_begin(), Data1);
  const RecordDecl *KmpCmplrdataUD =
      (*FI)->getType()->getAsUnionType()->getDecl();
  if (NeedsCleanup) {
    llvm::Value *DestructorFn = emitDestructorsFunction(
        CGM, Loc, KmpInt32Ty, KmpTaskTWithPrivatesPtrQTy,
        KmpTaskTWithPrivatesQTy);
    LValue Data1LV = CGF.EmitLValueForField(TDBase, *FI);
    LValue DestructorsLV = CGF.EmitLValueForField(
        Data1LV, *std::next(KmpCmplrdataUD->field_begin(), Destructors));
    CGF.EmitStoreOfScalar(CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
                              DestructorFn, KmpRoutineEntryPtrTy),
                          DestructorsLV);
  }
  // Set priority.
  if (Data.Priority.getInt()) {
    LValue Data2LV = CGF.EmitLValueForField(
        TDBase, *std::next(KmpTaskTQTyRD->field_begin(), Data2));
    LValue PriorityLV = CGF.EmitLValueForField(
        Data2LV, *std::next(KmpCmplrdataUD->field_begin(), Priority));
    CGF.EmitStoreOfScalar(Data.Priority.getPointer(), PriorityLV);
  }
  Result.NewTask = NewTask;
  Result.TaskEntry = TaskEntry;
  Result.NewTaskNewTaskTTy = NewTaskNewTaskTTy;
  Result.TDBase = TDBase;
  Result.KmpTaskTQTyRD = KmpTaskTQTyRD;
  return Result;
}

namespace {
/// Dependence kind for RTL.
enum RTLDependenceKindTy {
  DepIn = 0x01,
  DepInOut = 0x3,
  DepMutexInOutSet = 0x4
};
/// Fields ids in kmp_depend_info record.
enum RTLDependInfoFieldsTy { BaseAddr, Len, Flags };
} // namespace

/// Translates internal dependency kind into the runtime kind.
static RTLDependenceKindTy translateDependencyKind(OpenMPDependClauseKind K) {
  RTLDependenceKindTy DepKind;
  switch (K) {
  case OMPC_DEPEND_in:
    DepKind = DepIn;
    break;
  // Out and InOut dependencies must use the same code.
  case OMPC_DEPEND_out:
  case OMPC_DEPEND_inout:
    DepKind = DepInOut;
    break;
  case OMPC_DEPEND_mutexinoutset:
    DepKind = DepMutexInOutSet;
    break;
  case OMPC_DEPEND_source:
  case OMPC_DEPEND_sink:
  case OMPC_DEPEND_depobj:
  case OMPC_DEPEND_unknown:
    llvm_unreachable("Unknown task dependence type");
  }
  return DepKind;
}

/// Builds kmp_depend_info, if it is not built yet, and builds flags type.
static void getDependTypes(ASTContext &C, QualType &KmpDependInfoTy,
                           QualType &FlagsTy) {
  FlagsTy = C.getIntTypeForBitwidth(C.getTypeSize(C.BoolTy), /*Signed=*/false);
  if (KmpDependInfoTy.isNull()) {
    RecordDecl *KmpDependInfoRD = C.buildImplicitRecord("kmp_depend_info");
    KmpDependInfoRD->startDefinition();
    addFieldToRecordDecl(C, KmpDependInfoRD, C.getIntPtrType());
    addFieldToRecordDecl(C, KmpDependInfoRD, C.getSizeType());
    addFieldToRecordDecl(C, KmpDependInfoRD, FlagsTy);
    KmpDependInfoRD->completeDefinition();
    KmpDependInfoTy = C.getRecordType(KmpDependInfoRD);
  }
}

std::pair<llvm::Value *, LValue>
CGOpenMPRuntime::getDepobjElements(CodeGenFunction &CGF, LValue DepobjLVal,
                                   SourceLocation Loc) {
  ASTContext &C = CGM.getContext();
  QualType FlagsTy;
  getDependTypes(C, KmpDependInfoTy, FlagsTy);
  RecordDecl *KmpDependInfoRD =
      cast<RecordDecl>(KmpDependInfoTy->getAsTagDecl());
  LValue Base = CGF.EmitLoadOfPointerLValue(
      DepobjLVal.getAddress(CGF),
      C.getPointerType(C.VoidPtrTy).castAs<PointerType>());
  QualType KmpDependInfoPtrTy = C.getPointerType(KmpDependInfoTy);
  Address Addr = CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
          Base.getAddress(CGF), CGF.ConvertTypeForMem(KmpDependInfoPtrTy));
  Base = CGF.MakeAddrLValue(Addr, KmpDependInfoTy, Base.getBaseInfo(),
                            Base.getTBAAInfo());
  Address DepObjAddr = CGF.Builder.CreateGEP(
      Addr, llvm::ConstantInt::get(CGF.IntPtrTy, -1, /*isSigned=*/true));
  LValue NumDepsBase = CGF.MakeAddrLValue(
      DepObjAddr, KmpDependInfoTy, Base.getBaseInfo(), Base.getTBAAInfo());
  // NumDeps = deps[i].base_addr;
  LValue BaseAddrLVal = CGF.EmitLValueForField(
      NumDepsBase, *std::next(KmpDependInfoRD->field_begin(), BaseAddr));
  llvm::Value *NumDeps = CGF.EmitLoadOfScalar(BaseAddrLVal, Loc);
  return std::make_pair(NumDeps, Base);
}

static void emitDependData(CodeGenFunction &CGF, QualType &KmpDependInfoTy,
                           llvm::PointerUnion<unsigned *, LValue *> Pos,
                           const OMPTaskDataTy::DependData &Data,
                           Address DependenciesArray) {
  CodeGenModule &CGM = CGF.CGM;
  ASTContext &C = CGM.getContext();
  QualType FlagsTy;
  getDependTypes(C, KmpDependInfoTy, FlagsTy);
  RecordDecl *KmpDependInfoRD =
      cast<RecordDecl>(KmpDependInfoTy->getAsTagDecl());
  llvm::Type *LLVMFlagsTy = CGF.ConvertTypeForMem(FlagsTy);

  OMPIteratorGeneratorScope IteratorScope(
      CGF, cast_or_null<OMPIteratorExpr>(
               Data.IteratorExpr ? Data.IteratorExpr->IgnoreParenImpCasts()
                                 : nullptr));
  for (const Expr *E : Data.DepExprs) {
    llvm::Value *Addr;
    llvm::Value *Size;
    std::tie(Addr, Size) = getPointerAndSize(CGF, E);
    LValue Base;
    if (unsigned *P = Pos.dyn_cast<unsigned *>()) {
      Base = CGF.MakeAddrLValue(
          CGF.Builder.CreateConstGEP(DependenciesArray, *P), KmpDependInfoTy);
    } else {
      LValue &PosLVal = *Pos.get<LValue *>();
      llvm::Value *Idx = CGF.EmitLoadOfScalar(PosLVal, E->getExprLoc());
      Base = CGF.MakeAddrLValue(
          CGF.Builder.CreateGEP(DependenciesArray, Idx), KmpDependInfoTy);
    }
    // deps[i].base_addr = &<Dependencies[i].second>;
    LValue BaseAddrLVal = CGF.EmitLValueForField(
        Base, *std::next(KmpDependInfoRD->field_begin(), BaseAddr));
    CGF.EmitStoreOfScalar(CGF.Builder.CreatePtrToInt(Addr, CGF.IntPtrTy),
                          BaseAddrLVal);
    // deps[i].len = sizeof(<Dependencies[i].second>);
    LValue LenLVal = CGF.EmitLValueForField(
        Base, *std::next(KmpDependInfoRD->field_begin(), Len));
    CGF.EmitStoreOfScalar(Size, LenLVal);
    // deps[i].flags = <Dependencies[i].first>;
    RTLDependenceKindTy DepKind = translateDependencyKind(Data.DepKind);
    LValue FlagsLVal = CGF.EmitLValueForField(
        Base, *std::next(KmpDependInfoRD->field_begin(), Flags));
    CGF.EmitStoreOfScalar(llvm::ConstantInt::get(LLVMFlagsTy, DepKind),
                          FlagsLVal);
    if (unsigned *P = Pos.dyn_cast<unsigned *>()) {
      ++(*P);
    } else {
      LValue &PosLVal = *Pos.get<LValue *>();
      llvm::Value *Idx = CGF.EmitLoadOfScalar(PosLVal, E->getExprLoc());
      Idx = CGF.Builder.CreateNUWAdd(Idx,
                                     llvm::ConstantInt::get(Idx->getType(), 1));
      CGF.EmitStoreOfScalar(Idx, PosLVal);
    }
  }
}

static SmallVector<llvm::Value *, 4>
emitDepobjElementsSizes(CodeGenFunction &CGF, QualType &KmpDependInfoTy,
                        const OMPTaskDataTy::DependData &Data) {
  assert(Data.DepKind == OMPC_DEPEND_depobj &&
         "Expected depobj dependecy kind.");
  SmallVector<llvm::Value *, 4> Sizes;
  SmallVector<LValue, 4> SizeLVals;
  ASTContext &C = CGF.getContext();
  QualType FlagsTy;
  getDependTypes(C, KmpDependInfoTy, FlagsTy);
  RecordDecl *KmpDependInfoRD =
      cast<RecordDecl>(KmpDependInfoTy->getAsTagDecl());
  QualType KmpDependInfoPtrTy = C.getPointerType(KmpDependInfoTy);
  llvm::Type *KmpDependInfoPtrT = CGF.ConvertTypeForMem(KmpDependInfoPtrTy);
  {
    OMPIteratorGeneratorScope IteratorScope(
        CGF, cast_or_null<OMPIteratorExpr>(
                 Data.IteratorExpr ? Data.IteratorExpr->IgnoreParenImpCasts()
                                   : nullptr));
    for (const Expr *E : Data.DepExprs) {
      LValue DepobjLVal = CGF.EmitLValue(E->IgnoreParenImpCasts());
      LValue Base = CGF.EmitLoadOfPointerLValue(
          DepobjLVal.getAddress(CGF),
          C.getPointerType(C.VoidPtrTy).castAs<PointerType>());
      Address Addr = CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
          Base.getAddress(CGF), KmpDependInfoPtrT);
      Base = CGF.MakeAddrLValue(Addr, KmpDependInfoTy, Base.getBaseInfo(),
                                Base.getTBAAInfo());
      Address DepObjAddr = CGF.Builder.CreateGEP(
          Addr, llvm::ConstantInt::get(CGF.IntPtrTy, -1, /*isSigned=*/true));
      LValue NumDepsBase = CGF.MakeAddrLValue(
          DepObjAddr, KmpDependInfoTy, Base.getBaseInfo(), Base.getTBAAInfo());
      // NumDeps = deps[i].base_addr;
      LValue BaseAddrLVal = CGF.EmitLValueForField(
          NumDepsBase, *std::next(KmpDependInfoRD->field_begin(), BaseAddr));
      llvm::Value *NumDeps =
          CGF.EmitLoadOfScalar(BaseAddrLVal, E->getExprLoc());
      LValue NumLVal = CGF.MakeAddrLValue(
          CGF.CreateMemTemp(C.getUIntPtrType(), "depobj.size.addr"),
          C.getUIntPtrType());
      CGF.Builder.CreateStore(llvm::ConstantInt::get(CGF.IntPtrTy, 0),
                              NumLVal.getAddress(CGF));
      llvm::Value *PrevVal = CGF.EmitLoadOfScalar(NumLVal, E->getExprLoc());
      llvm::Value *Add = CGF.Builder.CreateNUWAdd(PrevVal, NumDeps);
      CGF.EmitStoreOfScalar(Add, NumLVal);
      SizeLVals.push_back(NumLVal);
    }
  }
  for (unsigned I = 0, E = SizeLVals.size(); I < E; ++I) {
    llvm::Value *Size =
        CGF.EmitLoadOfScalar(SizeLVals[I], Data.DepExprs[I]->getExprLoc());
    Sizes.push_back(Size);
  }
  return Sizes;
}

static void emitDepobjElements(CodeGenFunction &CGF, QualType &KmpDependInfoTy,
                               LValue PosLVal,
                               const OMPTaskDataTy::DependData &Data,
                               Address DependenciesArray) {
  assert(Data.DepKind == OMPC_DEPEND_depobj &&
         "Expected depobj dependecy kind.");
  ASTContext &C = CGF.getContext();
  QualType FlagsTy;
  getDependTypes(C, KmpDependInfoTy, FlagsTy);
  RecordDecl *KmpDependInfoRD =
      cast<RecordDecl>(KmpDependInfoTy->getAsTagDecl());
  QualType KmpDependInfoPtrTy = C.getPointerType(KmpDependInfoTy);
  llvm::Type *KmpDependInfoPtrT = CGF.ConvertTypeForMem(KmpDependInfoPtrTy);
  llvm::Value *ElSize = CGF.getTypeSize(KmpDependInfoTy);
  {
    OMPIteratorGeneratorScope IteratorScope(
        CGF, cast_or_null<OMPIteratorExpr>(
                 Data.IteratorExpr ? Data.IteratorExpr->IgnoreParenImpCasts()
                                   : nullptr));
    for (unsigned I = 0, End = Data.DepExprs.size(); I < End; ++I) {
      const Expr *E = Data.DepExprs[I];
      LValue DepobjLVal = CGF.EmitLValue(E->IgnoreParenImpCasts());
      LValue Base = CGF.EmitLoadOfPointerLValue(
          DepobjLVal.getAddress(CGF),
          C.getPointerType(C.VoidPtrTy).castAs<PointerType>());
      Address Addr = CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
          Base.getAddress(CGF), KmpDependInfoPtrT);
      Base = CGF.MakeAddrLValue(Addr, KmpDependInfoTy, Base.getBaseInfo(),
                                Base.getTBAAInfo());

      // Get number of elements in a single depobj.
      Address DepObjAddr = CGF.Builder.CreateGEP(
          Addr, llvm::ConstantInt::get(CGF.IntPtrTy, -1, /*isSigned=*/true));
      LValue NumDepsBase = CGF.MakeAddrLValue(
          DepObjAddr, KmpDependInfoTy, Base.getBaseInfo(), Base.getTBAAInfo());
      // NumDeps = deps[i].base_addr;
      LValue BaseAddrLVal = CGF.EmitLValueForField(
          NumDepsBase, *std::next(KmpDependInfoRD->field_begin(), BaseAddr));
      llvm::Value *NumDeps =
          CGF.EmitLoadOfScalar(BaseAddrLVal, E->getExprLoc());

      // memcopy dependency data.
      llvm::Value *Size = CGF.Builder.CreateNUWMul(
          ElSize,
          CGF.Builder.CreateIntCast(NumDeps, CGF.SizeTy, /*isSigned=*/false));
      llvm::Value *Pos = CGF.EmitLoadOfScalar(PosLVal, E->getExprLoc());
      Address DepAddr = CGF.Builder.CreateGEP(DependenciesArray, Pos);
      CGF.Builder.CreateMemCpy(DepAddr, Base.getAddress(CGF), Size);

      // Increase pos.
      // pos += size;
      llvm::Value *Add = CGF.Builder.CreateNUWAdd(Pos, NumDeps);
      CGF.EmitStoreOfScalar(Add, PosLVal);
    }
  }
}

std::pair<llvm::Value *, Address> CGOpenMPRuntime::emitDependClause(
    CodeGenFunction &CGF, ArrayRef<OMPTaskDataTy::DependData> Dependencies,
    SourceLocation Loc) {
  if (llvm::all_of(Dependencies, [](const OMPTaskDataTy::DependData &D) {
        return D.DepExprs.empty();
      }))
    return std::make_pair(nullptr, Address::invalid());
  // Process list of dependencies.
  ASTContext &C = CGM.getContext();
  Address DependenciesArray = Address::invalid();
  llvm::Value *NumOfElements = nullptr;
  unsigned NumDependencies = std::accumulate(
      Dependencies.begin(), Dependencies.end(), 0,
      [](unsigned V, const OMPTaskDataTy::DependData &D) {
        return D.DepKind == OMPC_DEPEND_depobj
                   ? V
                   : (V + (D.IteratorExpr ? 0 : D.DepExprs.size()));
      });
  QualType FlagsTy;
  getDependTypes(C, KmpDependInfoTy, FlagsTy);
  bool HasDepobjDeps = false;
  bool HasRegularWithIterators = false;
  llvm::Value *NumOfDepobjElements = llvm::ConstantInt::get(CGF.IntPtrTy, 0);
  llvm::Value *NumOfRegularWithIterators =
      llvm::ConstantInt::get(CGF.IntPtrTy, 0);
  // Calculate number of depobj dependecies and regular deps with the iterators.
  for (const OMPTaskDataTy::DependData &D : Dependencies) {
    if (D.DepKind == OMPC_DEPEND_depobj) {
      SmallVector<llvm::Value *, 4> Sizes =
          emitDepobjElementsSizes(CGF, KmpDependInfoTy, D);
      for (llvm::Value *Size : Sizes) {
        NumOfDepobjElements =
            CGF.Builder.CreateNUWAdd(NumOfDepobjElements, Size);
      }
      HasDepobjDeps = true;
      continue;
    }
    // Include number of iterations, if any.

    if (const auto *IE = cast_or_null<OMPIteratorExpr>(D.IteratorExpr)) {
      for (unsigned I = 0, E = IE->numOfIterators(); I < E; ++I) {
        llvm::Value *Sz = CGF.EmitScalarExpr(IE->getHelper(I).Upper);
        Sz = CGF.Builder.CreateIntCast(Sz, CGF.IntPtrTy, /*isSigned=*/false);
        llvm::Value *NumClauseDeps = CGF.Builder.CreateNUWMul(
            Sz, llvm::ConstantInt::get(CGF.IntPtrTy, D.DepExprs.size()));
        NumOfRegularWithIterators =
            CGF.Builder.CreateNUWAdd(NumOfRegularWithIterators, NumClauseDeps);
      }
      HasRegularWithIterators = true;
      continue;
    }
  }

  QualType KmpDependInfoArrayTy;
  if (HasDepobjDeps || HasRegularWithIterators) {
    NumOfElements = llvm::ConstantInt::get(CGM.IntPtrTy, NumDependencies,
                                           /*isSigned=*/false);
    if (HasDepobjDeps) {
      NumOfElements =
          CGF.Builder.CreateNUWAdd(NumOfDepobjElements, NumOfElements);
    }
    if (HasRegularWithIterators) {
      NumOfElements =
          CGF.Builder.CreateNUWAdd(NumOfRegularWithIterators, NumOfElements);
    }
    auto *OVE = new (C) OpaqueValueExpr(
        Loc, C.getIntTypeForBitwidth(/*DestWidth=*/64, /*Signed=*/0),
        VK_PRValue);
    CodeGenFunction::OpaqueValueMapping OpaqueMap(CGF, OVE,
                                                  RValue::get(NumOfElements));
    KmpDependInfoArrayTy =
        C.getVariableArrayType(KmpDependInfoTy, OVE, ArrayType::Normal,
                               /*IndexTypeQuals=*/0, SourceRange(Loc, Loc));
    // CGF.EmitVariablyModifiedType(KmpDependInfoArrayTy);
    // Properly emit variable-sized array.
    auto *PD = ImplicitParamDecl::Create(C, KmpDependInfoArrayTy,
                                         ImplicitParamDecl::Other);
    CGF.EmitVarDecl(*PD);
    DependenciesArray = CGF.GetAddrOfLocalVar(PD);
    NumOfElements = CGF.Builder.CreateIntCast(NumOfElements, CGF.Int32Ty,
                                              /*isSigned=*/false);
  } else {
    KmpDependInfoArrayTy = C.getConstantArrayType(
        KmpDependInfoTy, llvm::APInt(/*numBits=*/64, NumDependencies), nullptr,
        ArrayType::Normal, /*IndexTypeQuals=*/0);
    DependenciesArray =
        CGF.CreateMemTemp(KmpDependInfoArrayTy, ".dep.arr.addr");
    DependenciesArray = CGF.Builder.CreateConstArrayGEP(DependenciesArray, 0);
    NumOfElements = llvm::ConstantInt::get(CGM.Int32Ty, NumDependencies,
                                           /*isSigned=*/false);
  }
  unsigned Pos = 0;
  for (unsigned I = 0, End = Dependencies.size(); I < End; ++I) {
    if (Dependencies[I].DepKind == OMPC_DEPEND_depobj ||
        Dependencies[I].IteratorExpr)
      continue;
    emitDependData(CGF, KmpDependInfoTy, &Pos, Dependencies[I],
                   DependenciesArray);
  }
  // Copy regular dependecies with iterators.
  LValue PosLVal = CGF.MakeAddrLValue(
      CGF.CreateMemTemp(C.getSizeType(), "dep.counter.addr"), C.getSizeType());
  CGF.EmitStoreOfScalar(llvm::ConstantInt::get(CGF.SizeTy, Pos), PosLVal);
  for (unsigned I = 0, End = Dependencies.size(); I < End; ++I) {
    if (Dependencies[I].DepKind == OMPC_DEPEND_depobj ||
        !Dependencies[I].IteratorExpr)
      continue;
    emitDependData(CGF, KmpDependInfoTy, &PosLVal, Dependencies[I],
                   DependenciesArray);
  }
  // Copy final depobj arrays without iterators.
  if (HasDepobjDeps) {
    for (unsigned I = 0, End = Dependencies.size(); I < End; ++I) {
      if (Dependencies[I].DepKind != OMPC_DEPEND_depobj)
        continue;
      emitDepobjElements(CGF, KmpDependInfoTy, PosLVal, Dependencies[I],
                         DependenciesArray);
    }
  }
  DependenciesArray = CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
      DependenciesArray, CGF.VoidPtrTy);
  return std::make_pair(NumOfElements, DependenciesArray);
}

Address CGOpenMPRuntime::emitDepobjDependClause(
    CodeGenFunction &CGF, const OMPTaskDataTy::DependData &Dependencies,
    SourceLocation Loc) {
  if (Dependencies.DepExprs.empty())
    return Address::invalid();
  // Process list of dependencies.
  ASTContext &C = CGM.getContext();
  Address DependenciesArray = Address::invalid();
  unsigned NumDependencies = Dependencies.DepExprs.size();
  QualType FlagsTy;
  getDependTypes(C, KmpDependInfoTy, FlagsTy);
  RecordDecl *KmpDependInfoRD =
      cast<RecordDecl>(KmpDependInfoTy->getAsTagDecl());

  llvm::Value *Size;
  // Define type kmp_depend_info[<Dependencies.size()>];
  // For depobj reserve one extra element to store the number of elements.
  // It is required to handle depobj(x) update(in) construct.
  // kmp_depend_info[<Dependencies.size()>] deps;
  llvm::Value *NumDepsVal;
  CharUnits Align = C.getTypeAlignInChars(KmpDependInfoTy);
  if (const auto *IE =
          cast_or_null<OMPIteratorExpr>(Dependencies.IteratorExpr)) {
    NumDepsVal = llvm::ConstantInt::get(CGF.SizeTy, 1);
    for (unsigned I = 0, E = IE->numOfIterators(); I < E; ++I) {
      llvm::Value *Sz = CGF.EmitScalarExpr(IE->getHelper(I).Upper);
      Sz = CGF.Builder.CreateIntCast(Sz, CGF.SizeTy, /*isSigned=*/false);
      NumDepsVal = CGF.Builder.CreateNUWMul(NumDepsVal, Sz);
    }
    Size = CGF.Builder.CreateNUWAdd(llvm::ConstantInt::get(CGF.SizeTy, 1),
                                    NumDepsVal);
    CharUnits SizeInBytes =
        C.getTypeSizeInChars(KmpDependInfoTy).alignTo(Align);
    llvm::Value *RecSize = CGM.getSize(SizeInBytes);
    Size = CGF.Builder.CreateNUWMul(Size, RecSize);
    NumDepsVal =
        CGF.Builder.CreateIntCast(NumDepsVal, CGF.IntPtrTy, /*isSigned=*/false);
  } else {
    QualType KmpDependInfoArrayTy = C.getConstantArrayType(
        KmpDependInfoTy, llvm::APInt(/*numBits=*/64, NumDependencies + 1),
        nullptr, ArrayType::Normal, /*IndexTypeQuals=*/0);
    CharUnits Sz = C.getTypeSizeInChars(KmpDependInfoArrayTy);
    Size = CGM.getSize(Sz.alignTo(Align));
    NumDepsVal = llvm::ConstantInt::get(CGF.IntPtrTy, NumDependencies);
  }
  // Need to allocate on the dynamic memory.
  llvm::Value *ThreadID = getThreadID(CGF, Loc);
  // Use default allocator.
  llvm::Value *Allocator = llvm::ConstantPointerNull::get(CGF.VoidPtrTy);
  llvm::Value *Args[] = {ThreadID, Size, Allocator};

  llvm::Value *Addr =
      CGF.EmitRuntimeCall(OMPBuilder.getOrCreateRuntimeFunction(
                              CGM.getModule(), OMPRTL___kmpc_alloc),
                          Args, ".dep.arr.addr");
  Addr = CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
      Addr, CGF.ConvertTypeForMem(KmpDependInfoTy)->getPointerTo());
  DependenciesArray = Address(Addr, Align);
  // Write number of elements in the first element of array for depobj.
  LValue Base = CGF.MakeAddrLValue(DependenciesArray, KmpDependInfoTy);
  // deps[i].base_addr = NumDependencies;
  LValue BaseAddrLVal = CGF.EmitLValueForField(
      Base, *std::next(KmpDependInfoRD->field_begin(), BaseAddr));
  CGF.EmitStoreOfScalar(NumDepsVal, BaseAddrLVal);
  llvm::PointerUnion<unsigned *, LValue *> Pos;
  unsigned Idx = 1;
  LValue PosLVal;
  if (Dependencies.IteratorExpr) {
    PosLVal = CGF.MakeAddrLValue(
        CGF.CreateMemTemp(C.getSizeType(), "iterator.counter.addr"),
        C.getSizeType());
    CGF.EmitStoreOfScalar(llvm::ConstantInt::get(CGF.SizeTy, Idx), PosLVal,
                          /*IsInit=*/true);
    Pos = &PosLVal;
  } else {
    Pos = &Idx;
  }
  emitDependData(CGF, KmpDependInfoTy, Pos, Dependencies, DependenciesArray);
  DependenciesArray = CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
      CGF.Builder.CreateConstGEP(DependenciesArray, 1), CGF.VoidPtrTy);
  return DependenciesArray;
}

void CGOpenMPRuntime::emitDestroyClause(CodeGenFunction &CGF, LValue DepobjLVal,
                                        SourceLocation Loc) {
  ASTContext &C = CGM.getContext();
  QualType FlagsTy;
  getDependTypes(C, KmpDependInfoTy, FlagsTy);
  LValue Base = CGF.EmitLoadOfPointerLValue(
      DepobjLVal.getAddress(CGF),
      C.getPointerType(C.VoidPtrTy).castAs<PointerType>());
  QualType KmpDependInfoPtrTy = C.getPointerType(KmpDependInfoTy);
  Address Addr = CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
      Base.getAddress(CGF), CGF.ConvertTypeForMem(KmpDependInfoPtrTy));
  llvm::Value *DepObjAddr = CGF.Builder.CreateGEP(
      Addr.getElementType(), Addr.getPointer(),
      llvm::ConstantInt::get(CGF.IntPtrTy, -1, /*isSigned=*/true));
  DepObjAddr = CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(DepObjAddr,
                                                               CGF.VoidPtrTy);
  llvm::Value *ThreadID = getThreadID(CGF, Loc);
  // Use default allocator.
  llvm::Value *Allocator = llvm::ConstantPointerNull::get(CGF.VoidPtrTy);
  llvm::Value *Args[] = {ThreadID, DepObjAddr, Allocator};

  // _kmpc_free(gtid, addr, nullptr);
  (void)CGF.EmitRuntimeCall(OMPBuilder.getOrCreateRuntimeFunction(
                                CGM.getModule(), OMPRTL___kmpc_free),
                            Args);
}

void CGOpenMPRuntime::emitUpdateClause(CodeGenFunction &CGF, LValue DepobjLVal,
                                       OpenMPDependClauseKind NewDepKind,
                                       SourceLocation Loc) {
  ASTContext &C = CGM.getContext();
  QualType FlagsTy;
  getDependTypes(C, KmpDependInfoTy, FlagsTy);
  RecordDecl *KmpDependInfoRD =
      cast<RecordDecl>(KmpDependInfoTy->getAsTagDecl());
  llvm::Type *LLVMFlagsTy = CGF.ConvertTypeForMem(FlagsTy);
  llvm::Value *NumDeps;
  LValue Base;
  std::tie(NumDeps, Base) = getDepobjElements(CGF, DepobjLVal, Loc);

  Address Begin = Base.getAddress(CGF);
  // Cast from pointer to array type to pointer to single element.
  llvm::Value *End = CGF.Builder.CreateGEP(
      Begin.getElementType(), Begin.getPointer(), NumDeps);
  // The basic structure here is a while-do loop.
  llvm::BasicBlock *BodyBB = CGF.createBasicBlock("omp.body");
  llvm::BasicBlock *DoneBB = CGF.createBasicBlock("omp.done");
  llvm::BasicBlock *EntryBB = CGF.Builder.GetInsertBlock();
  CGF.EmitBlock(BodyBB);
  llvm::PHINode *ElementPHI =
      CGF.Builder.CreatePHI(Begin.getType(), 2, "omp.elementPast");
  ElementPHI->addIncoming(Begin.getPointer(), EntryBB);
  Begin = Address(ElementPHI, Begin.getAlignment());
  Base = CGF.MakeAddrLValue(Begin, KmpDependInfoTy, Base.getBaseInfo(),
                            Base.getTBAAInfo());
  // deps[i].flags = NewDepKind;
  RTLDependenceKindTy DepKind = translateDependencyKind(NewDepKind);
  LValue FlagsLVal = CGF.EmitLValueForField(
      Base, *std::next(KmpDependInfoRD->field_begin(), Flags));
  CGF.EmitStoreOfScalar(llvm::ConstantInt::get(LLVMFlagsTy, DepKind),
                        FlagsLVal);

  // Shift the address forward by one element.
  Address ElementNext =
      CGF.Builder.CreateConstGEP(Begin, /*Index=*/1, "omp.elementNext");
  ElementPHI->addIncoming(ElementNext.getPointer(),
                          CGF.Builder.GetInsertBlock());
  llvm::Value *IsEmpty =
      CGF.Builder.CreateICmpEQ(ElementNext.getPointer(), End, "omp.isempty");
  CGF.Builder.CreateCondBr(IsEmpty, DoneBB, BodyBB);
  // Done.
  CGF.EmitBlock(DoneBB, /*IsFinished=*/true);
}

void CGOpenMPRuntime::emitTaskCall(CodeGenFunction &CGF, SourceLocation Loc,
                                   const OMPExecutableDirective &D,
                                   llvm::Function *TaskFunction,
                                   QualType SharedsTy, Address Shareds,
                                   const Expr *IfCond,
                                   const OMPTaskDataTy &Data) {
  if (!CGF.HaveInsertPoint())
    return;

  TaskResultTy Result =
      emitTaskInit(CGF, Loc, D, TaskFunction, SharedsTy, Shareds, Data);
  llvm::Value *NewTask = Result.NewTask;
  llvm::Function *TaskEntry = Result.TaskEntry;
  llvm::Value *NewTaskNewTaskTTy = Result.NewTaskNewTaskTTy;
  LValue TDBase = Result.TDBase;
  const RecordDecl *KmpTaskTQTyRD = Result.KmpTaskTQTyRD;
  // Process list of dependences.
  Address DependenciesArray = Address::invalid();
  llvm::Value *NumOfElements;
  std::tie(NumOfElements, DependenciesArray) =
      emitDependClause(CGF, Data.Dependences, Loc);

  // NOTE: routine and part_id fields are initialized by __kmpc_omp_task_alloc()
  // libcall.
  // Build kmp_int32 __kmpc_omp_task_with_deps(ident_t *, kmp_int32 gtid,
  // kmp_task_t *new_task, kmp_int32 ndeps, kmp_depend_info_t *dep_list,
  // kmp_int32 ndeps_noalias, kmp_depend_info_t *noalias_dep_list) if dependence
  // list is not empty
  llvm::Value *ThreadID = getThreadID(CGF, Loc);
  llvm::Value *UpLoc = emitUpdateLocation(CGF, Loc);
  llvm::Value *TaskArgs[] = { UpLoc, ThreadID, NewTask };
  llvm::Value *DepTaskArgs[7];
  if (!Data.Dependences.empty()) {
    DepTaskArgs[0] = UpLoc;
    DepTaskArgs[1] = ThreadID;
    DepTaskArgs[2] = NewTask;
    DepTaskArgs[3] = NumOfElements;
    DepTaskArgs[4] = DependenciesArray.getPointer();
    DepTaskArgs[5] = CGF.Builder.getInt32(0);
    DepTaskArgs[6] = llvm::ConstantPointerNull::get(CGF.VoidPtrTy);
  }
  auto &&ThenCodeGen = [this, &Data, TDBase, KmpTaskTQTyRD, &TaskArgs,
                        &DepTaskArgs](CodeGenFunction &CGF, PrePostActionTy &) {
    if (!Data.Tied) {
      auto PartIdFI = std::next(KmpTaskTQTyRD->field_begin(), KmpTaskTPartId);
      LValue PartIdLVal = CGF.EmitLValueForField(TDBase, *PartIdFI);
      CGF.EmitStoreOfScalar(CGF.Builder.getInt32(0), PartIdLVal);
    }
    if (!Data.Dependences.empty()) {
      CGF.EmitRuntimeCall(
          OMPBuilder.getOrCreateRuntimeFunction(
              CGM.getModule(), OMPRTL___kmpc_omp_task_with_deps),
          DepTaskArgs);
    } else {
      CGF.EmitRuntimeCall(OMPBuilder.getOrCreateRuntimeFunction(
                              CGM.getModule(), OMPRTL___kmpc_omp_task),
                          TaskArgs);
    }
    // Check if parent region is untied and build return for untied task;
    if (auto *Region =
            dyn_cast_or_null<CGOpenMPRegionInfo>(CGF.CapturedStmtInfo))
      Region->emitUntiedSwitch(CGF);
  };

  llvm::Value *DepWaitTaskArgs[6];
  if (!Data.Dependences.empty()) {
    DepWaitTaskArgs[0] = UpLoc;
    DepWaitTaskArgs[1] = ThreadID;
    DepWaitTaskArgs[2] = NumOfElements;
    DepWaitTaskArgs[3] = DependenciesArray.getPointer();
    DepWaitTaskArgs[4] = CGF.Builder.getInt32(0);
    DepWaitTaskArgs[5] = llvm::ConstantPointerNull::get(CGF.VoidPtrTy);
  }
  auto &M = CGM.getModule();
  auto &&ElseCodeGen = [this, &M, &TaskArgs, ThreadID, NewTaskNewTaskTTy,
                        TaskEntry, &Data, &DepWaitTaskArgs,
                        Loc](CodeGenFunction &CGF, PrePostActionTy &) {
    CodeGenFunction::RunCleanupsScope LocalScope(CGF);
    // Build void __kmpc_omp_wait_deps(ident_t *, kmp_int32 gtid,
    // kmp_int32 ndeps, kmp_depend_info_t *dep_list, kmp_int32
    // ndeps_noalias, kmp_depend_info_t *noalias_dep_list); if dependence info
    // is specified.
    if (!Data.Dependences.empty())
      CGF.EmitRuntimeCall(
          OMPBuilder.getOrCreateRuntimeFunction(M, OMPRTL___kmpc_omp_wait_deps),
          DepWaitTaskArgs);
    // Call proxy_task_entry(gtid, new_task);
    auto &&CodeGen = [TaskEntry, ThreadID, NewTaskNewTaskTTy,
                      Loc](CodeGenFunction &CGF, PrePostActionTy &Action) {
      Action.Enter(CGF);
      llvm::Value *OutlinedFnArgs[] = {ThreadID, NewTaskNewTaskTTy};
      CGF.CGM.getOpenMPRuntime().emitOutlinedFunctionCall(CGF, Loc, TaskEntry,
                                                          OutlinedFnArgs);
    };

    // Build void __kmpc_omp_task_begin_if0(ident_t *, kmp_int32 gtid,
    // kmp_task_t *new_task);
    // Build void __kmpc_omp_task_complete_if0(ident_t *, kmp_int32 gtid,
    // kmp_task_t *new_task);
    RegionCodeGenTy RCG(CodeGen);
    CommonActionTy Action(OMPBuilder.getOrCreateRuntimeFunction(
                              M, OMPRTL___kmpc_omp_task_begin_if0),
                          TaskArgs,
                          OMPBuilder.getOrCreateRuntimeFunction(
                              M, OMPRTL___kmpc_omp_task_complete_if0),
                          TaskArgs);
    RCG.setAction(Action);
    RCG(CGF);
  };

  if (IfCond) {
    emitIfClause(CGF, IfCond, ThenCodeGen, ElseCodeGen);
  } else {
    RegionCodeGenTy ThenRCG(ThenCodeGen);
    ThenRCG(CGF);
  }
}

void CGOpenMPRuntime::emitTaskLoopCall(CodeGenFunction &CGF, SourceLocation Loc,
                                       const OMPLoopDirective &D,
                                       llvm::Function *TaskFunction,
                                       QualType SharedsTy, Address Shareds,
                                       const Expr *IfCond,
                                       const OMPTaskDataTy &Data) {
  if (!CGF.HaveInsertPoint())
    return;
  TaskResultTy Result =
      emitTaskInit(CGF, Loc, D, TaskFunction, SharedsTy, Shareds, Data);
  // NOTE: routine and part_id fields are initialized by __kmpc_omp_task_alloc()
  // libcall.
  // Call to void __kmpc_taskloop(ident_t *loc, int gtid, kmp_task_t *task, int
  // if_val, kmp_uint64 *lb, kmp_uint64 *ub, kmp_int64 st, int nogroup, int
  // sched, kmp_uint64 grainsize, void *task_dup);
  llvm::Value *ThreadID = getThreadID(CGF, Loc);
  llvm::Value *UpLoc = emitUpdateLocation(CGF, Loc);
  llvm::Value *IfVal;
  if (IfCond) {
    IfVal = CGF.Builder.CreateIntCast(CGF.EvaluateExprAsBool(IfCond), CGF.IntTy,
                                      /*isSigned=*/true);
  } else {
    IfVal = llvm::ConstantInt::getSigned(CGF.IntTy, /*V=*/1);
  }

  LValue LBLVal = CGF.EmitLValueForField(
      Result.TDBase,
      *std::next(Result.KmpTaskTQTyRD->field_begin(), KmpTaskTLowerBound));
  const auto *LBVar =
      cast<VarDecl>(cast<DeclRefExpr>(D.getLowerBoundVariable())->getDecl());
  CGF.EmitAnyExprToMem(LBVar->getInit(), LBLVal.getAddress(CGF),
                       LBLVal.getQuals(),
                       /*IsInitializer=*/true);
  LValue UBLVal = CGF.EmitLValueForField(
      Result.TDBase,
      *std::next(Result.KmpTaskTQTyRD->field_begin(), KmpTaskTUpperBound));
  const auto *UBVar =
      cast<VarDecl>(cast<DeclRefExpr>(D.getUpperBoundVariable())->getDecl());
  CGF.EmitAnyExprToMem(UBVar->getInit(), UBLVal.getAddress(CGF),
                       UBLVal.getQuals(),
                       /*IsInitializer=*/true);
  LValue StLVal = CGF.EmitLValueForField(
      Result.TDBase,
      *std::next(Result.KmpTaskTQTyRD->field_begin(), KmpTaskTStride));
  const auto *StVar =
      cast<VarDecl>(cast<DeclRefExpr>(D.getStrideVariable())->getDecl());
  CGF.EmitAnyExprToMem(StVar->getInit(), StLVal.getAddress(CGF),
                       StLVal.getQuals(),
                       /*IsInitializer=*/true);
  // Store reductions address.
  LValue RedLVal = CGF.EmitLValueForField(
      Result.TDBase,
      *std::next(Result.KmpTaskTQTyRD->field_begin(), KmpTaskTReductions));
  if (Data.Reductions) {
    CGF.EmitStoreOfScalar(Data.Reductions, RedLVal);
  } else {
    CGF.EmitNullInitialization(RedLVal.getAddress(CGF),
                               CGF.getContext().VoidPtrTy);
  }
  enum { NoSchedule = 0, Grainsize = 1, NumTasks = 2 };
  llvm::Value *TaskArgs[] = {
      UpLoc,
      ThreadID,
      Result.NewTask,
      IfVal,
      LBLVal.getPointer(CGF),
      UBLVal.getPointer(CGF),
      CGF.EmitLoadOfScalar(StLVal, Loc),
      llvm::ConstantInt::getSigned(
          CGF.IntTy, 1), // Always 1 because taskgroup emitted by the compiler
      llvm::ConstantInt::getSigned(
          CGF.IntTy, Data.Schedule.getPointer()
                         ? Data.Schedule.getInt() ? NumTasks : Grainsize
                         : NoSchedule),
      Data.Schedule.getPointer()
          ? CGF.Builder.CreateIntCast(Data.Schedule.getPointer(), CGF.Int64Ty,
                                      /*isSigned=*/false)
          : llvm::ConstantInt::get(CGF.Int64Ty, /*V=*/0),
      Result.TaskDupFn ? CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
                             Result.TaskDupFn, CGF.VoidPtrTy)
                       : llvm::ConstantPointerNull::get(CGF.VoidPtrTy)};
  CGF.EmitRuntimeCall(OMPBuilder.getOrCreateRuntimeFunction(
                          CGM.getModule(), OMPRTL___kmpc_taskloop),
                      TaskArgs);
}

/// Emit reduction operation for each element of array (required for
/// array sections) LHS op = RHS.
/// \param Type Type of array.
/// \param LHSVar Variable on the left side of the reduction operation
/// (references element of array in original variable).
/// \param RHSVar Variable on the right side of the reduction operation
/// (references element of array in original variable).
/// \param RedOpGen Generator of reduction operation with use of LHSVar and
/// RHSVar.
static void EmitOMPAggregateReduction(
    CodeGenFunction &CGF, QualType Type, const VarDecl *LHSVar,
    const VarDecl *RHSVar,
    const llvm::function_ref<void(CodeGenFunction &CGF, const Expr *,
                                  const Expr *, const Expr *)> &RedOpGen,
    const Expr *XExpr = nullptr, const Expr *EExpr = nullptr,
    const Expr *UpExpr = nullptr) {
  // Perform element-by-element initialization.
  QualType ElementTy;
  Address LHSAddr = CGF.GetAddrOfLocalVar(LHSVar);
  Address RHSAddr = CGF.GetAddrOfLocalVar(RHSVar);

  // Drill down to the base element type on both arrays.
  const ArrayType *ArrayTy = Type->getAsArrayTypeUnsafe();
  llvm::Value *NumElements = CGF.emitArrayLength(ArrayTy, ElementTy, LHSAddr);

  llvm::Value *RHSBegin = RHSAddr.getPointer();
  llvm::Value *LHSBegin = LHSAddr.getPointer();
  // Cast from pointer to array type to pointer to single element.
  llvm::Value *LHSEnd =
      CGF.Builder.CreateGEP(LHSAddr.getElementType(), LHSBegin, NumElements);
  // The basic structure here is a while-do loop.
  llvm::BasicBlock *BodyBB = CGF.createBasicBlock("omp.arraycpy.body");
  llvm::BasicBlock *DoneBB = CGF.createBasicBlock("omp.arraycpy.done");
  llvm::Value *IsEmpty =
      CGF.Builder.CreateICmpEQ(LHSBegin, LHSEnd, "omp.arraycpy.isempty");
  CGF.Builder.CreateCondBr(IsEmpty, DoneBB, BodyBB);

  // Enter the loop body, making that address the current address.
  llvm::BasicBlock *EntryBB = CGF.Builder.GetInsertBlock();
  CGF.EmitBlock(BodyBB);

  CharUnits ElementSize = CGF.getContext().getTypeSizeInChars(ElementTy);

  llvm::PHINode *RHSElementPHI = CGF.Builder.CreatePHI(
      RHSBegin->getType(), 2, "omp.arraycpy.srcElementPast");
  RHSElementPHI->addIncoming(RHSBegin, EntryBB);
  Address RHSElementCurrent =
      Address(RHSElementPHI,
              RHSAddr.getAlignment().alignmentOfArrayElement(ElementSize));

  llvm::PHINode *LHSElementPHI = CGF.Builder.CreatePHI(
      LHSBegin->getType(), 2, "omp.arraycpy.destElementPast");
  LHSElementPHI->addIncoming(LHSBegin, EntryBB);
  Address LHSElementCurrent =
      Address(LHSElementPHI,
              LHSAddr.getAlignment().alignmentOfArrayElement(ElementSize));

  // Emit copy.
  CodeGenFunction::OMPPrivateScope Scope(CGF);
  Scope.addPrivate(LHSVar, [=]() { return LHSElementCurrent; });
  Scope.addPrivate(RHSVar, [=]() { return RHSElementCurrent; });
  Scope.Privatize();
  RedOpGen(CGF, XExpr, EExpr, UpExpr);
  Scope.ForceCleanup();

  // Shift the address forward by one element.
  llvm::Value *LHSElementNext = CGF.Builder.CreateConstGEP1_32(
      LHSAddr.getElementType(), LHSElementPHI, /*Idx0=*/1,
      "omp.arraycpy.dest.element");
  llvm::Value *RHSElementNext = CGF.Builder.CreateConstGEP1_32(
      RHSAddr.getElementType(), RHSElementPHI, /*Idx0=*/1,
      "omp.arraycpy.src.element");
  // Check whether we've reached the end.
  llvm::Value *Done =
      CGF.Builder.CreateICmpEQ(LHSElementNext, LHSEnd, "omp.arraycpy.done");
  CGF.Builder.CreateCondBr(Done, DoneBB, BodyBB);
  LHSElementPHI->addIncoming(LHSElementNext, CGF.Builder.GetInsertBlock());
  RHSElementPHI->addIncoming(RHSElementNext, CGF.Builder.GetInsertBlock());

  // Done.
  CGF.EmitBlock(DoneBB, /*IsFinished=*/true);
}

/// Emit reduction combiner. If the combiner is a simple expression emit it as
/// is, otherwise consider it as combiner of UDR decl and emit it as a call of
/// UDR combiner function.
static void emitReductionCombiner(CodeGenFunction &CGF,
                                  const Expr *ReductionOp) {
  if (const auto *CE = dyn_cast<CallExpr>(ReductionOp))
    if (const auto *OVE = dyn_cast<OpaqueValueExpr>(CE->getCallee()))
      if (const auto *DRE =
              dyn_cast<DeclRefExpr>(OVE->getSourceExpr()->IgnoreImpCasts()))
        if (const auto *DRD =
                dyn_cast<OMPDeclareReductionDecl>(DRE->getDecl())) {
          std::pair<llvm::Function *, llvm::Function *> Reduction =
              CGF.CGM.getOpenMPRuntime().getUserDefinedReduction(DRD);
          RValue Func = RValue::get(Reduction.first);
          CodeGenFunction::OpaqueValueMapping Map(CGF, OVE, Func);
          CGF.EmitIgnoredExpr(ReductionOp);
          return;
        }
  CGF.EmitIgnoredExpr(ReductionOp);
}

llvm::Function *CGOpenMPRuntime::emitReductionFunction(
    SourceLocation Loc, llvm::Type *ArgsType, ArrayRef<const Expr *> Privates,
    ArrayRef<const Expr *> LHSExprs, ArrayRef<const Expr *> RHSExprs,
    ArrayRef<const Expr *> ReductionOps) {
  ASTContext &C = CGM.getContext();

  // void reduction_func(void *LHSArg, void *RHSArg);
  FunctionArgList Args;
  ImplicitParamDecl LHSArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr, C.VoidPtrTy,
                           ImplicitParamDecl::Other);
  ImplicitParamDecl RHSArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr, C.VoidPtrTy,
                           ImplicitParamDecl::Other);
  Args.push_back(&LHSArg);
  Args.push_back(&RHSArg);
  const auto &CGFI =
      CGM.getTypes().arrangeBuiltinFunctionDeclaration(C.VoidTy, Args);
  std::string Name = getName({"omp", "reduction", "reduction_func"});
  auto *Fn = llvm::Function::Create(CGM.getTypes().GetFunctionType(CGFI),
                                    llvm::GlobalValue::InternalLinkage, Name,
                                    &CGM.getModule());
  CGM.SetInternalFunctionAttributes(GlobalDecl(), Fn, CGFI);
  Fn->setDoesNotRecurse();
  CodeGenFunction CGF(CGM);
  CGF.StartFunction(GlobalDecl(), C.VoidTy, Fn, CGFI, Args, Loc, Loc);

  // Dst = (void*[n])(LHSArg);
  // Src = (void*[n])(RHSArg);
  Address LHS(CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
      CGF.Builder.CreateLoad(CGF.GetAddrOfLocalVar(&LHSArg)),
      ArgsType), CGF.getPointerAlign());
  Address RHS(CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
      CGF.Builder.CreateLoad(CGF.GetAddrOfLocalVar(&RHSArg)),
      ArgsType), CGF.getPointerAlign());

  //  ...
  //  *(Type<i>*)lhs[i] = RedOp<i>(*(Type<i>*)lhs[i], *(Type<i>*)rhs[i]);
  //  ...
  CodeGenFunction::OMPPrivateScope Scope(CGF);
  auto IPriv = Privates.begin();
  unsigned Idx = 0;
  for (unsigned I = 0, E = ReductionOps.size(); I < E; ++I, ++IPriv, ++Idx) {
    const auto *RHSVar =
        cast<VarDecl>(cast<DeclRefExpr>(RHSExprs[I])->getDecl());
    Scope.addPrivate(RHSVar, [&CGF, RHS, Idx, RHSVar]() {
      return emitAddrOfVarFromArray(CGF, RHS, Idx, RHSVar);
    });
    const auto *LHSVar =
        cast<VarDecl>(cast<DeclRefExpr>(LHSExprs[I])->getDecl());
    Scope.addPrivate(LHSVar, [&CGF, LHS, Idx, LHSVar]() {
      return emitAddrOfVarFromArray(CGF, LHS, Idx, LHSVar);
    });
    QualType PrivTy = (*IPriv)->getType();
    if (PrivTy->isVariablyModifiedType()) {
      // Get array size and emit VLA type.
      ++Idx;
      Address Elem = CGF.Builder.CreateConstArrayGEP(LHS, Idx);
      llvm::Value *Ptr = CGF.Builder.CreateLoad(Elem);
      const VariableArrayType *VLA =
          CGF.getContext().getAsVariableArrayType(PrivTy);
      const auto *OVE = cast<OpaqueValueExpr>(VLA->getSizeExpr());
      CodeGenFunction::OpaqueValueMapping OpaqueMap(
          CGF, OVE, RValue::get(CGF.Builder.CreatePtrToInt(Ptr, CGF.SizeTy)));
      CGF.EmitVariablyModifiedType(PrivTy);
    }
  }
  Scope.Privatize();
  IPriv = Privates.begin();
  auto ILHS = LHSExprs.begin();
  auto IRHS = RHSExprs.begin();
  for (const Expr *E : ReductionOps) {
    if ((*IPriv)->getType()->isArrayType()) {
      // Emit reduction for array section.
      const auto *LHSVar = cast<VarDecl>(cast<DeclRefExpr>(*ILHS)->getDecl());
      const auto *RHSVar = cast<VarDecl>(cast<DeclRefExpr>(*IRHS)->getDecl());
      EmitOMPAggregateReduction(
          CGF, (*IPriv)->getType(), LHSVar, RHSVar,
          [=](CodeGenFunction &CGF, const Expr *, const Expr *, const Expr *) {
            emitReductionCombiner(CGF, E);
          });
    } else {
      // Emit reduction for array subscript or single variable.
      emitReductionCombiner(CGF, E);
    }
    ++IPriv;
    ++ILHS;
    ++IRHS;
  }
  Scope.ForceCleanup();
  CGF.FinishFunction();
  return Fn;
}

void CGOpenMPRuntime::emitSingleReductionCombiner(CodeGenFunction &CGF,
                                                  const Expr *ReductionOp,
                                                  const Expr *PrivateRef,
                                                  const DeclRefExpr *LHS,
                                                  const DeclRefExpr *RHS) {
  if (PrivateRef->getType()->isArrayType()) {
    // Emit reduction for array section.
    const auto *LHSVar = cast<VarDecl>(LHS->getDecl());
    const auto *RHSVar = cast<VarDecl>(RHS->getDecl());
    EmitOMPAggregateReduction(
        CGF, PrivateRef->getType(), LHSVar, RHSVar,
        [=](CodeGenFunction &CGF, const Expr *, const Expr *, const Expr *) {
          emitReductionCombiner(CGF, ReductionOp);
        });
  } else {
    // Emit reduction for array subscript or single variable.
    emitReductionCombiner(CGF, ReductionOp);
  }
}

void CGOpenMPRuntime::emitReduction(CodeGenFunction &CGF, SourceLocation Loc,
                                    ArrayRef<const Expr *> Privates,
                                    ArrayRef<const Expr *> LHSExprs,
                                    ArrayRef<const Expr *> RHSExprs,
                                    ArrayRef<const Expr *> ReductionOps,
                                    ReductionOptionsTy Options) {
  if (!CGF.HaveInsertPoint())
    return;

  bool WithNowait = Options.WithNowait;
  bool SimpleReduction = Options.SimpleReduction;

  // Next code should be emitted for reduction:
  //
  // static kmp_critical_name lock = { 0 };
  //
  // void reduce_func(void *lhs[<n>], void *rhs[<n>]) {
  //  *(Type0*)lhs[0] = ReductionOperation0(*(Type0*)lhs[0], *(Type0*)rhs[0]);
  //  ...
  //  *(Type<n>-1*)lhs[<n>-1] = ReductionOperation<n>-1(*(Type<n>-1*)lhs[<n>-1],
  //  *(Type<n>-1*)rhs[<n>-1]);
  // }
  //
  // ...
  // void *RedList[<n>] = {&<RHSExprs>[0], ..., &<RHSExprs>[<n>-1]};
  // switch (__kmpc_reduce{_nowait}(<loc>, <gtid>, <n>, sizeof(RedList),
  // RedList, reduce_func, &<lock>)) {
  // case 1:
  //  ...
  //  <LHSExprs>[i] = RedOp<i>(*<LHSExprs>[i], *<RHSExprs>[i]);
  //  ...
  // __kmpc_end_reduce{_nowait}(<loc>, <gtid>, &<lock>);
  // break;
  // case 2:
  //  ...
  //  Atomic(<LHSExprs>[i] = RedOp<i>(*<LHSExprs>[i], *<RHSExprs>[i]));
  //  ...
  // [__kmpc_end_reduce(<loc>, <gtid>, &<lock>);]
  // break;
  // default:;
  // }
  //
  // if SimpleReduction is true, only the next code is generated:
  //  ...
  //  <LHSExprs>[i] = RedOp<i>(*<LHSExprs>[i], *<RHSExprs>[i]);
  //  ...

  ASTContext &C = CGM.getContext();

  if (SimpleReduction) {
    CodeGenFunction::RunCleanupsScope Scope(CGF);
    auto IPriv = Privates.begin();
    auto ILHS = LHSExprs.begin();
    auto IRHS = RHSExprs.begin();
    for (const Expr *E : ReductionOps) {
      emitSingleReductionCombiner(CGF, E, *IPriv, cast<DeclRefExpr>(*ILHS),
                                  cast<DeclRefExpr>(*IRHS));
      ++IPriv;
      ++ILHS;
      ++IRHS;
    }
    return;
  }

  // 1. Build a list of reduction variables.
  // void *RedList[<n>] = {<ReductionVars>[0], ..., <ReductionVars>[<n>-1]};
  auto Size = RHSExprs.size();
  for (const Expr *E : Privates) {
    if (E->getType()->isVariablyModifiedType())
      // Reserve place for array size.
      ++Size;
  }
  llvm::APInt ArraySize(/*unsigned int numBits=*/32, Size);
  QualType ReductionArrayTy =
      C.getConstantArrayType(C.VoidPtrTy, ArraySize, nullptr, ArrayType::Normal,
                             /*IndexTypeQuals=*/0);
  Address ReductionList =
      CGF.CreateMemTemp(ReductionArrayTy, ".omp.reduction.red_list");
  auto IPriv = Privates.begin();
  unsigned Idx = 0;
  for (unsigned I = 0, E = RHSExprs.size(); I < E; ++I, ++IPriv, ++Idx) {
    Address Elem = CGF.Builder.CreateConstArrayGEP(ReductionList, Idx);
    CGF.Builder.CreateStore(
        CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
            CGF.EmitLValue(RHSExprs[I]).getPointer(CGF), CGF.VoidPtrTy),
        Elem);
    if ((*IPriv)->getType()->isVariablyModifiedType()) {
      // Store array size.
      ++Idx;
      Elem = CGF.Builder.CreateConstArrayGEP(ReductionList, Idx);
      llvm::Value *Size = CGF.Builder.CreateIntCast(
          CGF.getVLASize(
                 CGF.getContext().getAsVariableArrayType((*IPriv)->getType()))
              .NumElts,
          CGF.SizeTy, /*isSigned=*/false);
      CGF.Builder.CreateStore(CGF.Builder.CreateIntToPtr(Size, CGF.VoidPtrTy),
                              Elem);
    }
  }

  // 2. Emit reduce_func().
  llvm::Function *ReductionFn = emitReductionFunction(
      Loc, CGF.ConvertTypeForMem(ReductionArrayTy)->getPointerTo(), Privates,
      LHSExprs, RHSExprs, ReductionOps);

  // 3. Create static kmp_critical_name lock = { 0 };
  std::string Name = getName({"reduction"});
  llvm::Value *Lock = getCriticalRegionLock(Name);

  // 4. Build res = __kmpc_reduce{_nowait}(<loc>, <gtid>, <n>, sizeof(RedList),
  // RedList, reduce_func, &<lock>);
  llvm::Value *IdentTLoc = emitUpdateLocation(CGF, Loc, OMP_ATOMIC_REDUCE);
  llvm::Value *ThreadId = getThreadID(CGF, Loc);
  llvm::Value *ReductionArrayTySize = CGF.getTypeSize(ReductionArrayTy);
  llvm::Value *RL = CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
      ReductionList.getPointer(), CGF.VoidPtrTy);
  llvm::Value *Args[] = {
      IdentTLoc,                             // ident_t *<loc>
      ThreadId,                              // i32 <gtid>
      CGF.Builder.getInt32(RHSExprs.size()), // i32 <n>
      ReductionArrayTySize,                  // size_type sizeof(RedList)
      RL,                                    // void *RedList
      ReductionFn, // void (*) (void *, void *) <reduce_func>
      Lock         // kmp_critical_name *&<lock>
  };
  llvm::Value *Res = CGF.EmitRuntimeCall(
      OMPBuilder.getOrCreateRuntimeFunction(
          CGM.getModule(),
          WithNowait ? OMPRTL___kmpc_reduce_nowait : OMPRTL___kmpc_reduce),
      Args);

  // 5. Build switch(res)
  llvm::BasicBlock *DefaultBB = CGF.createBasicBlock(".omp.reduction.default");
  llvm::SwitchInst *SwInst =
      CGF.Builder.CreateSwitch(Res, DefaultBB, /*NumCases=*/2);

  // 6. Build case 1:
  //  ...
  //  <LHSExprs>[i] = RedOp<i>(*<LHSExprs>[i], *<RHSExprs>[i]);
  //  ...
  // __kmpc_end_reduce{_nowait}(<loc>, <gtid>, &<lock>);
  // break;
  llvm::BasicBlock *Case1BB = CGF.createBasicBlock(".omp.reduction.case1");
  SwInst->addCase(CGF.Builder.getInt32(1), Case1BB);
  CGF.EmitBlock(Case1BB);

  // Add emission of __kmpc_end_reduce{_nowait}(<loc>, <gtid>, &<lock>);
  llvm::Value *EndArgs[] = {
      IdentTLoc, // ident_t *<loc>
      ThreadId,  // i32 <gtid>
      Lock       // kmp_critical_name *&<lock>
  };
  auto &&CodeGen = [Privates, LHSExprs, RHSExprs, ReductionOps](
                       CodeGenFunction &CGF, PrePostActionTy &Action) {
    CGOpenMPRuntime &RT = CGF.CGM.getOpenMPRuntime();
    auto IPriv = Privates.begin();
    auto ILHS = LHSExprs.begin();
    auto IRHS = RHSExprs.begin();
    for (const Expr *E : ReductionOps) {
      RT.emitSingleReductionCombiner(CGF, E, *IPriv, cast<DeclRefExpr>(*ILHS),
                                     cast<DeclRefExpr>(*IRHS));
      ++IPriv;
      ++ILHS;
      ++IRHS;
    }
  };
  RegionCodeGenTy RCG(CodeGen);
  CommonActionTy Action(
      nullptr, llvm::None,
      OMPBuilder.getOrCreateRuntimeFunction(
          CGM.getModule(), WithNowait ? OMPRTL___kmpc_end_reduce_nowait
                                      : OMPRTL___kmpc_end_reduce),
      EndArgs);
  RCG.setAction(Action);
  RCG(CGF);

  CGF.EmitBranch(DefaultBB);

  // 7. Build case 2:
  //  ...
  //  Atomic(<LHSExprs>[i] = RedOp<i>(*<LHSExprs>[i], *<RHSExprs>[i]));
  //  ...
  // break;
  llvm::BasicBlock *Case2BB = CGF.createBasicBlock(".omp.reduction.case2");
  SwInst->addCase(CGF.Builder.getInt32(2), Case2BB);
  CGF.EmitBlock(Case2BB);

  auto &&AtomicCodeGen = [Loc, Privates, LHSExprs, RHSExprs, ReductionOps](
                             CodeGenFunction &CGF, PrePostActionTy &Action) {
    auto ILHS = LHSExprs.begin();
    auto IRHS = RHSExprs.begin();
    auto IPriv = Privates.begin();
    for (const Expr *E : ReductionOps) {
      const Expr *XExpr = nullptr;
      const Expr *EExpr = nullptr;
      const Expr *UpExpr = nullptr;
      BinaryOperatorKind BO = BO_Comma;
      if (const auto *BO = dyn_cast<BinaryOperator>(E)) {
        if (BO->getOpcode() == BO_Assign) {
          XExpr = BO->getLHS();
          UpExpr = BO->getRHS();
        }
      }
      // Try to emit update expression as a simple atomic.
      const Expr *RHSExpr = UpExpr;
      if (RHSExpr) {
        // Analyze RHS part of the whole expression.
        if (const auto *ACO = dyn_cast<AbstractConditionalOperator>(
                RHSExpr->IgnoreParenImpCasts())) {
          // If this is a conditional operator, analyze its condition for
          // min/max reduction operator.
          RHSExpr = ACO->getCond();
        }
        if (const auto *BORHS =
                dyn_cast<BinaryOperator>(RHSExpr->IgnoreParenImpCasts())) {
          EExpr = BORHS->getRHS();
          BO = BORHS->getOpcode();
        }
      }
      if (XExpr) {
        const auto *VD = cast<VarDecl>(cast<DeclRefExpr>(*ILHS)->getDecl());
        auto &&AtomicRedGen = [BO, VD,
                               Loc](CodeGenFunction &CGF, const Expr *XExpr,
                                    const Expr *EExpr, const Expr *UpExpr) {
          LValue X = CGF.EmitLValue(XExpr);
          RValue E;
          if (EExpr)
            E = CGF.EmitAnyExpr(EExpr);
          CGF.EmitOMPAtomicSimpleUpdateExpr(
              X, E, BO, /*IsXLHSInRHSPart=*/true,
              llvm::AtomicOrdering::Monotonic, Loc,
              [&CGF, UpExpr, VD, Loc](RValue XRValue) {
                CodeGenFunction::OMPPrivateScope PrivateScope(CGF);
                PrivateScope.addPrivate(
                    VD, [&CGF, VD, XRValue, Loc]() {
                      Address LHSTemp = CGF.CreateMemTemp(VD->getType());
                      CGF.emitOMPSimpleStore(
                          CGF.MakeAddrLValue(LHSTemp, VD->getType()), XRValue,
                          VD->getType().getNonReferenceType(), Loc);
                      return LHSTemp;
                    });
                (void)PrivateScope.Privatize();
                return CGF.EmitAnyExpr(UpExpr);
              });
        };
        if ((*IPriv)->getType()->isArrayType()) {
          // Emit atomic reduction for array section.
          const auto *RHSVar =
              cast<VarDecl>(cast<DeclRefExpr>(*IRHS)->getDecl());
          EmitOMPAggregateReduction(CGF, (*IPriv)->getType(), VD, RHSVar,
                                    AtomicRedGen, XExpr, EExpr, UpExpr);
        } else {
          // Emit atomic reduction for array subscript or single variable.
          AtomicRedGen(CGF, XExpr, EExpr, UpExpr);
        }
      } else {
        // Emit as a critical region.
        auto &&CritRedGen = [E, Loc](CodeGenFunction &CGF, const Expr *,
                                           const Expr *, const Expr *) {
          CGOpenMPRuntime &RT = CGF.CGM.getOpenMPRuntime();
          std::string Name = RT.getName({"atomic_reduction"});
          RT.emitCriticalRegion(
              CGF, Name,
              [=](CodeGenFunction &CGF, PrePostActionTy &Action) {
                Action.Enter(CGF);
                emitReductionCombiner(CGF, E);
              },
              Loc);
        };
        if ((*IPriv)->getType()->isArrayType()) {
          const auto *LHSVar =
              cast<VarDecl>(cast<DeclRefExpr>(*ILHS)->getDecl());
          const auto *RHSVar =
              cast<VarDecl>(cast<DeclRefExpr>(*IRHS)->getDecl());
          EmitOMPAggregateReduction(CGF, (*IPriv)->getType(), LHSVar, RHSVar,
                                    CritRedGen);
        } else {
          CritRedGen(CGF, nullptr, nullptr, nullptr);
        }
      }
      ++ILHS;
      ++IRHS;
      ++IPriv;
    }
  };
  RegionCodeGenTy AtomicRCG(AtomicCodeGen);
  if (!WithNowait) {
    // Add emission of __kmpc_end_reduce(<loc>, <gtid>, &<lock>);
    llvm::Value *EndArgs[] = {
        IdentTLoc, // ident_t *<loc>
        ThreadId,  // i32 <gtid>
        Lock       // kmp_critical_name *&<lock>
    };
    CommonActionTy Action(nullptr, llvm::None,
                          OMPBuilder.getOrCreateRuntimeFunction(
                              CGM.getModule(), OMPRTL___kmpc_end_reduce),
                          EndArgs);
    AtomicRCG.setAction(Action);
    AtomicRCG(CGF);
  } else {
    AtomicRCG(CGF);
  }

  CGF.EmitBranch(DefaultBB);
  CGF.EmitBlock(DefaultBB, /*IsFinished=*/true);
}

/// Generates unique name for artificial threadprivate variables.
/// Format is: <Prefix> "." <Decl_mangled_name> "_" "<Decl_start_loc_raw_enc>"
static std::string generateUniqueName(CodeGenModule &CGM, StringRef Prefix,
                                      const Expr *Ref) {
  SmallString<256> Buffer;
  llvm::raw_svector_ostream Out(Buffer);
  const clang::DeclRefExpr *DE;
  const VarDecl *D = ::getBaseDecl(Ref, DE);
  if (!D)
    D = cast<VarDecl>(cast<DeclRefExpr>(Ref)->getDecl());
  D = D->getCanonicalDecl();
  std::string Name = CGM.getOpenMPRuntime().getName(
      {D->isLocalVarDeclOrParm() ? D->getName() : CGM.getMangledName(D)});
  Out << Prefix << Name << "_"
      << D->getCanonicalDecl()->getBeginLoc().getRawEncoding();
  return std::string(Out.str());
}

/// Emits reduction initializer function:
/// \code
/// void @.red_init(void* %arg, void* %orig) {
/// %0 = bitcast void* %arg to <type>*
/// store <type> <init>, <type>* %0
/// ret void
/// }
/// \endcode
static llvm::Value *emitReduceInitFunction(CodeGenModule &CGM,
                                           SourceLocation Loc,
                                           ReductionCodeGen &RCG, unsigned N) {
  ASTContext &C = CGM.getContext();
  QualType VoidPtrTy = C.VoidPtrTy;
  VoidPtrTy.addRestrict();
  FunctionArgList Args;
  ImplicitParamDecl Param(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr, VoidPtrTy,
                          ImplicitParamDecl::Other);
  ImplicitParamDecl ParamOrig(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr, VoidPtrTy,
                              ImplicitParamDecl::Other);
  Args.emplace_back(&Param);
  Args.emplace_back(&ParamOrig);
  const auto &FnInfo =
      CGM.getTypes().arrangeBuiltinFunctionDeclaration(C.VoidTy, Args);
  llvm::FunctionType *FnTy = CGM.getTypes().GetFunctionType(FnInfo);
  std::string Name = CGM.getOpenMPRuntime().getName({"red_init", ""});
  auto *Fn = llvm::Function::Create(FnTy, llvm::GlobalValue::InternalLinkage,
                                    Name, &CGM.getModule());
  CGM.SetInternalFunctionAttributes(GlobalDecl(), Fn, FnInfo);
  Fn->setDoesNotRecurse();
  CodeGenFunction CGF(CGM);
  CGF.StartFunction(GlobalDecl(), C.VoidTy, Fn, FnInfo, Args, Loc, Loc);
  Address PrivateAddr = CGF.EmitLoadOfPointer(
      CGF.GetAddrOfLocalVar(&Param),
      C.getPointerType(C.VoidPtrTy).castAs<PointerType>());
  llvm::Value *Size = nullptr;
  // If the size of the reduction item is non-constant, load it from global
  // threadprivate variable.
  if (RCG.getSizes(N).second) {
    Address SizeAddr = CGM.getOpenMPRuntime().getAddrOfArtificialThreadPrivate(
        CGF, CGM.getContext().getSizeType(),
        generateUniqueName(CGM, "reduction_size", RCG.getRefExpr(N)));
    Size = CGF.EmitLoadOfScalar(SizeAddr, /*Volatile=*/false,
                                CGM.getContext().getSizeType(), Loc);
  }
  RCG.emitAggregateType(CGF, N, Size);
  Address OrigAddr = Address::invalid();
  // If initializer uses initializer from declare reduction construct, emit a
  // pointer to the address of the original reduction item (reuired by reduction
  // initializer)
  if (RCG.usesReductionInitializer(N)) {
    Address SharedAddr = CGF.GetAddrOfLocalVar(&ParamOrig);
    OrigAddr = CGF.EmitLoadOfPointer(
        SharedAddr,
        CGM.getContext().VoidPtrTy.castAs<PointerType>()->getTypePtr());
  }
  // Emit the initializer:
  // %0 = bitcast void* %arg to <type>*
  // store <type> <init>, <type>* %0
  RCG.emitInitialization(CGF, N, PrivateAddr, OrigAddr,
                         [](CodeGenFunction &) { return false; });
  CGF.FinishFunction();
  return Fn;
}

/// Emits reduction combiner function:
/// \code
/// void @.red_comb(void* %arg0, void* %arg1) {
/// %lhs = bitcast void* %arg0 to <type>*
/// %rhs = bitcast void* %arg1 to <type>*
/// %2 = <ReductionOp>(<type>* %lhs, <type>* %rhs)
/// store <type> %2, <type>* %lhs
/// ret void
/// }
/// \endcode
static llvm::Value *emitReduceCombFunction(CodeGenModule &CGM,
                                           SourceLocation Loc,
                                           ReductionCodeGen &RCG, unsigned N,
                                           const Expr *ReductionOp,
                                           const Expr *LHS, const Expr *RHS,
                                           const Expr *PrivateRef) {
  ASTContext &C = CGM.getContext();
  const auto *LHSVD = cast<VarDecl>(cast<DeclRefExpr>(LHS)->getDecl());
  const auto *RHSVD = cast<VarDecl>(cast<DeclRefExpr>(RHS)->getDecl());
  FunctionArgList Args;
  ImplicitParamDecl ParamInOut(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr,
                               C.VoidPtrTy, ImplicitParamDecl::Other);
  ImplicitParamDecl ParamIn(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr, C.VoidPtrTy,
                            ImplicitParamDecl::Other);
  Args.emplace_back(&ParamInOut);
  Args.emplace_back(&ParamIn);
  const auto &FnInfo =
      CGM.getTypes().arrangeBuiltinFunctionDeclaration(C.VoidTy, Args);
  llvm::FunctionType *FnTy = CGM.getTypes().GetFunctionType(FnInfo);
  std::string Name = CGM.getOpenMPRuntime().getName({"red_comb", ""});
  auto *Fn = llvm::Function::Create(FnTy, llvm::GlobalValue::InternalLinkage,
                                    Name, &CGM.getModule());
  CGM.SetInternalFunctionAttributes(GlobalDecl(), Fn, FnInfo);
  Fn->setDoesNotRecurse();
  CodeGenFunction CGF(CGM);
  CGF.StartFunction(GlobalDecl(), C.VoidTy, Fn, FnInfo, Args, Loc, Loc);
  llvm::Value *Size = nullptr;
  // If the size of the reduction item is non-constant, load it from global
  // threadprivate variable.
  if (RCG.getSizes(N).second) {
    Address SizeAddr = CGM.getOpenMPRuntime().getAddrOfArtificialThreadPrivate(
        CGF, CGM.getContext().getSizeType(),
        generateUniqueName(CGM, "reduction_size", RCG.getRefExpr(N)));
    Size = CGF.EmitLoadOfScalar(SizeAddr, /*Volatile=*/false,
                                CGM.getContext().getSizeType(), Loc);
  }
  RCG.emitAggregateType(CGF, N, Size);
  // Remap lhs and rhs variables to the addresses of the function arguments.
  // %lhs = bitcast void* %arg0 to <type>*
  // %rhs = bitcast void* %arg1 to <type>*
  CodeGenFunction::OMPPrivateScope PrivateScope(CGF);
  PrivateScope.addPrivate(LHSVD, [&C, &CGF, &ParamInOut, LHSVD]() {
    // Pull out the pointer to the variable.
    Address PtrAddr = CGF.EmitLoadOfPointer(
        CGF.GetAddrOfLocalVar(&ParamInOut),
        C.getPointerType(C.VoidPtrTy).castAs<PointerType>());
    return CGF.Builder.CreateElementBitCast(
        PtrAddr, CGF.ConvertTypeForMem(LHSVD->getType()));
  });
  PrivateScope.addPrivate(RHSVD, [&C, &CGF, &ParamIn, RHSVD]() {
    // Pull out the pointer to the variable.
    Address PtrAddr = CGF.EmitLoadOfPointer(
        CGF.GetAddrOfLocalVar(&ParamIn),
        C.getPointerType(C.VoidPtrTy).castAs<PointerType>());
    return CGF.Builder.CreateElementBitCast(
        PtrAddr, CGF.ConvertTypeForMem(RHSVD->getType()));
  });
  PrivateScope.Privatize();
  // Emit the combiner body:
  // %2 = <ReductionOp>(<type> *%lhs, <type> *%rhs)
  // store <type> %2, <type>* %lhs
  CGM.getOpenMPRuntime().emitSingleReductionCombiner(
      CGF, ReductionOp, PrivateRef, cast<DeclRefExpr>(LHS),
      cast<DeclRefExpr>(RHS));
  CGF.FinishFunction();
  return Fn;
}

/// Emits reduction finalizer function:
/// \code
/// void @.red_fini(void* %arg) {
/// %0 = bitcast void* %arg to <type>*
/// <destroy>(<type>* %0)
/// ret void
/// }
/// \endcode
static llvm::Value *emitReduceFiniFunction(CodeGenModule &CGM,
                                           SourceLocation Loc,
                                           ReductionCodeGen &RCG, unsigned N) {
  if (!RCG.needCleanups(N))
    return nullptr;
  ASTContext &C = CGM.getContext();
  FunctionArgList Args;
  ImplicitParamDecl Param(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr, C.VoidPtrTy,
                          ImplicitParamDecl::Other);
  Args.emplace_back(&Param);
  const auto &FnInfo =
      CGM.getTypes().arrangeBuiltinFunctionDeclaration(C.VoidTy, Args);
  llvm::FunctionType *FnTy = CGM.getTypes().GetFunctionType(FnInfo);
  std::string Name = CGM.getOpenMPRuntime().getName({"red_fini", ""});
  auto *Fn = llvm::Function::Create(FnTy, llvm::GlobalValue::InternalLinkage,
                                    Name, &CGM.getModule());
  CGM.SetInternalFunctionAttributes(GlobalDecl(), Fn, FnInfo);
  Fn->setDoesNotRecurse();
  CodeGenFunction CGF(CGM);
  CGF.StartFunction(GlobalDecl(), C.VoidTy, Fn, FnInfo, Args, Loc, Loc);
  Address PrivateAddr = CGF.EmitLoadOfPointer(
      CGF.GetAddrOfLocalVar(&Param),
      C.getPointerType(C.VoidPtrTy).castAs<PointerType>());
  llvm::Value *Size = nullptr;
  // If the size of the reduction item is non-constant, load it from global
  // threadprivate variable.
  if (RCG.getSizes(N).second) {
    Address SizeAddr = CGM.getOpenMPRuntime().getAddrOfArtificialThreadPrivate(
        CGF, CGM.getContext().getSizeType(),
        generateUniqueName(CGM, "reduction_size", RCG.getRefExpr(N)));
    Size = CGF.EmitLoadOfScalar(SizeAddr, /*Volatile=*/false,
                                CGM.getContext().getSizeType(), Loc);
  }
  RCG.emitAggregateType(CGF, N, Size);
  // Emit the finalizer body:
  // <destroy>(<type>* %0)
  RCG.emitCleanups(CGF, N, PrivateAddr);
  CGF.FinishFunction(Loc);
  return Fn;
}

llvm::Value *CGOpenMPRuntime::emitTaskReductionInit(
    CodeGenFunction &CGF, SourceLocation Loc, ArrayRef<const Expr *> LHSExprs,
    ArrayRef<const Expr *> RHSExprs, const OMPTaskDataTy &Data) {
  if (!CGF.HaveInsertPoint() || Data.ReductionVars.empty())
    return nullptr;

  // Build typedef struct:
  // kmp_taskred_input {
  //   void *reduce_shar; // shared reduction item
  //   void *reduce_orig; // original reduction item used for initialization
  //   size_t reduce_size; // size of data item
  //   void *reduce_init; // data initialization routine
  //   void *reduce_fini; // data finalization routine
  //   void *reduce_comb; // data combiner routine
  //   kmp_task_red_flags_t flags; // flags for additional info from compiler
  // } kmp_taskred_input_t;
  ASTContext &C = CGM.getContext();
  RecordDecl *RD = C.buildImplicitRecord("kmp_taskred_input_t");
  RD->startDefinition();
  const FieldDecl *SharedFD = addFieldToRecordDecl(C, RD, C.VoidPtrTy);
  const FieldDecl *OrigFD = addFieldToRecordDecl(C, RD, C.VoidPtrTy);
  const FieldDecl *SizeFD = addFieldToRecordDecl(C, RD, C.getSizeType());
  const FieldDecl *InitFD  = addFieldToRecordDecl(C, RD, C.VoidPtrTy);
  const FieldDecl *FiniFD = addFieldToRecordDecl(C, RD, C.VoidPtrTy);
  const FieldDecl *CombFD = addFieldToRecordDecl(C, RD, C.VoidPtrTy);
  const FieldDecl *FlagsFD = addFieldToRecordDecl(
      C, RD, C.getIntTypeForBitwidth(/*DestWidth=*/32, /*Signed=*/false));
  RD->completeDefinition();
  QualType RDType = C.getRecordType(RD);
  unsigned Size = Data.ReductionVars.size();
  llvm::APInt ArraySize(/*numBits=*/64, Size);
  QualType ArrayRDType = C.getConstantArrayType(
      RDType, ArraySize, nullptr, ArrayType::Normal, /*IndexTypeQuals=*/0);
  // kmp_task_red_input_t .rd_input.[Size];
  Address TaskRedInput = CGF.CreateMemTemp(ArrayRDType, ".rd_input.");
  ReductionCodeGen RCG(Data.ReductionVars, Data.ReductionOrigs,
                       Data.ReductionCopies, Data.ReductionOps);
  for (unsigned Cnt = 0; Cnt < Size; ++Cnt) {
    // kmp_task_red_input_t &ElemLVal = .rd_input.[Cnt];
    llvm::Value *Idxs[] = {llvm::ConstantInt::get(CGM.SizeTy, /*V=*/0),
                           llvm::ConstantInt::get(CGM.SizeTy, Cnt)};
    llvm::Value *GEP = CGF.EmitCheckedInBoundsGEP(
        TaskRedInput.getElementType(), TaskRedInput.getPointer(), Idxs,
        /*SignedIndices=*/false, /*IsSubtraction=*/false, Loc,
        ".rd_input.gep.");
    LValue ElemLVal = CGF.MakeNaturalAlignAddrLValue(GEP, RDType);
    // ElemLVal.reduce_shar = &Shareds[Cnt];
    LValue SharedLVal = CGF.EmitLValueForField(ElemLVal, SharedFD);
    RCG.emitSharedOrigLValue(CGF, Cnt);
    llvm::Value *CastedShared =
        CGF.EmitCastToVoidPtr(RCG.getSharedLValue(Cnt).getPointer(CGF));
    CGF.EmitStoreOfScalar(CastedShared, SharedLVal);
    // ElemLVal.reduce_orig = &Origs[Cnt];
    LValue OrigLVal = CGF.EmitLValueForField(ElemLVal, OrigFD);
    llvm::Value *CastedOrig =
        CGF.EmitCastToVoidPtr(RCG.getOrigLValue(Cnt).getPointer(CGF));
    CGF.EmitStoreOfScalar(CastedOrig, OrigLVal);
    RCG.emitAggregateType(CGF, Cnt);
    llvm::Value *SizeValInChars;
    llvm::Value *SizeVal;
    std::tie(SizeValInChars, SizeVal) = RCG.getSizes(Cnt);
    // We use delayed creation/initialization for VLAs and array sections. It is
    // required because runtime does not provide the way to pass the sizes of
    // VLAs/array sections to initializer/combiner/finalizer functions. Instead
    // threadprivate global variables are used to store these values and use
    // them in the functions.
    bool DelayedCreation = !!SizeVal;
    SizeValInChars = CGF.Builder.CreateIntCast(SizeValInChars, CGM.SizeTy,
                                               /*isSigned=*/false);
    LValue SizeLVal = CGF.EmitLValueForField(ElemLVal, SizeFD);
    CGF.EmitStoreOfScalar(SizeValInChars, SizeLVal);
    // ElemLVal.reduce_init = init;
    LValue InitLVal = CGF.EmitLValueForField(ElemLVal, InitFD);
    llvm::Value *InitAddr =
        CGF.EmitCastToVoidPtr(emitReduceInitFunction(CGM, Loc, RCG, Cnt));
    CGF.EmitStoreOfScalar(InitAddr, InitLVal);
    // ElemLVal.reduce_fini = fini;
    LValue FiniLVal = CGF.EmitLValueForField(ElemLVal, FiniFD);
    llvm::Value *Fini = emitReduceFiniFunction(CGM, Loc, RCG, Cnt);
    llvm::Value *FiniAddr = Fini
                                ? CGF.EmitCastToVoidPtr(Fini)
                                : llvm::ConstantPointerNull::get(CGM.VoidPtrTy);
    CGF.EmitStoreOfScalar(FiniAddr, FiniLVal);
    // ElemLVal.reduce_comb = comb;
    LValue CombLVal = CGF.EmitLValueForField(ElemLVal, CombFD);
    llvm::Value *CombAddr = CGF.EmitCastToVoidPtr(emitReduceCombFunction(
        CGM, Loc, RCG, Cnt, Data.ReductionOps[Cnt], LHSExprs[Cnt],
        RHSExprs[Cnt], Data.ReductionCopies[Cnt]));
    CGF.EmitStoreOfScalar(CombAddr, CombLVal);
    // ElemLVal.flags = 0;
    LValue FlagsLVal = CGF.EmitLValueForField(ElemLVal, FlagsFD);
    if (DelayedCreation) {
      CGF.EmitStoreOfScalar(
          llvm::ConstantInt::get(CGM.Int32Ty, /*V=*/1, /*isSigned=*/true),
          FlagsLVal);
    } else
      CGF.EmitNullInitialization(FlagsLVal.getAddress(CGF),
                                 FlagsLVal.getType());
  }
  if (Data.IsReductionWithTaskMod) {
    // Build call void *__kmpc_taskred_modifier_init(ident_t *loc, int gtid, int
    // is_ws, int num, void *data);
    llvm::Value *IdentTLoc = emitUpdateLocation(CGF, Loc);
    llvm::Value *GTid = CGF.Builder.CreateIntCast(getThreadID(CGF, Loc),
                                                  CGM.IntTy, /*isSigned=*/true);
    llvm::Value *Args[] = {
        IdentTLoc, GTid,
        llvm::ConstantInt::get(CGM.IntTy, Data.IsWorksharingReduction ? 1 : 0,
                               /*isSigned=*/true),
        llvm::ConstantInt::get(CGM.IntTy, Size, /*isSigned=*/true),
        CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
            TaskRedInput.getPointer(), CGM.VoidPtrTy)};
    return CGF.EmitRuntimeCall(
        OMPBuilder.getOrCreateRuntimeFunction(
            CGM.getModule(), OMPRTL___kmpc_taskred_modifier_init),
        Args);
  }
  // Build call void *__kmpc_taskred_init(int gtid, int num_data, void *data);
  llvm::Value *Args[] = {
      CGF.Builder.CreateIntCast(getThreadID(CGF, Loc), CGM.IntTy,
                                /*isSigned=*/true),
      llvm::ConstantInt::get(CGM.IntTy, Size, /*isSigned=*/true),
      CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(TaskRedInput.getPointer(),
                                                      CGM.VoidPtrTy)};
  return CGF.EmitRuntimeCall(OMPBuilder.getOrCreateRuntimeFunction(
                                 CGM.getModule(), OMPRTL___kmpc_taskred_init),
                             Args);
}

void CGOpenMPRuntime::emitTaskReductionFini(CodeGenFunction &CGF,
                                            SourceLocation Loc,
                                            bool IsWorksharingReduction) {
  // Build call void *__kmpc_taskred_modifier_init(ident_t *loc, int gtid, int
  // is_ws, int num, void *data);
  llvm::Value *IdentTLoc = emitUpdateLocation(CGF, Loc);
  llvm::Value *GTid = CGF.Builder.CreateIntCast(getThreadID(CGF, Loc),
                                                CGM.IntTy, /*isSigned=*/true);
  llvm::Value *Args[] = {IdentTLoc, GTid,
                         llvm::ConstantInt::get(CGM.IntTy,
                                                IsWorksharingReduction ? 1 : 0,
                                                /*isSigned=*/true)};
  (void)CGF.EmitRuntimeCall(
      OMPBuilder.getOrCreateRuntimeFunction(
          CGM.getModule(), OMPRTL___kmpc_task_reduction_modifier_fini),
      Args);
}

void CGOpenMPRuntime::emitTaskReductionFixups(CodeGenFunction &CGF,
                                              SourceLocation Loc,
                                              ReductionCodeGen &RCG,
                                              unsigned N) {
  auto Sizes = RCG.getSizes(N);
  // Emit threadprivate global variable if the type is non-constant
  // (Sizes.second = nullptr).
  if (Sizes.second) {
    llvm::Value *SizeVal = CGF.Builder.CreateIntCast(Sizes.second, CGM.SizeTy,
                                                     /*isSigned=*/false);
    Address SizeAddr = getAddrOfArtificialThreadPrivate(
        CGF, CGM.getContext().getSizeType(),
        generateUniqueName(CGM, "reduction_size", RCG.getRefExpr(N)));
    CGF.Builder.CreateStore(SizeVal, SizeAddr, /*IsVolatile=*/false);
  }
}

Address CGOpenMPRuntime::getTaskReductionItem(CodeGenFunction &CGF,
                                              SourceLocation Loc,
                                              llvm::Value *ReductionsPtr,
                                              LValue SharedLVal) {
  // Build call void *__kmpc_task_reduction_get_th_data(int gtid, void *tg, void
  // *d);
  llvm::Value *Args[] = {CGF.Builder.CreateIntCast(getThreadID(CGF, Loc),
                                                   CGM.IntTy,
                                                   /*isSigned=*/true),
                         ReductionsPtr,
                         CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
                             SharedLVal.getPointer(CGF), CGM.VoidPtrTy)};
  return Address(
      CGF.EmitRuntimeCall(
          OMPBuilder.getOrCreateRuntimeFunction(
              CGM.getModule(), OMPRTL___kmpc_task_reduction_get_th_data),
          Args),
      SharedLVal.getAlignment());
}

void CGOpenMPRuntime::emitTaskwaitCall(CodeGenFunction &CGF, SourceLocation Loc,
                                       const OMPTaskDataTy &Data) {
  if (!CGF.HaveInsertPoint())
    return;

  if (CGF.CGM.getLangOpts().OpenMPIRBuilder && Data.Dependences.empty()) {
    // TODO: Need to support taskwait with dependences in the OpenMPIRBuilder.
    OMPBuilder.createTaskwait(CGF.Builder);
  } else {
    llvm::Value *ThreadID = getThreadID(CGF, Loc);
    llvm::Value *UpLoc = emitUpdateLocation(CGF, Loc);
    auto &M = CGM.getModule();
    Address DependenciesArray = Address::invalid();
    llvm::Value *NumOfElements;
    std::tie(NumOfElements, DependenciesArray) =
        emitDependClause(CGF, Data.Dependences, Loc);
    llvm::Value *DepWaitTaskArgs[6];
    if (!Data.Dependences.empty()) {
      DepWaitTaskArgs[0] = UpLoc;
      DepWaitTaskArgs[1] = ThreadID;
      DepWaitTaskArgs[2] = NumOfElements;
      DepWaitTaskArgs[3] = DependenciesArray.getPointer();
      DepWaitTaskArgs[4] = CGF.Builder.getInt32(0);
      DepWaitTaskArgs[5] = llvm::ConstantPointerNull::get(CGF.VoidPtrTy);

      CodeGenFunction::RunCleanupsScope LocalScope(CGF);

      // Build void __kmpc_omp_wait_deps(ident_t *, kmp_int32 gtid,
      // kmp_int32 ndeps, kmp_depend_info_t *dep_list, kmp_int32
      // ndeps_noalias, kmp_depend_info_t *noalias_dep_list); if dependence info
      // is specified.
      CGF.EmitRuntimeCall(
          OMPBuilder.getOrCreateRuntimeFunction(M, OMPRTL___kmpc_omp_wait_deps),
          DepWaitTaskArgs);

    } else {

      // Build call kmp_int32 __kmpc_omp_taskwait(ident_t *loc, kmp_int32
      // global_tid);
      llvm::Value *Args[] = {UpLoc, ThreadID};
      // Ignore return result until untied tasks are supported.
      CGF.EmitRuntimeCall(
          OMPBuilder.getOrCreateRuntimeFunction(M, OMPRTL___kmpc_omp_taskwait),
          Args);
    }
  }

  if (auto *Region = dyn_cast_or_null<CGOpenMPRegionInfo>(CGF.CapturedStmtInfo))
    Region->emitUntiedSwitch(CGF);
}

void CGOpenMPRuntime::emitInlinedDirective(CodeGenFunction &CGF,
                                           OpenMPDirectiveKind InnerKind,
                                           const RegionCodeGenTy &CodeGen,
                                           bool HasCancel) {
  if (!CGF.HaveInsertPoint())
    return;
  InlinedOpenMPRegionRAII Region(CGF, CodeGen, InnerKind, HasCancel,
                                 InnerKind != OMPD_critical &&
                                     InnerKind != OMPD_master &&
                                     InnerKind != OMPD_masked);
  CGF.CapturedStmtInfo->EmitBody(CGF, /*S=*/nullptr);
}

namespace {
enum RTCancelKind {
  CancelNoreq = 0,
  CancelParallel = 1,
  CancelLoop = 2,
  CancelSections = 3,
  CancelTaskgroup = 4
};
} // anonymous namespace

static RTCancelKind getCancellationKind(OpenMPDirectiveKind CancelRegion) {
  RTCancelKind CancelKind = CancelNoreq;
  if (CancelRegion == OMPD_parallel)
    CancelKind = CancelParallel;
  else if (CancelRegion == OMPD_for)
    CancelKind = CancelLoop;
  else if (CancelRegion == OMPD_sections)
    CancelKind = CancelSections;
  else {
    assert(CancelRegion == OMPD_taskgroup);
    CancelKind = CancelTaskgroup;
  }
  return CancelKind;
}

void CGOpenMPRuntime::emitCancellationPointCall(
    CodeGenFunction &CGF, SourceLocation Loc,
    OpenMPDirectiveKind CancelRegion) {
  if (!CGF.HaveInsertPoint())
    return;
  // Build call kmp_int32 __kmpc_cancellationpoint(ident_t *loc, kmp_int32
  // global_tid, kmp_int32 cncl_kind);
  if (auto *OMPRegionInfo =
          dyn_cast_or_null<CGOpenMPRegionInfo>(CGF.CapturedStmtInfo)) {
    // For 'cancellation point taskgroup', the task region info may not have a
    // cancel. This may instead happen in another adjacent task.
    if (CancelRegion == OMPD_taskgroup || OMPRegionInfo->hasCancel()) {
      llvm::Value *Args[] = {
          emitUpdateLocation(CGF, Loc), getThreadID(CGF, Loc),
          CGF.Builder.getInt32(getCancellationKind(CancelRegion))};
      // Ignore return result until untied tasks are supported.
      llvm::Value *Result = CGF.EmitRuntimeCall(
          OMPBuilder.getOrCreateRuntimeFunction(
              CGM.getModule(), OMPRTL___kmpc_cancellationpoint),
          Args);
      // if (__kmpc_cancellationpoint()) {
      //   call i32 @__kmpc_cancel_barrier( // for parallel cancellation only
      //   exit from construct;
      // }
      llvm::BasicBlock *ExitBB = CGF.createBasicBlock(".cancel.exit");
      llvm::BasicBlock *ContBB = CGF.createBasicBlock(".cancel.continue");
      llvm::Value *Cmp = CGF.Builder.CreateIsNotNull(Result);
      CGF.Builder.CreateCondBr(Cmp, ExitBB, ContBB);
      CGF.EmitBlock(ExitBB);
      if (CancelRegion == OMPD_parallel)
        emitBarrierCall(CGF, Loc, OMPD_unknown, /*EmitChecks=*/false);
      // exit from construct;
      CodeGenFunction::JumpDest CancelDest =
          CGF.getOMPCancelDestination(OMPRegionInfo->getDirectiveKind());
      CGF.EmitBranchThroughCleanup(CancelDest);
      CGF.EmitBlock(ContBB, /*IsFinished=*/true);
    }
  }
}

void CGOpenMPRuntime::emitCancelCall(CodeGenFunction &CGF, SourceLocation Loc,
                                     const Expr *IfCond,
                                     OpenMPDirectiveKind CancelRegion) {
  if (!CGF.HaveInsertPoint())
    return;
  // Build call kmp_int32 __kmpc_cancel(ident_t *loc, kmp_int32 global_tid,
  // kmp_int32 cncl_kind);
  auto &M = CGM.getModule();
  if (auto *OMPRegionInfo =
          dyn_cast_or_null<CGOpenMPRegionInfo>(CGF.CapturedStmtInfo)) {
    auto &&ThenGen = [this, &M, Loc, CancelRegion,
                      OMPRegionInfo](CodeGenFunction &CGF, PrePostActionTy &) {
      CGOpenMPRuntime &RT = CGF.CGM.getOpenMPRuntime();
      llvm::Value *Args[] = {
          RT.emitUpdateLocation(CGF, Loc), RT.getThreadID(CGF, Loc),
          CGF.Builder.getInt32(getCancellationKind(CancelRegion))};
      // Ignore return result until untied tasks are supported.
      llvm::Value *Result = CGF.EmitRuntimeCall(
          OMPBuilder.getOrCreateRuntimeFunction(M, OMPRTL___kmpc_cancel), Args);
      // if (__kmpc_cancel()) {
      //   call i32 @__kmpc_cancel_barrier( // for parallel cancellation only
      //   exit from construct;
      // }
      llvm::BasicBlock *ExitBB = CGF.createBasicBlock(".cancel.exit");
      llvm::BasicBlock *ContBB = CGF.createBasicBlock(".cancel.continue");
      llvm::Value *Cmp = CGF.Builder.CreateIsNotNull(Result);
      CGF.Builder.CreateCondBr(Cmp, ExitBB, ContBB);
      CGF.EmitBlock(ExitBB);
      if (CancelRegion == OMPD_parallel)
        RT.emitBarrierCall(CGF, Loc, OMPD_unknown, /*EmitChecks=*/false);
      // exit from construct;
      CodeGenFunction::JumpDest CancelDest =
          CGF.getOMPCancelDestination(OMPRegionInfo->getDirectiveKind());
      CGF.EmitBranchThroughCleanup(CancelDest);
      CGF.EmitBlock(ContBB, /*IsFinished=*/true);
    };
    if (IfCond) {
      emitIfClause(CGF, IfCond, ThenGen,
                   [](CodeGenFunction &, PrePostActionTy &) {});
    } else {
      RegionCodeGenTy ThenRCG(ThenGen);
      ThenRCG(CGF);
    }
  }
}

namespace {
/// Cleanup action for uses_allocators support.
class OMPUsesAllocatorsActionTy final : public PrePostActionTy {
  ArrayRef<std::pair<const Expr *, const Expr *>> Allocators;

public:
  OMPUsesAllocatorsActionTy(
      ArrayRef<std::pair<const Expr *, const Expr *>> Allocators)
      : Allocators(Allocators) {}
  void Enter(CodeGenFunction &CGF) override {
    if (!CGF.HaveInsertPoint())
      return;
    for (const auto &AllocatorData : Allocators) {
      CGF.CGM.getOpenMPRuntime().emitUsesAllocatorsInit(
          CGF, AllocatorData.first, AllocatorData.second);
    }
  }
  void Exit(CodeGenFunction &CGF) override {
    if (!CGF.HaveInsertPoint())
      return;
    for (const auto &AllocatorData : Allocators) {
      CGF.CGM.getOpenMPRuntime().emitUsesAllocatorsFini(CGF,
                                                        AllocatorData.first);
    }
  }
};
} // namespace

void CGOpenMPRuntime::emitTargetOutlinedFunction(
    const OMPExecutableDirective &D, StringRef ParentName,
    llvm::Function *&OutlinedFn, llvm::Constant *&OutlinedFnID,
    bool IsOffloadEntry, const RegionCodeGenTy &CodeGen) {
  assert(!ParentName.empty() && "Invalid target region parent name!");
  HasEmittedTargetRegion = true;
  SmallVector<std::pair<const Expr *, const Expr *>, 4> Allocators;
  for (const auto *C : D.getClausesOfKind<OMPUsesAllocatorsClause>()) {
    for (unsigned I = 0, E = C->getNumberOfAllocators(); I < E; ++I) {
      const OMPUsesAllocatorsClause::Data D = C->getAllocatorData(I);
      if (!D.AllocatorTraits)
        continue;
      Allocators.emplace_back(D.Allocator, D.AllocatorTraits);
    }
  }
  OMPUsesAllocatorsActionTy UsesAllocatorAction(Allocators);
  CodeGen.setAction(UsesAllocatorAction);
  emitTargetOutlinedFunctionHelper(D, ParentName, OutlinedFn, OutlinedFnID,
                                   IsOffloadEntry, CodeGen);
}

void CGOpenMPRuntime::emitUsesAllocatorsInit(CodeGenFunction &CGF,
                                             const Expr *Allocator,
                                             const Expr *AllocatorTraits) {
  llvm::Value *ThreadId = getThreadID(CGF, Allocator->getExprLoc());
  ThreadId = CGF.Builder.CreateIntCast(ThreadId, CGF.IntTy, /*isSigned=*/true);
  // Use default memspace handle.
  llvm::Value *MemSpaceHandle = llvm::ConstantPointerNull::get(CGF.VoidPtrTy);
  llvm::Value *NumTraits = llvm::ConstantInt::get(
      CGF.IntTy, cast<ConstantArrayType>(
                     AllocatorTraits->getType()->getAsArrayTypeUnsafe())
                     ->getSize()
                     .getLimitedValue());
  LValue AllocatorTraitsLVal = CGF.EmitLValue(AllocatorTraits);
  Address Addr = CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
      AllocatorTraitsLVal.getAddress(CGF), CGF.VoidPtrPtrTy);
  AllocatorTraitsLVal = CGF.MakeAddrLValue(Addr, CGF.getContext().VoidPtrTy,
                                           AllocatorTraitsLVal.getBaseInfo(),
                                           AllocatorTraitsLVal.getTBAAInfo());
  llvm::Value *Traits =
      CGF.EmitLoadOfScalar(AllocatorTraitsLVal, AllocatorTraits->getExprLoc());

  llvm::Value *AllocatorVal =
      CGF.EmitRuntimeCall(OMPBuilder.getOrCreateRuntimeFunction(
                              CGM.getModule(), OMPRTL___kmpc_init_allocator),
                          {ThreadId, MemSpaceHandle, NumTraits, Traits});
  // Store to allocator.
  CGF.EmitVarDecl(*cast<VarDecl>(
      cast<DeclRefExpr>(Allocator->IgnoreParenImpCasts())->getDecl()));
  LValue AllocatorLVal = CGF.EmitLValue(Allocator->IgnoreParenImpCasts());
  AllocatorVal =
      CGF.EmitScalarConversion(AllocatorVal, CGF.getContext().VoidPtrTy,
                               Allocator->getType(), Allocator->getExprLoc());
  CGF.EmitStoreOfScalar(AllocatorVal, AllocatorLVal);
}

void CGOpenMPRuntime::emitUsesAllocatorsFini(CodeGenFunction &CGF,
                                             const Expr *Allocator) {
  llvm::Value *ThreadId = getThreadID(CGF, Allocator->getExprLoc());
  ThreadId = CGF.Builder.CreateIntCast(ThreadId, CGF.IntTy, /*isSigned=*/true);
  LValue AllocatorLVal = CGF.EmitLValue(Allocator->IgnoreParenImpCasts());
  llvm::Value *AllocatorVal =
      CGF.EmitLoadOfScalar(AllocatorLVal, Allocator->getExprLoc());
  AllocatorVal = CGF.EmitScalarConversion(AllocatorVal, Allocator->getType(),
                                          CGF.getContext().VoidPtrTy,
                                          Allocator->getExprLoc());
  (void)CGF.EmitRuntimeCall(
      OMPBuilder.getOrCreateRuntimeFunction(CGM.getModule(),
                                            OMPRTL___kmpc_destroy_allocator),
      {ThreadId, AllocatorVal});
}

void CGOpenMPRuntime::emitTargetOutlinedFunctionHelper(
    const OMPExecutableDirective &D, StringRef ParentName,
    llvm::Function *&OutlinedFn, llvm::Constant *&OutlinedFnID,
    bool IsOffloadEntry, const RegionCodeGenTy &CodeGen) {
  // Create a unique name for the entry function using the source location
  // information of the current target region. The name will be something like:
  //
  // __omp_offloading_DD_FFFF_PP_lBB
  //
  // where DD_FFFF is an ID unique to the file (device and file IDs), PP is the
  // mangled name of the function that encloses the target region and BB is the
  // line number of the target region.

  unsigned DeviceID;
  unsigned FileID;
  unsigned Line;
  getTargetEntryUniqueInfo(CGM.getContext(), D.getBeginLoc(), DeviceID, FileID,
                           Line);
  SmallString<64> EntryFnName;
  {
    llvm::raw_svector_ostream OS(EntryFnName);
    OS << "__omp_offloading" << llvm::format("_%x", DeviceID)
       << llvm::format("_%x_", FileID) << ParentName << "_l" << Line;
  }

  const CapturedStmt &CS = *D.getCapturedStmt(OMPD_target);

  CodeGenFunction CGF(CGM, true);
  CGOpenMPTargetRegionInfo CGInfo(CS, CodeGen, EntryFnName);
  CodeGenFunction::CGCapturedStmtRAII CapInfoRAII(CGF, &CGInfo);

  OutlinedFn = CGF.GenerateOpenMPCapturedStmtFunction(CS, D.getBeginLoc());

  // If this target outline function is not an offload entry, we don't need to
  // register it.
  if (!IsOffloadEntry)
    return;

  // The target region ID is used by the runtime library to identify the current
  // target region, so it only has to be unique and not necessarily point to
  // anything. It could be the pointer to the outlined function that implements
  // the target region, but we aren't using that so that the compiler doesn't
  // need to keep that, and could therefore inline the host function if proven
  // worthwhile during optimization. In the other hand, if emitting code for the
  // device, the ID has to be the function address so that it can retrieved from
  // the offloading entry and launched by the runtime library. We also mark the
  // outlined function to have external linkage in case we are emitting code for
  // the device, because these functions will be entry points to the device.

  if (CGM.getLangOpts().OpenMPIsDevice) {
    OutlinedFnID = llvm::ConstantExpr::getBitCast(OutlinedFn, CGM.Int8PtrTy);
    OutlinedFn->setLinkage(llvm::GlobalValue::WeakAnyLinkage);
    OutlinedFn->setDSOLocal(false);
    if (CGM.getTriple().isAMDGCN())
      OutlinedFn->setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);
  } else {
    std::string Name = getName({EntryFnName, "region_id"});
    OutlinedFnID = new llvm::GlobalVariable(
        CGM.getModule(), CGM.Int8Ty, /*isConstant=*/true,
        llvm::GlobalValue::WeakAnyLinkage,
        llvm::Constant::getNullValue(CGM.Int8Ty), Name);
  }

  // Register the information for the entry associated with this target region.
  OffloadEntriesInfoManager.registerTargetRegionEntryInfo(
      DeviceID, FileID, ParentName, Line, OutlinedFn, OutlinedFnID,
      OffloadEntriesInfoManagerTy::OMPTargetRegionEntryTargetRegion);

  // Add NumTeams and ThreadLimit attributes to the outlined GPU function
  int32_t DefaultValTeams = -1;
  getNumTeamsExprForTargetDirective(CGF, D, DefaultValTeams);
  if (DefaultValTeams > 0) {
    OutlinedFn->addFnAttr("omp_target_num_teams",
                          std::to_string(DefaultValTeams));
  }
  int32_t DefaultValThreads = -1;
  getNumThreadsExprForTargetDirective(CGF, D, DefaultValThreads);
  if (DefaultValThreads > 0) {
    OutlinedFn->addFnAttr("omp_target_thread_limit",
                          std::to_string(DefaultValThreads));
  }

  CGM.getTargetCodeGenInfo().setTargetAttributes(nullptr, OutlinedFn, CGM);
}

/// Checks if the expression is constant or does not have non-trivial function
/// calls.
static bool isTrivial(ASTContext &Ctx, const Expr * E) {
  // We can skip constant expressions.
  // We can skip expressions with trivial calls or simple expressions.
  return (E->isEvaluatable(Ctx, Expr::SE_AllowUndefinedBehavior) ||
          !E->hasNonTrivialCall(Ctx)) &&
         !E->HasSideEffects(Ctx, /*IncludePossibleEffects=*/true);
}

const Stmt *CGOpenMPRuntime::getSingleCompoundChild(ASTContext &Ctx,
                                                    const Stmt *Body) {
  const Stmt *Child = Body->IgnoreContainers();
  while (const auto *C = dyn_cast_or_null<CompoundStmt>(Child)) {
    Child = nullptr;
    for (const Stmt *S : C->body()) {
      if (const auto *E = dyn_cast<Expr>(S)) {
        if (isTrivial(Ctx, E))
          continue;
      }
      // Some of the statements can be ignored.
      if (isa<AsmStmt>(S) || isa<NullStmt>(S) || isa<OMPFlushDirective>(S) ||
          isa<OMPBarrierDirective>(S) || isa<OMPTaskyieldDirective>(S))
        continue;
      // Analyze declarations.
      if (const auto *DS = dyn_cast<DeclStmt>(S)) {
        if (llvm::all_of(DS->decls(), [](const Decl *D) {
              if (isa<EmptyDecl>(D) || isa<DeclContext>(D) ||
                  isa<TypeDecl>(D) || isa<PragmaCommentDecl>(D) ||
                  isa<PragmaDetectMismatchDecl>(D) || isa<UsingDecl>(D) ||
                  isa<UsingDirectiveDecl>(D) ||
                  isa<OMPDeclareReductionDecl>(D) ||
                  isa<OMPThreadPrivateDecl>(D) || isa<OMPAllocateDecl>(D))
                return true;
              const auto *VD = dyn_cast<VarDecl>(D);
              if (!VD)
                return false;
              return VD->hasGlobalStorage() || !VD->isUsed();
            }))
          continue;
      }
      // Found multiple children - cannot get the one child only.
      if (Child)
        return nullptr;
      Child = S;
    }
    if (Child)
      Child = Child->IgnoreContainers();
  }
  return Child;
}

const Expr *CGOpenMPRuntime::getNumTeamsExprForTargetDirective(
    CodeGenFunction &CGF, const OMPExecutableDirective &D,
    int32_t &DefaultVal) {

  OpenMPDirectiveKind DirectiveKind = D.getDirectiveKind();
  assert(isOpenMPTargetExecutionDirective(DirectiveKind) &&
         "Expected target-based executable directive.");
  switch (DirectiveKind) {
  case OMPD_target: {
    const auto *CS = D.getInnermostCapturedStmt();
    const auto *Body =
        CS->getCapturedStmt()->IgnoreContainers(/*IgnoreCaptured=*/true);
    const Stmt *ChildStmt =
        CGOpenMPRuntime::getSingleCompoundChild(CGF.getContext(), Body);
    if (const auto *NestedDir =
            dyn_cast_or_null<OMPExecutableDirective>(ChildStmt)) {
      if (isOpenMPTeamsDirective(NestedDir->getDirectiveKind())) {
        if (NestedDir->hasClausesOfKind<OMPNumTeamsClause>()) {
          const Expr *NumTeams =
              NestedDir->getSingleClause<OMPNumTeamsClause>()->getNumTeams();
          if (NumTeams->isIntegerConstantExpr(CGF.getContext()))
            if (auto Constant =
                    NumTeams->getIntegerConstantExpr(CGF.getContext()))
              DefaultVal = Constant->getExtValue();
          return NumTeams;
        }
        DefaultVal = 0;
        return nullptr;
      }
      if (isOpenMPParallelDirective(NestedDir->getDirectiveKind()) ||
          isOpenMPSimdDirective(NestedDir->getDirectiveKind())) {
        DefaultVal = 1;
        return nullptr;
      }
      DefaultVal = 1;
      return nullptr;
    }
    // A value of -1 is used to check if we need to emit no teams region
    DefaultVal = -1;
    return nullptr;
  }
  case OMPD_target_teams:
  case OMPD_target_teams_distribute:
  case OMPD_target_teams_distribute_simd:
  case OMPD_target_teams_distribute_parallel_for:
  case OMPD_target_teams_distribute_parallel_for_simd: {
    if (D.hasClausesOfKind<OMPNumTeamsClause>()) {
      const Expr *NumTeams =
          D.getSingleClause<OMPNumTeamsClause>()->getNumTeams();
      if (NumTeams->isIntegerConstantExpr(CGF.getContext()))
        if (auto Constant = NumTeams->getIntegerConstantExpr(CGF.getContext()))
          DefaultVal = Constant->getExtValue();
      return NumTeams;
    }
    DefaultVal = 0;
    return nullptr;
  }
  case OMPD_target_parallel:
  case OMPD_target_parallel_for:
  case OMPD_target_parallel_for_simd:
  case OMPD_target_simd:
    DefaultVal = 1;
    return nullptr;
  case OMPD_parallel:
  case OMPD_for:
  case OMPD_parallel_for:
  case OMPD_parallel_master:
  case OMPD_parallel_sections:
  case OMPD_for_simd:
  case OMPD_parallel_for_simd:
  case OMPD_cancel:
  case OMPD_cancellation_point:
  case OMPD_ordered:
  case OMPD_threadprivate:
  case OMPD_allocate:
  case OMPD_task:
  case OMPD_simd:
  case OMPD_tile:
  case OMPD_unroll:
  case OMPD_sections:
  case OMPD_section:
  case OMPD_single:
  case OMPD_master:
  case OMPD_critical:
  case OMPD_taskyield:
  case OMPD_barrier:
  case OMPD_taskwait:
  case OMPD_taskgroup:
  case OMPD_atomic:
  case OMPD_flush:
  case OMPD_depobj:
  case OMPD_scan:
  case OMPD_teams:
  case OMPD_target_data:
  case OMPD_target_exit_data:
  case OMPD_target_enter_data:
  case OMPD_distribute:
  case OMPD_distribute_simd:
  case OMPD_distribute_parallel_for:
  case OMPD_distribute_parallel_for_simd:
  case OMPD_teams_distribute:
  case OMPD_teams_distribute_simd:
  case OMPD_teams_distribute_parallel_for:
  case OMPD_teams_distribute_parallel_for_simd:
  case OMPD_target_update:
  case OMPD_declare_simd:
  case OMPD_declare_variant:
  case OMPD_begin_declare_variant:
  case OMPD_end_declare_variant:
  case OMPD_declare_target:
  case OMPD_end_declare_target:
  case OMPD_declare_reduction:
  case OMPD_declare_mapper:
  case OMPD_taskloop:
  case OMPD_taskloop_simd:
  case OMPD_master_taskloop:
  case OMPD_master_taskloop_simd:
  case OMPD_parallel_master_taskloop:
  case OMPD_parallel_master_taskloop_simd:
  case OMPD_requires:
  case OMPD_metadirective:
  case OMPD_unknown:
    break;
  default:
    break;
  }
  llvm_unreachable("Unexpected directive kind.");
}

llvm::Value *CGOpenMPRuntime::emitNumTeamsForTargetDirective(
    CodeGenFunction &CGF, const OMPExecutableDirective &D) {
  assert(!CGF.getLangOpts().OpenMPIsDevice &&
         "Clauses associated with the teams directive expected to be emitted "
         "only for the host!");
  CGBuilderTy &Bld = CGF.Builder;
  int32_t DefaultNT = -1;
  const Expr *NumTeams = getNumTeamsExprForTargetDirective(CGF, D, DefaultNT);
  if (NumTeams != nullptr) {
    OpenMPDirectiveKind DirectiveKind = D.getDirectiveKind();

    switch (DirectiveKind) {
    case OMPD_target: {
      const auto *CS = D.getInnermostCapturedStmt();
      CGOpenMPInnerExprInfo CGInfo(CGF, *CS);
      CodeGenFunction::CGCapturedStmtRAII CapInfoRAII(CGF, &CGInfo);
      llvm::Value *NumTeamsVal = CGF.EmitScalarExpr(NumTeams,
                                                  /*IgnoreResultAssign*/ true);
      return Bld.CreateIntCast(NumTeamsVal, CGF.Int32Ty,
                             /*isSigned=*/true);
    }
    case OMPD_target_teams:
    case OMPD_target_teams_distribute:
    case OMPD_target_teams_distribute_simd:
    case OMPD_target_teams_distribute_parallel_for:
    case OMPD_target_teams_distribute_parallel_for_simd: {
      CodeGenFunction::RunCleanupsScope NumTeamsScope(CGF);
      llvm::Value *NumTeamsVal = CGF.EmitScalarExpr(NumTeams,
                                                  /*IgnoreResultAssign*/ true);
      return Bld.CreateIntCast(NumTeamsVal, CGF.Int32Ty,
                             /*isSigned=*/true);
    }
    default:
      break;
    }
  } else if (DefaultNT == -1) {
    return nullptr;
  }

  return Bld.getInt32(DefaultNT);
}

static llvm::Value *getNumThreads(CodeGenFunction &CGF, const CapturedStmt *CS,
                                  llvm::Value *DefaultThreadLimitVal) {
  const Stmt *Child = CGOpenMPRuntime::getSingleCompoundChild(
      CGF.getContext(), CS->getCapturedStmt());
  if (const auto *Dir = dyn_cast_or_null<OMPExecutableDirective>(Child)) {
    if (isOpenMPParallelDirective(Dir->getDirectiveKind())) {
      llvm::Value *NumThreads = nullptr;
      llvm::Value *CondVal = nullptr;
      // Handle if clause. If if clause present, the number of threads is
      // calculated as <cond> ? (<numthreads> ? <numthreads> : 0 ) : 1.
      if (Dir->hasClausesOfKind<OMPIfClause>()) {
        CGOpenMPInnerExprInfo CGInfo(CGF, *CS);
        CodeGenFunction::CGCapturedStmtRAII CapInfoRAII(CGF, &CGInfo);
        const OMPIfClause *IfClause = nullptr;
        for (const auto *C : Dir->getClausesOfKind<OMPIfClause>()) {
          if (C->getNameModifier() == OMPD_unknown ||
              C->getNameModifier() == OMPD_parallel) {
            IfClause = C;
            break;
          }
        }
        if (IfClause) {
          const Expr *Cond = IfClause->getCondition();
          bool Result;
          if (Cond->EvaluateAsBooleanCondition(Result, CGF.getContext())) {
            if (!Result)
              return CGF.Builder.getInt32(1);
          } else {
            CodeGenFunction::LexicalScope Scope(CGF, Cond->getSourceRange());
            if (const auto *PreInit =
                    cast_or_null<DeclStmt>(IfClause->getPreInitStmt())) {
              for (const auto *I : PreInit->decls()) {
                if (!I->hasAttr<OMPCaptureNoInitAttr>()) {
                  CGF.EmitVarDecl(cast<VarDecl>(*I));
                } else {
                  CodeGenFunction::AutoVarEmission Emission =
                      CGF.EmitAutoVarAlloca(cast<VarDecl>(*I));
                  CGF.EmitAutoVarCleanups(Emission);
                }
              }
            }
            CondVal = CGF.EvaluateExprAsBool(Cond);
          }
        }
      }
      // Check the value of num_threads clause iff if clause was not specified
      // or is not evaluated to false.
      if (Dir->hasClausesOfKind<OMPNumThreadsClause>()) {
        CGOpenMPInnerExprInfo CGInfo(CGF, *CS);
        CodeGenFunction::CGCapturedStmtRAII CapInfoRAII(CGF, &CGInfo);
        const auto *NumThreadsClause =
            Dir->getSingleClause<OMPNumThreadsClause>();
        CodeGenFunction::LexicalScope Scope(
            CGF, NumThreadsClause->getNumThreads()->getSourceRange());
        if (const auto *PreInit =
                cast_or_null<DeclStmt>(NumThreadsClause->getPreInitStmt())) {
          for (const auto *I : PreInit->decls()) {
            if (!I->hasAttr<OMPCaptureNoInitAttr>()) {
              CGF.EmitVarDecl(cast<VarDecl>(*I));
            } else {
              CodeGenFunction::AutoVarEmission Emission =
                  CGF.EmitAutoVarAlloca(cast<VarDecl>(*I));
              CGF.EmitAutoVarCleanups(Emission);
            }
          }
        }
        NumThreads = CGF.EmitScalarExpr(NumThreadsClause->getNumThreads());
        NumThreads = CGF.Builder.CreateIntCast(NumThreads, CGF.Int32Ty,
                                               /*isSigned=*/false);
        if (DefaultThreadLimitVal)
          NumThreads = CGF.Builder.CreateSelect(
              CGF.Builder.CreateICmpULT(DefaultThreadLimitVal, NumThreads),
              DefaultThreadLimitVal, NumThreads);
      } else {
        NumThreads = DefaultThreadLimitVal ? DefaultThreadLimitVal
                                           : CGF.Builder.getInt32(0);
      }
      // Process condition of the if clause.
      if (CondVal) {
        NumThreads = CGF.Builder.CreateSelect(CondVal, NumThreads,
                                              CGF.Builder.getInt32(1));
      }
      return NumThreads;
    }
    if (isOpenMPSimdDirective(Dir->getDirectiveKind()))
      return CGF.Builder.getInt32(1);
    return DefaultThreadLimitVal;
  }
  return DefaultThreadLimitVal ? DefaultThreadLimitVal
                               : CGF.Builder.getInt32(0);
}

const Expr *CGOpenMPRuntime::getNumThreadsExprForTargetDirective(
    CodeGenFunction &CGF, const OMPExecutableDirective &D,
    int32_t &DefaultVal) {
  OpenMPDirectiveKind DirectiveKind = D.getDirectiveKind();
  assert(isOpenMPTargetExecutionDirective(DirectiveKind) &&
         "Expected target-based executable directive.");

  switch (DirectiveKind) {
  case OMPD_target:
    // Teams have no clause thread_limit
    return nullptr;
  case OMPD_target_teams:
  case OMPD_target_teams_distribute:
    if (D.hasClausesOfKind<OMPThreadLimitClause>()) {
      const auto *ThreadLimitClause = D.getSingleClause<OMPThreadLimitClause>();
      const Expr *ThreadLimit = ThreadLimitClause->getThreadLimit();
      if (ThreadLimit->isIntegerConstantExpr(CGF.getContext()))
        if (auto Constant =
                ThreadLimit->getIntegerConstantExpr(CGF.getContext()))
          DefaultVal = Constant->getExtValue();
      return ThreadLimit;
    }
    return nullptr;
  case OMPD_target_parallel:
  case OMPD_target_parallel_for:
  case OMPD_target_parallel_for_simd:
  case OMPD_target_teams_distribute_parallel_for:
  case OMPD_target_teams_distribute_parallel_for_simd: {
    Expr *ThreadLimit = nullptr;
    Expr *NumThreads = nullptr;
    if (D.hasClausesOfKind<OMPThreadLimitClause>()) {
      const auto *ThreadLimitClause = D.getSingleClause<OMPThreadLimitClause>();
      ThreadLimit = ThreadLimitClause->getThreadLimit();
      if (ThreadLimit->isIntegerConstantExpr(CGF.getContext()))
        if (auto Constant =
                ThreadLimit->getIntegerConstantExpr(CGF.getContext()))
          DefaultVal = Constant->getExtValue();
    }
    if (D.hasClausesOfKind<OMPNumThreadsClause>()) {
      const auto *NumThreadsClause = D.getSingleClause<OMPNumThreadsClause>();
      NumThreads = NumThreadsClause->getNumThreads();
      if (NumThreads->isIntegerConstantExpr(CGF.getContext())) {
        if (auto Constant =
                NumThreads->getIntegerConstantExpr(CGF.getContext())) {
          if (Constant->getExtValue() < DefaultVal) {
            DefaultVal = Constant->getExtValue();
            ThreadLimit = NumThreads;
          }
        }
      }
    }
    return ThreadLimit;
  }
  case OMPD_target_teams_distribute_simd:
  case OMPD_target_simd:
    DefaultVal = 1;
    return nullptr;
  case OMPD_parallel:
  case OMPD_for:
  case OMPD_parallel_for:
  case OMPD_parallel_master:
  case OMPD_parallel_sections:
  case OMPD_for_simd:
  case OMPD_parallel_for_simd:
  case OMPD_cancel:
  case OMPD_cancellation_point:
  case OMPD_ordered:
  case OMPD_threadprivate:
  case OMPD_allocate:
  case OMPD_task:
  case OMPD_simd:
  case OMPD_tile:
  case OMPD_unroll:
  case OMPD_sections:
  case OMPD_section:
  case OMPD_single:
  case OMPD_master:
  case OMPD_critical:
  case OMPD_taskyield:
  case OMPD_barrier:
  case OMPD_taskwait:
  case OMPD_taskgroup:
  case OMPD_atomic:
  case OMPD_flush:
  case OMPD_depobj:
  case OMPD_scan:
  case OMPD_teams:
  case OMPD_target_data:
  case OMPD_target_exit_data:
  case OMPD_target_enter_data:
  case OMPD_distribute:
  case OMPD_distribute_simd:
  case OMPD_distribute_parallel_for:
  case OMPD_distribute_parallel_for_simd:
  case OMPD_teams_distribute:
  case OMPD_teams_distribute_simd:
  case OMPD_teams_distribute_parallel_for:
  case OMPD_teams_distribute_parallel_for_simd:
  case OMPD_target_update:
  case OMPD_declare_simd:
  case OMPD_declare_variant:
  case OMPD_begin_declare_variant:
  case OMPD_end_declare_variant:
  case OMPD_declare_target:
  case OMPD_end_declare_target:
  case OMPD_declare_reduction:
  case OMPD_declare_mapper:
  case OMPD_taskloop:
  case OMPD_taskloop_simd:
  case OMPD_master_taskloop:
  case OMPD_master_taskloop_simd:
  case OMPD_parallel_master_taskloop:
  case OMPD_parallel_master_taskloop_simd:
  case OMPD_requires:
  case OMPD_unknown:
    break;
  default:
    break;
  }
  llvm_unreachable("Unsupported directive kind.");
}

llvm::Value *CGOpenMPRuntime::emitNumThreadsForTargetDirective(
    CodeGenFunction &CGF, const OMPExecutableDirective &D) {
  assert(!CGF.getLangOpts().OpenMPIsDevice &&
         "Clauses associated with the teams directive expected to be emitted "
         "only for the host!");
  OpenMPDirectiveKind DirectiveKind = D.getDirectiveKind();
  assert(isOpenMPTargetExecutionDirective(DirectiveKind) &&
         "Expected target-based executable directive.");
  CGBuilderTy &Bld = CGF.Builder;
  llvm::Value *ThreadLimitVal = nullptr;
  llvm::Value *NumThreadsVal = nullptr;
  switch (DirectiveKind) {
  case OMPD_target: {
    const CapturedStmt *CS = D.getInnermostCapturedStmt();
    if (llvm::Value *NumThreads = getNumThreads(CGF, CS, ThreadLimitVal))
      return NumThreads;
    const Stmt *Child = CGOpenMPRuntime::getSingleCompoundChild(
        CGF.getContext(), CS->getCapturedStmt());
    if (const auto *Dir = dyn_cast_or_null<OMPExecutableDirective>(Child)) {
      if (Dir->hasClausesOfKind<OMPThreadLimitClause>()) {
        CGOpenMPInnerExprInfo CGInfo(CGF, *CS);
        CodeGenFunction::CGCapturedStmtRAII CapInfoRAII(CGF, &CGInfo);
        const auto *ThreadLimitClause =
            Dir->getSingleClause<OMPThreadLimitClause>();
        CodeGenFunction::LexicalScope Scope(
            CGF, ThreadLimitClause->getThreadLimit()->getSourceRange());
        if (const auto *PreInit =
                cast_or_null<DeclStmt>(ThreadLimitClause->getPreInitStmt())) {
          for (const auto *I : PreInit->decls()) {
            if (!I->hasAttr<OMPCaptureNoInitAttr>()) {
              CGF.EmitVarDecl(cast<VarDecl>(*I));
            } else {
              CodeGenFunction::AutoVarEmission Emission =
                  CGF.EmitAutoVarAlloca(cast<VarDecl>(*I));
              CGF.EmitAutoVarCleanups(Emission);
            }
          }
        }
        llvm::Value *ThreadLimit = CGF.EmitScalarExpr(
            ThreadLimitClause->getThreadLimit(), /*IgnoreResultAssign=*/true);
        ThreadLimitVal =
            Bld.CreateIntCast(ThreadLimit, CGF.Int32Ty, /*isSigned=*/false);
      }
      if (isOpenMPTeamsDirective(Dir->getDirectiveKind()) &&
          !isOpenMPDistributeDirective(Dir->getDirectiveKind())) {
        CS = Dir->getInnermostCapturedStmt();
        const Stmt *Child = CGOpenMPRuntime::getSingleCompoundChild(
            CGF.getContext(), CS->getCapturedStmt());
        Dir = dyn_cast_or_null<OMPExecutableDirective>(Child);
      }
      if (Dir && isOpenMPDistributeDirective(Dir->getDirectiveKind()) &&
          !isOpenMPSimdDirective(Dir->getDirectiveKind())) {
        CS = Dir->getInnermostCapturedStmt();
        if (llvm::Value *NumThreads = getNumThreads(CGF, CS, ThreadLimitVal))
          return NumThreads;
      }
      if (Dir && isOpenMPSimdDirective(Dir->getDirectiveKind()))
        return Bld.getInt32(1);
    }
    return ThreadLimitVal ? ThreadLimitVal : Bld.getInt32(0);
  }
  case OMPD_target_teams: {
    if (D.hasClausesOfKind<OMPThreadLimitClause>()) {
      CodeGenFunction::RunCleanupsScope ThreadLimitScope(CGF);
      const auto *ThreadLimitClause = D.getSingleClause<OMPThreadLimitClause>();
      llvm::Value *ThreadLimit = CGF.EmitScalarExpr(
          ThreadLimitClause->getThreadLimit(), /*IgnoreResultAssign=*/true);
      ThreadLimitVal =
          Bld.CreateIntCast(ThreadLimit, CGF.Int32Ty, /*isSigned=*/false);
    }
    const CapturedStmt *CS = D.getInnermostCapturedStmt();
    if (llvm::Value *NumThreads = getNumThreads(CGF, CS, ThreadLimitVal))
      return NumThreads;
    const Stmt *Child = CGOpenMPRuntime::getSingleCompoundChild(
        CGF.getContext(), CS->getCapturedStmt());
    if (const auto *Dir = dyn_cast_or_null<OMPExecutableDirective>(Child)) {
      if (Dir->getDirectiveKind() == OMPD_distribute) {
        CS = Dir->getInnermostCapturedStmt();
        if (llvm::Value *NumThreads = getNumThreads(CGF, CS, ThreadLimitVal))
          return NumThreads;
      }
    }
    return ThreadLimitVal ? ThreadLimitVal : Bld.getInt32(0);
  }
  case OMPD_target_teams_distribute:
    if (D.hasClausesOfKind<OMPThreadLimitClause>()) {
      CodeGenFunction::RunCleanupsScope ThreadLimitScope(CGF);
      const auto *ThreadLimitClause = D.getSingleClause<OMPThreadLimitClause>();
      llvm::Value *ThreadLimit = CGF.EmitScalarExpr(
          ThreadLimitClause->getThreadLimit(), /*IgnoreResultAssign=*/true);
      ThreadLimitVal =
          Bld.CreateIntCast(ThreadLimit, CGF.Int32Ty, /*isSigned=*/false);
    }
    return getNumThreads(CGF, D.getInnermostCapturedStmt(), ThreadLimitVal);
  case OMPD_target_parallel:
  case OMPD_target_parallel_for:
  case OMPD_target_parallel_for_simd:
  case OMPD_target_teams_distribute_parallel_for:
  case OMPD_target_teams_distribute_parallel_for_simd: {
    llvm::Value *CondVal = nullptr;
    // Handle if clause. If if clause present, the number of threads is
    // calculated as <cond> ? (<numthreads> ? <numthreads> : 0 ) : 1.
    if (D.hasClausesOfKind<OMPIfClause>()) {
      const OMPIfClause *IfClause = nullptr;
      for (const auto *C : D.getClausesOfKind<OMPIfClause>()) {
        if (C->getNameModifier() == OMPD_unknown ||
            C->getNameModifier() == OMPD_parallel) {
          IfClause = C;
          break;
        }
      }
      if (IfClause) {
        const Expr *Cond = IfClause->getCondition();
        bool Result;
        if (Cond->EvaluateAsBooleanCondition(Result, CGF.getContext())) {
          if (!Result)
            return Bld.getInt32(1);
        } else {
          CodeGenFunction::RunCleanupsScope Scope(CGF);
          CondVal = CGF.EvaluateExprAsBool(Cond);
        }
      }
    }
    if (D.hasClausesOfKind<OMPThreadLimitClause>()) {
      CodeGenFunction::RunCleanupsScope ThreadLimitScope(CGF);
      const auto *ThreadLimitClause = D.getSingleClause<OMPThreadLimitClause>();
      llvm::Value *ThreadLimit = CGF.EmitScalarExpr(
          ThreadLimitClause->getThreadLimit(), /*IgnoreResultAssign=*/true);
      ThreadLimitVal =
          Bld.CreateIntCast(ThreadLimit, CGF.Int32Ty, /*isSigned=*/false);
    }
    if (D.hasClausesOfKind<OMPNumThreadsClause>()) {
      CodeGenFunction::RunCleanupsScope NumThreadsScope(CGF);
      const auto *NumThreadsClause = D.getSingleClause<OMPNumThreadsClause>();
      llvm::Value *NumThreads = CGF.EmitScalarExpr(
          NumThreadsClause->getNumThreads(), /*IgnoreResultAssign=*/true);
      NumThreadsVal =
          Bld.CreateIntCast(NumThreads, CGF.Int32Ty, /*isSigned=*/false);
      ThreadLimitVal = ThreadLimitVal
                           ? Bld.CreateSelect(Bld.CreateICmpULT(NumThreadsVal,
                                                                ThreadLimitVal),
                                              NumThreadsVal, ThreadLimitVal)
                           : NumThreadsVal;
    }
    if (!ThreadLimitVal)
      ThreadLimitVal = Bld.getInt32(0);
    if (CondVal)
      return Bld.CreateSelect(CondVal, ThreadLimitVal, Bld.getInt32(1));
    return ThreadLimitVal;
  }
  case OMPD_target_teams_distribute_simd:
  case OMPD_target_simd:
    return Bld.getInt32(1);
  case OMPD_parallel:
  case OMPD_for:
  case OMPD_parallel_for:
  case OMPD_parallel_master:
  case OMPD_parallel_sections:
  case OMPD_for_simd:
  case OMPD_parallel_for_simd:
  case OMPD_cancel:
  case OMPD_cancellation_point:
  case OMPD_ordered:
  case OMPD_threadprivate:
  case OMPD_allocate:
  case OMPD_task:
  case OMPD_simd:
  case OMPD_tile:
  case OMPD_unroll:
  case OMPD_sections:
  case OMPD_section:
  case OMPD_single:
  case OMPD_master:
  case OMPD_critical:
  case OMPD_taskyield:
  case OMPD_barrier:
  case OMPD_taskwait:
  case OMPD_taskgroup:
  case OMPD_atomic:
  case OMPD_flush:
  case OMPD_depobj:
  case OMPD_scan:
  case OMPD_teams:
  case OMPD_target_data:
  case OMPD_target_exit_data:
  case OMPD_target_enter_data:
  case OMPD_distribute:
  case OMPD_distribute_simd:
  case OMPD_distribute_parallel_for:
  case OMPD_distribute_parallel_for_simd:
  case OMPD_teams_distribute:
  case OMPD_teams_distribute_simd:
  case OMPD_teams_distribute_parallel_for:
  case OMPD_teams_distribute_parallel_for_simd:
  case OMPD_target_update:
  case OMPD_declare_simd:
  case OMPD_declare_variant:
  case OMPD_begin_declare_variant:
  case OMPD_end_declare_variant:
  case OMPD_declare_target:
  case OMPD_end_declare_target:
  case OMPD_declare_reduction:
  case OMPD_declare_mapper:
  case OMPD_taskloop:
  case OMPD_taskloop_simd:
  case OMPD_master_taskloop:
  case OMPD_master_taskloop_simd:
  case OMPD_parallel_master_taskloop:
  case OMPD_parallel_master_taskloop_simd:
  case OMPD_requires:
  case OMPD_metadirective:
  case OMPD_unknown:
    break;
  default:
    break;
  }
  llvm_unreachable("Unsupported directive kind.");
}

namespace {
LLVM_ENABLE_BITMASK_ENUMS_IN_NAMESPACE();

// Utility to handle information from clauses associated with a given
// construct that use mappable expressions (e.g. 'map' clause, 'to' clause).
// It provides a convenient interface to obtain the information and generate
// code for that information.
class MappableExprsHandler {
public:
  /// Values for bit flags used to specify the mapping type for
  /// offloading.
  enum OpenMPOffloadMappingFlags : uint64_t {
    /// No flags
    OMP_MAP_NONE = 0x0,
    /// Allocate memory on the device and move data from host to device.
    OMP_MAP_TO = 0x01,
    /// Allocate memory on the device and move data from device to host.
    OMP_MAP_FROM = 0x02,
    /// Always perform the requested mapping action on the element, even
    /// if it was already mapped before.
    OMP_MAP_ALWAYS = 0x04,
    /// Delete the element from the device environment, ignoring the
    /// current reference count associated with the element.
    OMP_MAP_DELETE = 0x08,
    /// The element being mapped is a pointer-pointee pair; both the
    /// pointer and the pointee should be mapped.
    OMP_MAP_PTR_AND_OBJ = 0x10,
    /// This flags signals that the base address of an entry should be
    /// passed to the target kernel as an argument.
    OMP_MAP_TARGET_PARAM = 0x20,
    /// Signal that the runtime library has to return the device pointer
    /// in the current position for the data being mapped. Used when we have the
    /// use_device_ptr or use_device_addr clause.
    OMP_MAP_RETURN_PARAM = 0x40,
    /// This flag signals that the reference being passed is a pointer to
    /// private data.
    OMP_MAP_PRIVATE = 0x80,
    /// Pass the element to the device by value.
    OMP_MAP_LITERAL = 0x100,
    /// Implicit map
    OMP_MAP_IMPLICIT = 0x200,
    /// Close is a hint to the runtime to allocate memory close to
    /// the target device.
    OMP_MAP_CLOSE = 0x400,
    /// 0x800 is reserved for compatibility with XLC.
    /// Produce a runtime error if the data is not already allocated.
    OMP_MAP_PRESENT = 0x1000,
    // Increment and decrement a separate reference counter so that the data
    // cannot be unmapped within the associated region.  Thus, this flag is
    // intended to be used on 'target' and 'target data' directives because they
    // are inherently structured.  It is not intended to be used on 'target
    // enter data' and 'target exit data' directives because they are inherently
    // dynamic.
    // This is an OpenMP extension for the sake of OpenACC support.
    OMP_MAP_OMPX_HOLD = 0x2000,
    /// Signal that the runtime library should use args as an array of
    /// descriptor_dim pointers and use args_size as dims. Used when we have
    /// non-contiguous list items in target update directive
    OMP_MAP_NON_CONTIG = 0x100000000000,
    /// The 16 MSBs of the flags indicate whether the entry is member of some
    /// struct/class.
    OMP_MAP_MEMBER_OF = 0xffff000000000000,
    LLVM_MARK_AS_BITMASK_ENUM(/* LargestFlag = */ OMP_MAP_MEMBER_OF),
  };

  /// Get the offset of the OMP_MAP_MEMBER_OF field.
  static unsigned getFlagMemberOffset() {
    unsigned Offset = 0;
    for (uint64_t Remain = OMP_MAP_MEMBER_OF; !(Remain & 1);
         Remain = Remain >> 1)
      Offset++;
    return Offset;
  }

  /// Class that holds debugging information for a data mapping to be passed to
  /// the runtime library.
  class MappingExprInfo {
    /// The variable declaration used for the data mapping.
    const ValueDecl *MapDecl = nullptr;
    /// The original expression used in the map clause, or null if there is
    /// none.
    const Expr *MapExpr = nullptr;

  public:
    MappingExprInfo(const ValueDecl *MapDecl, const Expr *MapExpr = nullptr)
        : MapDecl(MapDecl), MapExpr(MapExpr) {}

    const ValueDecl *getMapDecl() const { return MapDecl; }
    const Expr *getMapExpr() const { return MapExpr; }
  };

  /// Class that associates information with a base pointer to be passed to the
  /// runtime library.
  class BasePointerInfo {
    /// The base pointer.
    llvm::Value *Ptr = nullptr;
    /// The base declaration that refers to this device pointer, or null if
    /// there is none.
    const ValueDecl *DevPtrDecl = nullptr;

  public:
    BasePointerInfo(llvm::Value *Ptr, const ValueDecl *DevPtrDecl = nullptr)
        : Ptr(Ptr), DevPtrDecl(DevPtrDecl) {}
    llvm::Value *operator*() const { return Ptr; }
    const ValueDecl *getDevicePtrDecl() const { return DevPtrDecl; }
    void setDevicePtrDecl(const ValueDecl *D) { DevPtrDecl = D; }
  };

  using MapExprsArrayTy = SmallVector<MappingExprInfo, 4>;
  using MapBaseValuesArrayTy = SmallVector<BasePointerInfo, 4>;
  using MapValuesArrayTy = SmallVector<llvm::Value *, 4>;
  using MapFlagsArrayTy = SmallVector<OpenMPOffloadMappingFlags, 4>;
  using MapMappersArrayTy = SmallVector<const ValueDecl *, 4>;
  using MapDimArrayTy = SmallVector<uint64_t, 4>;
  using MapNonContiguousArrayTy = SmallVector<MapValuesArrayTy, 4>;

  /// This structure contains combined information generated for mappable
  /// clauses, including base pointers, pointers, sizes, map types, user-defined
  /// mappers, and non-contiguous information.
  struct MapCombinedInfoTy {
    struct StructNonContiguousInfo {
      bool IsNonContiguous = false;
      MapDimArrayTy Dims;
      MapNonContiguousArrayTy Offsets;
      MapNonContiguousArrayTy Counts;
      MapNonContiguousArrayTy Strides;
    };
    MapExprsArrayTy Exprs;
    MapBaseValuesArrayTy BasePointers;
    MapValuesArrayTy Pointers;
    MapValuesArrayTy Sizes;
    MapFlagsArrayTy Types;
    MapMappersArrayTy Mappers;
    StructNonContiguousInfo NonContigInfo;

    /// Append arrays in \a CurInfo.
    void append(MapCombinedInfoTy &CurInfo) {
      Exprs.append(CurInfo.Exprs.begin(), CurInfo.Exprs.end());
      BasePointers.append(CurInfo.BasePointers.begin(),
                          CurInfo.BasePointers.end());
      Pointers.append(CurInfo.Pointers.begin(), CurInfo.Pointers.end());
      Sizes.append(CurInfo.Sizes.begin(), CurInfo.Sizes.end());
      Types.append(CurInfo.Types.begin(), CurInfo.Types.end());
      Mappers.append(CurInfo.Mappers.begin(), CurInfo.Mappers.end());
      NonContigInfo.Dims.append(CurInfo.NonContigInfo.Dims.begin(),
                                 CurInfo.NonContigInfo.Dims.end());
      NonContigInfo.Offsets.append(CurInfo.NonContigInfo.Offsets.begin(),
                                    CurInfo.NonContigInfo.Offsets.end());
      NonContigInfo.Counts.append(CurInfo.NonContigInfo.Counts.begin(),
                                   CurInfo.NonContigInfo.Counts.end());
      NonContigInfo.Strides.append(CurInfo.NonContigInfo.Strides.begin(),
                                    CurInfo.NonContigInfo.Strides.end());
    }
  };

  /// Map between a struct and the its lowest & highest elements which have been
  /// mapped.
  /// [ValueDecl *] --> {LE(FieldIndex, Pointer),
  ///                    HE(FieldIndex, Pointer)}
  struct StructRangeInfoTy {
    MapCombinedInfoTy PreliminaryMapData;
    std::pair<unsigned /*FieldIndex*/, Address /*Pointer*/> LowestElem = {
        0, Address::invalid()};
    std::pair<unsigned /*FieldIndex*/, Address /*Pointer*/> HighestElem = {
        0, Address::invalid()};
    Address Base = Address::invalid();
    Address LB = Address::invalid();
    bool IsArraySection = false;
    bool HasCompleteRecord = false;
  };

private:
  /// Kind that defines how a device pointer has to be returned.
  struct MapInfo {
    OMPClauseMappableExprCommon::MappableExprComponentListRef Components;
    OpenMPMapClauseKind MapType = OMPC_MAP_unknown;
    ArrayRef<OpenMPMapModifierKind> MapModifiers;
    ArrayRef<OpenMPMotionModifierKind> MotionModifiers;
    bool ReturnDevicePointer = false;
    bool IsImplicit = false;
    const ValueDecl *Mapper = nullptr;
    const Expr *VarRef = nullptr;
    bool ForDeviceAddr = false;

    MapInfo() = default;
    MapInfo(
        OMPClauseMappableExprCommon::MappableExprComponentListRef Components,
        OpenMPMapClauseKind MapType,
        ArrayRef<OpenMPMapModifierKind> MapModifiers,
        ArrayRef<OpenMPMotionModifierKind> MotionModifiers,
        bool ReturnDevicePointer, bool IsImplicit,
        const ValueDecl *Mapper = nullptr, const Expr *VarRef = nullptr,
        bool ForDeviceAddr = false)
        : Components(Components), MapType(MapType), MapModifiers(MapModifiers),
          MotionModifiers(MotionModifiers),
          ReturnDevicePointer(ReturnDevicePointer), IsImplicit(IsImplicit),
          Mapper(Mapper), VarRef(VarRef), ForDeviceAddr(ForDeviceAddr) {}
  };

  /// If use_device_ptr or use_device_addr is used on a decl which is a struct
  /// member and there is no map information about it, then emission of that
  /// entry is deferred until the whole struct has been processed.
  struct DeferredDevicePtrEntryTy {
    const Expr *IE = nullptr;
    const ValueDecl *VD = nullptr;
    bool ForDeviceAddr = false;

    DeferredDevicePtrEntryTy(const Expr *IE, const ValueDecl *VD,
                             bool ForDeviceAddr)
        : IE(IE), VD(VD), ForDeviceAddr(ForDeviceAddr) {}
  };

  /// The target directive from where the mappable clauses were extracted. It
  /// is either a executable directive or a user-defined mapper directive.
  llvm::PointerUnion<const OMPExecutableDirective *,
                     const OMPDeclareMapperDecl *>
      CurDir;

  /// Function the directive is being generated for.
  CodeGenFunction &CGF;

  /// Set of all first private variables in the current directive.
  /// bool data is set to true if the variable is implicitly marked as
  /// firstprivate, false otherwise.
  llvm::DenseMap<CanonicalDeclPtr<const VarDecl>, bool> FirstPrivateDecls;

  /// Map between device pointer declarations and their expression components.
  /// The key value for declarations in 'this' is null.
  llvm::DenseMap<
      const ValueDecl *,
      SmallVector<OMPClauseMappableExprCommon::MappableExprComponentListRef, 4>>
      DevPointersMap;

  /// Map between lambda declarations and their map type.
  llvm::DenseMap<const ValueDecl *, const OMPMapClause *> LambdasMap;

  llvm::Value *getExprTypeSize(const Expr *E) const {
    QualType ExprTy = E->getType().getCanonicalType();

    // Calculate the size for array shaping expression.
    if (const auto *OAE = dyn_cast<OMPArrayShapingExpr>(E)) {
      llvm::Value *Size =
          CGF.getTypeSize(OAE->getBase()->getType()->getPointeeType());
      for (const Expr *SE : OAE->getDimensions()) {
        llvm::Value *Sz = CGF.EmitScalarExpr(SE);
        Sz = CGF.EmitScalarConversion(Sz, SE->getType(),
                                      CGF.getContext().getSizeType(),
                                      SE->getExprLoc());
        Size = CGF.Builder.CreateNUWMul(Size, Sz);
      }
      return Size;
    }

    // Reference types are ignored for mapping purposes.
    if (const auto *RefTy = ExprTy->getAs<ReferenceType>())
      ExprTy = RefTy->getPointeeType().getCanonicalType();

    // Given that an array section is considered a built-in type, we need to
    // do the calculation based on the length of the section instead of relying
    // on CGF.getTypeSize(E->getType()).
    if (const auto *OAE = dyn_cast<OMPArraySectionExpr>(E)) {
      QualType BaseTy = OMPArraySectionExpr::getBaseOriginalType(
                            OAE->getBase()->IgnoreParenImpCasts())
                            .getCanonicalType();

      // If there is no length associated with the expression and lower bound is
      // not specified too, that means we are using the whole length of the
      // base.
      if (!OAE->getLength() && OAE->getColonLocFirst().isValid() &&
          !OAE->getLowerBound())
        return CGF.getTypeSize(BaseTy);

      llvm::Value *ElemSize;
      if (const auto *PTy = BaseTy->getAs<PointerType>()) {
        ElemSize = CGF.getTypeSize(PTy->getPointeeType().getCanonicalType());
      } else {
        const auto *ATy = cast<ArrayType>(BaseTy.getTypePtr());
        assert(ATy && "Expecting array type if not a pointer type.");
        ElemSize = CGF.getTypeSize(ATy->getElementType().getCanonicalType());
      }

      // If we don't have a length at this point, that is because we have an
      // array section with a single element.
      if (!OAE->getLength() && OAE->getColonLocFirst().isInvalid())
        return ElemSize;

      if (const Expr *LenExpr = OAE->getLength()) {
        llvm::Value *LengthVal = CGF.EmitScalarExpr(LenExpr);
        LengthVal = CGF.EmitScalarConversion(LengthVal, LenExpr->getType(),
                                             CGF.getContext().getSizeType(),
                                             LenExpr->getExprLoc());
        return CGF.Builder.CreateNUWMul(LengthVal, ElemSize);
      }
      assert(!OAE->getLength() && OAE->getColonLocFirst().isValid() &&
             OAE->getLowerBound() && "expected array_section[lb:].");
      // Size = sizetype - lb * elemtype;
      llvm::Value *LengthVal = CGF.getTypeSize(BaseTy);
      llvm::Value *LBVal = CGF.EmitScalarExpr(OAE->getLowerBound());
      LBVal = CGF.EmitScalarConversion(LBVal, OAE->getLowerBound()->getType(),
                                       CGF.getContext().getSizeType(),
                                       OAE->getLowerBound()->getExprLoc());
      LBVal = CGF.Builder.CreateNUWMul(LBVal, ElemSize);
      llvm::Value *Cmp = CGF.Builder.CreateICmpUGT(LengthVal, LBVal);
      llvm::Value *TrueVal = CGF.Builder.CreateNUWSub(LengthVal, LBVal);
      LengthVal = CGF.Builder.CreateSelect(
          Cmp, TrueVal, llvm::ConstantInt::get(CGF.SizeTy, 0));
      return LengthVal;
    }
    return CGF.getTypeSize(ExprTy);
  }

  /// Return the corresponding bits for a given map clause modifier. Add
  /// a flag marking the map as a pointer if requested. Add a flag marking the
  /// map as the first one of a series of maps that relate to the same map
  /// expression.
  OpenMPOffloadMappingFlags getMapTypeBits(
      OpenMPMapClauseKind MapType, ArrayRef<OpenMPMapModifierKind> MapModifiers,
      ArrayRef<OpenMPMotionModifierKind> MotionModifiers, bool IsImplicit,
      bool AddPtrFlag, bool AddIsTargetParamFlag, bool IsNonContiguous) const {
    OpenMPOffloadMappingFlags Bits =
        IsImplicit ? OMP_MAP_IMPLICIT : OMP_MAP_NONE;
    switch (MapType) {
    case OMPC_MAP_alloc:
    case OMPC_MAP_release:
      // alloc and release is the default behavior in the runtime library,  i.e.
      // if we don't pass any bits alloc/release that is what the runtime is
      // going to do. Therefore, we don't need to signal anything for these two
      // type modifiers.
      break;
    case OMPC_MAP_to:
      Bits |= OMP_MAP_TO;
      break;
    case OMPC_MAP_from:
      Bits |= OMP_MAP_FROM;
      break;
    case OMPC_MAP_tofrom:
      Bits |= OMP_MAP_TO | OMP_MAP_FROM;
      break;
    case OMPC_MAP_delete:
      Bits |= OMP_MAP_DELETE;
      break;
    case OMPC_MAP_unknown:
      llvm_unreachable("Unexpected map type!");
    }
    if (AddPtrFlag)
      Bits |= OMP_MAP_PTR_AND_OBJ;
    if (AddIsTargetParamFlag)
      Bits |= OMP_MAP_TARGET_PARAM;
    if (llvm::is_contained(MapModifiers, OMPC_MAP_MODIFIER_always))
      Bits |= OMP_MAP_ALWAYS;
    if (llvm::is_contained(MapModifiers, OMPC_MAP_MODIFIER_close))
      Bits |= OMP_MAP_CLOSE;
    if (llvm::is_contained(MapModifiers, OMPC_MAP_MODIFIER_present) ||
        llvm::is_contained(MotionModifiers, OMPC_MOTION_MODIFIER_present))
      Bits |= OMP_MAP_PRESENT;
    if (llvm::is_contained(MapModifiers, OMPC_MAP_MODIFIER_ompx_hold))
      Bits |= OMP_MAP_OMPX_HOLD;
    if (IsNonContiguous)
      Bits |= OMP_MAP_NON_CONTIG;
    return Bits;
  }

  /// Return true if the provided expression is a final array section. A
  /// final array section, is one whose length can't be proved to be one.
  bool isFinalArraySectionExpression(const Expr *E) const {
    const auto *OASE = dyn_cast<OMPArraySectionExpr>(E);

    // It is not an array section and therefore not a unity-size one.
    if (!OASE)
      return false;

    // An array section with no colon always refer to a single element.
    if (OASE->getColonLocFirst().isInvalid())
      return false;

    const Expr *Length = OASE->getLength();

    // If we don't have a length we have to check if the array has size 1
    // for this dimension. Also, we should always expect a length if the
    // base type is pointer.
    if (!Length) {
      QualType BaseQTy = OMPArraySectionExpr::getBaseOriginalType(
                             OASE->getBase()->IgnoreParenImpCasts())
                             .getCanonicalType();
      if (const auto *ATy = dyn_cast<ConstantArrayType>(BaseQTy.getTypePtr()))
        return ATy->getSize().getSExtValue() != 1;
      // If we don't have a constant dimension length, we have to consider
      // the current section as having any size, so it is not necessarily
      // unitary. If it happen to be unity size, that's user fault.
      return true;
    }

    // Check if the length evaluates to 1.
    Expr::EvalResult Result;
    if (!Length->EvaluateAsInt(Result, CGF.getContext()))
      return true; // Can have more that size 1.

    llvm::APSInt ConstLength = Result.Val.getInt();
    return ConstLength.getSExtValue() != 1;
  }

  /// Generate the base pointers, section pointers, sizes, map type bits, and
  /// user-defined mappers (all included in \a CombinedInfo) for the provided
  /// map type, map or motion modifiers, and expression components.
  /// \a IsFirstComponent should be set to true if the provided set of
  /// components is the first associated with a capture.
  void generateInfoForComponentList(
      OpenMPMapClauseKind MapType, ArrayRef<OpenMPMapModifierKind> MapModifiers,
      ArrayRef<OpenMPMotionModifierKind> MotionModifiers,
      OMPClauseMappableExprCommon::MappableExprComponentListRef Components,
      MapCombinedInfoTy &CombinedInfo, StructRangeInfoTy &PartialStruct,
      bool IsFirstComponentList, bool IsImplicit,
      const ValueDecl *Mapper = nullptr, bool ForDeviceAddr = false,
      const ValueDecl *BaseDecl = nullptr, const Expr *MapExpr = nullptr,
      ArrayRef<OMPClauseMappableExprCommon::MappableExprComponentListRef>
          OverlappedElements = llvm::None) const {
    // The following summarizes what has to be generated for each map and the
    // types below. The generated information is expressed in this order:
    // base pointer, section pointer, size, flags
    // (to add to the ones that come from the map type and modifier).
    //
    // double d;
    // int i[100];
    // float *p;
    //
    // struct S1 {
    //   int i;
    //   float f[50];
    // }
    // struct S2 {
    //   int i;
    //   float f[50];
    //   S1 s;
    //   double *p;
    //   struct S2 *ps;
    //   int &ref;
    // }
    // S2 s;
    // S2 *ps;
    //
    // map(d)
    // &d, &d, sizeof(double), TARGET_PARAM | TO | FROM
    //
    // map(i)
    // &i, &i, 100*sizeof(int), TARGET_PARAM | TO | FROM
    //
    // map(i[1:23])
    // &i(=&i[0]), &i[1], 23*sizeof(int), TARGET_PARAM | TO | FROM
    //
    // map(p)
    // &p, &p, sizeof(float*), TARGET_PARAM | TO | FROM
    //
    // map(p[1:24])
    // &p, &p[1], 24*sizeof(float), TARGET_PARAM | TO | FROM | PTR_AND_OBJ
    // in unified shared memory mode or for local pointers
    // p, &p[1], 24*sizeof(float), TARGET_PARAM | TO | FROM
    //
    // map(s)
    // &s, &s, sizeof(S2), TARGET_PARAM | TO | FROM
    //
    // map(s.i)
    // &s, &(s.i), sizeof(int), TARGET_PARAM | TO | FROM
    //
    // map(s.s.f)
    // &s, &(s.s.f[0]), 50*sizeof(float), TARGET_PARAM | TO | FROM
    //
    // map(s.p)
    // &s, &(s.p), sizeof(double*), TARGET_PARAM | TO | FROM
    //
    // map(to: s.p[:22])
    // &s, &(s.p), sizeof(double*), TARGET_PARAM (*)
    // &s, &(s.p), sizeof(double*), MEMBER_OF(1) (**)
    // &(s.p), &(s.p[0]), 22*sizeof(double),
    //   MEMBER_OF(1) | PTR_AND_OBJ | TO (***)
    // (*) alloc space for struct members, only this is a target parameter
    // (**) map the pointer (nothing to be mapped in this example) (the compiler
    //      optimizes this entry out, same in the examples below)
    // (***) map the pointee (map: to)
    //
    // map(to: s.ref)
    // &s, &(s.ref), sizeof(int*), TARGET_PARAM (*)
    // &s, &(s.ref), sizeof(int), MEMBER_OF(1) | PTR_AND_OBJ | TO (***)
    // (*) alloc space for struct members, only this is a target parameter
    // (**) map the pointer (nothing to be mapped in this example) (the compiler
    //      optimizes this entry out, same in the examples below)
    // (***) map the pointee (map: to)
    //
    // map(s.ps)
    // &s, &(s.ps), sizeof(S2*), TARGET_PARAM | TO | FROM
    //
    // map(from: s.ps->s.i)
    // &s, &(s.ps), sizeof(S2*), TARGET_PARAM
    // &s, &(s.ps), sizeof(S2*), MEMBER_OF(1)
    // &(s.ps), &(s.ps->s.i), sizeof(int), MEMBER_OF(1) | PTR_AND_OBJ  | FROM
    //
    // map(to: s.ps->ps)
    // &s, &(s.ps), sizeof(S2*), TARGET_PARAM
    // &s, &(s.ps), sizeof(S2*), MEMBER_OF(1)
    // &(s.ps), &(s.ps->ps), sizeof(S2*), MEMBER_OF(1) | PTR_AND_OBJ  | TO
    //
    // map(s.ps->ps->ps)
    // &s, &(s.ps), sizeof(S2*), TARGET_PARAM
    // &s, &(s.ps), sizeof(S2*), MEMBER_OF(1)
    // &(s.ps), &(s.ps->ps), sizeof(S2*), MEMBER_OF(1) | PTR_AND_OBJ
    // &(s.ps->ps), &(s.ps->ps->ps), sizeof(S2*), PTR_AND_OBJ | TO | FROM
    //
    // map(to: s.ps->ps->s.f[:22])
    // &s, &(s.ps), sizeof(S2*), TARGET_PARAM
    // &s, &(s.ps), sizeof(S2*), MEMBER_OF(1)
    // &(s.ps), &(s.ps->ps), sizeof(S2*), MEMBER_OF(1) | PTR_AND_OBJ
    // &(s.ps->ps), &(s.ps->ps->s.f[0]), 22*sizeof(float), PTR_AND_OBJ | TO
    //
    // map(ps)
    // &ps, &ps, sizeof(S2*), TARGET_PARAM | TO | FROM
    //
    // map(ps->i)
    // ps, &(ps->i), sizeof(int), TARGET_PARAM | TO | FROM
    //
    // map(ps->s.f)
    // ps, &(ps->s.f[0]), 50*sizeof(float), TARGET_PARAM | TO | FROM
    //
    // map(from: ps->p)
    // ps, &(ps->p), sizeof(double*), TARGET_PARAM | FROM
    //
    // map(to: ps->p[:22])
    // ps, &(ps->p), sizeof(double*), TARGET_PARAM
    // ps, &(ps->p), sizeof(double*), MEMBER_OF(1)
    // &(ps->p), &(ps->p[0]), 22*sizeof(double), MEMBER_OF(1) | PTR_AND_OBJ | TO
    //
    // map(ps->ps)
    // ps, &(ps->ps), sizeof(S2*), TARGET_PARAM | TO | FROM
    //
    // map(from: ps->ps->s.i)
    // ps, &(ps->ps), sizeof(S2*), TARGET_PARAM
    // ps, &(ps->ps), sizeof(S2*), MEMBER_OF(1)
    // &(ps->ps), &(ps->ps->s.i), sizeof(int), MEMBER_OF(1) | PTR_AND_OBJ | FROM
    //
    // map(from: ps->ps->ps)
    // ps, &(ps->ps), sizeof(S2*), TARGET_PARAM
    // ps, &(ps->ps), sizeof(S2*), MEMBER_OF(1)
    // &(ps->ps), &(ps->ps->ps), sizeof(S2*), MEMBER_OF(1) | PTR_AND_OBJ | FROM
    //
    // map(ps->ps->ps->ps)
    // ps, &(ps->ps), sizeof(S2*), TARGET_PARAM
    // ps, &(ps->ps), sizeof(S2*), MEMBER_OF(1)
    // &(ps->ps), &(ps->ps->ps), sizeof(S2*), MEMBER_OF(1) | PTR_AND_OBJ
    // &(ps->ps->ps), &(ps->ps->ps->ps), sizeof(S2*), PTR_AND_OBJ | TO | FROM
    //
    // map(to: ps->ps->ps->s.f[:22])
    // ps, &(ps->ps), sizeof(S2*), TARGET_PARAM
    // ps, &(ps->ps), sizeof(S2*), MEMBER_OF(1)
    // &(ps->ps), &(ps->ps->ps), sizeof(S2*), MEMBER_OF(1) | PTR_AND_OBJ
    // &(ps->ps->ps), &(ps->ps->ps->s.f[0]), 22*sizeof(float), PTR_AND_OBJ | TO
    //
    // map(to: s.f[:22]) map(from: s.p[:33])
    // &s, &(s.f[0]), 50*sizeof(float) + sizeof(struct S1) +
    //     sizeof(double*) (**), TARGET_PARAM
    // &s, &(s.f[0]), 22*sizeof(float), MEMBER_OF(1) | TO
    // &s, &(s.p), sizeof(double*), MEMBER_OF(1)
    // &(s.p), &(s.p[0]), 33*sizeof(double), MEMBER_OF(1) | PTR_AND_OBJ | FROM
    // (*) allocate contiguous space needed to fit all mapped members even if
    //     we allocate space for members not mapped (in this example,
    //     s.f[22..49] and s.s are not mapped, yet we must allocate space for
    //     them as well because they fall between &s.f[0] and &s.p)
    //
    // map(from: s.f[:22]) map(to: ps->p[:33])
    // &s, &(s.f[0]), 22*sizeof(float), TARGET_PARAM | FROM
    // ps, &(ps->p), sizeof(S2*), TARGET_PARAM
    // ps, &(ps->p), sizeof(double*), MEMBER_OF(2) (*)
    // &(ps->p), &(ps->p[0]), 33*sizeof(double), MEMBER_OF(2) | PTR_AND_OBJ | TO
    // (*) the struct this entry pertains to is the 2nd element in the list of
    //     arguments, hence MEMBER_OF(2)
    //
    // map(from: s.f[:22], s.s) map(to: ps->p[:33])
    // &s, &(s.f[0]), 50*sizeof(float) + sizeof(struct S1), TARGET_PARAM
    // &s, &(s.f[0]), 22*sizeof(float), MEMBER_OF(1) | FROM
    // &s, &(s.s), sizeof(struct S1), MEMBER_OF(1) | FROM
    // ps, &(ps->p), sizeof(S2*), TARGET_PARAM
    // ps, &(ps->p), sizeof(double*), MEMBER_OF(4) (*)
    // &(ps->p), &(ps->p[0]), 33*sizeof(double), MEMBER_OF(4) | PTR_AND_OBJ | TO
    // (*) the struct this entry pertains to is the 4th element in the list
    //     of arguments, hence MEMBER_OF(4)

    // Track if the map information being generated is the first for a capture.
    bool IsCaptureFirstInfo = IsFirstComponentList;
    // When the variable is on a declare target link or in a to clause with
    // unified memory, a reference is needed to hold the host/device address
    // of the variable.
    bool RequiresReference = false;

    // Scan the components from the base to the complete expression.
    auto CI = Components.rbegin();
    auto CE = Components.rend();
    auto I = CI;

    // Track if the map information being generated is the first for a list of
    // components.
    bool IsExpressionFirstInfo = true;
    bool FirstPointerInComplexData = false;
    Address BP = Address::invalid();
    const Expr *AssocExpr = I->getAssociatedExpression();
    const auto *AE = dyn_cast<ArraySubscriptExpr>(AssocExpr);
    const auto *OASE = dyn_cast<OMPArraySectionExpr>(AssocExpr);
    const auto *OAShE = dyn_cast<OMPArrayShapingExpr>(AssocExpr);

    if (isa<MemberExpr>(AssocExpr)) {
      // The base is the 'this' pointer. The content of the pointer is going
      // to be the base of the field being mapped.
      BP = CGF.LoadCXXThisAddress();
    } else if ((AE && isa<CXXThisExpr>(AE->getBase()->IgnoreParenImpCasts())) ||
               (OASE &&
                isa<CXXThisExpr>(OASE->getBase()->IgnoreParenImpCasts()))) {
      BP = CGF.EmitOMPSharedLValue(AssocExpr).getAddress(CGF);
    } else if (OAShE &&
               isa<CXXThisExpr>(OAShE->getBase()->IgnoreParenCasts())) {
      BP = Address(
          CGF.EmitScalarExpr(OAShE->getBase()),
          CGF.getContext().getTypeAlignInChars(OAShE->getBase()->getType()));
    } else {
      // The base is the reference to the variable.
      // BP = &Var.
      BP = CGF.EmitOMPSharedLValue(AssocExpr).getAddress(CGF);
      if (const auto *VD =
              dyn_cast_or_null<VarDecl>(I->getAssociatedDeclaration())) {
        if (llvm::Optional<OMPDeclareTargetDeclAttr::MapTypeTy> Res =
                OMPDeclareTargetDeclAttr::isDeclareTargetDeclaration(VD)) {
          if ((*Res == OMPDeclareTargetDeclAttr::MT_Link) ||
              (*Res == OMPDeclareTargetDeclAttr::MT_To &&
               CGF.CGM.getOpenMPRuntime().hasRequiresUnifiedSharedMemory())) {
            RequiresReference = true;
            BP = CGF.CGM.getOpenMPRuntime().getAddrOfDeclareTargetVar(VD);
          }
        }
      }

      // If the variable is a pointer and is being dereferenced (i.e. is not
      // the last component), the base has to be the pointer itself, not its
      // reference. References are ignored for mapping purposes.
      QualType Ty =
          I->getAssociatedDeclaration()->getType().getNonReferenceType();
      if (Ty->isAnyPointerType() && std::next(I) != CE) {
        // No need to generate individual map information for the pointer, it
        // can be associated with the combined storage if shared memory mode is
        // active or the base declaration is not global variable.
        const auto *VD = dyn_cast<VarDecl>(I->getAssociatedDeclaration());
        if (CGF.CGM.getOpenMPRuntime().hasRequiresUnifiedSharedMemory() ||
            !VD || VD->hasLocalStorage())
          BP = CGF.EmitLoadOfPointer(BP, Ty->castAs<PointerType>());
        else
          FirstPointerInComplexData = true;
        ++I;
      }
    }

    // Track whether a component of the list should be marked as MEMBER_OF some
    // combined entry (for partial structs). Only the first PTR_AND_OBJ entry
    // in a component list should be marked as MEMBER_OF, all subsequent entries
    // do not belong to the base struct. E.g.
    // struct S2 s;
    // s.ps->ps->ps->f[:]
    //   (1) (2) (3) (4)
    // ps(1) is a member pointer, ps(2) is a pointee of ps(1), so it is a
    // PTR_AND_OBJ entry; the PTR is ps(1), so MEMBER_OF the base struct. ps(3)
    // is the pointee of ps(2) which is not member of struct s, so it should not
    // be marked as such (it is still PTR_AND_OBJ).
    // The variable is initialized to false so that PTR_AND_OBJ entries which
    // are not struct members are not considered (e.g. array of pointers to
    // data).
    bool ShouldBeMemberOf = false;

    // Variable keeping track of whether or not we have encountered a component
    // in the component list which is a member expression. Useful when we have a
    // pointer or a final array section, in which case it is the previous
    // component in the list which tells us whether we have a member expression.
    // E.g. X.f[:]
    // While processing the final array section "[:]" it is "f" which tells us
    // whether we are dealing with a member of a declared struct.
    const MemberExpr *EncounteredME = nullptr;

    // Track for the total number of dimension. Start from one for the dummy
    // dimension.
    uint64_t DimSize = 1;

    bool IsNonContiguous = CombinedInfo.NonContigInfo.IsNonContiguous;
    bool IsPrevMemberReference = false;

    for (; I != CE; ++I) {
      // If the current component is member of a struct (parent struct) mark it.
      if (!EncounteredME) {
        EncounteredME = dyn_cast<MemberExpr>(I->getAssociatedExpression());
        // If we encounter a PTR_AND_OBJ entry from now on it should be marked
        // as MEMBER_OF the parent struct.
        if (EncounteredME) {
          ShouldBeMemberOf = true;
          // Do not emit as complex pointer if this is actually not array-like
          // expression.
          if (FirstPointerInComplexData) {
            QualType Ty = std::prev(I)
                              ->getAssociatedDeclaration()
                              ->getType()
                              .getNonReferenceType();
            BP = CGF.EmitLoadOfPointer(BP, Ty->castAs<PointerType>());
            FirstPointerInComplexData = false;
          }
        }
      }

      auto Next = std::next(I);

      // We need to generate the addresses and sizes if this is the last
      // component, if the component is a pointer or if it is an array section
      // whose length can't be proved to be one. If this is a pointer, it
      // becomes the base address for the following components.

      // A final array section, is one whose length can't be proved to be one.
      // If the map item is non-contiguous then we don't treat any array section
      // as final array section.
      bool IsFinalArraySection =
          !IsNonContiguous &&
          isFinalArraySectionExpression(I->getAssociatedExpression());

      // If we have a declaration for the mapping use that, otherwise use
      // the base declaration of the map clause.
      const ValueDecl *MapDecl = (I->getAssociatedDeclaration())
                                     ? I->getAssociatedDeclaration()
                                     : BaseDecl;
      MapExpr = (I->getAssociatedExpression()) ? I->getAssociatedExpression()
                                               : MapExpr;

      // Get information on whether the element is a pointer. Have to do a
      // special treatment for array sections given that they are built-in
      // types.
      const auto *OASE =
          dyn_cast<OMPArraySectionExpr>(I->getAssociatedExpression());
      const auto *OAShE =
          dyn_cast<OMPArrayShapingExpr>(I->getAssociatedExpression());
      const auto *UO = dyn_cast<UnaryOperator>(I->getAssociatedExpression());
      const auto *BO = dyn_cast<BinaryOperator>(I->getAssociatedExpression());
      bool IsPointer =
          OAShE ||
          (OASE && OMPArraySectionExpr::getBaseOriginalType(OASE)
                       .getCanonicalType()
                       ->isAnyPointerType()) ||
          I->getAssociatedExpression()->getType()->isAnyPointerType();
      bool IsMemberReference = isa<MemberExpr>(I->getAssociatedExpression()) &&
                               MapDecl &&
                               MapDecl->getType()->isLValueReferenceType();
      bool IsNonDerefPointer = IsPointer && !UO && !BO && !IsNonContiguous;

      if (OASE)
        ++DimSize;

      if (Next == CE || IsMemberReference || IsNonDerefPointer ||
          IsFinalArraySection) {
        // If this is not the last component, we expect the pointer to be
        // associated with an array expression or member expression.
        assert((Next == CE ||
                isa<MemberExpr>(Next->getAssociatedExpression()) ||
                isa<ArraySubscriptExpr>(Next->getAssociatedExpression()) ||
                isa<OMPArraySectionExpr>(Next->getAssociatedExpression()) ||
                isa<OMPArrayShapingExpr>(Next->getAssociatedExpression()) ||
                isa<UnaryOperator>(Next->getAssociatedExpression()) ||
                isa<BinaryOperator>(Next->getAssociatedExpression())) &&
               "Unexpected expression");

        Address LB = Address::invalid();
        Address LowestElem = Address::invalid();
        auto &&EmitMemberExprBase = [](CodeGenFunction &CGF,
                                       const MemberExpr *E) {
          const Expr *BaseExpr = E->getBase();
          // If this is s.x, emit s as an lvalue.  If it is s->x, emit s as a
          // scalar.
          LValue BaseLV;
          if (E->isArrow()) {
            LValueBaseInfo BaseInfo;
            TBAAAccessInfo TBAAInfo;
            Address Addr =
                CGF.EmitPointerWithAlignment(BaseExpr, &BaseInfo, &TBAAInfo);
            QualType PtrTy = BaseExpr->getType()->getPointeeType();
            BaseLV = CGF.MakeAddrLValue(Addr, PtrTy, BaseInfo, TBAAInfo);
          } else {
            BaseLV = CGF.EmitOMPSharedLValue(BaseExpr);
          }
          return BaseLV;
        };
        if (OAShE) {
          LowestElem = LB = Address(CGF.EmitScalarExpr(OAShE->getBase()),
                                    CGF.getContext().getTypeAlignInChars(
                                        OAShE->getBase()->getType()));
        } else if (IsMemberReference) {
          const auto *ME = cast<MemberExpr>(I->getAssociatedExpression());
          LValue BaseLVal = EmitMemberExprBase(CGF, ME);
          LowestElem = CGF.EmitLValueForFieldInitialization(
                              BaseLVal, cast<FieldDecl>(MapDecl))
                           .getAddress(CGF);
          LB = CGF.EmitLoadOfReferenceLValue(LowestElem, MapDecl->getType())
                   .getAddress(CGF);
        } else {
          LowestElem = LB =
              CGF.EmitOMPSharedLValue(I->getAssociatedExpression())
                  .getAddress(CGF);
        }

        // If this component is a pointer inside the base struct then we don't
        // need to create any entry for it - it will be combined with the object
        // it is pointing to into a single PTR_AND_OBJ entry.
        bool IsMemberPointerOrAddr =
            EncounteredME &&
            (((IsPointer || ForDeviceAddr) &&
              I->getAssociatedExpression() == EncounteredME) ||
             (IsPrevMemberReference && !IsPointer) ||
             (IsMemberReference && Next != CE &&
              !Next->getAssociatedExpression()->getType()->isPointerType()));
        if (!OverlappedElements.empty() && Next == CE) {
          // Handle base element with the info for overlapped elements.
          assert(!PartialStruct.Base.isValid() && "The base element is set.");
          assert(!IsPointer &&
                 "Unexpected base element with the pointer type.");
          // Mark the whole struct as the struct that requires allocation on the
          // device.
          PartialStruct.LowestElem = {0, LowestElem};
          CharUnits TypeSize = CGF.getContext().getTypeSizeInChars(
              I->getAssociatedExpression()->getType());
          Address HB = CGF.Builder.CreateConstGEP(
              CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(LowestElem,
                                                              CGF.VoidPtrTy),
              TypeSize.getQuantity() - 1);
          PartialStruct.HighestElem = {
              std::numeric_limits<decltype(
                  PartialStruct.HighestElem.first)>::max(),
              HB};
          PartialStruct.Base = BP;
          PartialStruct.LB = LB;
          assert(
              PartialStruct.PreliminaryMapData.BasePointers.empty() &&
              "Overlapped elements must be used only once for the variable.");
          std::swap(PartialStruct.PreliminaryMapData, CombinedInfo);
          // Emit data for non-overlapped data.
          OpenMPOffloadMappingFlags Flags =
              OMP_MAP_MEMBER_OF |
              getMapTypeBits(MapType, MapModifiers, MotionModifiers, IsImplicit,
                             /*AddPtrFlag=*/false,
                             /*AddIsTargetParamFlag=*/false, IsNonContiguous);
          llvm::Value *Size = nullptr;
          // Do bitcopy of all non-overlapped structure elements.
          for (OMPClauseMappableExprCommon::MappableExprComponentListRef
                   Component : OverlappedElements) {
            Address ComponentLB = Address::invalid();
            for (const OMPClauseMappableExprCommon::MappableComponent &MC :
                 Component) {
              if (const ValueDecl *VD = MC.getAssociatedDeclaration()) {
                const auto *FD = dyn_cast<FieldDecl>(VD);
                if (FD && FD->getType()->isLValueReferenceType()) {
                  const auto *ME =
                      cast<MemberExpr>(MC.getAssociatedExpression());
                  LValue BaseLVal = EmitMemberExprBase(CGF, ME);
                  ComponentLB =
                      CGF.EmitLValueForFieldInitialization(BaseLVal, FD)
                          .getAddress(CGF);
                } else {
                  ComponentLB =
                      CGF.EmitOMPSharedLValue(MC.getAssociatedExpression())
                          .getAddress(CGF);
                }
                Size = CGF.Builder.CreatePtrDiff(
                    CGF.EmitCastToVoidPtr(ComponentLB.getPointer()),
                    CGF.EmitCastToVoidPtr(LB.getPointer()));
                break;
              }
            }
            assert(Size && "Failed to determine structure size");
            CombinedInfo.Exprs.emplace_back(MapDecl, MapExpr);
            CombinedInfo.BasePointers.push_back(BP.getPointer());
            CombinedInfo.Pointers.push_back(LB.getPointer());
            CombinedInfo.Sizes.push_back(CGF.Builder.CreateIntCast(
                Size, CGF.Int64Ty, /*isSigned=*/true));
            CombinedInfo.Types.push_back(Flags);
            CombinedInfo.Mappers.push_back(nullptr);
            CombinedInfo.NonContigInfo.Dims.push_back(IsNonContiguous ? DimSize
                                                                      : 1);
            LB = CGF.Builder.CreateConstGEP(ComponentLB, 1);
          }
          CombinedInfo.Exprs.emplace_back(MapDecl, MapExpr);
          CombinedInfo.BasePointers.push_back(BP.getPointer());
          CombinedInfo.Pointers.push_back(LB.getPointer());
          Size = CGF.Builder.CreatePtrDiff(
              CGF.Builder.CreateConstGEP(HB, 1).getPointer(),
              CGF.EmitCastToVoidPtr(LB.getPointer()));
          CombinedInfo.Sizes.push_back(
              CGF.Builder.CreateIntCast(Size, CGF.Int64Ty, /*isSigned=*/true));
          CombinedInfo.Types.push_back(Flags);
          CombinedInfo.Mappers.push_back(nullptr);
          CombinedInfo.NonContigInfo.Dims.push_back(IsNonContiguous ? DimSize
                                                                    : 1);
          break;
        }
        llvm::Value *Size = getExprTypeSize(I->getAssociatedExpression());
        if (!IsMemberPointerOrAddr ||
            (Next == CE && MapType != OMPC_MAP_unknown)) {
          CombinedInfo.Exprs.emplace_back(MapDecl, MapExpr);
          CombinedInfo.BasePointers.push_back(BP.getPointer());
          CombinedInfo.Pointers.push_back(LB.getPointer());
          CombinedInfo.Sizes.push_back(
              CGF.Builder.CreateIntCast(Size, CGF.Int64Ty, /*isSigned=*/true));
          CombinedInfo.NonContigInfo.Dims.push_back(IsNonContiguous ? DimSize
                                                                    : 1);

          // If Mapper is valid, the last component inherits the mapper.
          bool HasMapper = Mapper && Next == CE;
          CombinedInfo.Mappers.push_back(HasMapper ? Mapper : nullptr);

          // We need to add a pointer flag for each map that comes from the
          // same expression except for the first one. We also need to signal
          // this map is the first one that relates with the current capture
          // (there is a set of entries for each capture).
          OpenMPOffloadMappingFlags Flags = getMapTypeBits(
              MapType, MapModifiers, MotionModifiers, IsImplicit,
              !IsExpressionFirstInfo || RequiresReference ||
                  FirstPointerInComplexData || IsMemberReference,
              IsCaptureFirstInfo && !RequiresReference, IsNonContiguous);

          if (!IsExpressionFirstInfo || IsMemberReference) {
            // If we have a PTR_AND_OBJ pair where the OBJ is a pointer as well,
            // then we reset the TO/FROM/ALWAYS/DELETE/CLOSE flags.
            if (IsPointer || (IsMemberReference && Next != CE))
              Flags &= ~(OMP_MAP_TO | OMP_MAP_FROM | OMP_MAP_ALWAYS |
                         OMP_MAP_DELETE | OMP_MAP_CLOSE);

            if (ShouldBeMemberOf) {
              // Set placeholder value MEMBER_OF=FFFF to indicate that the flag
              // should be later updated with the correct value of MEMBER_OF.
              Flags |= OMP_MAP_MEMBER_OF;
              // From now on, all subsequent PTR_AND_OBJ entries should not be
              // marked as MEMBER_OF.
              ShouldBeMemberOf = false;
            }
          }

          CombinedInfo.Types.push_back(Flags);
        }

        // If we have encountered a member expression so far, keep track of the
        // mapped member. If the parent is "*this", then the value declaration
        // is nullptr.
        if (EncounteredME) {
          const auto *FD = cast<FieldDecl>(EncounteredME->getMemberDecl());
          unsigned FieldIndex = FD->getFieldIndex();

          // Update info about the lowest and highest elements for this struct
          if (!PartialStruct.Base.isValid()) {
            PartialStruct.LowestElem = {FieldIndex, LowestElem};
            if (IsFinalArraySection) {
              Address HB =
                  CGF.EmitOMPArraySectionExpr(OASE, /*IsLowerBound=*/false)
                      .getAddress(CGF);
              PartialStruct.HighestElem = {FieldIndex, HB};
            } else {
              PartialStruct.HighestElem = {FieldIndex, LowestElem};
            }
            PartialStruct.Base = BP;
            PartialStruct.LB = BP;
          } else if (FieldIndex < PartialStruct.LowestElem.first) {
            PartialStruct.LowestElem = {FieldIndex, LowestElem};
          } else if (FieldIndex > PartialStruct.HighestElem.first) {
            PartialStruct.HighestElem = {FieldIndex, LowestElem};
          }
        }

        // Need to emit combined struct for array sections.
        if (IsFinalArraySection || IsNonContiguous)
          PartialStruct.IsArraySection = true;

        // If we have a final array section, we are done with this expression.
        if (IsFinalArraySection)
          break;

        // The pointer becomes the base for the next element.
        if (Next != CE)
          BP = IsMemberReference ? LowestElem : LB;

        IsExpressionFirstInfo = false;
        IsCaptureFirstInfo = false;
        FirstPointerInComplexData = false;
        IsPrevMemberReference = IsMemberReference;
      } else if (FirstPointerInComplexData) {
        QualType Ty = Components.rbegin()
                          ->getAssociatedDeclaration()
                          ->getType()
                          .getNonReferenceType();
        BP = CGF.EmitLoadOfPointer(BP, Ty->castAs<PointerType>());
        FirstPointerInComplexData = false;
      }
    }
    // If ran into the whole component - allocate the space for the whole
    // record.
    if (!EncounteredME)
      PartialStruct.HasCompleteRecord = true;

    if (!IsNonContiguous)
      return;

    const ASTContext &Context = CGF.getContext();

    // For supporting stride in array section, we need to initialize the first
    // dimension size as 1, first offset as 0, and first count as 1
    MapValuesArrayTy CurOffsets = {llvm::ConstantInt::get(CGF.CGM.Int64Ty, 0)};
    MapValuesArrayTy CurCounts = {llvm::ConstantInt::get(CGF.CGM.Int64Ty, 1)};
    MapValuesArrayTy CurStrides;
    MapValuesArrayTy DimSizes{llvm::ConstantInt::get(CGF.CGM.Int64Ty, 1)};
    uint64_t ElementTypeSize;

    // Collect Size information for each dimension and get the element size as
    // the first Stride. For example, for `int arr[10][10]`, the DimSizes
    // should be [10, 10] and the first stride is 4 btyes.
    for (const OMPClauseMappableExprCommon::MappableComponent &Component :
         Components) {
      const Expr *AssocExpr = Component.getAssociatedExpression();
      const auto *OASE = dyn_cast<OMPArraySectionExpr>(AssocExpr);

      if (!OASE)
        continue;

      QualType Ty = OMPArraySectionExpr::getBaseOriginalType(OASE->getBase());
      auto *CAT = Context.getAsConstantArrayType(Ty);
      auto *VAT = Context.getAsVariableArrayType(Ty);

      // We need all the dimension size except for the last dimension.
      assert((VAT || CAT || &Component == &*Components.begin()) &&
             "Should be either ConstantArray or VariableArray if not the "
             "first Component");

      // Get element size if CurStrides is empty.
      if (CurStrides.empty()) {
        const Type *ElementType = nullptr;
        if (CAT)
          ElementType = CAT->getElementType().getTypePtr();
        else if (VAT)
          ElementType = VAT->getElementType().getTypePtr();
        else
          assert(&Component == &*Components.begin() &&
                 "Only expect pointer (non CAT or VAT) when this is the "
                 "first Component");
        // If ElementType is null, then it means the base is a pointer
        // (neither CAT nor VAT) and we'll attempt to get ElementType again
        // for next iteration.
        if (ElementType) {
          // For the case that having pointer as base, we need to remove one
          // level of indirection.
          if (&Component != &*Components.begin())
            ElementType = ElementType->getPointeeOrArrayElementType();
          ElementTypeSize =
              Context.getTypeSizeInChars(ElementType).getQuantity();
          CurStrides.push_back(
              llvm::ConstantInt::get(CGF.Int64Ty, ElementTypeSize));
        }
      }
      // Get dimension value except for the last dimension since we don't need
      // it.
      if (DimSizes.size() < Components.size() - 1) {
        if (CAT)
          DimSizes.push_back(llvm::ConstantInt::get(
              CGF.Int64Ty, CAT->getSize().getZExtValue()));
        else if (VAT)
          DimSizes.push_back(CGF.Builder.CreateIntCast(
              CGF.EmitScalarExpr(VAT->getSizeExpr()), CGF.Int64Ty,
              /*IsSigned=*/false));
      }
    }

    // Skip the dummy dimension since we have already have its information.
    auto DI = DimSizes.begin() + 1;
    // Product of dimension.
    llvm::Value *DimProd =
        llvm::ConstantInt::get(CGF.CGM.Int64Ty, ElementTypeSize);

    // Collect info for non-contiguous. Notice that offset, count, and stride
    // are only meaningful for array-section, so we insert a null for anything
    // other than array-section.
    // Also, the size of offset, count, and stride are not the same as
    // pointers, base_pointers, sizes, or dims. Instead, the size of offset,
    // count, and stride are the same as the number of non-contiguous
    // declaration in target update to/from clause.
    for (const OMPClauseMappableExprCommon::MappableComponent &Component :
         Components) {
      const Expr *AssocExpr = Component.getAssociatedExpression();

      if (const auto *AE = dyn_cast<ArraySubscriptExpr>(AssocExpr)) {
        llvm::Value *Offset = CGF.Builder.CreateIntCast(
            CGF.EmitScalarExpr(AE->getIdx()), CGF.Int64Ty,
            /*isSigned=*/false);
        CurOffsets.push_back(Offset);
        CurCounts.push_back(llvm::ConstantInt::get(CGF.Int64Ty, /*V=*/1));
        CurStrides.push_back(CurStrides.back());
        continue;
      }

      const auto *OASE = dyn_cast<OMPArraySectionExpr>(AssocExpr);

      if (!OASE)
        continue;

      // Offset
      const Expr *OffsetExpr = OASE->getLowerBound();
      llvm::Value *Offset = nullptr;
      if (!OffsetExpr) {
        // If offset is absent, then we just set it to zero.
        Offset = llvm::ConstantInt::get(CGF.Int64Ty, 0);
      } else {
        Offset = CGF.Builder.CreateIntCast(CGF.EmitScalarExpr(OffsetExpr),
                                           CGF.Int64Ty,
                                           /*isSigned=*/false);
      }
      CurOffsets.push_back(Offset);

      // Count
      const Expr *CountExpr = OASE->getLength();
      llvm::Value *Count = nullptr;
      if (!CountExpr) {
        // In Clang, once a high dimension is an array section, we construct all
        // the lower dimension as array section, however, for case like
        // arr[0:2][2], Clang construct the inner dimension as an array section
        // but it actually is not in an array section form according to spec.
        if (!OASE->getColonLocFirst().isValid() &&
            !OASE->getColonLocSecond().isValid()) {
          Count = llvm::ConstantInt::get(CGF.Int64Ty, 1);
        } else {
          // OpenMP 5.0, 2.1.5 Array Sections, Description.
          // When the length is absent it defaults to ⌈(size −
          // lower-bound)/stride⌉, where size is the size of the array
          // dimension.
          const Expr *StrideExpr = OASE->getStride();
          llvm::Value *Stride =
              StrideExpr
                  ? CGF.Builder.CreateIntCast(CGF.EmitScalarExpr(StrideExpr),
                                              CGF.Int64Ty, /*isSigned=*/false)
                  : nullptr;
          if (Stride)
            Count = CGF.Builder.CreateUDiv(
                CGF.Builder.CreateNUWSub(*DI, Offset), Stride);
          else
            Count = CGF.Builder.CreateNUWSub(*DI, Offset);
        }
      } else {
        Count = CGF.EmitScalarExpr(CountExpr);
      }
      Count = CGF.Builder.CreateIntCast(Count, CGF.Int64Ty, /*isSigned=*/false);
      CurCounts.push_back(Count);

      // Stride_n' = Stride_n * (D_0 * D_1 ... * D_n-1) * Unit size
      // Take `int arr[5][5][5]` and `arr[0:2:2][1:2:1][0:2:2]` as an example:
      //              Offset      Count     Stride
      //    D0          0           1         4    (int)    <- dummy dimension
      //    D1          0           2         8    (2 * (1) * 4)
      //    D2          1           2         20   (1 * (1 * 5) * 4)
      //    D3          0           2         200  (2 * (1 * 5 * 4) * 4)
      const Expr *StrideExpr = OASE->getStride();
      llvm::Value *Stride =
          StrideExpr
              ? CGF.Builder.CreateIntCast(CGF.EmitScalarExpr(StrideExpr),
                                          CGF.Int64Ty, /*isSigned=*/false)
              : nullptr;
      DimProd = CGF.Builder.CreateNUWMul(DimProd, *(DI - 1));
      if (Stride)
        CurStrides.push_back(CGF.Builder.CreateNUWMul(DimProd, Stride));
      else
        CurStrides.push_back(DimProd);
      if (DI != DimSizes.end())
        ++DI;
    }

    CombinedInfo.NonContigInfo.Offsets.push_back(CurOffsets);
    CombinedInfo.NonContigInfo.Counts.push_back(CurCounts);
    CombinedInfo.NonContigInfo.Strides.push_back(CurStrides);
  }

  /// Return the adjusted map modifiers if the declaration a capture refers to
  /// appears in a first-private clause. This is expected to be used only with
  /// directives that start with 'target'.
  MappableExprsHandler::OpenMPOffloadMappingFlags
  getMapModifiersForPrivateClauses(const CapturedStmt::Capture &Cap) const {
    assert(Cap.capturesVariable() && "Expected capture by reference only!");

    // A first private variable captured by reference will use only the
    // 'private ptr' and 'map to' flag. Return the right flags if the captured
    // declaration is known as first-private in this handler.
    if (FirstPrivateDecls.count(Cap.getCapturedVar())) {
      if (Cap.getCapturedVar()->getType()->isAnyPointerType())
        return MappableExprsHandler::OMP_MAP_TO |
               MappableExprsHandler::OMP_MAP_PTR_AND_OBJ;
      return MappableExprsHandler::OMP_MAP_PRIVATE |
             MappableExprsHandler::OMP_MAP_TO;
    }
    auto I = LambdasMap.find(Cap.getCapturedVar()->getCanonicalDecl());
    if (I != LambdasMap.end())
      // for map(to: lambda): using user specified map type.
      return getMapTypeBits(
          I->getSecond()->getMapType(), I->getSecond()->getMapTypeModifiers(),
          /*MotionModifiers=*/llvm::None, I->getSecond()->isImplicit(),
          /*AddPtrFlag=*/false,
          /*AddIsTargetParamFlag=*/false,
          /*isNonContiguous=*/false);
    return MappableExprsHandler::OMP_MAP_TO |
           MappableExprsHandler::OMP_MAP_FROM;
  }

  static OpenMPOffloadMappingFlags getMemberOfFlag(unsigned Position) {
    // Rotate by getFlagMemberOffset() bits.
    return static_cast<OpenMPOffloadMappingFlags>(((uint64_t)Position + 1)
                                                  << getFlagMemberOffset());
  }

  static void setCorrectMemberOfFlag(OpenMPOffloadMappingFlags &Flags,
                                     OpenMPOffloadMappingFlags MemberOfFlag) {
    // If the entry is PTR_AND_OBJ but has not been marked with the special
    // placeholder value 0xFFFF in the MEMBER_OF field, then it should not be
    // marked as MEMBER_OF.
    if ((Flags & OMP_MAP_PTR_AND_OBJ) &&
        ((Flags & OMP_MAP_MEMBER_OF) != OMP_MAP_MEMBER_OF))
      return;

    // Reset the placeholder value to prepare the flag for the assignment of the
    // proper MEMBER_OF value.
    Flags &= ~OMP_MAP_MEMBER_OF;
    Flags |= MemberOfFlag;
  }

  void getPlainLayout(const CXXRecordDecl *RD,
                      llvm::SmallVectorImpl<const FieldDecl *> &Layout,
                      bool AsBase) const {
    const CGRecordLayout &RL = CGF.getTypes().getCGRecordLayout(RD);

    llvm::StructType *St =
        AsBase ? RL.getBaseSubobjectLLVMType() : RL.getLLVMType();

    unsigned NumElements = St->getNumElements();
    llvm::SmallVector<
        llvm::PointerUnion<const CXXRecordDecl *, const FieldDecl *>, 4>
        RecordLayout(NumElements);

    // Fill bases.
    for (const auto &I : RD->bases()) {
      if (I.isVirtual())
        continue;
      const auto *Base = I.getType()->getAsCXXRecordDecl();
      // Ignore empty bases.
      if (Base->isEmpty() || CGF.getContext()
                                 .getASTRecordLayout(Base)
                                 .getNonVirtualSize()
                                 .isZero())
        continue;

      unsigned FieldIndex = RL.getNonVirtualBaseLLVMFieldNo(Base);
      RecordLayout[FieldIndex] = Base;
    }
    // Fill in virtual bases.
    for (const auto &I : RD->vbases()) {
      const auto *Base = I.getType()->getAsCXXRecordDecl();
      // Ignore empty bases.
      if (Base->isEmpty())
        continue;
      unsigned FieldIndex = RL.getVirtualBaseIndex(Base);
      if (RecordLayout[FieldIndex])
        continue;
      RecordLayout[FieldIndex] = Base;
    }
    // Fill in all the fields.
    assert(!RD->isUnion() && "Unexpected union.");
    for (const auto *Field : RD->fields()) {
      // Fill in non-bitfields. (Bitfields always use a zero pattern, which we
      // will fill in later.)
      if (!Field->isBitField() && !Field->isZeroSize(CGF.getContext())) {
        unsigned FieldIndex = RL.getLLVMFieldNo(Field);
        RecordLayout[FieldIndex] = Field;
      }
    }
    for (const llvm::PointerUnion<const CXXRecordDecl *, const FieldDecl *>
             &Data : RecordLayout) {
      if (Data.isNull())
        continue;
      if (const auto *Base = Data.dyn_cast<const CXXRecordDecl *>())
        getPlainLayout(Base, Layout, /*AsBase=*/true);
      else
        Layout.push_back(Data.get<const FieldDecl *>());
    }
  }

  /// Generate all the base pointers, section pointers, sizes, map types, and
  /// mappers for the extracted mappable expressions (all included in \a
  /// CombinedInfo). Also, for each item that relates with a device pointer, a
  /// pair of the relevant declaration and index where it occurs is appended to
  /// the device pointers info array.
  void generateAllInfoForClauses(
      ArrayRef<const OMPClause *> Clauses, MapCombinedInfoTy &CombinedInfo,
      const llvm::DenseSet<CanonicalDeclPtr<const Decl>> &SkipVarSet =
          llvm::DenseSet<CanonicalDeclPtr<const Decl>>()) const {
    // We have to process the component lists that relate with the same
    // declaration in a single chunk so that we can generate the map flags
    // correctly. Therefore, we organize all lists in a map.
    enum MapKind { Present, Allocs, Other, Total };
    llvm::MapVector<CanonicalDeclPtr<const Decl>,
                    SmallVector<SmallVector<MapInfo, 8>, 4>>
        Info;

    // Helper function to fill the information map for the different supported
    // clauses.
    auto &&InfoGen =
        [&Info, &SkipVarSet](
            const ValueDecl *D, MapKind Kind,
            OMPClauseMappableExprCommon::MappableExprComponentListRef L,
            OpenMPMapClauseKind MapType,
            ArrayRef<OpenMPMapModifierKind> MapModifiers,
            ArrayRef<OpenMPMotionModifierKind> MotionModifiers,
            bool ReturnDevicePointer, bool IsImplicit, const ValueDecl *Mapper,
            const Expr *VarRef = nullptr, bool ForDeviceAddr = false) {
          if (SkipVarSet.contains(D))
            return;
          auto It = Info.find(D);
          if (It == Info.end())
            It = Info
                     .insert(std::make_pair(
                         D, SmallVector<SmallVector<MapInfo, 8>, 4>(Total)))
                     .first;
          It->second[Kind].emplace_back(
              L, MapType, MapModifiers, MotionModifiers, ReturnDevicePointer,
              IsImplicit, Mapper, VarRef, ForDeviceAddr);
        };

    for (const auto *Cl : Clauses) {
      const auto *C = dyn_cast<OMPMapClause>(Cl);
      if (!C)
        continue;
      MapKind Kind = Other;
      if (llvm::is_contained(C->getMapTypeModifiers(),
                             OMPC_MAP_MODIFIER_present))
        Kind = Present;
      else if (C->getMapType() == OMPC_MAP_alloc)
        Kind = Allocs;
      const auto *EI = C->getVarRefs().begin();
      for (const auto L : C->component_lists()) {
        const Expr *E = (C->getMapLoc().isValid()) ? *EI : nullptr;
        InfoGen(std::get<0>(L), Kind, std::get<1>(L), C->getMapType(),
                C->getMapTypeModifiers(), llvm::None,
                /*ReturnDevicePointer=*/false, C->isImplicit(), std::get<2>(L),
                E);
        ++EI;
      }
    }
    for (const auto *Cl : Clauses) {
      const auto *C = dyn_cast<OMPToClause>(Cl);
      if (!C)
        continue;
      MapKind Kind = Other;
      if (llvm::is_contained(C->getMotionModifiers(),
                             OMPC_MOTION_MODIFIER_present))
        Kind = Present;
      const auto *EI = C->getVarRefs().begin();
      for (const auto L : C->component_lists()) {
        InfoGen(std::get<0>(L), Kind, std::get<1>(L), OMPC_MAP_to, llvm::None,
                C->getMotionModifiers(), /*ReturnDevicePointer=*/false,
                C->isImplicit(), std::get<2>(L), *EI);
        ++EI;
      }
    }
    for (const auto *Cl : Clauses) {
      const auto *C = dyn_cast<OMPFromClause>(Cl);
      if (!C)
        continue;
      MapKind Kind = Other;
      if (llvm::is_contained(C->getMotionModifiers(),
                             OMPC_MOTION_MODIFIER_present))
        Kind = Present;
      const auto *EI = C->getVarRefs().begin();
      for (const auto L : C->component_lists()) {
        InfoGen(std::get<0>(L), Kind, std::get<1>(L), OMPC_MAP_from, llvm::None,
                C->getMotionModifiers(), /*ReturnDevicePointer=*/false,
                C->isImplicit(), std::get<2>(L), *EI);
        ++EI;
      }
    }

    // Look at the use_device_ptr clause information and mark the existing map
    // entries as such. If there is no map information for an entry in the
    // use_device_ptr list, we create one with map type 'alloc' and zero size
    // section. It is the user fault if that was not mapped before. If there is
    // no map information and the pointer is a struct member, then we defer the
    // emission of that entry until the whole struct has been processed.
    llvm::MapVector<CanonicalDeclPtr<const Decl>,
                    SmallVector<DeferredDevicePtrEntryTy, 4>>
        DeferredInfo;
    MapCombinedInfoTy UseDevicePtrCombinedInfo;

    for (const auto *Cl : Clauses) {
      const auto *C = dyn_cast<OMPUseDevicePtrClause>(Cl);
      if (!C)
        continue;
      for (const auto L : C->component_lists()) {
        OMPClauseMappableExprCommon::MappableExprComponentListRef Components =
            std::get<1>(L);
        assert(!Components.empty() &&
               "Not expecting empty list of components!");
        const ValueDecl *VD = Components.back().getAssociatedDeclaration();
        VD = cast<ValueDecl>(VD->getCanonicalDecl());
        const Expr *IE = Components.back().getAssociatedExpression();
        // If the first component is a member expression, we have to look into
        // 'this', which maps to null in the map of map information. Otherwise
        // look directly for the information.
        auto It = Info.find(isa<MemberExpr>(IE) ? nullptr : VD);

        // We potentially have map information for this declaration already.
        // Look for the first set of components that refer to it.
        if (It != Info.end()) {
          bool Found = false;
          for (auto &Data : It->second) {
            auto *CI = llvm::find_if(Data, [VD](const MapInfo &MI) {
              return MI.Components.back().getAssociatedDeclaration() == VD;
            });
            // If we found a map entry, signal that the pointer has to be
            // returned and move on to the next declaration. Exclude cases where
            // the base pointer is mapped as array subscript, array section or
            // array shaping. The base address is passed as a pointer to base in
            // this case and cannot be used as a base for use_device_ptr list
            // item.
            if (CI != Data.end()) {
              auto PrevCI = std::next(CI->Components.rbegin());
              const auto *VarD = dyn_cast<VarDecl>(VD);
              if (CGF.CGM.getOpenMPRuntime().hasRequiresUnifiedSharedMemory() ||
                  isa<MemberExpr>(IE) ||
                  !VD->getType().getNonReferenceType()->isPointerType() ||
                  PrevCI == CI->Components.rend() ||
                  isa<MemberExpr>(PrevCI->getAssociatedExpression()) || !VarD ||
                  VarD->hasLocalStorage()) {
                CI->ReturnDevicePointer = true;
                Found = true;
                break;
              }
            }
          }
          if (Found)
            continue;
        }

        // We didn't find any match in our map information - generate a zero
        // size array section - if the pointer is a struct member we defer this
        // action until the whole struct has been processed.
        if (isa<MemberExpr>(IE)) {
          // Insert the pointer into Info to be processed by
          // generateInfoForComponentList. Because it is a member pointer
          // without a pointee, no entry will be generated for it, therefore
          // we need to generate one after the whole struct has been processed.
          // Nonetheless, generateInfoForComponentList must be called to take
          // the pointer into account for the calculation of the range of the
          // partial struct.
          InfoGen(nullptr, Other, Components, OMPC_MAP_unknown, llvm::None,
                  llvm::None, /*ReturnDevicePointer=*/false, C->isImplicit(),
                  nullptr);
          DeferredInfo[nullptr].emplace_back(IE, VD, /*ForDeviceAddr=*/false);
        } else {
          llvm::Value *Ptr =
              CGF.EmitLoadOfScalar(CGF.EmitLValue(IE), IE->getExprLoc());
          UseDevicePtrCombinedInfo.Exprs.push_back(VD);
          UseDevicePtrCombinedInfo.BasePointers.emplace_back(Ptr, VD);
          UseDevicePtrCombinedInfo.Pointers.push_back(Ptr);
          UseDevicePtrCombinedInfo.Sizes.push_back(
              llvm::Constant::getNullValue(CGF.Int64Ty));
          UseDevicePtrCombinedInfo.Types.push_back(OMP_MAP_RETURN_PARAM);
          UseDevicePtrCombinedInfo.Mappers.push_back(nullptr);
        }
      }
    }

    // Look at the use_device_addr clause information and mark the existing map
    // entries as such. If there is no map information for an entry in the
    // use_device_addr list, we create one with map type 'alloc' and zero size
    // section. It is the user fault if that was not mapped before. If there is
    // no map information and the pointer is a struct member, then we defer the
    // emission of that entry until the whole struct has been processed.
    llvm::SmallDenseSet<CanonicalDeclPtr<const Decl>, 4> Processed;
    for (const auto *Cl : Clauses) {
      const auto *C = dyn_cast<OMPUseDeviceAddrClause>(Cl);
      if (!C)
        continue;
      for (const auto L : C->component_lists()) {
        assert(!std::get<1>(L).empty() &&
               "Not expecting empty list of components!");
        const ValueDecl *VD = std::get<1>(L).back().getAssociatedDeclaration();
        if (!Processed.insert(VD).second)
          continue;
        VD = cast<ValueDecl>(VD->getCanonicalDecl());
        const Expr *IE = std::get<1>(L).back().getAssociatedExpression();
        // If the first component is a member expression, we have to look into
        // 'this', which maps to null in the map of map information. Otherwise
        // look directly for the information.
        auto It = Info.find(isa<MemberExpr>(IE) ? nullptr : VD);

        // We potentially have map information for this declaration already.
        // Look for the first set of components that refer to it.
        if (It != Info.end()) {
          bool Found = false;
          for (auto &Data : It->second) {
            auto *CI = llvm::find_if(Data, [VD](const MapInfo &MI) {
              return MI.Components.back().getAssociatedDeclaration() == VD;
            });
            // If we found a map entry, signal that the pointer has to be
            // returned and move on to the next declaration.
            if (CI != Data.end()) {
              CI->ReturnDevicePointer = true;
              Found = true;
              break;
            }
          }
          if (Found)
            continue;
        }

        // We didn't find any match in our map information - generate a zero
        // size array section - if the pointer is a struct member we defer this
        // action until the whole struct has been processed.
        if (isa<MemberExpr>(IE)) {
          // Insert the pointer into Info to be processed by
          // generateInfoForComponentList. Because it is a member pointer
          // without a pointee, no entry will be generated for it, therefore
          // we need to generate one after the whole struct has been processed.
          // Nonetheless, generateInfoForComponentList must be called to take
          // the pointer into account for the calculation of the range of the
          // partial struct.
          InfoGen(nullptr, Other, std::get<1>(L), OMPC_MAP_unknown, llvm::None,
                  llvm::None, /*ReturnDevicePointer=*/false, C->isImplicit(),
                  nullptr, nullptr, /*ForDeviceAddr=*/true);
          DeferredInfo[nullptr].emplace_back(IE, VD, /*ForDeviceAddr=*/true);
        } else {
          llvm::Value *Ptr;
          if (IE->isGLValue())
            Ptr = CGF.EmitLValue(IE).getPointer(CGF);
          else
            Ptr = CGF.EmitScalarExpr(IE);
          CombinedInfo.Exprs.push_back(VD);
          CombinedInfo.BasePointers.emplace_back(Ptr, VD);
          CombinedInfo.Pointers.push_back(Ptr);
          CombinedInfo.Sizes.push_back(
              llvm::Constant::getNullValue(CGF.Int64Ty));
          CombinedInfo.Types.push_back(OMP_MAP_RETURN_PARAM);
          CombinedInfo.Mappers.push_back(nullptr);
        }
      }
    }

    for (const auto &Data : Info) {
      StructRangeInfoTy PartialStruct;
      // Temporary generated information.
      MapCombinedInfoTy CurInfo;
      const Decl *D = Data.first;
      const ValueDecl *VD = cast_or_null<ValueDecl>(D);
      for (const auto &M : Data.second) {
        for (const MapInfo &L : M) {
          assert(!L.Components.empty() &&
                 "Not expecting declaration with no component lists.");

          // Remember the current base pointer index.
          unsigned CurrentBasePointersIdx = CurInfo.BasePointers.size();
          CurInfo.NonContigInfo.IsNonContiguous =
              L.Components.back().isNonContiguous();
          generateInfoForComponentList(
              L.MapType, L.MapModifiers, L.MotionModifiers, L.Components,
              CurInfo, PartialStruct, /*IsFirstComponentList=*/false,
              L.IsImplicit, L.Mapper, L.ForDeviceAddr, VD, L.VarRef);

          // If this entry relates with a device pointer, set the relevant
          // declaration and add the 'return pointer' flag.
          if (L.ReturnDevicePointer) {
            assert(CurInfo.BasePointers.size() > CurrentBasePointersIdx &&
                   "Unexpected number of mapped base pointers.");

            const ValueDecl *RelevantVD =
                L.Components.back().getAssociatedDeclaration();
            assert(RelevantVD &&
                   "No relevant declaration related with device pointer??");

            CurInfo.BasePointers[CurrentBasePointersIdx].setDevicePtrDecl(
                RelevantVD);
            CurInfo.Types[CurrentBasePointersIdx] |= OMP_MAP_RETURN_PARAM;
          }
        }
      }

      // Append any pending zero-length pointers which are struct members and
      // used with use_device_ptr or use_device_addr.
      auto CI = DeferredInfo.find(Data.first);
      if (CI != DeferredInfo.end()) {
        for (const DeferredDevicePtrEntryTy &L : CI->second) {
          llvm::Value *BasePtr;
          llvm::Value *Ptr;
          if (L.ForDeviceAddr) {
            if (L.IE->isGLValue())
              Ptr = this->CGF.EmitLValue(L.IE).getPointer(CGF);
            else
              Ptr = this->CGF.EmitScalarExpr(L.IE);
            BasePtr = Ptr;
            // Entry is RETURN_PARAM. Also, set the placeholder value
            // MEMBER_OF=FFFF so that the entry is later updated with the
            // correct value of MEMBER_OF.
            CurInfo.Types.push_back(OMP_MAP_RETURN_PARAM | OMP_MAP_MEMBER_OF);
          } else {
            BasePtr = this->CGF.EmitLValue(L.IE).getPointer(CGF);
            Ptr = this->CGF.EmitLoadOfScalar(this->CGF.EmitLValue(L.IE),
                                             L.IE->getExprLoc());
            // Entry is PTR_AND_OBJ and RETURN_PARAM. Also, set the
            // placeholder value MEMBER_OF=FFFF so that the entry is later
            // updated with the correct value of MEMBER_OF.
            CurInfo.Types.push_back(OMP_MAP_PTR_AND_OBJ | OMP_MAP_RETURN_PARAM |
                                    OMP_MAP_MEMBER_OF);
          }
          CurInfo.Exprs.push_back(L.VD);
          CurInfo.BasePointers.emplace_back(BasePtr, L.VD);
          CurInfo.Pointers.push_back(Ptr);
          CurInfo.Sizes.push_back(
              llvm::Constant::getNullValue(this->CGF.Int64Ty));
          CurInfo.Mappers.push_back(nullptr);
        }
      }
      // If there is an entry in PartialStruct it means we have a struct with
      // individual members mapped. Emit an extra combined entry.
      if (PartialStruct.Base.isValid()) {
        CurInfo.NonContigInfo.Dims.push_back(0);
        emitCombinedEntry(CombinedInfo, CurInfo.Types, PartialStruct, VD);
      }

      // We need to append the results of this capture to what we already
      // have.
      CombinedInfo.append(CurInfo);
    }
    // Append data for use_device_ptr clauses.
    CombinedInfo.append(UseDevicePtrCombinedInfo);
  }

public:
  MappableExprsHandler(const OMPExecutableDirective &Dir, CodeGenFunction &CGF)
      : CurDir(&Dir), CGF(CGF) {
    // Extract firstprivate clause information.
    for (const auto *C : Dir.getClausesOfKind<OMPFirstprivateClause>())
      for (const auto *D : C->varlists())
        FirstPrivateDecls.try_emplace(
            cast<VarDecl>(cast<DeclRefExpr>(D)->getDecl()), C->isImplicit());
    // Extract implicit firstprivates from uses_allocators clauses.
    for (const auto *C : Dir.getClausesOfKind<OMPUsesAllocatorsClause>()) {
      for (unsigned I = 0, E = C->getNumberOfAllocators(); I < E; ++I) {
        OMPUsesAllocatorsClause::Data D = C->getAllocatorData(I);
        if (const auto *DRE = dyn_cast_or_null<DeclRefExpr>(D.AllocatorTraits))
          FirstPrivateDecls.try_emplace(cast<VarDecl>(DRE->getDecl()),
                                        /*Implicit=*/true);
        else if (const auto *VD = dyn_cast<VarDecl>(
                     cast<DeclRefExpr>(D.Allocator->IgnoreParenImpCasts())
                         ->getDecl()))
          FirstPrivateDecls.try_emplace(VD, /*Implicit=*/true);
      }
    }
    // Extract device pointer clause information.
    for (const auto *C : Dir.getClausesOfKind<OMPIsDevicePtrClause>())
      for (auto L : C->component_lists())
        DevPointersMap[std::get<0>(L)].push_back(std::get<1>(L));
    // Extract map information.
    for (const auto *C : Dir.getClausesOfKind<OMPMapClause>()) {
      if (C->getMapType() != OMPC_MAP_to)
        continue;
      for (auto L : C->component_lists()) {
        const ValueDecl *VD = std::get<0>(L);
        const auto *RD = VD ? VD->getType()
                                  .getCanonicalType()
                                  .getNonReferenceType()
                                  ->getAsCXXRecordDecl()
                            : nullptr;
        if (RD && RD->isLambda())
          LambdasMap.try_emplace(std::get<0>(L), C);
      }
    }
  }

  /// Constructor for the declare mapper directive.
  MappableExprsHandler(const OMPDeclareMapperDecl &Dir, CodeGenFunction &CGF)
      : CurDir(&Dir), CGF(CGF) {}

  /// Generate code for the combined entry if we have a partially mapped struct
  /// and take care of the mapping flags of the arguments corresponding to
  /// individual struct members.
  void emitCombinedEntry(MapCombinedInfoTy &CombinedInfo,
                         MapFlagsArrayTy &CurTypes,
                         const StructRangeInfoTy &PartialStruct,
                         const ValueDecl *VD = nullptr,
                         bool NotTargetParams = true) const {
    if (CurTypes.size() == 1 &&
        ((CurTypes.back() & OMP_MAP_MEMBER_OF) != OMP_MAP_MEMBER_OF) &&
        !PartialStruct.IsArraySection)
      return;
    Address LBAddr = PartialStruct.LowestElem.second;
    Address HBAddr = PartialStruct.HighestElem.second;
    if (PartialStruct.HasCompleteRecord) {
      LBAddr = PartialStruct.LB;
      HBAddr = PartialStruct.LB;
    }
    CombinedInfo.Exprs.push_back(VD);
    // Base is the base of the struct
    CombinedInfo.BasePointers.push_back(PartialStruct.Base.getPointer());
    // Pointer is the address of the lowest element
    llvm::Value *LB = LBAddr.getPointer();
    CombinedInfo.Pointers.push_back(LB);
    // There should not be a mapper for a combined entry.
    CombinedInfo.Mappers.push_back(nullptr);
    // Size is (addr of {highest+1} element) - (addr of lowest element)
    llvm::Value *HB = HBAddr.getPointer();
    llvm::Value *HAddr =
        CGF.Builder.CreateConstGEP1_32(HBAddr.getElementType(), HB, /*Idx0=*/1);
    llvm::Value *CLAddr = CGF.Builder.CreatePointerCast(LB, CGF.VoidPtrTy);
    llvm::Value *CHAddr = CGF.Builder.CreatePointerCast(HAddr, CGF.VoidPtrTy);
    llvm::Value *Diff = CGF.Builder.CreatePtrDiff(CHAddr, CLAddr);
    llvm::Value *Size = CGF.Builder.CreateIntCast(Diff, CGF.Int64Ty,
                                                  /*isSigned=*/false);
    CombinedInfo.Sizes.push_back(Size);
    // Map type is always TARGET_PARAM, if generate info for captures.
    CombinedInfo.Types.push_back(NotTargetParams ? OMP_MAP_NONE
                                                 : OMP_MAP_TARGET_PARAM);
    // If any element has the present modifier, then make sure the runtime
    // doesn't attempt to allocate the struct.
    if (CurTypes.end() !=
        llvm::find_if(CurTypes, [](OpenMPOffloadMappingFlags Type) {
          return Type & OMP_MAP_PRESENT;
        }))
      CombinedInfo.Types.back() |= OMP_MAP_PRESENT;
    // Remove TARGET_PARAM flag from the first element
    (*CurTypes.begin()) &= ~OMP_MAP_TARGET_PARAM;
    // If any element has the ompx_hold modifier, then make sure the runtime
    // uses the hold reference count for the struct as a whole so that it won't
    // be unmapped by an extra dynamic reference count decrement.  Add it to all
    // elements as well so the runtime knows which reference count to check
    // when determining whether it's time for device-to-host transfers of
    // individual elements.
    if (CurTypes.end() !=
        llvm::find_if(CurTypes, [](OpenMPOffloadMappingFlags Type) {
          return Type & OMP_MAP_OMPX_HOLD;
        })) {
      CombinedInfo.Types.back() |= OMP_MAP_OMPX_HOLD;
      for (auto &M : CurTypes)
        M |= OMP_MAP_OMPX_HOLD;
    }

    // All other current entries will be MEMBER_OF the combined entry
    // (except for PTR_AND_OBJ entries which do not have a placeholder value
    // 0xFFFF in the MEMBER_OF field).
    OpenMPOffloadMappingFlags MemberOfFlag =
        getMemberOfFlag(CombinedInfo.BasePointers.size() - 1);
    for (auto &M : CurTypes)
      setCorrectMemberOfFlag(M, MemberOfFlag);
  }

  /// Generate all the base pointers, section pointers, sizes, map types, and
  /// mappers for the extracted mappable expressions (all included in \a
  /// CombinedInfo). Also, for each item that relates with a device pointer, a
  /// pair of the relevant declaration and index where it occurs is appended to
  /// the device pointers info array.
  void generateAllInfo(
      MapCombinedInfoTy &CombinedInfo,
      const llvm::DenseSet<CanonicalDeclPtr<const Decl>> &SkipVarSet =
          llvm::DenseSet<CanonicalDeclPtr<const Decl>>()) const {
    assert(CurDir.is<const OMPExecutableDirective *>() &&
           "Expect a executable directive");
    const auto *CurExecDir = CurDir.get<const OMPExecutableDirective *>();
    generateAllInfoForClauses(CurExecDir->clauses(), CombinedInfo, SkipVarSet);
  }

  /// Generate all the base pointers, section pointers, sizes, map types, and
  /// mappers for the extracted map clauses of user-defined mapper (all included
  /// in \a CombinedInfo).
  void generateAllInfoForMapper(MapCombinedInfoTy &CombinedInfo) const {
    assert(CurDir.is<const OMPDeclareMapperDecl *>() &&
           "Expect a declare mapper directive");
    const auto *CurMapperDir = CurDir.get<const OMPDeclareMapperDecl *>();
    generateAllInfoForClauses(CurMapperDir->clauses(), CombinedInfo);
  }

  /// Emit capture info for lambdas for variables captured by reference.
  void generateInfoForLambdaCaptures(
      const ValueDecl *VD, llvm::Value *Arg, MapCombinedInfoTy &CombinedInfo,
      llvm::DenseMap<llvm::Value *, llvm::Value *> &LambdaPointers) const {
    const auto *RD = VD->getType()
                         .getCanonicalType()
                         .getNonReferenceType()
                         ->getAsCXXRecordDecl();
    if (!RD || !RD->isLambda())
      return;
    Address VDAddr = Address(Arg, CGF.getContext().getDeclAlign(VD));
    LValue VDLVal = CGF.MakeAddrLValue(
        VDAddr, VD->getType().getCanonicalType().getNonReferenceType());
    llvm::DenseMap<const VarDecl *, FieldDecl *> Captures;
    FieldDecl *ThisCapture = nullptr;
    RD->getCaptureFields(Captures, ThisCapture);
    if (ThisCapture) {
      LValue ThisLVal =
          CGF.EmitLValueForFieldInitialization(VDLVal, ThisCapture);
      LValue ThisLValVal = CGF.EmitLValueForField(VDLVal, ThisCapture);
      LambdaPointers.try_emplace(ThisLVal.getPointer(CGF),
                                 VDLVal.getPointer(CGF));
      CombinedInfo.Exprs.push_back(VD);
      CombinedInfo.BasePointers.push_back(ThisLVal.getPointer(CGF));
      CombinedInfo.Pointers.push_back(ThisLValVal.getPointer(CGF));
      CombinedInfo.Sizes.push_back(
          CGF.Builder.CreateIntCast(CGF.getTypeSize(CGF.getContext().VoidPtrTy),
                                    CGF.Int64Ty, /*isSigned=*/true));
      CombinedInfo.Types.push_back(OMP_MAP_PTR_AND_OBJ | OMP_MAP_LITERAL |
                                   OMP_MAP_MEMBER_OF | OMP_MAP_IMPLICIT);
      CombinedInfo.Mappers.push_back(nullptr);
    }
    for (const LambdaCapture &LC : RD->captures()) {
      if (!LC.capturesVariable())
        continue;
      const VarDecl *VD = LC.getCapturedVar();
      if (LC.getCaptureKind() != LCK_ByRef && !VD->getType()->isPointerType())
        continue;
      auto It = Captures.find(VD);
      assert(It != Captures.end() && "Found lambda capture without field.");
      LValue VarLVal = CGF.EmitLValueForFieldInitialization(VDLVal, It->second);
      if (LC.getCaptureKind() == LCK_ByRef) {
        LValue VarLValVal = CGF.EmitLValueForField(VDLVal, It->second);
        LambdaPointers.try_emplace(VarLVal.getPointer(CGF),
                                   VDLVal.getPointer(CGF));
        CombinedInfo.Exprs.push_back(VD);
        CombinedInfo.BasePointers.push_back(VarLVal.getPointer(CGF));
        CombinedInfo.Pointers.push_back(VarLValVal.getPointer(CGF));
        CombinedInfo.Sizes.push_back(CGF.Builder.CreateIntCast(
            CGF.getTypeSize(
                VD->getType().getCanonicalType().getNonReferenceType()),
            CGF.Int64Ty, /*isSigned=*/true));
      } else {
        RValue VarRVal = CGF.EmitLoadOfLValue(VarLVal, RD->getLocation());
        LambdaPointers.try_emplace(VarLVal.getPointer(CGF),
                                   VDLVal.getPointer(CGF));
        CombinedInfo.Exprs.push_back(VD);
        CombinedInfo.BasePointers.push_back(VarLVal.getPointer(CGF));
        CombinedInfo.Pointers.push_back(VarRVal.getScalarVal());
        CombinedInfo.Sizes.push_back(llvm::ConstantInt::get(CGF.Int64Ty, 0));
      }
      CombinedInfo.Types.push_back(OMP_MAP_PTR_AND_OBJ | OMP_MAP_LITERAL |
                                   OMP_MAP_MEMBER_OF | OMP_MAP_IMPLICIT);
      CombinedInfo.Mappers.push_back(nullptr);
    }
  }

  /// Set correct indices for lambdas captures.
  void adjustMemberOfForLambdaCaptures(
      const llvm::DenseMap<llvm::Value *, llvm::Value *> &LambdaPointers,
      MapBaseValuesArrayTy &BasePointers, MapValuesArrayTy &Pointers,
      MapFlagsArrayTy &Types) const {
    for (unsigned I = 0, E = Types.size(); I < E; ++I) {
      // Set correct member_of idx for all implicit lambda captures.
      if (Types[I] != (OMP_MAP_PTR_AND_OBJ | OMP_MAP_LITERAL |
                       OMP_MAP_MEMBER_OF | OMP_MAP_IMPLICIT))
        continue;
      llvm::Value *BasePtr = LambdaPointers.lookup(*BasePointers[I]);
      assert(BasePtr && "Unable to find base lambda address.");
      int TgtIdx = -1;
      for (unsigned J = I; J > 0; --J) {
        unsigned Idx = J - 1;
        if (Pointers[Idx] != BasePtr)
          continue;
        TgtIdx = Idx;
        break;
      }
      assert(TgtIdx != -1 && "Unable to find parent lambda.");
      // All other current entries will be MEMBER_OF the combined entry
      // (except for PTR_AND_OBJ entries which do not have a placeholder value
      // 0xFFFF in the MEMBER_OF field).
      OpenMPOffloadMappingFlags MemberOfFlag = getMemberOfFlag(TgtIdx);
      setCorrectMemberOfFlag(Types[I], MemberOfFlag);
    }
  }

  /// Generate the base pointers, section pointers, sizes, map types, and
  /// mappers associated to a given capture (all included in \a CombinedInfo).
  void generateInfoForCapture(const CapturedStmt::Capture *Cap,
                              llvm::Value *Arg, MapCombinedInfoTy &CombinedInfo,
                              StructRangeInfoTy &PartialStruct) const {
    assert(!Cap->capturesVariableArrayType() &&
           "Not expecting to generate map info for a variable array type!");

    // We need to know when we generating information for the first component
    const ValueDecl *VD = Cap->capturesThis()
                              ? nullptr
                              : Cap->getCapturedVar()->getCanonicalDecl();

    // for map(to: lambda): skip here, processing it in
    // generateDefaultMapInfo
    if (LambdasMap.count(VD))
      return;

    // If this declaration appears in a is_device_ptr clause we just have to
    // pass the pointer by value. If it is a reference to a declaration, we just
    // pass its value.
    if (DevPointersMap.count(VD)) {
      CombinedInfo.Exprs.push_back(VD);
      CombinedInfo.BasePointers.emplace_back(Arg, VD);
      CombinedInfo.Pointers.push_back(Arg);
      CombinedInfo.Sizes.push_back(CGF.Builder.CreateIntCast(
          CGF.getTypeSize(CGF.getContext().VoidPtrTy), CGF.Int64Ty,
          /*isSigned=*/true));
      CombinedInfo.Types.push_back(
          (Cap->capturesVariable() ? OMP_MAP_TO : OMP_MAP_LITERAL) |
          OMP_MAP_TARGET_PARAM);
      CombinedInfo.Mappers.push_back(nullptr);
      return;
    }

    using MapData =
        std::tuple<OMPClauseMappableExprCommon::MappableExprComponentListRef,
                   OpenMPMapClauseKind, ArrayRef<OpenMPMapModifierKind>, bool,
                   const ValueDecl *, const Expr *>;
    SmallVector<MapData, 4> DeclComponentLists;
    assert(CurDir.is<const OMPExecutableDirective *>() &&
           "Expect a executable directive");
    const auto *CurExecDir = CurDir.get<const OMPExecutableDirective *>();
    for (const auto *C : CurExecDir->getClausesOfKind<OMPMapClause>()) {
      const auto *EI = C->getVarRefs().begin();
      for (const auto L : C->decl_component_lists(VD)) {
        const ValueDecl *VDecl, *Mapper;
        // The Expression is not correct if the mapping is implicit
        const Expr *E = (C->getMapLoc().isValid()) ? *EI : nullptr;
        OMPClauseMappableExprCommon::MappableExprComponentListRef Components;
        std::tie(VDecl, Components, Mapper) = L;
        assert(VDecl == VD && "We got information for the wrong declaration??");
        assert(!Components.empty() &&
               "Not expecting declaration with no component lists.");
        DeclComponentLists.emplace_back(Components, C->getMapType(),
                                        C->getMapTypeModifiers(),
                                        C->isImplicit(), Mapper, E);
        ++EI;
      }
    }
    llvm::stable_sort(DeclComponentLists, [](const MapData &LHS,
                                             const MapData &RHS) {
      ArrayRef<OpenMPMapModifierKind> MapModifiers = std::get<2>(LHS);
      OpenMPMapClauseKind MapType = std::get<1>(RHS);
      bool HasPresent =
          llvm::is_contained(MapModifiers, clang::OMPC_MAP_MODIFIER_present);
      bool HasAllocs = MapType == OMPC_MAP_alloc;
      MapModifiers = std::get<2>(RHS);
      MapType = std::get<1>(LHS);
      bool HasPresentR =
          llvm::is_contained(MapModifiers, clang::OMPC_MAP_MODIFIER_present);
      bool HasAllocsR = MapType == OMPC_MAP_alloc;
      return (HasPresent && !HasPresentR) || (HasAllocs && !HasAllocsR);
    });

    // Find overlapping elements (including the offset from the base element).
    llvm::SmallDenseMap<
        const MapData *,
        llvm::SmallVector<
            OMPClauseMappableExprCommon::MappableExprComponentListRef, 4>,
        4>
        OverlappedData;
    size_t Count = 0;
    for (const MapData &L : DeclComponentLists) {
      OMPClauseMappableExprCommon::MappableExprComponentListRef Components;
      OpenMPMapClauseKind MapType;
      ArrayRef<OpenMPMapModifierKind> MapModifiers;
      bool IsImplicit;
      const ValueDecl *Mapper;
      const Expr *VarRef;
      std::tie(Components, MapType, MapModifiers, IsImplicit, Mapper, VarRef) =
          L;
      ++Count;
      for (const MapData &L1 : makeArrayRef(DeclComponentLists).slice(Count)) {
        OMPClauseMappableExprCommon::MappableExprComponentListRef Components1;
        std::tie(Components1, MapType, MapModifiers, IsImplicit, Mapper,
                 VarRef) = L1;
        auto CI = Components.rbegin();
        auto CE = Components.rend();
        auto SI = Components1.rbegin();
        auto SE = Components1.rend();
        for (; CI != CE && SI != SE; ++CI, ++SI) {
          if (CI->getAssociatedExpression()->getStmtClass() !=
              SI->getAssociatedExpression()->getStmtClass())
            break;
          // Are we dealing with different variables/fields?
          if (CI->getAssociatedDeclaration() != SI->getAssociatedDeclaration())
            break;
        }
        // Found overlapping if, at least for one component, reached the head
        // of the components list.
        if (CI == CE || SI == SE) {
          // Ignore it if it is the same component.
          if (CI == CE && SI == SE)
            continue;
          const auto It = (SI == SE) ? CI : SI;
          // If one component is a pointer and another one is a kind of
          // dereference of this pointer (array subscript, section, dereference,
          // etc.), it is not an overlapping.
          // Same, if one component is a base and another component is a
          // dereferenced pointer memberexpr with the same base.
          if (!isa<MemberExpr>(It->getAssociatedExpression()) ||
              (std::prev(It)->getAssociatedDeclaration() &&
               std::prev(It)
                   ->getAssociatedDeclaration()
                   ->getType()
                   ->isPointerType()) ||
              (It->getAssociatedDeclaration() &&
               It->getAssociatedDeclaration()->getType()->isPointerType() &&
               std::next(It) != CE && std::next(It) != SE))
            continue;
          const MapData &BaseData = CI == CE ? L : L1;
          OMPClauseMappableExprCommon::MappableExprComponentListRef SubData =
              SI == SE ? Components : Components1;
          auto &OverlappedElements = OverlappedData.FindAndConstruct(&BaseData);
          OverlappedElements.getSecond().push_back(SubData);
        }
      }
    }
    // Sort the overlapped elements for each item.
    llvm::SmallVector<const FieldDecl *, 4> Layout;
    if (!OverlappedData.empty()) {
      const Type *BaseType = VD->getType().getCanonicalType().getTypePtr();
      const Type *OrigType = BaseType->getPointeeOrArrayElementType();
      while (BaseType != OrigType) {
        BaseType = OrigType->getCanonicalTypeInternal().getTypePtr();
        OrigType = BaseType->getPointeeOrArrayElementType();
      }

      if (const auto *CRD = BaseType->getAsCXXRecordDecl())
        getPlainLayout(CRD, Layout, /*AsBase=*/false);
      else {
        const auto *RD = BaseType->getAsRecordDecl();
        Layout.append(RD->field_begin(), RD->field_end());
      }
    }
    for (auto &Pair : OverlappedData) {
      llvm::stable_sort(
          Pair.getSecond(),
          [&Layout](
              OMPClauseMappableExprCommon::MappableExprComponentListRef First,
              OMPClauseMappableExprCommon::MappableExprComponentListRef
                  Second) {
            auto CI = First.rbegin();
            auto CE = First.rend();
            auto SI = Second.rbegin();
            auto SE = Second.rend();
            for (; CI != CE && SI != SE; ++CI, ++SI) {
              if (CI->getAssociatedExpression()->getStmtClass() !=
                  SI->getAssociatedExpression()->getStmtClass())
                break;
              // Are we dealing with different variables/fields?
              if (CI->getAssociatedDeclaration() !=
                  SI->getAssociatedDeclaration())
                break;
            }

            // Lists contain the same elements.
            if (CI == CE && SI == SE)
              return false;

            // List with less elements is less than list with more elements.
            if (CI == CE || SI == SE)
              return CI == CE;

            const auto *FD1 = cast<FieldDecl>(CI->getAssociatedDeclaration());
            const auto *FD2 = cast<FieldDecl>(SI->getAssociatedDeclaration());
            if (FD1->getParent() == FD2->getParent())
              return FD1->getFieldIndex() < FD2->getFieldIndex();
            const auto *It =
                llvm::find_if(Layout, [FD1, FD2](const FieldDecl *FD) {
                  return FD == FD1 || FD == FD2;
                });
            return *It == FD1;
          });
    }

    // Associated with a capture, because the mapping flags depend on it.
    // Go through all of the elements with the overlapped elements.
    bool IsFirstComponentList = true;
    for (const auto &Pair : OverlappedData) {
      const MapData &L = *Pair.getFirst();
      OMPClauseMappableExprCommon::MappableExprComponentListRef Components;
      OpenMPMapClauseKind MapType;
      ArrayRef<OpenMPMapModifierKind> MapModifiers;
      bool IsImplicit;
      const ValueDecl *Mapper;
      const Expr *VarRef;
      std::tie(Components, MapType, MapModifiers, IsImplicit, Mapper, VarRef) =
          L;
      ArrayRef<OMPClauseMappableExprCommon::MappableExprComponentListRef>
          OverlappedComponents = Pair.getSecond();
      generateInfoForComponentList(
          MapType, MapModifiers, llvm::None, Components, CombinedInfo,
          PartialStruct, IsFirstComponentList, IsImplicit, Mapper,
          /*ForDeviceAddr=*/false, VD, VarRef, OverlappedComponents);
      IsFirstComponentList = false;
    }
    // Go through other elements without overlapped elements.
    for (const MapData &L : DeclComponentLists) {
      OMPClauseMappableExprCommon::MappableExprComponentListRef Components;
      OpenMPMapClauseKind MapType;
      ArrayRef<OpenMPMapModifierKind> MapModifiers;
      bool IsImplicit;
      const ValueDecl *Mapper;
      const Expr *VarRef;
      std::tie(Components, MapType, MapModifiers, IsImplicit, Mapper, VarRef) =
          L;
      auto It = OverlappedData.find(&L);
      if (It == OverlappedData.end())
        generateInfoForComponentList(MapType, MapModifiers, llvm::None,
                                     Components, CombinedInfo, PartialStruct,
                                     IsFirstComponentList, IsImplicit, Mapper,
                                     /*ForDeviceAddr=*/false, VD, VarRef);
      IsFirstComponentList = false;
    }
  }

  /// Generate the default map information for a given capture \a CI,
  /// record field declaration \a RI and captured value \a CV.
  void generateDefaultMapInfo(const CapturedStmt::Capture &CI,
                              const FieldDecl &RI, llvm::Value *CV,
                              MapCombinedInfoTy &CombinedInfo) const {
    bool IsImplicit = true;
    // Do the default mapping.
    if (CI.capturesThis()) {
      CombinedInfo.Exprs.push_back(nullptr);
      CombinedInfo.BasePointers.push_back(CV);
      CombinedInfo.Pointers.push_back(CV);
      const auto *PtrTy = cast<PointerType>(RI.getType().getTypePtr());
      CombinedInfo.Sizes.push_back(
          CGF.Builder.CreateIntCast(CGF.getTypeSize(PtrTy->getPointeeType()),
                                    CGF.Int64Ty, /*isSigned=*/true));
      // Default map type.
      CombinedInfo.Types.push_back(OMP_MAP_TO | OMP_MAP_FROM);
    } else if (CI.capturesVariableByCopy()) {
      const VarDecl *VD = CI.getCapturedVar();
      CombinedInfo.Exprs.push_back(VD->getCanonicalDecl());
      CombinedInfo.BasePointers.push_back(CV);
      CombinedInfo.Pointers.push_back(CV);
      if (!RI.getType()->isAnyPointerType()) {
        // We have to signal to the runtime captures passed by value that are
        // not pointers.
        CombinedInfo.Types.push_back(OMP_MAP_LITERAL);
        CombinedInfo.Sizes.push_back(CGF.Builder.CreateIntCast(
            CGF.getTypeSize(RI.getType()), CGF.Int64Ty, /*isSigned=*/true));
      } else {
        // Pointers are implicitly mapped with a zero size and no flags
        // (other than first map that is added for all implicit maps).
        CombinedInfo.Types.push_back(OMP_MAP_NONE);
        CombinedInfo.Sizes.push_back(llvm::Constant::getNullValue(CGF.Int64Ty));
      }
      auto I = FirstPrivateDecls.find(VD);
      if (I != FirstPrivateDecls.end())
        IsImplicit = I->getSecond();
    } else {
      assert(CI.capturesVariable() && "Expected captured reference.");
      const auto *PtrTy = cast<ReferenceType>(RI.getType().getTypePtr());
      QualType ElementType = PtrTy->getPointeeType();
      CombinedInfo.Sizes.push_back(CGF.Builder.CreateIntCast(
          CGF.getTypeSize(ElementType), CGF.Int64Ty, /*isSigned=*/true));
      // The default map type for a scalar/complex type is 'to' because by
      // default the value doesn't have to be retrieved. For an aggregate
      // type, the default is 'tofrom'.
      CombinedInfo.Types.push_back(getMapModifiersForPrivateClauses(CI));
      const VarDecl *VD = CI.getCapturedVar();
      auto I = FirstPrivateDecls.find(VD);
      CombinedInfo.Exprs.push_back(VD->getCanonicalDecl());
      CombinedInfo.BasePointers.push_back(CV);
      if (I != FirstPrivateDecls.end() && ElementType->isAnyPointerType()) {
        Address PtrAddr = CGF.EmitLoadOfReference(CGF.MakeAddrLValue(
            CV, ElementType, CGF.getContext().getDeclAlign(VD),
            AlignmentSource::Decl));
        CombinedInfo.Pointers.push_back(PtrAddr.getPointer());
      } else {
        CombinedInfo.Pointers.push_back(CV);
      }
      if (I != FirstPrivateDecls.end())
        IsImplicit = I->getSecond();
    }
    // Every default map produces a single argument which is a target parameter.
    CombinedInfo.Types.back() |= OMP_MAP_TARGET_PARAM;

    // Add flag stating this is an implicit map.
    if (IsImplicit)
      CombinedInfo.Types.back() |= OMP_MAP_IMPLICIT;

    // No user-defined mapper for default mapping.
    CombinedInfo.Mappers.push_back(nullptr);
  }
};
} // anonymous namespace

static void emitNonContiguousDescriptor(
    CodeGenFunction &CGF, MappableExprsHandler::MapCombinedInfoTy &CombinedInfo,
    CGOpenMPRuntime::TargetDataInfo &Info) {
  CodeGenModule &CGM = CGF.CGM;
  MappableExprsHandler::MapCombinedInfoTy::StructNonContiguousInfo
      &NonContigInfo = CombinedInfo.NonContigInfo;

  // Build an array of struct descriptor_dim and then assign it to
  // offload_args.
  //
  // struct descriptor_dim {
  //  uint64_t offset;
  //  uint64_t count;
  //  uint64_t stride
  // };
  ASTContext &C = CGF.getContext();
  QualType Int64Ty = C.getIntTypeForBitwidth(/*DestWidth=*/64, /*Signed=*/0);
  RecordDecl *RD;
  RD = C.buildImplicitRecord("descriptor_dim");
  RD->startDefinition();
  addFieldToRecordDecl(C, RD, Int64Ty);
  addFieldToRecordDecl(C, RD, Int64Ty);
  addFieldToRecordDecl(C, RD, Int64Ty);
  RD->completeDefinition();
  QualType DimTy = C.getRecordType(RD);

  enum { OffsetFD = 0, CountFD, StrideFD };
  // We need two index variable here since the size of "Dims" is the same as the
  // size of Components, however, the size of offset, count, and stride is equal
  // to the size of base declaration that is non-contiguous.
  for (unsigned I = 0, L = 0, E = NonContigInfo.Dims.size(); I < E; ++I) {
    // Skip emitting ir if dimension size is 1 since it cannot be
    // non-contiguous.
    if (NonContigInfo.Dims[I] == 1)
      continue;
    llvm::APInt Size(/*numBits=*/32, NonContigInfo.Dims[I]);
    QualType ArrayTy =
        C.getConstantArrayType(DimTy, Size, nullptr, ArrayType::Normal, 0);
    Address DimsAddr = CGF.CreateMemTemp(ArrayTy, "dims");
    for (unsigned II = 0, EE = NonContigInfo.Dims[I]; II < EE; ++II) {
      unsigned RevIdx = EE - II - 1;
      LValue DimsLVal = CGF.MakeAddrLValue(
          CGF.Builder.CreateConstArrayGEP(DimsAddr, II), DimTy);
      // Offset
      LValue OffsetLVal = CGF.EmitLValueForField(
          DimsLVal, *std::next(RD->field_begin(), OffsetFD));
      CGF.EmitStoreOfScalar(NonContigInfo.Offsets[L][RevIdx], OffsetLVal);
      // Count
      LValue CountLVal = CGF.EmitLValueForField(
          DimsLVal, *std::next(RD->field_begin(), CountFD));
      CGF.EmitStoreOfScalar(NonContigInfo.Counts[L][RevIdx], CountLVal);
      // Stride
      LValue StrideLVal = CGF.EmitLValueForField(
          DimsLVal, *std::next(RD->field_begin(), StrideFD));
      CGF.EmitStoreOfScalar(NonContigInfo.Strides[L][RevIdx], StrideLVal);
    }
    // args[I] = &dims
    Address DAddr = CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
        DimsAddr, CGM.Int8PtrTy);
    llvm::Value *P = CGF.Builder.CreateConstInBoundsGEP2_32(
        llvm::ArrayType::get(CGM.VoidPtrTy, Info.NumberOfPtrs),
        Info.PointersArray, 0, I);
    Address PAddr(P, CGF.getPointerAlign());
    CGF.Builder.CreateStore(DAddr.getPointer(), PAddr);
    ++L;
  }
}

// Try to extract the base declaration from a `this->x` expression if possible.
static ValueDecl *getDeclFromThisExpr(const Expr *E) {
  if (!E)
    return nullptr;

  if (const auto *OASE = dyn_cast<OMPArraySectionExpr>(E->IgnoreParenCasts()))
    if (const MemberExpr *ME =
            dyn_cast<MemberExpr>(OASE->getBase()->IgnoreParenImpCasts()))
      return ME->getMemberDecl();
  return nullptr;
}

/// Emit a string constant containing the names of the values mapped to the
/// offloading runtime library.
llvm::Constant *
emitMappingInformation(CodeGenFunction &CGF, llvm::OpenMPIRBuilder &OMPBuilder,
                       MappableExprsHandler::MappingExprInfo &MapExprs) {

  uint32_t SrcLocStrSize;
  if (!MapExprs.getMapDecl() && !MapExprs.getMapExpr())
    return OMPBuilder.getOrCreateDefaultSrcLocStr(SrcLocStrSize);

  SourceLocation Loc;
  if (!MapExprs.getMapDecl() && MapExprs.getMapExpr()) {
    if (const ValueDecl *VD = getDeclFromThisExpr(MapExprs.getMapExpr()))
      Loc = VD->getLocation();
    else
      Loc = MapExprs.getMapExpr()->getExprLoc();
  } else {
    Loc = MapExprs.getMapDecl()->getLocation();
  }

  std::string ExprName;
  if (MapExprs.getMapExpr()) {
    PrintingPolicy P(CGF.getContext().getLangOpts());
    llvm::raw_string_ostream OS(ExprName);
    MapExprs.getMapExpr()->printPretty(OS, nullptr, P);
    OS.flush();
  } else {
    ExprName = MapExprs.getMapDecl()->getNameAsString();
  }

  PresumedLoc PLoc = CGF.getContext().getSourceManager().getPresumedLoc(Loc);
  return OMPBuilder.getOrCreateSrcLocStr(PLoc.getFilename(), ExprName,
                                         PLoc.getLine(), PLoc.getColumn(),
                                         SrcLocStrSize);
}

/// Emit the arrays used to pass the captures and map information to the
/// offloading runtime library. If there is no map or capture information,
/// return nullptr by reference.
static void emitOffloadingArrays(
    CodeGenFunction &CGF, MappableExprsHandler::MapCombinedInfoTy &CombinedInfo,
    CGOpenMPRuntime::TargetDataInfo &Info, llvm::OpenMPIRBuilder &OMPBuilder,
    bool IsNonContiguous = false) {
  CodeGenModule &CGM = CGF.CGM;
  ASTContext &Ctx = CGF.getContext();

  // Reset the array information.
  Info.clearArrayInfo();
  Info.NumberOfPtrs = CombinedInfo.BasePointers.size();

  if (Info.NumberOfPtrs) {
    // Detect if we have any capture size requiring runtime evaluation of the
    // size so that a constant array could be eventually used.
    bool hasRuntimeEvaluationCaptureSize = false;
    for (llvm::Value *S : CombinedInfo.Sizes)
      if (!isa<llvm::Constant>(S)) {
        hasRuntimeEvaluationCaptureSize = true;
        break;
      }

    llvm::APInt PointerNumAP(32, Info.NumberOfPtrs, /*isSigned=*/true);
    QualType PointerArrayType = Ctx.getConstantArrayType(
        Ctx.VoidPtrTy, PointerNumAP, nullptr, ArrayType::Normal,
        /*IndexTypeQuals=*/0);

    Info.BasePointersArray =
        CGF.CreateMemTemp(PointerArrayType, ".offload_baseptrs").getPointer();
    Info.PointersArray =
        CGF.CreateMemTemp(PointerArrayType, ".offload_ptrs").getPointer();
    Address MappersArray =
        CGF.CreateMemTemp(PointerArrayType, ".offload_mappers");
    Info.MappersArray = MappersArray.getPointer();

    // If we don't have any VLA types or other types that require runtime
    // evaluation, we can use a constant array for the map sizes, otherwise we
    // need to fill up the arrays as we do for the pointers.
    QualType Int64Ty =
        Ctx.getIntTypeForBitwidth(/*DestWidth=*/64, /*Signed=*/1);
    if (hasRuntimeEvaluationCaptureSize) {
      QualType SizeArrayType = Ctx.getConstantArrayType(
          Int64Ty, PointerNumAP, nullptr, ArrayType::Normal,
          /*IndexTypeQuals=*/0);
      Info.SizesArray =
          CGF.CreateMemTemp(SizeArrayType, ".offload_sizes").getPointer();
    } else {
      // We expect all the sizes to be constant, so we collect them to create
      // a constant array.
      SmallVector<llvm::Constant *, 16> ConstSizes;
      for (unsigned I = 0, E = CombinedInfo.Sizes.size(); I < E; ++I) {
        if (IsNonContiguous &&
            (CombinedInfo.Types[I] & MappableExprsHandler::OMP_MAP_NON_CONTIG)) {
          ConstSizes.push_back(llvm::ConstantInt::get(
              CGF.Int64Ty, CombinedInfo.NonContigInfo.Dims[I]));
        } else {
          ConstSizes.push_back(cast<llvm::Constant>(CombinedInfo.Sizes[I]));
        }
      }

      auto *SizesArrayInit = llvm::ConstantArray::get(
          llvm::ArrayType::get(CGM.Int64Ty, ConstSizes.size()), ConstSizes);
      std::string Name = CGM.getOpenMPRuntime().getName({"offload_sizes"});
      auto *SizesArrayGbl = new llvm::GlobalVariable(
          CGM.getModule(), SizesArrayInit->getType(),
          /*isConstant=*/true, llvm::GlobalValue::PrivateLinkage,
          SizesArrayInit, Name);
      SizesArrayGbl->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Global);
      Info.SizesArray = SizesArrayGbl;
    }

    // The map types are always constant so we don't need to generate code to
    // fill arrays. Instead, we create an array constant.
    SmallVector<uint64_t, 4> Mapping(CombinedInfo.Types.size(), 0);
    llvm::copy(CombinedInfo.Types, Mapping.begin());
    std::string MaptypesName =
        CGM.getOpenMPRuntime().getName({"offload_maptypes"});
    auto *MapTypesArrayGbl =
        OMPBuilder.createOffloadMaptypes(Mapping, MaptypesName);
    Info.MapTypesArray = MapTypesArrayGbl;

    // The information types are only built if there is debug information
    // requested.
    if (CGM.getCodeGenOpts().getDebugInfo() == codegenoptions::NoDebugInfo) {
      Info.MapNamesArray = llvm::Constant::getNullValue(
          llvm::Type::getInt8Ty(CGF.Builder.getContext())->getPointerTo());
    } else {
      auto fillInfoMap = [&](MappableExprsHandler::MappingExprInfo &MapExpr) {
        return emitMappingInformation(CGF, OMPBuilder, MapExpr);
      };
      SmallVector<llvm::Constant *, 4> InfoMap(CombinedInfo.Exprs.size());
      llvm::transform(CombinedInfo.Exprs, InfoMap.begin(), fillInfoMap);
      std::string MapnamesName =
          CGM.getOpenMPRuntime().getName({"offload_mapnames"});
      auto *MapNamesArrayGbl =
          OMPBuilder.createOffloadMapnames(InfoMap, MapnamesName);
      Info.MapNamesArray = MapNamesArrayGbl;
    }

    // If there's a present map type modifier, it must not be applied to the end
    // of a region, so generate a separate map type array in that case.
    if (Info.separateBeginEndCalls()) {
      bool EndMapTypesDiffer = false;
      for (uint64_t &Type : Mapping) {
        if (Type & MappableExprsHandler::OMP_MAP_PRESENT) {
          Type &= ~MappableExprsHandler::OMP_MAP_PRESENT;
          EndMapTypesDiffer = true;
        }
      }
      if (EndMapTypesDiffer) {
        MapTypesArrayGbl =
            OMPBuilder.createOffloadMaptypes(Mapping, MaptypesName);
        Info.MapTypesArrayEnd = MapTypesArrayGbl;
      }
    }

    for (unsigned I = 0; I < Info.NumberOfPtrs; ++I) {
      llvm::Value *BPVal = *CombinedInfo.BasePointers[I];
      llvm::Value *BP = CGF.Builder.CreateConstInBoundsGEP2_32(
          llvm::ArrayType::get(CGM.VoidPtrTy, Info.NumberOfPtrs),
          Info.BasePointersArray, 0, I);
      BP = CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
          BP, BPVal->getType()->getPointerTo(/*AddrSpace=*/0));
      Address BPAddr(BP, Ctx.getTypeAlignInChars(Ctx.VoidPtrTy));
      CGF.Builder.CreateStore(BPVal, BPAddr);

      if (Info.requiresDevicePointerInfo())
        if (const ValueDecl *DevVD =
                CombinedInfo.BasePointers[I].getDevicePtrDecl())
          Info.CaptureDeviceAddrMap.try_emplace(DevVD, BPAddr);

      llvm::Value *PVal = CombinedInfo.Pointers[I];
      llvm::Value *P = CGF.Builder.CreateConstInBoundsGEP2_32(
          llvm::ArrayType::get(CGM.VoidPtrTy, Info.NumberOfPtrs),
          Info.PointersArray, 0, I);
      P = CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
          P, PVal->getType()->getPointerTo(/*AddrSpace=*/0));
      Address PAddr(P, Ctx.getTypeAlignInChars(Ctx.VoidPtrTy));
      CGF.Builder.CreateStore(PVal, PAddr);

      if (hasRuntimeEvaluationCaptureSize) {
        llvm::Value *S = CGF.Builder.CreateConstInBoundsGEP2_32(
            llvm::ArrayType::get(CGM.Int64Ty, Info.NumberOfPtrs),
            Info.SizesArray,
            /*Idx0=*/0,
            /*Idx1=*/I);
        Address SAddr(S, Ctx.getTypeAlignInChars(Int64Ty));
        CGF.Builder.CreateStore(CGF.Builder.CreateIntCast(CombinedInfo.Sizes[I],
                                                          CGM.Int64Ty,
                                                          /*isSigned=*/true),
                                SAddr);
      }

      // Fill up the mapper array.
      llvm::Value *MFunc = llvm::ConstantPointerNull::get(CGM.VoidPtrTy);
      if (CombinedInfo.Mappers[I]) {
        MFunc = CGM.getOpenMPRuntime().getOrCreateUserDefinedMapperFunc(
            cast<OMPDeclareMapperDecl>(CombinedInfo.Mappers[I]));
        MFunc = CGF.Builder.CreatePointerCast(MFunc, CGM.VoidPtrTy);
        Info.HasMapper = true;
      }
      Address MAddr = CGF.Builder.CreateConstArrayGEP(MappersArray, I);
      CGF.Builder.CreateStore(MFunc, MAddr);
    }
  }

  if (!IsNonContiguous || CombinedInfo.NonContigInfo.Offsets.empty() ||
      Info.NumberOfPtrs == 0)
    return;

  emitNonContiguousDescriptor(CGF, CombinedInfo, Info);
}

namespace {
/// Additional arguments for emitOffloadingArraysArgument function.
struct ArgumentsOptions {
  bool ForEndCall = false;
  ArgumentsOptions() = default;
  ArgumentsOptions(bool ForEndCall) : ForEndCall(ForEndCall) {}
};
} // namespace

/// Emit the arguments to be passed to the runtime library based on the
/// arrays of base pointers, pointers, sizes, map types, and mappers.  If
/// ForEndCall, emit map types to be passed for the end of the region instead of
/// the beginning.
static void emitOffloadingArraysArgument(
    CodeGenFunction &CGF, llvm::Value *&BasePointersArrayArg,
    llvm::Value *&PointersArrayArg, llvm::Value *&SizesArrayArg,
    llvm::Value *&MapTypesArrayArg, llvm::Value *&MapNamesArrayArg,
    llvm::Value *&MappersArrayArg, CGOpenMPRuntime::TargetDataInfo &Info,
    const ArgumentsOptions &Options = ArgumentsOptions()) {
  assert((!Options.ForEndCall || Info.separateBeginEndCalls()) &&
         "expected region end call to runtime only when end call is separate");
  CodeGenModule &CGM = CGF.CGM;
  if (Info.NumberOfPtrs) {
    BasePointersArrayArg = CGF.Builder.CreateConstInBoundsGEP2_32(
        llvm::ArrayType::get(CGM.VoidPtrTy, Info.NumberOfPtrs),
        Info.BasePointersArray,
        /*Idx0=*/0, /*Idx1=*/0);
    PointersArrayArg = CGF.Builder.CreateConstInBoundsGEP2_32(
        llvm::ArrayType::get(CGM.VoidPtrTy, Info.NumberOfPtrs),
        Info.PointersArray,
        /*Idx0=*/0,
        /*Idx1=*/0);
    SizesArrayArg = CGF.Builder.CreateConstInBoundsGEP2_32(
        llvm::ArrayType::get(CGM.Int64Ty, Info.NumberOfPtrs), Info.SizesArray,
        /*Idx0=*/0, /*Idx1=*/0);
    MapTypesArrayArg = CGF.Builder.CreateConstInBoundsGEP2_32(
        llvm::ArrayType::get(CGM.Int64Ty, Info.NumberOfPtrs),
        Options.ForEndCall && Info.MapTypesArrayEnd ? Info.MapTypesArrayEnd
                                                    : Info.MapTypesArray,
        /*Idx0=*/0,
        /*Idx1=*/0);

    // Only emit the mapper information arrays if debug information is
    // requested.
    if (CGF.CGM.getCodeGenOpts().getDebugInfo() == codegenoptions::NoDebugInfo)
      MapNamesArrayArg = llvm::ConstantPointerNull::get(CGM.VoidPtrPtrTy);
    else
      MapNamesArrayArg = CGF.Builder.CreateConstInBoundsGEP2_32(
          llvm::ArrayType::get(CGM.VoidPtrTy, Info.NumberOfPtrs),
          Info.MapNamesArray,
          /*Idx0=*/0,
          /*Idx1=*/0);
    // If there is no user-defined mapper, set the mapper array to nullptr to
    // avoid an unnecessary data privatization
    if (!Info.HasMapper)
      MappersArrayArg = llvm::ConstantPointerNull::get(CGM.VoidPtrPtrTy);
    else
      MappersArrayArg =
          CGF.Builder.CreatePointerCast(Info.MappersArray, CGM.VoidPtrPtrTy);
  } else {
    BasePointersArrayArg = llvm::ConstantPointerNull::get(CGM.VoidPtrPtrTy);
    PointersArrayArg = llvm::ConstantPointerNull::get(CGM.VoidPtrPtrTy);
    SizesArrayArg = llvm::ConstantPointerNull::get(CGM.Int64Ty->getPointerTo());
    MapTypesArrayArg =
        llvm::ConstantPointerNull::get(CGM.Int64Ty->getPointerTo());
    MapNamesArrayArg = llvm::ConstantPointerNull::get(CGM.VoidPtrPtrTy);
    MappersArrayArg = llvm::ConstantPointerNull::get(CGM.VoidPtrPtrTy);
  }
}

/// Check for inner distribute directive.
static const OMPExecutableDirective *
getNestedDistributeDirective(ASTContext &Ctx, const OMPExecutableDirective &D) {
  const auto *CS = D.getInnermostCapturedStmt();
  const auto *Body =
      CS->getCapturedStmt()->IgnoreContainers(/*IgnoreCaptured=*/true);
  const Stmt *ChildStmt =
      CGOpenMPSIMDRuntime::getSingleCompoundChild(Ctx, Body);

  if (const auto *NestedDir =
          dyn_cast_or_null<OMPExecutableDirective>(ChildStmt)) {
    OpenMPDirectiveKind DKind = NestedDir->getDirectiveKind();
    switch (D.getDirectiveKind()) {
    case OMPD_target:
      if (isOpenMPDistributeDirective(DKind))
        return NestedDir;
      if (DKind == OMPD_teams) {
        Body = NestedDir->getInnermostCapturedStmt()->IgnoreContainers(
            /*IgnoreCaptured=*/true);
        if (!Body)
          return nullptr;
        ChildStmt = CGOpenMPSIMDRuntime::getSingleCompoundChild(Ctx, Body);
        if (const auto *NND =
                dyn_cast_or_null<OMPExecutableDirective>(ChildStmt)) {
          DKind = NND->getDirectiveKind();
          if (isOpenMPDistributeDirective(DKind))
            return NND;
        }
      }
      return nullptr;
    case OMPD_target_teams:
      if (isOpenMPDistributeDirective(DKind))
        return NestedDir;
      return nullptr;
    case OMPD_target_parallel:
    case OMPD_target_simd:
    case OMPD_target_parallel_for:
    case OMPD_target_parallel_for_simd:
      return nullptr;
    case OMPD_target_teams_distribute:
    case OMPD_target_teams_distribute_simd:
    case OMPD_target_teams_distribute_parallel_for:
    case OMPD_target_teams_distribute_parallel_for_simd:
    case OMPD_parallel:
    case OMPD_for:
    case OMPD_parallel_for:
    case OMPD_parallel_master:
    case OMPD_parallel_sections:
    case OMPD_for_simd:
    case OMPD_parallel_for_simd:
    case OMPD_cancel:
    case OMPD_cancellation_point:
    case OMPD_ordered:
    case OMPD_threadprivate:
    case OMPD_allocate:
    case OMPD_task:
    case OMPD_simd:
    case OMPD_tile:
    case OMPD_unroll:
    case OMPD_sections:
    case OMPD_section:
    case OMPD_single:
    case OMPD_master:
    case OMPD_critical:
    case OMPD_taskyield:
    case OMPD_barrier:
    case OMPD_taskwait:
    case OMPD_taskgroup:
    case OMPD_atomic:
    case OMPD_flush:
    case OMPD_depobj:
    case OMPD_scan:
    case OMPD_teams:
    case OMPD_target_data:
    case OMPD_target_exit_data:
    case OMPD_target_enter_data:
    case OMPD_distribute:
    case OMPD_distribute_simd:
    case OMPD_distribute_parallel_for:
    case OMPD_distribute_parallel_for_simd:
    case OMPD_teams_distribute:
    case OMPD_teams_distribute_simd:
    case OMPD_teams_distribute_parallel_for:
    case OMPD_teams_distribute_parallel_for_simd:
    case OMPD_target_update:
    case OMPD_declare_simd:
    case OMPD_declare_variant:
    case OMPD_begin_declare_variant:
    case OMPD_end_declare_variant:
    case OMPD_declare_target:
    case OMPD_end_declare_target:
    case OMPD_declare_reduction:
    case OMPD_declare_mapper:
    case OMPD_taskloop:
    case OMPD_taskloop_simd:
    case OMPD_master_taskloop:
    case OMPD_master_taskloop_simd:
    case OMPD_parallel_master_taskloop:
    case OMPD_parallel_master_taskloop_simd:
    case OMPD_requires:
    case OMPD_metadirective:
    case OMPD_unknown:
    default:
      llvm_unreachable("Unexpected directive.");
    }
  }

  return nullptr;
}

/// Emit the user-defined mapper function. The code generation follows the
/// pattern in the example below.
/// \code
/// void .omp_mapper.<type_name>.<mapper_id>.(void *rt_mapper_handle,
///                                           void *base, void *begin,
///                                           int64_t size, int64_t type,
///                                           void *name = nullptr) {
///   // Allocate space for an array section first or add a base/begin for
///   // pointer dereference.
///   if ((size > 1 || (base != begin && maptype.IsPtrAndObj)) &&
///       !maptype.IsDelete)
///     __tgt_push_mapper_component(rt_mapper_handle, base, begin,
///                                 size*sizeof(Ty), clearToFromMember(type));
///   // Map members.
///   for (unsigned i = 0; i < size; i++) {
///     // For each component specified by this mapper:
///     for (auto c : begin[i]->all_components) {
///       if (c.hasMapper())
///         (*c.Mapper())(rt_mapper_handle, c.arg_base, c.arg_begin, c.arg_size,
///                       c.arg_type, c.arg_name);
///       else
///         __tgt_push_mapper_component(rt_mapper_handle, c.arg_base,
///                                     c.arg_begin, c.arg_size, c.arg_type,
///                                     c.arg_name);
///     }
///   }
///   // Delete the array section.
///   if (size > 1 && maptype.IsDelete)
///     __tgt_push_mapper_component(rt_mapper_handle, base, begin,
///                                 size*sizeof(Ty), clearToFromMember(type));
/// }
/// \endcode
void CGOpenMPRuntime::emitUserDefinedMapper(const OMPDeclareMapperDecl *D,
                                            CodeGenFunction *CGF) {
  if (UDMMap.count(D) > 0)
    return;
  ASTContext &C = CGM.getContext();
  QualType Ty = D->getType();
  QualType PtrTy = C.getPointerType(Ty).withRestrict();
  QualType Int64Ty = C.getIntTypeForBitwidth(/*DestWidth=*/64, /*Signed=*/true);
  auto *MapperVarDecl =
      cast<VarDecl>(cast<DeclRefExpr>(D->getMapperVarRef())->getDecl());
  SourceLocation Loc = D->getLocation();
  CharUnits ElementSize = C.getTypeSizeInChars(Ty);

  // Prepare mapper function arguments and attributes.
  ImplicitParamDecl HandleArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr,
                              C.VoidPtrTy, ImplicitParamDecl::Other);
  ImplicitParamDecl BaseArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr, C.VoidPtrTy,
                            ImplicitParamDecl::Other);
  ImplicitParamDecl BeginArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr,
                             C.VoidPtrTy, ImplicitParamDecl::Other);
  ImplicitParamDecl SizeArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr, Int64Ty,
                            ImplicitParamDecl::Other);
  ImplicitParamDecl TypeArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr, Int64Ty,
                            ImplicitParamDecl::Other);
  ImplicitParamDecl NameArg(C, /*DC=*/nullptr, Loc, /*Id=*/nullptr, C.VoidPtrTy,
                            ImplicitParamDecl::Other);
  FunctionArgList Args;
  Args.push_back(&HandleArg);
  Args.push_back(&BaseArg);
  Args.push_back(&BeginArg);
  Args.push_back(&SizeArg);
  Args.push_back(&TypeArg);
  Args.push_back(&NameArg);
  const CGFunctionInfo &FnInfo =
      CGM.getTypes().arrangeBuiltinFunctionDeclaration(C.VoidTy, Args);
  llvm::FunctionType *FnTy = CGM.getTypes().GetFunctionType(FnInfo);
  SmallString<64> TyStr;
  llvm::raw_svector_ostream Out(TyStr);
  CGM.getCXXABI().getMangleContext().mangleTypeName(Ty, Out);
  std::string Name = getName({"omp_mapper", TyStr, D->getName()});
  auto *Fn = llvm::Function::Create(FnTy, llvm::GlobalValue::InternalLinkage,
                                    Name, &CGM.getModule());
  CGM.SetInternalFunctionAttributes(GlobalDecl(), Fn, FnInfo);
  Fn->removeFnAttr(llvm::Attribute::OptimizeNone);
  // Start the mapper function code generation.
  CodeGenFunction MapperCGF(CGM);
  MapperCGF.StartFunction(GlobalDecl(), C.VoidTy, Fn, FnInfo, Args, Loc, Loc);
  // Compute the starting and end addresses of array elements.
  llvm::Value *Size = MapperCGF.EmitLoadOfScalar(
      MapperCGF.GetAddrOfLocalVar(&SizeArg), /*Volatile=*/false,
      C.getPointerType(Int64Ty), Loc);
  // Prepare common arguments for array initiation and deletion.
  llvm::Value *Handle = MapperCGF.EmitLoadOfScalar(
      MapperCGF.GetAddrOfLocalVar(&HandleArg),
      /*Volatile=*/false, C.getPointerType(C.VoidPtrTy), Loc);
  llvm::Value *BaseIn = MapperCGF.EmitLoadOfScalar(
      MapperCGF.GetAddrOfLocalVar(&BaseArg),
      /*Volatile=*/false, C.getPointerType(C.VoidPtrTy), Loc);
  llvm::Value *BeginIn = MapperCGF.EmitLoadOfScalar(
      MapperCGF.GetAddrOfLocalVar(&BeginArg),
      /*Volatile=*/false, C.getPointerType(C.VoidPtrTy), Loc);
  // Convert the size in bytes into the number of array elements.
  Size = MapperCGF.Builder.CreateExactUDiv(
      Size, MapperCGF.Builder.getInt64(ElementSize.getQuantity()));
  llvm::Value *PtrBegin = MapperCGF.Builder.CreateBitCast(
      BeginIn, CGM.getTypes().ConvertTypeForMem(PtrTy));
  llvm::Value *PtrEnd = MapperCGF.Builder.CreateGEP(
      PtrBegin->getType()->getPointerElementType(), PtrBegin, Size);
  llvm::Value *MapType = MapperCGF.EmitLoadOfScalar(
      MapperCGF.GetAddrOfLocalVar(&TypeArg), /*Volatile=*/false,
      C.getPointerType(Int64Ty), Loc);
  llvm::Value *MapName = MapperCGF.EmitLoadOfScalar(
      MapperCGF.GetAddrOfLocalVar(&NameArg),
      /*Volatile=*/false, C.getPointerType(C.VoidPtrTy), Loc);

  // Emit array initiation if this is an array section and \p MapType indicates
  // that memory allocation is required.
  llvm::BasicBlock *HeadBB = MapperCGF.createBasicBlock("omp.arraymap.head");
  emitUDMapperArrayInitOrDel(MapperCGF, Handle, BaseIn, BeginIn, Size, MapType,
                             MapName, ElementSize, HeadBB, /*IsInit=*/true);

  // Emit a for loop to iterate through SizeArg of elements and map all of them.

  // Emit the loop header block.
  MapperCGF.EmitBlock(HeadBB);
  llvm::BasicBlock *BodyBB = MapperCGF.createBasicBlock("omp.arraymap.body");
  llvm::BasicBlock *DoneBB = MapperCGF.createBasicBlock("omp.done");
  // Evaluate whether the initial condition is satisfied.
  llvm::Value *IsEmpty =
      MapperCGF.Builder.CreateICmpEQ(PtrBegin, PtrEnd, "omp.arraymap.isempty");
  MapperCGF.Builder.CreateCondBr(IsEmpty, DoneBB, BodyBB);
  llvm::BasicBlock *EntryBB = MapperCGF.Builder.GetInsertBlock();

  // Emit the loop body block.
  MapperCGF.EmitBlock(BodyBB);
  llvm::BasicBlock *LastBB = BodyBB;
  llvm::PHINode *PtrPHI = MapperCGF.Builder.CreatePHI(
      PtrBegin->getType(), 2, "omp.arraymap.ptrcurrent");
  PtrPHI->addIncoming(PtrBegin, EntryBB);
  Address PtrCurrent =
      Address(PtrPHI, MapperCGF.GetAddrOfLocalVar(&BeginArg)
                          .getAlignment()
                          .alignmentOfArrayElement(ElementSize));
  // Privatize the declared variable of mapper to be the current array element.
  CodeGenFunction::OMPPrivateScope Scope(MapperCGF);
  Scope.addPrivate(MapperVarDecl, [PtrCurrent]() { return PtrCurrent; });
  (void)Scope.Privatize();

  // Get map clause information. Fill up the arrays with all mapped variables.
  MappableExprsHandler::MapCombinedInfoTy Info;
  MappableExprsHandler MEHandler(*D, MapperCGF);
  MEHandler.generateAllInfoForMapper(Info);

  // Call the runtime API __tgt_mapper_num_components to get the number of
  // pre-existing components.
  llvm::Value *OffloadingArgs[] = {Handle};
  llvm::Value *PreviousSize = MapperCGF.EmitRuntimeCall(
      OMPBuilder.getOrCreateRuntimeFunction(CGM.getModule(),
                                            OMPRTL___tgt_mapper_num_components),
      OffloadingArgs);
  llvm::Value *ShiftedPreviousSize = MapperCGF.Builder.CreateShl(
      PreviousSize,
      MapperCGF.Builder.getInt64(MappableExprsHandler::getFlagMemberOffset()));

  // Fill up the runtime mapper handle for all components.
  for (unsigned I = 0; I < Info.BasePointers.size(); ++I) {
    llvm::Value *CurBaseArg = MapperCGF.Builder.CreateBitCast(
        *Info.BasePointers[I], CGM.getTypes().ConvertTypeForMem(C.VoidPtrTy));
    llvm::Value *CurBeginArg = MapperCGF.Builder.CreateBitCast(
        Info.Pointers[I], CGM.getTypes().ConvertTypeForMem(C.VoidPtrTy));
    llvm::Value *CurSizeArg = Info.Sizes[I];
    llvm::Value *CurNameArg =
        (CGM.getCodeGenOpts().getDebugInfo() == codegenoptions::NoDebugInfo)
            ? llvm::ConstantPointerNull::get(CGM.VoidPtrTy)
            : emitMappingInformation(MapperCGF, OMPBuilder, Info.Exprs[I]);

    // Extract the MEMBER_OF field from the map type.
    llvm::Value *OriMapType = MapperCGF.Builder.getInt64(Info.Types[I]);
    llvm::Value *MemberMapType =
        MapperCGF.Builder.CreateNUWAdd(OriMapType, ShiftedPreviousSize);

    // Combine the map type inherited from user-defined mapper with that
    // specified in the program. According to the OMP_MAP_TO and OMP_MAP_FROM
    // bits of the \a MapType, which is the input argument of the mapper
    // function, the following code will set the OMP_MAP_TO and OMP_MAP_FROM
    // bits of MemberMapType.
    // [OpenMP 5.0], 1.2.6. map-type decay.
    //        | alloc |  to   | from  | tofrom | release | delete
    // ----------------------------------------------------------
    // alloc  | alloc | alloc | alloc | alloc  | release | delete
    // to     | alloc |  to   | alloc |   to   | release | delete
    // from   | alloc | alloc | from  |  from  | release | delete
    // tofrom | alloc |  to   | from  | tofrom | release | delete
    llvm::Value *LeftToFrom = MapperCGF.Builder.CreateAnd(
        MapType,
        MapperCGF.Builder.getInt64(MappableExprsHandler::OMP_MAP_TO |
                                   MappableExprsHandler::OMP_MAP_FROM));
    llvm::BasicBlock *AllocBB = MapperCGF.createBasicBlock("omp.type.alloc");
    llvm::BasicBlock *AllocElseBB =
        MapperCGF.createBasicBlock("omp.type.alloc.else");
    llvm::BasicBlock *ToBB = MapperCGF.createBasicBlock("omp.type.to");
    llvm::BasicBlock *ToElseBB = MapperCGF.createBasicBlock("omp.type.to.else");
    llvm::BasicBlock *FromBB = MapperCGF.createBasicBlock("omp.type.from");
    llvm::BasicBlock *EndBB = MapperCGF.createBasicBlock("omp.type.end");
    llvm::Value *IsAlloc = MapperCGF.Builder.CreateIsNull(LeftToFrom);
    MapperCGF.Builder.CreateCondBr(IsAlloc, AllocBB, AllocElseBB);
    // In case of alloc, clear OMP_MAP_TO and OMP_MAP_FROM.
    MapperCGF.EmitBlock(AllocBB);
    llvm::Value *AllocMapType = MapperCGF.Builder.CreateAnd(
        MemberMapType,
        MapperCGF.Builder.getInt64(~(MappableExprsHandler::OMP_MAP_TO |
                                     MappableExprsHandler::OMP_MAP_FROM)));
    MapperCGF.Builder.CreateBr(EndBB);
    MapperCGF.EmitBlock(AllocElseBB);
    llvm::Value *IsTo = MapperCGF.Builder.CreateICmpEQ(
        LeftToFrom,
        MapperCGF.Builder.getInt64(MappableExprsHandler::OMP_MAP_TO));
    MapperCGF.Builder.CreateCondBr(IsTo, ToBB, ToElseBB);
    // In case of to, clear OMP_MAP_FROM.
    MapperCGF.EmitBlock(ToBB);
    llvm::Value *ToMapType = MapperCGF.Builder.CreateAnd(
        MemberMapType,
        MapperCGF.Builder.getInt64(~MappableExprsHandler::OMP_MAP_FROM));
    MapperCGF.Builder.CreateBr(EndBB);
    MapperCGF.EmitBlock(ToElseBB);
    llvm::Value *IsFrom = MapperCGF.Builder.CreateICmpEQ(
        LeftToFrom,
        MapperCGF.Builder.getInt64(MappableExprsHandler::OMP_MAP_FROM));
    MapperCGF.Builder.CreateCondBr(IsFrom, FromBB, EndBB);
    // In case of from, clear OMP_MAP_TO.
    MapperCGF.EmitBlock(FromBB);
    llvm::Value *FromMapType = MapperCGF.Builder.CreateAnd(
        MemberMapType,
        MapperCGF.Builder.getInt64(~MappableExprsHandler::OMP_MAP_TO));
    // In case of tofrom, do nothing.
    MapperCGF.EmitBlock(EndBB);
    LastBB = EndBB;
    llvm::PHINode *CurMapType =
        MapperCGF.Builder.CreatePHI(CGM.Int64Ty, 4, "omp.maptype");
    CurMapType->addIncoming(AllocMapType, AllocBB);
    CurMapType->addIncoming(ToMapType, ToBB);
    CurMapType->addIncoming(FromMapType, FromBB);
    CurMapType->addIncoming(MemberMapType, ToElseBB);

    llvm::Value *OffloadingArgs[] = {Handle,     CurBaseArg, CurBeginArg,
                                     CurSizeArg, CurMapType, CurNameArg};
    if (Info.Mappers[I]) {
      // Call the corresponding mapper function.
      llvm::Function *MapperFunc = getOrCreateUserDefinedMapperFunc(
          cast<OMPDeclareMapperDecl>(Info.Mappers[I]));
      assert(MapperFunc && "Expect a valid mapper function is available.");
      MapperCGF.EmitNounwindRuntimeCall(MapperFunc, OffloadingArgs);
    } else {
      // Call the runtime API __tgt_push_mapper_component to fill up the runtime
      // data structure.
      MapperCGF.EmitRuntimeCall(
          OMPBuilder.getOrCreateRuntimeFunction(
              CGM.getModule(), OMPRTL___tgt_push_mapper_component),
          OffloadingArgs);
    }
  }

  // Update the pointer to point to the next element that needs to be mapped,
  // and check whether we have mapped all elements.
  llvm::Type *ElemTy = PtrPHI->getType()->getPointerElementType();
  llvm::Value *PtrNext = MapperCGF.Builder.CreateConstGEP1_32(
      ElemTy, PtrPHI, /*Idx0=*/1, "omp.arraymap.next");
  PtrPHI->addIncoming(PtrNext, LastBB);
  llvm::Value *IsDone =
      MapperCGF.Builder.CreateICmpEQ(PtrNext, PtrEnd, "omp.arraymap.isdone");
  llvm::BasicBlock *ExitBB = MapperCGF.createBasicBlock("omp.arraymap.exit");
  MapperCGF.Builder.CreateCondBr(IsDone, ExitBB, BodyBB);

  MapperCGF.EmitBlock(ExitBB);
  // Emit array deletion if this is an array section and \p MapType indicates
  // that deletion is required.
  emitUDMapperArrayInitOrDel(MapperCGF, Handle, BaseIn, BeginIn, Size, MapType,
                             MapName, ElementSize, DoneBB, /*IsInit=*/false);

  // Emit the function exit block.
  MapperCGF.EmitBlock(DoneBB, /*IsFinished=*/true);
  MapperCGF.FinishFunction();
  UDMMap.try_emplace(D, Fn);
  if (CGF) {
    auto &Decls = FunctionUDMMap.FindAndConstruct(CGF->CurFn);
    Decls.second.push_back(D);
  }
}

/// Emit the array initialization or deletion portion for user-defined mapper
/// code generation. First, it evaluates whether an array section is mapped and
/// whether the \a MapType instructs to delete this section. If \a IsInit is
/// true, and \a MapType indicates to not delete this array, array
/// initialization code is generated. If \a IsInit is false, and \a MapType
/// indicates to not this array, array deletion code is generated.
void CGOpenMPRuntime::emitUDMapperArrayInitOrDel(
    CodeGenFunction &MapperCGF, llvm::Value *Handle, llvm::Value *Base,
    llvm::Value *Begin, llvm::Value *Size, llvm::Value *MapType,
    llvm::Value *MapName, CharUnits ElementSize, llvm::BasicBlock *ExitBB,
    bool IsInit) {
  StringRef Prefix = IsInit ? ".init" : ".del";

  // Evaluate if this is an array section.
  llvm::BasicBlock *BodyBB =
      MapperCGF.createBasicBlock(getName({"omp.array", Prefix}));
  llvm::Value *IsArray = MapperCGF.Builder.CreateICmpSGT(
      Size, MapperCGF.Builder.getInt64(1), "omp.arrayinit.isarray");
  llvm::Value *DeleteBit = MapperCGF.Builder.CreateAnd(
      MapType,
      MapperCGF.Builder.getInt64(MappableExprsHandler::OMP_MAP_DELETE));
  llvm::Value *DeleteCond;
  llvm::Value *Cond;
  if (IsInit) {
    // base != begin?
    llvm::Value *BaseIsBegin = MapperCGF.Builder.CreateIsNotNull(
        MapperCGF.Builder.CreatePtrDiff(Base, Begin));
    // IsPtrAndObj?
    llvm::Value *PtrAndObjBit = MapperCGF.Builder.CreateAnd(
        MapType,
        MapperCGF.Builder.getInt64(MappableExprsHandler::OMP_MAP_PTR_AND_OBJ));
    PtrAndObjBit = MapperCGF.Builder.CreateIsNotNull(PtrAndObjBit);
    BaseIsBegin = MapperCGF.Builder.CreateAnd(BaseIsBegin, PtrAndObjBit);
    Cond = MapperCGF.Builder.CreateOr(IsArray, BaseIsBegin);
    DeleteCond = MapperCGF.Builder.CreateIsNull(
        DeleteBit, getName({"omp.array", Prefix, ".delete"}));
  } else {
    Cond = IsArray;
    DeleteCond = MapperCGF.Builder.CreateIsNotNull(
        DeleteBit, getName({"omp.array", Prefix, ".delete"}));
  }
  Cond = MapperCGF.Builder.CreateAnd(Cond, DeleteCond);
  MapperCGF.Builder.CreateCondBr(Cond, BodyBB, ExitBB);

  MapperCGF.EmitBlock(BodyBB);
  // Get the array size by multiplying element size and element number (i.e., \p
  // Size).
  llvm::Value *ArraySize = MapperCGF.Builder.CreateNUWMul(
      Size, MapperCGF.Builder.getInt64(ElementSize.getQuantity()));
  // Remove OMP_MAP_TO and OMP_MAP_FROM from the map type, so that it achieves
  // memory allocation/deletion purpose only.
  llvm::Value *MapTypeArg = MapperCGF.Builder.CreateAnd(
      MapType,
      MapperCGF.Builder.getInt64(~(MappableExprsHandler::OMP_MAP_TO |
                                   MappableExprsHandler::OMP_MAP_FROM)));
  MapTypeArg = MapperCGF.Builder.CreateOr(
      MapTypeArg,
      MapperCGF.Builder.getInt64(MappableExprsHandler::OMP_MAP_IMPLICIT));

  // Call the runtime API __tgt_push_mapper_component to fill up the runtime
  // data structure.
  llvm::Value *OffloadingArgs[] = {Handle,    Base,       Begin,
                                   ArraySize, MapTypeArg, MapName};
  MapperCGF.EmitRuntimeCall(
      OMPBuilder.getOrCreateRuntimeFunction(CGM.getModule(),
                                            OMPRTL___tgt_push_mapper_component),
      OffloadingArgs);
}

llvm::Function *CGOpenMPRuntime::getOrCreateUserDefinedMapperFunc(
    const OMPDeclareMapperDecl *D) {
  auto I = UDMMap.find(D);
  if (I != UDMMap.end())
    return I->second;
  emitUserDefinedMapper(D);
  return UDMMap.lookup(D);
}

void CGOpenMPRuntime::emitTargetNumIterationsCall(
    CodeGenFunction &CGF, const OMPExecutableDirective &D,
    llvm::Value *DeviceID,
    llvm::function_ref<llvm::Value *(CodeGenFunction &CGF,
                                     const OMPLoopDirective &D)>
        SizeEmitter) {
  OpenMPDirectiveKind Kind = D.getDirectiveKind();
  const OMPExecutableDirective *TD = &D;
  // Get nested teams distribute kind directive, if any.
  if (!isOpenMPDistributeDirective(Kind) || !isOpenMPTeamsDirective(Kind))
    TD = getNestedDistributeDirective(CGM.getContext(), D);
  if (!TD)
    return;
  const auto *LD = cast<OMPLoopDirective>(TD);
  auto &&CodeGen = [LD, DeviceID, SizeEmitter, &D, this](CodeGenFunction &CGF,
                                                         PrePostActionTy &) {
    if (llvm::Value *NumIterations = SizeEmitter(CGF, *LD)) {
      llvm::Value *RTLoc = emitUpdateLocation(CGF, D.getBeginLoc());
      llvm::Value *Args[] = {RTLoc, DeviceID, NumIterations};
      CGF.EmitRuntimeCall(
          OMPBuilder.getOrCreateRuntimeFunction(
              CGM.getModule(), OMPRTL___kmpc_push_target_tripcount_mapper),
          Args);
    }
  };
  emitInlinedDirective(CGF, OMPD_unknown, CodeGen);
}

void CGOpenMPRuntime::emitTargetCall(
    CodeGenFunction &CGF, const OMPExecutableDirective &D,
    llvm::Function *OutlinedFn, llvm::Value *OutlinedFnID, const Expr *IfCond,
    llvm::PointerIntPair<const Expr *, 2, OpenMPDeviceClauseModifier> Device,
    llvm::function_ref<llvm::Value *(CodeGenFunction &CGF,
                                     const OMPLoopDirective &D)>
        SizeEmitter) {
  if (!CGF.HaveInsertPoint())
    return;

  assert(OutlinedFn && "Invalid outlined function!");

  const bool RequiresOuterTask = D.hasClausesOfKind<OMPDependClause>() ||
                                 D.hasClausesOfKind<OMPNowaitClause>();
  llvm::SmallVector<llvm::Value *, 16> CapturedVars;
  const CapturedStmt &CS = *D.getCapturedStmt(OMPD_target);
  auto &&ArgsCodegen = [&CS, &CapturedVars](CodeGenFunction &CGF,
                                            PrePostActionTy &) {
    CGF.GenerateOpenMPCapturedVars(CS, CapturedVars);
  };
  emitInlinedDirective(CGF, OMPD_unknown, ArgsCodegen);

  CodeGenFunction::OMPTargetDataInfo InputInfo;
  llvm::Value *MapTypesArray = nullptr;
  llvm::Value *MapNamesArray = nullptr;
  // Fill up the pointer arrays and transfer execution to the device.
  auto &&ThenGen = [this, Device, OutlinedFn, OutlinedFnID, &D, &InputInfo,
                    &MapTypesArray, &MapNamesArray, &CS, RequiresOuterTask,
                    &CapturedVars,
                    SizeEmitter](CodeGenFunction &CGF, PrePostActionTy &) {
    if (Device.getInt() == OMPC_DEVICE_ancestor) {
      // Reverse offloading is not supported, so just execute on the host.
      if (RequiresOuterTask) {
        CapturedVars.clear();
        CGF.GenerateOpenMPCapturedVars(CS, CapturedVars);
      }
      emitOutlinedFunctionCall(CGF, D.getBeginLoc(), OutlinedFn, CapturedVars);
      return;
    }

    // On top of the arrays that were filled up, the target offloading call
    // takes as arguments the device id as well as the host pointer. The host
    // pointer is used by the runtime library to identify the current target
    // region, so it only has to be unique and not necessarily point to
    // anything. It could be the pointer to the outlined function that
    // implements the target region, but we aren't using that so that the
    // compiler doesn't need to keep that, and could therefore inline the host
    // function if proven worthwhile during optimization.

    // From this point on, we need to have an ID of the target region defined.
    assert(OutlinedFnID && "Invalid outlined function ID!");

    // Emit device ID if any.
    llvm::Value *DeviceID;
    if (Device.getPointer()) {
      assert((Device.getInt() == OMPC_DEVICE_unknown ||
              Device.getInt() == OMPC_DEVICE_device_num) &&
             "Expected device_num modifier.");
      llvm::Value *DevVal = CGF.EmitScalarExpr(Device.getPointer());
      DeviceID =
          CGF.Builder.CreateIntCast(DevVal, CGF.Int64Ty, /*isSigned=*/true);
    } else {
      DeviceID = CGF.Builder.getInt64(OMP_DEVICEID_UNDEF);
    }

    // Emit the number of elements in the offloading arrays.
    llvm::Value *PointerNum =
        CGF.Builder.getInt32(InputInfo.NumberOfTargetItems);

    // Return value of the runtime offloading call.
    llvm::Value *Return;

    llvm::Value *NumTeams = emitNumTeamsForTargetDirective(CGF, D);
    llvm::Value *NumThreads = emitNumThreadsForTargetDirective(CGF, D);

    // Source location for the ident struct
    llvm::Value *RTLoc = emitUpdateLocation(CGF, D.getBeginLoc());

    // Emit tripcount for the target loop-based directive.
    emitTargetNumIterationsCall(CGF, D, DeviceID, SizeEmitter);

    bool HasNowait = D.hasClausesOfKind<OMPNowaitClause>();
    // The target region is an outlined function launched by the runtime
    // via calls __tgt_target() or __tgt_target_teams().
    //
    // __tgt_target() launches a target region with one team and one thread,
    // executing a serial region.  This master thread may in turn launch
    // more threads within its team upon encountering a parallel region,
    // however, no additional teams can be launched on the device.
    //
    // __tgt_target_teams() launches a target region with one or more teams,
    // each with one or more threads.  This call is required for target
    // constructs such as:
    //  'target teams'
    //  'target' / 'teams'
    //  'target teams distribute parallel for'
    //  'target parallel'
    // and so on.
    //
    // Note that on the host and CPU targets, the runtime implementation of
    // these calls simply call the outlined function without forking threads.
    // The outlined functions themselves have runtime calls to
    // __kmpc_fork_teams() and __kmpc_fork() for this purpose, codegen'd by
    // the compiler in emitTeamsCall() and emitParallelCall().
    //
    // In contrast, on the NVPTX target, the implementation of
    // __tgt_target_teams() launches a GPU kernel with the requested number
    // of teams and threads so no additional calls to the runtime are required.
    if (NumTeams) {
      // If we have NumTeams defined this means that we have an enclosed teams
      // region. Therefore we also expect to have NumThreads defined. These two
      // values should be defined in the presence of a teams directive,
      // regardless of having any clauses associated. If the user is using teams
      // but no clauses, these two values will be the default that should be
      // passed to the runtime library - a 32-bit integer with the value zero.
      assert(NumThreads && "Thread limit expression should be available along "
                           "with number of teams.");
      SmallVector<llvm::Value *> OffloadingArgs = {
          RTLoc,
          DeviceID,
          OutlinedFnID,
          PointerNum,
          InputInfo.BasePointersArray.getPointer(),
          InputInfo.PointersArray.getPointer(),
          InputInfo.SizesArray.getPointer(),
          MapTypesArray,
          MapNamesArray,
          InputInfo.MappersArray.getPointer(),
          NumTeams,
          NumThreads};
      if (HasNowait) {
        // Add int32_t depNum = 0, void *depList = nullptr, int32_t
        // noAliasDepNum = 0, void *noAliasDepList = nullptr.
        OffloadingArgs.push_back(CGF.Builder.getInt32(0));
        OffloadingArgs.push_back(llvm::ConstantPointerNull::get(CGM.VoidPtrTy));
        OffloadingArgs.push_back(CGF.Builder.getInt32(0));
        OffloadingArgs.push_back(llvm::ConstantPointerNull::get(CGM.VoidPtrTy));
      }
      Return = CGF.EmitRuntimeCall(
          OMPBuilder.getOrCreateRuntimeFunction(
              CGM.getModule(), HasNowait
                                   ? OMPRTL___tgt_target_teams_nowait_mapper
                                   : OMPRTL___tgt_target_teams_mapper),
          OffloadingArgs);
    } else {
      SmallVector<llvm::Value *> OffloadingArgs = {
          RTLoc,
          DeviceID,
          OutlinedFnID,
          PointerNum,
          InputInfo.BasePointersArray.getPointer(),
          InputInfo.PointersArray.getPointer(),
          InputInfo.SizesArray.getPointer(),
          MapTypesArray,
          MapNamesArray,
          InputInfo.MappersArray.getPointer()};
      if (HasNowait) {
        // Add int32_t depNum = 0, void *depList = nullptr, int32_t
        // noAliasDepNum = 0, void *noAliasDepList = nullptr.
        OffloadingArgs.push_back(CGF.Builder.getInt32(0));
        OffloadingArgs.push_back(llvm::ConstantPointerNull::get(CGM.VoidPtrTy));
        OffloadingArgs.push_back(CGF.Builder.getInt32(0));
        OffloadingArgs.push_back(llvm::ConstantPointerNull::get(CGM.VoidPtrTy));
      }
      Return = CGF.EmitRuntimeCall(
          OMPBuilder.getOrCreateRuntimeFunction(
              CGM.getModule(), HasNowait ? OMPRTL___tgt_target_nowait_mapper
                                         : OMPRTL___tgt_target_mapper),
          OffloadingArgs);
    }

    // Check the error code and execute the host version if required.
    llvm::BasicBlock *OffloadFailedBlock =
        CGF.createBasicBlock("omp_offload.failed");
    llvm::BasicBlock *OffloadContBlock =
        CGF.createBasicBlock("omp_offload.cont");
    llvm::Value *Failed = CGF.Builder.CreateIsNotNull(Return);
    CGF.Builder.CreateCondBr(Failed, OffloadFailedBlock, OffloadContBlock);

    CGF.EmitBlock(OffloadFailedBlock);
    if (RequiresOuterTask) {
      CapturedVars.clear();
      CGF.GenerateOpenMPCapturedVars(CS, CapturedVars);
    }
    emitOutlinedFunctionCall(CGF, D.getBeginLoc(), OutlinedFn, CapturedVars);
    CGF.EmitBranch(OffloadContBlock);

    CGF.EmitBlock(OffloadContBlock, /*IsFinished=*/true);
  };

  // Notify that the host version must be executed.
  auto &&ElseGen = [this, &D, OutlinedFn, &CS, &CapturedVars,
                    RequiresOuterTask](CodeGenFunction &CGF,
                                       PrePostActionTy &) {
    if (RequiresOuterTask) {
      CapturedVars.clear();
      CGF.GenerateOpenMPCapturedVars(CS, CapturedVars);
    }
    emitOutlinedFunctionCall(CGF, D.getBeginLoc(), OutlinedFn, CapturedVars);
  };

  auto &&TargetThenGen = [this, &ThenGen, &D, &InputInfo, &MapTypesArray,
                          &MapNamesArray, &CapturedVars, RequiresOuterTask,
                          &CS](CodeGenFunction &CGF, PrePostActionTy &) {
    // Fill up the arrays with all the captured variables.
    MappableExprsHandler::MapCombinedInfoTy CombinedInfo;

    // Get mappable expression information.
    MappableExprsHandler MEHandler(D, CGF);
    llvm::DenseMap<llvm::Value *, llvm::Value *> LambdaPointers;
    llvm::DenseSet<CanonicalDeclPtr<const Decl>> MappedVarSet;

    auto RI = CS.getCapturedRecordDecl()->field_begin();
    auto *CV = CapturedVars.begin();
    for (CapturedStmt::const_capture_iterator CI = CS.capture_begin(),
                                              CE = CS.capture_end();
         CI != CE; ++CI, ++RI, ++CV) {
      MappableExprsHandler::MapCombinedInfoTy CurInfo;
      MappableExprsHandler::StructRangeInfoTy PartialStruct;

      // VLA sizes are passed to the outlined region by copy and do not have map
      // information associated.
      if (CI->capturesVariableArrayType()) {
        CurInfo.Exprs.push_back(nullptr);
        CurInfo.BasePointers.push_back(*CV);
        CurInfo.Pointers.push_back(*CV);
        CurInfo.Sizes.push_back(CGF.Builder.CreateIntCast(
            CGF.getTypeSize(RI->getType()), CGF.Int64Ty, /*isSigned=*/true));
        // Copy to the device as an argument. No need to retrieve it.
        CurInfo.Types.push_back(MappableExprsHandler::OMP_MAP_LITERAL |
                                MappableExprsHandler::OMP_MAP_TARGET_PARAM |
                                MappableExprsHandler::OMP_MAP_IMPLICIT);
        CurInfo.Mappers.push_back(nullptr);
      } else {
        // If we have any information in the map clause, we use it, otherwise we
        // just do a default mapping.
        MEHandler.generateInfoForCapture(CI, *CV, CurInfo, PartialStruct);
        if (!CI->capturesThis())
          MappedVarSet.insert(CI->getCapturedVar());
        else
          MappedVarSet.insert(nullptr);
        if (CurInfo.BasePointers.empty() && !PartialStruct.Base.isValid())
          MEHandler.generateDefaultMapInfo(*CI, **RI, *CV, CurInfo);
        // Generate correct mapping for variables captured by reference in
        // lambdas.
        if (CI->capturesVariable())
          MEHandler.generateInfoForLambdaCaptures(CI->getCapturedVar(), *CV,
                                                  CurInfo, LambdaPointers);
      }
      // We expect to have at least an element of information for this capture.
      assert((!CurInfo.BasePointers.empty() || PartialStruct.Base.isValid()) &&
             "Non-existing map pointer for capture!");
      assert(CurInfo.BasePointers.size() == CurInfo.Pointers.size() &&
             CurInfo.BasePointers.size() == CurInfo.Sizes.size() &&
             CurInfo.BasePointers.size() == CurInfo.Types.size() &&
             CurInfo.BasePointers.size() == CurInfo.Mappers.size() &&
             "Inconsistent map information sizes!");

      // If there is an entry in PartialStruct it means we have a struct with
      // individual members mapped. Emit an extra combined entry.
      if (PartialStruct.Base.isValid()) {
        CombinedInfo.append(PartialStruct.PreliminaryMapData);
        MEHandler.emitCombinedEntry(
            CombinedInfo, CurInfo.Types, PartialStruct, nullptr,
            !PartialStruct.PreliminaryMapData.BasePointers.empty());
      }

      // We need to append the results of this capture to what we already have.
      CombinedInfo.append(CurInfo);
    }
    // Adjust MEMBER_OF flags for the lambdas captures.
    MEHandler.adjustMemberOfForLambdaCaptures(
        LambdaPointers, CombinedInfo.BasePointers, CombinedInfo.Pointers,
        CombinedInfo.Types);
    // Map any list items in a map clause that were not captures because they
    // weren't referenced within the construct.
    MEHandler.generateAllInfo(CombinedInfo, MappedVarSet);

    TargetDataInfo Info;
    // Fill up the arrays and create the arguments.
    emitOffloadingArrays(CGF, CombinedInfo, Info, OMPBuilder);
    emitOffloadingArraysArgument(
        CGF, Info.BasePointersArray, Info.PointersArray, Info.SizesArray,
        Info.MapTypesArray, Info.MapNamesArray, Info.MappersArray, Info,
        {/*ForEndCall=*/false});

    InputInfo.NumberOfTargetItems = Info.NumberOfPtrs;
    InputInfo.BasePointersArray =
        Address(Info.BasePointersArray, CGM.getPointerAlign());
    InputInfo.PointersArray =
        Address(Info.PointersArray, CGM.getPointerAlign());
    InputInfo.SizesArray = Address(Info.SizesArray, CGM.getPointerAlign());
    InputInfo.MappersArray = Address(Info.MappersArray, CGM.getPointerAlign());
    MapTypesArray = Info.MapTypesArray;
    MapNamesArray = Info.MapNamesArray;
    if (RequiresOuterTask)
      CGF.EmitOMPTargetTaskBasedDirective(D, ThenGen, InputInfo);
    else
      emitInlinedDirective(CGF, D.getDirectiveKind(), ThenGen);
  };

  auto &&TargetElseGen = [this, &ElseGen, &D, RequiresOuterTask](
                             CodeGenFunction &CGF, PrePostActionTy &) {
    if (RequiresOuterTask) {
      CodeGenFunction::OMPTargetDataInfo InputInfo;
      CGF.EmitOMPTargetTaskBasedDirective(D, ElseGen, InputInfo);
    } else {
      emitInlinedDirective(CGF, D.getDirectiveKind(), ElseGen);
    }
  };

  // If we have a target function ID it means that we need to support
  // offloading, otherwise, just execute on the host. We need to execute on host
  // regardless of the conditional in the if clause if, e.g., the user do not
  // specify target triples.
  if (OutlinedFnID) {
    if (IfCond) {
      emitIfClause(CGF, IfCond, TargetThenGen, TargetElseGen);
    } else {
      RegionCodeGenTy ThenRCG(TargetThenGen);
      ThenRCG(CGF);
    }
  } else {
    RegionCodeGenTy ElseRCG(TargetElseGen);
    ElseRCG(CGF);
  }
}

void CGOpenMPRuntime::scanForTargetRegionsFunctions(const Stmt *S,
                                                    StringRef ParentName) {
  if (!S)
    return;

  // Codegen OMP target directives that offload compute to the device.
  bool RequiresDeviceCodegen =
      isa<OMPExecutableDirective>(S) &&
      isOpenMPTargetExecutionDirective(
          cast<OMPExecutableDirective>(S)->getDirectiveKind());

  if (RequiresDeviceCodegen) {
    const auto &E = *cast<OMPExecutableDirective>(S);
    unsigned DeviceID;
    unsigned FileID;
    unsigned Line;
    getTargetEntryUniqueInfo(CGM.getContext(), E.getBeginLoc(), DeviceID,
                             FileID, Line);

    // Is this a target region that should not be emitted as an entry point? If
    // so just signal we are done with this target region.
    if (!OffloadEntriesInfoManager.hasTargetRegionEntryInfo(DeviceID, FileID,
                                                            ParentName, Line))
      return;

    switch (E.getDirectiveKind()) {
    case OMPD_target:
      CodeGenFunction::EmitOMPTargetDeviceFunction(CGM, ParentName,
                                                   cast<OMPTargetDirective>(E));
      break;
    case OMPD_target_parallel:
      CodeGenFunction::EmitOMPTargetParallelDeviceFunction(
          CGM, ParentName, cast<OMPTargetParallelDirective>(E));
      break;
    case OMPD_target_teams:
      CodeGenFunction::EmitOMPTargetTeamsDeviceFunction(
          CGM, ParentName, cast<OMPTargetTeamsDirective>(E));
      break;
    case OMPD_target_teams_distribute:
      CodeGenFunction::EmitOMPTargetTeamsDistributeDeviceFunction(
          CGM, ParentName, cast<OMPTargetTeamsDistributeDirective>(E));
      break;
    case OMPD_target_teams_distribute_simd:
      CodeGenFunction::EmitOMPTargetTeamsDistributeSimdDeviceFunction(
          CGM, ParentName, cast<OMPTargetTeamsDistributeSimdDirective>(E));
      break;
    case OMPD_target_parallel_for:
      CodeGenFunction::EmitOMPTargetParallelForDeviceFunction(
          CGM, ParentName, cast<OMPTargetParallelForDirective>(E));
      break;
    case OMPD_target_parallel_for_simd:
      CodeGenFunction::EmitOMPTargetParallelForSimdDeviceFunction(
          CGM, ParentName, cast<OMPTargetParallelForSimdDirective>(E));
      break;
    case OMPD_target_simd:
      CodeGenFunction::EmitOMPTargetSimdDeviceFunction(
          CGM, ParentName, cast<OMPTargetSimdDirective>(E));
      break;
    case OMPD_target_teams_distribute_parallel_for:
      CodeGenFunction::EmitOMPTargetTeamsDistributeParallelForDeviceFunction(
          CGM, ParentName,
          cast<OMPTargetTeamsDistributeParallelForDirective>(E));
      break;
    case OMPD_target_teams_distribute_parallel_for_simd:
      CodeGenFunction::
          EmitOMPTargetTeamsDistributeParallelForSimdDeviceFunction(
              CGM, ParentName,
              cast<OMPTargetTeamsDistributeParallelForSimdDirective>(E));
      break;
    case OMPD_parallel:
    case OMPD_for:
    case OMPD_parallel_for:
    case OMPD_parallel_master:
    case OMPD_parallel_sections:
    case OMPD_for_simd:
    case OMPD_parallel_for_simd:
    case OMPD_cancel:
    case OMPD_cancellation_point:
    case OMPD_ordered:
    case OMPD_threadprivate:
    case OMPD_allocate:
    case OMPD_task:
    case OMPD_simd:
    case OMPD_tile:
    case OMPD_unroll:
    case OMPD_sections:
    case OMPD_section:
    case OMPD_single:
    case OMPD_master:
    case OMPD_critical:
    case OMPD_taskyield:
    case OMPD_barrier:
    case OMPD_taskwait:
    case OMPD_taskgroup:
    case OMPD_atomic:
    case OMPD_flush:
    case OMPD_depobj:
    case OMPD_scan:
    case OMPD_teams:
    case OMPD_target_data:
    case OMPD_target_exit_data:
    case OMPD_target_enter_data:
    case OMPD_distribute:
    case OMPD_distribute_simd:
    case OMPD_distribute_parallel_for:
    case OMPD_distribute_parallel_for_simd:
    case OMPD_teams_distribute:
    case OMPD_teams_distribute_simd:
    case OMPD_teams_distribute_parallel_for:
    case OMPD_teams_distribute_parallel_for_simd:
    case OMPD_target_update:
    case OMPD_declare_simd:
    case OMPD_declare_variant:
    case OMPD_begin_declare_variant:
    case OMPD_end_declare_variant:
    case OMPD_declare_target:
    case OMPD_end_declare_target:
    case OMPD_declare_reduction:
    case OMPD_declare_mapper:
    case OMPD_taskloop:
    case OMPD_taskloop_simd:
    case OMPD_master_taskloop:
    case OMPD_master_taskloop_simd:
    case OMPD_parallel_master_taskloop:
    case OMPD_parallel_master_taskloop_simd:
    case OMPD_requires:
    case OMPD_metadirective:
    case OMPD_unknown:
    default:
      llvm_unreachable("Unknown target directive for OpenMP device codegen.");
    }
    return;
  }

  if (const auto *E = dyn_cast<OMPExecutableDirective>(S)) {
    if (!E->hasAssociatedStmt() || !E->getAssociatedStmt())
      return;

    scanForTargetRegionsFunctions(E->getRawStmt(), ParentName);
    return;
  }

  // If this is a lambda function, look into its body.
  if (const auto *L = dyn_cast<LambdaExpr>(S))
    S = L->getBody();

  // Keep looking for target regions recursively.
  for (const Stmt *II : S->children())
    scanForTargetRegionsFunctions(II, ParentName);
}

static bool isAssumedToBeNotEmitted(const ValueDecl *VD, bool IsDevice) {
  Optional<OMPDeclareTargetDeclAttr::DevTypeTy> DevTy =
      OMPDeclareTargetDeclAttr::getDeviceType(VD);
  if (!DevTy)
    return false;
  // Do not emit device_type(nohost) functions for the host.
  if (!IsDevice && DevTy == OMPDeclareTargetDeclAttr::DT_NoHost)
    return true;
  // Do not emit device_type(host) functions for the device.
  if (IsDevice && DevTy == OMPDeclareTargetDeclAttr::DT_Host)
    return true;
  return false;
}

bool CGOpenMPRuntime::emitTargetFunctions(GlobalDecl GD) {
  // If emitting code for the host, we do not process FD here. Instead we do
  // the normal code generation.
  if (!CGM.getLangOpts().OpenMPIsDevice) {
    if (const auto *FD = dyn_cast<FunctionDecl>(GD.getDecl()))
      if (isAssumedToBeNotEmitted(cast<ValueDecl>(FD),
                                  CGM.getLangOpts().OpenMPIsDevice))
        return true;
    return false;
  }

  const ValueDecl *VD = cast<ValueDecl>(GD.getDecl());
  // Try to detect target regions in the function.
  if (const auto *FD = dyn_cast<FunctionDecl>(VD)) {
    StringRef Name = CGM.getMangledName(GD);
    scanForTargetRegionsFunctions(FD->getBody(), Name);
    if (isAssumedToBeNotEmitted(cast<ValueDecl>(FD),
                                CGM.getLangOpts().OpenMPIsDevice))
      return true;
  }

  // Do not to emit function if it is not marked as declare target.
  return !OMPDeclareTargetDeclAttr::isDeclareTargetDeclaration(VD) &&
         AlreadyEmittedTargetDecls.count(VD) == 0;
}

bool CGOpenMPRuntime::emitTargetGlobalVariable(GlobalDecl GD) {
  if (isAssumedToBeNotEmitted(cast<ValueDecl>(GD.getDecl()),
                              CGM.getLangOpts().OpenMPIsDevice))
    return true;

  if (!CGM.getLangOpts().OpenMPIsDevice)
    return false;

  // Check if there are Ctors/Dtors in this declaration and look for target
  // regions in it. We use the complete variant to produce the kernel name
  // mangling.
  QualType RDTy = cast<VarDecl>(GD.getDecl())->getType();
  if (const auto *RD = RDTy->getBaseElementTypeUnsafe()->getAsCXXRecordDecl()) {
    for (const CXXConstructorDecl *Ctor : RD->ctors()) {
      StringRef ParentName =
          CGM.getMangledName(GlobalDecl(Ctor, Ctor_Complete));
      scanForTargetRegionsFunctions(Ctor->getBody(), ParentName);
    }
    if (const CXXDestructorDecl *Dtor = RD->getDestructor()) {
      StringRef ParentName =
          CGM.getMangledName(GlobalDecl(Dtor, Dtor_Complete));
      scanForTargetRegionsFunctions(Dtor->getBody(), ParentName);
    }
  }

  // Do not to emit variable if it is not marked as declare target.
  llvm::Optional<OMPDeclareTargetDeclAttr::MapTypeTy> Res =
      OMPDeclareTargetDeclAttr::isDeclareTargetDeclaration(
          cast<VarDecl>(GD.getDecl()));
  if (!Res || *Res == OMPDeclareTargetDeclAttr::MT_Link ||
      (*Res == OMPDeclareTargetDeclAttr::MT_To &&
       HasRequiresUnifiedSharedMemory)) {
    DeferredGlobalVariables.insert(cast<VarDecl>(GD.getDecl()));
    return true;
  }
  return false;
}

void CGOpenMPRuntime::registerTargetGlobalVariable(const VarDecl *VD,
                                                   llvm::Constant *Addr) {
  if (CGM.getLangOpts().OMPTargetTriples.empty() &&
      !CGM.getLangOpts().OpenMPIsDevice)
    return;

  // If we have host/nohost variables, they do not need to be registered.
  Optional<OMPDeclareTargetDeclAttr::DevTypeTy> DevTy =
      OMPDeclareTargetDeclAttr::getDeviceType(VD);
  if (DevTy && DevTy.getValue() != OMPDeclareTargetDeclAttr::DT_Any)
    return;

  llvm::Optional<OMPDeclareTargetDeclAttr::MapTypeTy> Res =
      OMPDeclareTargetDeclAttr::isDeclareTargetDeclaration(VD);
  if (!Res) {
    if (CGM.getLangOpts().OpenMPIsDevice) {
      // Register non-target variables being emitted in device code (debug info
      // may cause this).
      StringRef VarName = CGM.getMangledName(VD);
      EmittedNonTargetVariables.try_emplace(VarName, Addr);
    }
    return;
  }
  // Register declare target variables.
  OffloadEntriesInfoManagerTy::OMPTargetGlobalVarEntryKind Flags;
  StringRef VarName;
  CharUnits VarSize;
  llvm::GlobalValue::LinkageTypes Linkage;

  if (*Res == OMPDeclareTargetDeclAttr::MT_To &&
      !HasRequiresUnifiedSharedMemory) {
    Flags = OffloadEntriesInfoManagerTy::OMPTargetGlobalVarEntryTo;
    VarName = CGM.getMangledName(VD);
    if (VD->hasDefinition(CGM.getContext()) != VarDecl::DeclarationOnly) {
      VarSize = CGM.getContext().getTypeSizeInChars(VD->getType());
      assert(!VarSize.isZero() && "Expected non-zero size of the variable");
    } else {
      VarSize = CharUnits::Zero();
    }
    Linkage = CGM.getLLVMLinkageVarDefinition(VD, /*IsConstant=*/false);
    // Temp solution to prevent optimizations of the internal variables.
    if (CGM.getLangOpts().OpenMPIsDevice && !VD->isExternallyVisible()) {
      // Do not create a "ref-variable" if the original is not also available
      // on the host.
      if (!OffloadEntriesInfoManager.hasDeviceGlobalVarEntryInfo(VarName))
        return;
      std::string RefName = getName({VarName, "ref"});
      if (!CGM.GetGlobalValue(RefName)) {
        llvm::Constant *AddrRef =
            getOrCreateInternalVariable(Addr->getType(), RefName);
        auto *GVAddrRef = cast<llvm::GlobalVariable>(AddrRef);
        GVAddrRef->setConstant(/*Val=*/true);
        GVAddrRef->setLinkage(llvm::GlobalValue::InternalLinkage);
        GVAddrRef->setInitializer(Addr);
        CGM.addCompilerUsedGlobal(GVAddrRef);
      }
    }
  } else {
    assert(((*Res == OMPDeclareTargetDeclAttr::MT_Link) ||
            (*Res == OMPDeclareTargetDeclAttr::MT_To &&
             HasRequiresUnifiedSharedMemory)) &&
           "Declare target attribute must link or to with unified memory.");
    if (*Res == OMPDeclareTargetDeclAttr::MT_Link)
      Flags = OffloadEntriesInfoManagerTy::OMPTargetGlobalVarEntryLink;
    else
      Flags = OffloadEntriesInfoManagerTy::OMPTargetGlobalVarEntryTo;

    if (CGM.getLangOpts().OpenMPIsDevice) {
      VarName = Addr->getName();
      Addr = nullptr;
    } else {
      VarName = getAddrOfDeclareTargetVar(VD).getName();
      Addr = cast<llvm::Constant>(getAddrOfDeclareTargetVar(VD).getPointer());
    }
    VarSize = CGM.getPointerSize();
    Linkage = llvm::GlobalValue::WeakAnyLinkage;
  }

  OffloadEntriesInfoManager.registerDeviceGlobalVarEntryInfo(
      VarName, Addr, VarSize, Flags, Linkage);
}

bool CGOpenMPRuntime::emitTargetGlobal(GlobalDecl GD) {
  if (isa<FunctionDecl>(GD.getDecl()) ||
      isa<OMPDeclareReductionDecl>(GD.getDecl()))
    return emitTargetFunctions(GD);

  return emitTargetGlobalVariable(GD);
}

void CGOpenMPRuntime::emitDeferredTargetDecls() const {
  for (const VarDecl *VD : DeferredGlobalVariables) {
    llvm::Optional<OMPDeclareTargetDeclAttr::MapTypeTy> Res =
        OMPDeclareTargetDeclAttr::isDeclareTargetDeclaration(VD);
    if (!Res)
      continue;
    if (*Res == OMPDeclareTargetDeclAttr::MT_To &&
        !HasRequiresUnifiedSharedMemory) {
      CGM.EmitGlobal(VD);
    } else {
      assert((*Res == OMPDeclareTargetDeclAttr::MT_Link ||
              (*Res == OMPDeclareTargetDeclAttr::MT_To &&
               HasRequiresUnifiedSharedMemory)) &&
             "Expected link clause or to clause with unified memory.");
      (void)CGM.getOpenMPRuntime().getAddrOfDeclareTargetVar(VD);
    }
  }
}

void CGOpenMPRuntime::adjustTargetSpecificDataForLambdas(
    CodeGenFunction &CGF, const OMPExecutableDirective &D) const {
  assert(isOpenMPTargetExecutionDirective(D.getDirectiveKind()) &&
         " Expected target-based directive.");
}

void CGOpenMPRuntime::processRequiresDirective(const OMPRequiresDecl *D) {
  for (const OMPClause *Clause : D->clauselists()) {
    if (Clause->getClauseKind() == OMPC_unified_shared_memory) {
      HasRequiresUnifiedSharedMemory = true;
    } else if (const auto *AC =
                   dyn_cast<OMPAtomicDefaultMemOrderClause>(Clause)) {
      switch (AC->getAtomicDefaultMemOrderKind()) {
      case OMPC_ATOMIC_DEFAULT_MEM_ORDER_acq_rel:
        RequiresAtomicOrdering = llvm::AtomicOrdering::AcquireRelease;
        break;
      case OMPC_ATOMIC_DEFAULT_MEM_ORDER_seq_cst:
        RequiresAtomicOrdering = llvm::AtomicOrdering::SequentiallyConsistent;
        break;
      case OMPC_ATOMIC_DEFAULT_MEM_ORDER_relaxed:
        RequiresAtomicOrdering = llvm::AtomicOrdering::Monotonic;
        break;
      case OMPC_ATOMIC_DEFAULT_MEM_ORDER_unknown:
        break;
      }
    }
  }
}

llvm::AtomicOrdering CGOpenMPRuntime::getDefaultMemoryOrdering() const {
  return RequiresAtomicOrdering;
}

bool CGOpenMPRuntime::hasAllocateAttributeForGlobalVar(const VarDecl *VD,
                                                       LangAS &AS) {
  if (!VD || !VD->hasAttr<OMPAllocateDeclAttr>())
    return false;
  const auto *A = VD->getAttr<OMPAllocateDeclAttr>();
  switch(A->getAllocatorType()) {
  case OMPAllocateDeclAttr::OMPNullMemAlloc:
  case OMPAllocateDeclAttr::OMPDefaultMemAlloc:
  // Not supported, fallback to the default mem space.
  case OMPAllocateDeclAttr::OMPLargeCapMemAlloc:
  case OMPAllocateDeclAttr::OMPCGroupMemAlloc:
  case OMPAllocateDeclAttr::OMPHighBWMemAlloc:
  case OMPAllocateDeclAttr::OMPLowLatMemAlloc:
  case OMPAllocateDeclAttr::OMPThreadMemAlloc:
  case OMPAllocateDeclAttr::OMPConstMemAlloc:
  case OMPAllocateDeclAttr::OMPPTeamMemAlloc:
    AS = LangAS::Default;
    return true;
  case OMPAllocateDeclAttr::OMPUserDefinedMemAlloc:
    llvm_unreachable("Expected predefined allocator for the variables with the "
                     "static storage.");
  }
  return false;
}

bool CGOpenMPRuntime::hasRequiresUnifiedSharedMemory() const {
  return HasRequiresUnifiedSharedMemory;
}

CGOpenMPRuntime::DisableAutoDeclareTargetRAII::DisableAutoDeclareTargetRAII(
    CodeGenModule &CGM)
    : CGM(CGM) {
  if (CGM.getLangOpts().OpenMPIsDevice) {
    SavedShouldMarkAsGlobal = CGM.getOpenMPRuntime().ShouldMarkAsGlobal;
    CGM.getOpenMPRuntime().ShouldMarkAsGlobal = false;
  }
}

CGOpenMPRuntime::DisableAutoDeclareTargetRAII::~DisableAutoDeclareTargetRAII() {
  if (CGM.getLangOpts().OpenMPIsDevice)
    CGM.getOpenMPRuntime().ShouldMarkAsGlobal = SavedShouldMarkAsGlobal;
}

bool CGOpenMPRuntime::markAsGlobalTarget(GlobalDecl GD) {
  if (!CGM.getLangOpts().OpenMPIsDevice || !ShouldMarkAsGlobal)
    return true;

  const auto *D = cast<FunctionDecl>(GD.getDecl());
  // Do not to emit function if it is marked as declare target as it was already
  // emitted.
  if (OMPDeclareTargetDeclAttr::isDeclareTargetDeclaration(D)) {
    if (D->hasBody() && AlreadyEmittedTargetDecls.count(D) == 0) {
      if (auto *F = dyn_cast_or_null<llvm::Function>(
              CGM.GetGlobalValue(CGM.getMangledName(GD))))
        return !F->isDeclaration();
      return false;
    }
    return true;
  }

  return !AlreadyEmittedTargetDecls.insert(D).second;
}

llvm::Function *CGOpenMPRuntime::emitRequiresDirectiveRegFun() {
  // If we don't have entries or if we are emitting code for the device, we
  // don't need to do anything.
  if (CGM.getLangOpts().OMPTargetTriples.empty() ||
      CGM.getLangOpts().OpenMPSimd || CGM.getLangOpts().OpenMPIsDevice ||
      (OffloadEntriesInfoManager.empty() &&
       !HasEmittedDeclareTargetRegion &&
       !HasEmittedTargetRegion))
    return nullptr;

  // Create and register the function that handles the requires directives.
  ASTContext &C = CGM.getContext();

  llvm::Function *RequiresRegFn;
  {
    CodeGenFunction CGF(CGM);
    const auto &FI = CGM.getTypes().arrangeNullaryFunction();
    llvm::FunctionType *FTy = CGM.getTypes().GetFunctionType(FI);
    std::string ReqName = getName({"omp_offloading", "requires_reg"});
    RequiresRegFn = CGM.CreateGlobalInitOrCleanUpFunction(FTy, ReqName, FI);
    CGF.StartFunction(GlobalDecl(), C.VoidTy, RequiresRegFn, FI, {});
    OpenMPOffloadingRequiresDirFlags Flags = OMP_REQ_NONE;
    // TODO: check for other requires clauses.
    // The requires directive takes effect only when a target region is
    // present in the compilation unit. Otherwise it is ignored and not
    // passed to the runtime. This avoids the runtime from throwing an error
    // for mismatching requires clauses across compilation units that don't
    // contain at least 1 target region.
    assert((HasEmittedTargetRegion ||
            HasEmittedDeclareTargetRegion ||
            !OffloadEntriesInfoManager.empty()) &&
           "Target or declare target region expected.");
    if (HasRequiresUnifiedSharedMemory)
      Flags = OMP_REQ_UNIFIED_SHARED_MEMORY;
    CGF.EmitRuntimeCall(OMPBuilder.getOrCreateRuntimeFunction(
                            CGM.getModule(), OMPRTL___tgt_register_requires),
                        llvm::ConstantInt::get(CGM.Int64Ty, Flags));
    CGF.FinishFunction();
  }
  return RequiresRegFn;
}

void CGOpenMPRuntime::emitTeamsCall(CodeGenFunction &CGF,
                                    const OMPExecutableDirective &D,
                                    SourceLocation Loc,
                                    llvm::Function *OutlinedFn,
                                    ArrayRef<llvm::Value *> CapturedVars) {
  if (!CGF.HaveInsertPoint())
    return;

  llvm::Value *RTLoc = emitUpdateLocation(CGF, Loc);
  CodeGenFunction::RunCleanupsScope Scope(CGF);

  // Build call __kmpc_fork_teams(loc, n, microtask, var1, .., varn);
  llvm::Value *Args[] = {
      RTLoc,
      CGF.Builder.getInt32(CapturedVars.size()), // Number of captured vars
      CGF.Builder.CreateBitCast(OutlinedFn, getKmpc_MicroPointerTy())};
  llvm::SmallVector<llvm::Value *, 16> RealArgs;
  RealArgs.append(std::begin(Args), std::end(Args));
  RealArgs.append(CapturedVars.begin(), CapturedVars.end());

  llvm::FunctionCallee RTLFn = OMPBuilder.getOrCreateRuntimeFunction(
      CGM.getModule(), OMPRTL___kmpc_fork_teams);
  CGF.EmitRuntimeCall(RTLFn, RealArgs);
}

void CGOpenMPRuntime::emitNumTeamsClause(CodeGenFunction &CGF,
                                         const Expr *NumTeams,
                                         const Expr *ThreadLimit,
                                         SourceLocation Loc) {
  if (!CGF.HaveInsertPoint())
    return;

  llvm::Value *RTLoc = emitUpdateLocation(CGF, Loc);

  llvm::Value *NumTeamsVal =
      NumTeams
          ? CGF.Builder.CreateIntCast(CGF.EmitScalarExpr(NumTeams),
                                      CGF.CGM.Int32Ty, /* isSigned = */ true)
          : CGF.Builder.getInt32(0);

  llvm::Value *ThreadLimitVal =
      ThreadLimit
          ? CGF.Builder.CreateIntCast(CGF.EmitScalarExpr(ThreadLimit),
                                      CGF.CGM.Int32Ty, /* isSigned = */ true)
          : CGF.Builder.getInt32(0);

  // Build call __kmpc_push_num_teamss(&loc, global_tid, num_teams, thread_limit)
  llvm::Value *PushNumTeamsArgs[] = {RTLoc, getThreadID(CGF, Loc), NumTeamsVal,
                                     ThreadLimitVal};
  CGF.EmitRuntimeCall(OMPBuilder.getOrCreateRuntimeFunction(
                          CGM.getModule(), OMPRTL___kmpc_push_num_teams),
                      PushNumTeamsArgs);
}

void CGOpenMPRuntime::emitTargetDataCalls(
    CodeGenFunction &CGF, const OMPExecutableDirective &D, const Expr *IfCond,
    const Expr *Device, const RegionCodeGenTy &CodeGen, TargetDataInfo &Info) {
  if (!CGF.HaveInsertPoint())
    return;

  // Action used to replace the default codegen action and turn privatization
  // off.
  PrePostActionTy NoPrivAction;

  // Generate the code for the opening of the data environment. Capture all the
  // arguments of the runtime call by reference because they are used in the
  // closing of the region.
  auto &&BeginThenGen = [this, &D, Device, &Info,
                         &CodeGen](CodeGenFunction &CGF, PrePostActionTy &) {
    // Fill up the arrays with all the mapped variables.
    MappableExprsHandler::MapCombinedInfoTy CombinedInfo;

    // Get map clause information.
    MappableExprsHandler MEHandler(D, CGF);
    MEHandler.generateAllInfo(CombinedInfo);

    // Fill up the arrays and create the arguments.
    emitOffloadingArrays(CGF, CombinedInfo, Info, OMPBuilder,
                         /*IsNonContiguous=*/true);

    llvm::Value *BasePointersArrayArg = nullptr;
    llvm::Value *PointersArrayArg = nullptr;
    llvm::Value *SizesArrayArg = nullptr;
    llvm::Value *MapTypesArrayArg = nullptr;
    llvm::Value *MapNamesArrayArg = nullptr;
    llvm::Value *MappersArrayArg = nullptr;
    emitOffloadingArraysArgument(CGF, BasePointersArrayArg, PointersArrayArg,
                                 SizesArrayArg, MapTypesArrayArg,
                                 MapNamesArrayArg, MappersArrayArg, Info);

    // Emit device ID if any.
    llvm::Value *DeviceID = nullptr;
    if (Device) {
      DeviceID = CGF.Builder.CreateIntCast(CGF.EmitScalarExpr(Device),
                                           CGF.Int64Ty, /*isSigned=*/true);
    } else {
      DeviceID = CGF.Builder.getInt64(OMP_DEVICEID_UNDEF);
    }

    // Emit the number of elements in the offloading arrays.
    llvm::Value *PointerNum = CGF.Builder.getInt32(Info.NumberOfPtrs);
    //
    // Source location for the ident struct
    llvm::Value *RTLoc = emitUpdateLocation(CGF, D.getBeginLoc());

    llvm::Value *OffloadingArgs[] = {RTLoc,
                                     DeviceID,
                                     PointerNum,
                                     BasePointersArrayArg,
                                     PointersArrayArg,
                                     SizesArrayArg,
                                     MapTypesArrayArg,
                                     MapNamesArrayArg,
                                     MappersArrayArg};
    CGF.EmitRuntimeCall(
        OMPBuilder.getOrCreateRuntimeFunction(
            CGM.getModule(), OMPRTL___tgt_target_data_begin_mapper),
        OffloadingArgs);

    // If device pointer privatization is required, emit the body of the region
    // here. It will have to be duplicated: with and without privatization.
    if (!Info.CaptureDeviceAddrMap.empty())
      CodeGen(CGF);
  };

  // Generate code for the closing of the data region.
  auto &&EndThenGen = [this, Device, &Info, &D](CodeGenFunction &CGF,
                                                PrePostActionTy &) {
    assert(Info.isValid() && "Invalid data environment closing arguments.");

    llvm::Value *BasePointersArrayArg = nullptr;
    llvm::Value *PointersArrayArg = nullptr;
    llvm::Value *SizesArrayArg = nullptr;
    llvm::Value *MapTypesArrayArg = nullptr;
    llvm::Value *MapNamesArrayArg = nullptr;
    llvm::Value *MappersArrayArg = nullptr;
    emitOffloadingArraysArgument(CGF, BasePointersArrayArg, PointersArrayArg,
                                 SizesArrayArg, MapTypesArrayArg,
                                 MapNamesArrayArg, MappersArrayArg, Info,
                                 {/*ForEndCall=*/true});

    // Emit device ID if any.
    llvm::Value *DeviceID = nullptr;
    if (Device) {
      DeviceID = CGF.Builder.CreateIntCast(CGF.EmitScalarExpr(Device),
                                           CGF.Int64Ty, /*isSigned=*/true);
    } else {
      DeviceID = CGF.Builder.getInt64(OMP_DEVICEID_UNDEF);
    }

    // Emit the number of elements in the offloading arrays.
    llvm::Value *PointerNum = CGF.Builder.getInt32(Info.NumberOfPtrs);

    // Source location for the ident struct
    llvm::Value *RTLoc = emitUpdateLocation(CGF, D.getBeginLoc());

    llvm::Value *OffloadingArgs[] = {RTLoc,
                                     DeviceID,
                                     PointerNum,
                                     BasePointersArrayArg,
                                     PointersArrayArg,
                                     SizesArrayArg,
                                     MapTypesArrayArg,
                                     MapNamesArrayArg,
                                     MappersArrayArg};
    CGF.EmitRuntimeCall(
        OMPBuilder.getOrCreateRuntimeFunction(
            CGM.getModule(), OMPRTL___tgt_target_data_end_mapper),
        OffloadingArgs);
  };

  // If we need device pointer privatization, we need to emit the body of the
  // region with no privatization in the 'else' branch of the conditional.
  // Otherwise, we don't have to do anything.
  auto &&BeginElseGen = [&Info, &CodeGen, &NoPrivAction](CodeGenFunction &CGF,
                                                         PrePostActionTy &) {
    if (!Info.CaptureDeviceAddrMap.empty()) {
      CodeGen.setAction(NoPrivAction);
      CodeGen(CGF);
    }
  };

  // We don't have to do anything to close the region if the if clause evaluates
  // to false.
  auto &&EndElseGen = [](CodeGenFunction &CGF, PrePostActionTy &) {};

  if (IfCond) {
    emitIfClause(CGF, IfCond, BeginThenGen, BeginElseGen);
  } else {
    RegionCodeGenTy RCG(BeginThenGen);
    RCG(CGF);
  }

  // If we don't require privatization of device pointers, we emit the body in
  // between the runtime calls. This avoids duplicating the body code.
  if (Info.CaptureDeviceAddrMap.empty()) {
    CodeGen.setAction(NoPrivAction);
    CodeGen(CGF);
  }

  if (IfCond) {
    emitIfClause(CGF, IfCond, EndThenGen, EndElseGen);
  } else {
    RegionCodeGenTy RCG(EndThenGen);
    RCG(CGF);
  }
}

void CGOpenMPRuntime::emitTargetDataStandAloneCall(
    CodeGenFunction &CGF, const OMPExecutableDirective &D, const Expr *IfCond,
    const Expr *Device) {
  if (!CGF.HaveInsertPoint())
    return;

  assert((isa<OMPTargetEnterDataDirective>(D) ||
          isa<OMPTargetExitDataDirective>(D) ||
          isa<OMPTargetUpdateDirective>(D)) &&
         "Expecting either target enter, exit data, or update directives.");

  CodeGenFunction::OMPTargetDataInfo InputInfo;
  llvm::Value *MapTypesArray = nullptr;
  llvm::Value *MapNamesArray = nullptr;
  // Generate the code for the opening of the data environment.
  auto &&ThenGen = [this, &D, Device, &InputInfo, &MapTypesArray,
                    &MapNamesArray](CodeGenFunction &CGF, PrePostActionTy &) {
    // Emit device ID if any.
    llvm::Value *DeviceID = nullptr;
    if (Device) {
      DeviceID = CGF.Builder.CreateIntCast(CGF.EmitScalarExpr(Device),
                                           CGF.Int64Ty, /*isSigned=*/true);
    } else {
      DeviceID = CGF.Builder.getInt64(OMP_DEVICEID_UNDEF);
    }

    // Emit the number of elements in the offloading arrays.
    llvm::Constant *PointerNum =
        CGF.Builder.getInt32(InputInfo.NumberOfTargetItems);

    // Source location for the ident struct
    llvm::Value *RTLoc = emitUpdateLocation(CGF, D.getBeginLoc());

    llvm::Value *OffloadingArgs[] = {RTLoc,
                                     DeviceID,
                                     PointerNum,
                                     InputInfo.BasePointersArray.getPointer(),
                                     InputInfo.PointersArray.getPointer(),
                                     InputInfo.SizesArray.getPointer(),
                                     MapTypesArray,
                                     MapNamesArray,
                                     InputInfo.MappersArray.getPointer()};

    // Select the right runtime function call for each standalone
    // directive.
    const bool HasNowait = D.hasClausesOfKind<OMPNowaitClause>();
    RuntimeFunction RTLFn;
    switch (D.getDirectiveKind()) {
    case OMPD_target_enter_data:
      RTLFn = HasNowait ? OMPRTL___tgt_target_data_begin_nowait_mapper
                        : OMPRTL___tgt_target_data_begin_mapper;
      break;
    case OMPD_target_exit_data:
      RTLFn = HasNowait ? OMPRTL___tgt_target_data_end_nowait_mapper
                        : OMPRTL___tgt_target_data_end_mapper;
      break;
    case OMPD_target_update:
      RTLFn = HasNowait ? OMPRTL___tgt_target_data_update_nowait_mapper
                        : OMPRTL___tgt_target_data_update_mapper;
      break;
    case OMPD_parallel:
    case OMPD_for:
    case OMPD_parallel_for:
    case OMPD_parallel_master:
    case OMPD_parallel_sections:
    case OMPD_for_simd:
    case OMPD_parallel_for_simd:
    case OMPD_cancel:
    case OMPD_cancellation_point:
    case OMPD_ordered:
    case OMPD_threadprivate:
    case OMPD_allocate:
    case OMPD_task:
    case OMPD_simd:
    case OMPD_tile:
    case OMPD_unroll:
    case OMPD_sections:
    case OMPD_section:
    case OMPD_single:
    case OMPD_master:
    case OMPD_critical:
    case OMPD_taskyield:
    case OMPD_barrier:
    case OMPD_taskwait:
    case OMPD_taskgroup:
    case OMPD_atomic:
    case OMPD_flush:
    case OMPD_depobj:
    case OMPD_scan:
    case OMPD_teams:
    case OMPD_target_data:
    case OMPD_distribute:
    case OMPD_distribute_simd:
    case OMPD_distribute_parallel_for:
    case OMPD_distribute_parallel_for_simd:
    case OMPD_teams_distribute:
    case OMPD_teams_distribute_simd:
    case OMPD_teams_distribute_parallel_for:
    case OMPD_teams_distribute_parallel_for_simd:
    case OMPD_declare_simd:
    case OMPD_declare_variant:
    case OMPD_begin_declare_variant:
    case OMPD_end_declare_variant:
    case OMPD_declare_target:
    case OMPD_end_declare_target:
    case OMPD_declare_reduction:
    case OMPD_declare_mapper:
    case OMPD_taskloop:
    case OMPD_taskloop_simd:
    case OMPD_master_taskloop:
    case OMPD_master_taskloop_simd:
    case OMPD_parallel_master_taskloop:
    case OMPD_parallel_master_taskloop_simd:
    case OMPD_target:
    case OMPD_target_simd:
    case OMPD_target_teams_distribute:
    case OMPD_target_teams_distribute_simd:
    case OMPD_target_teams_distribute_parallel_for:
    case OMPD_target_teams_distribute_parallel_for_simd:
    case OMPD_target_teams:
    case OMPD_target_parallel:
    case OMPD_target_parallel_for:
    case OMPD_target_parallel_for_simd:
    case OMPD_requires:
    case OMPD_metadirective:
    case OMPD_unknown:
    default:
      llvm_unreachable("Unexpected standalone target data directive.");
      break;
    }
    CGF.EmitRuntimeCall(
        OMPBuilder.getOrCreateRuntimeFunction(CGM.getModule(), RTLFn),
        OffloadingArgs);
  };

  auto &&TargetThenGen = [this, &ThenGen, &D, &InputInfo, &MapTypesArray,
                          &MapNamesArray](CodeGenFunction &CGF,
                                          PrePostActionTy &) {
    // Fill up the arrays with all the mapped variables.
    MappableExprsHandler::MapCombinedInfoTy CombinedInfo;

    // Get map clause information.
    MappableExprsHandler MEHandler(D, CGF);
    MEHandler.generateAllInfo(CombinedInfo);

    TargetDataInfo Info;
    // Fill up the arrays and create the arguments.
    emitOffloadingArrays(CGF, CombinedInfo, Info, OMPBuilder,
                         /*IsNonContiguous=*/true);
    bool RequiresOuterTask = D.hasClausesOfKind<OMPDependClause>() ||
                             D.hasClausesOfKind<OMPNowaitClause>();
    emitOffloadingArraysArgument(
        CGF, Info.BasePointersArray, Info.PointersArray, Info.SizesArray,
        Info.MapTypesArray, Info.MapNamesArray, Info.MappersArray, Info,
        {/*ForEndCall=*/false});
    InputInfo.NumberOfTargetItems = Info.NumberOfPtrs;
    InputInfo.BasePointersArray =
        Address(Info.BasePointersArray, CGM.getPointerAlign());
    InputInfo.PointersArray =
        Address(Info.PointersArray, CGM.getPointerAlign());
    InputInfo.SizesArray =
        Address(Info.SizesArray, CGM.getPointerAlign());
    InputInfo.MappersArray = Address(Info.MappersArray, CGM.getPointerAlign());
    MapTypesArray = Info.MapTypesArray;
    MapNamesArray = Info.MapNamesArray;
    if (RequiresOuterTask)
      CGF.EmitOMPTargetTaskBasedDirective(D, ThenGen, InputInfo);
    else
      emitInlinedDirective(CGF, D.getDirectiveKind(), ThenGen);
  };

  if (IfCond) {
    emitIfClause(CGF, IfCond, TargetThenGen,
                 [](CodeGenFunction &CGF, PrePostActionTy &) {});
  } else {
    RegionCodeGenTy ThenRCG(TargetThenGen);
    ThenRCG(CGF);
  }
}

namespace {
  /// Kind of parameter in a function with 'declare simd' directive.
  enum ParamKindTy { LinearWithVarStride, Linear, Uniform, Vector };
  /// Attribute set of the parameter.
  struct ParamAttrTy {
    ParamKindTy Kind = Vector;
    llvm::APSInt StrideOrArg;
    llvm::APSInt Alignment;
  };
} // namespace

static unsigned evaluateCDTSize(const FunctionDecl *FD,
                                ArrayRef<ParamAttrTy> ParamAttrs) {
  // Every vector variant of a SIMD-enabled function has a vector length (VLEN).
  // If OpenMP clause "simdlen" is used, the VLEN is the value of the argument
  // of that clause. The VLEN value must be power of 2.
  // In other case the notion of the function`s "characteristic data type" (CDT)
  // is used to compute the vector length.
  // CDT is defined in the following order:
  //   a) For non-void function, the CDT is the return type.
  //   b) If the function has any non-uniform, non-linear parameters, then the
  //   CDT is the type of the first such parameter.
  //   c) If the CDT determined by a) or b) above is struct, union, or class
  //   type which is pass-by-value (except for the type that maps to the
  //   built-in complex data type), the characteristic data type is int.
  //   d) If none of the above three cases is applicable, the CDT is int.
  // The VLEN is then determined based on the CDT and the size of vector
  // register of that ISA for which current vector version is generated. The
  // VLEN is computed using the formula below:
  //   VLEN  = sizeof(vector_register) / sizeof(CDT),
  // where vector register size specified in section 3.2.1 Registers and the
  // Stack Frame of original AMD64 ABI document.
  QualType RetType = FD->getReturnType();
  if (RetType.isNull())
    return 0;
  ASTContext &C = FD->getASTContext();
  QualType CDT;
  if (!RetType.isNull() && !RetType->isVoidType()) {
    CDT = RetType;
  } else {
    unsigned Offset = 0;
    if (const auto *MD = dyn_cast<CXXMethodDecl>(FD)) {
      if (ParamAttrs[Offset].Kind == Vector)
        CDT = C.getPointerType(C.getRecordType(MD->getParent()));
      ++Offset;
    }
    if (CDT.isNull()) {
      for (unsigned I = 0, E = FD->getNumParams(); I < E; ++I) {
        if (ParamAttrs[I + Offset].Kind == Vector) {
          CDT = FD->getParamDecl(I)->getType();
          break;
        }
      }
    }
  }
  if (CDT.isNull())
    CDT = C.IntTy;
  CDT = CDT->getCanonicalTypeUnqualified();
  if (CDT->isRecordType() || CDT->isUnionType())
    CDT = C.IntTy;
  return C.getTypeSize(CDT);
}

static void
emitX86DeclareSimdFunction(const FunctionDecl *FD, llvm::Function *Fn,
                           const llvm::APSInt &VLENVal,
                           ArrayRef<ParamAttrTy> ParamAttrs,
                           OMPDeclareSimdDeclAttr::BranchStateTy State) {
  struct ISADataTy {
    char ISA;
    unsigned VecRegSize;
  };
  ISADataTy ISAData[] = {
      {
          'b', 128
      }, // SSE
      {
          'c', 256
      }, // AVX
      {
          'd', 256
      }, // AVX2
      {
          'e', 512
      }, // AVX512
  };
  llvm::SmallVector<char, 2> Masked;
  switch (State) {
  case OMPDeclareSimdDeclAttr::BS_Undefined:
    Masked.push_back('N');
    Masked.push_back('M');
    break;
  case OMPDeclareSimdDeclAttr::BS_Notinbranch:
    Masked.push_back('N');
    break;
  case OMPDeclareSimdDeclAttr::BS_Inbranch:
    Masked.push_back('M');
    break;
  }
  for (char Mask : Masked) {
    for (const ISADataTy &Data : ISAData) {
      SmallString<256> Buffer;
      llvm::raw_svector_ostream Out(Buffer);
      Out << "_ZGV" << Data.ISA << Mask;
      if (!VLENVal) {
        unsigned NumElts = evaluateCDTSize(FD, ParamAttrs);
        assert(NumElts && "Non-zero simdlen/cdtsize expected");
        Out << llvm::APSInt::getUnsigned(Data.VecRegSize / NumElts);
      } else {
        Out << VLENVal;
      }
      for (const ParamAttrTy &ParamAttr : ParamAttrs) {
        switch (ParamAttr.Kind){
        case LinearWithVarStride:
          Out << 's' << ParamAttr.StrideOrArg;
          break;
        case Linear:
          Out << 'l';
          if (ParamAttr.StrideOrArg != 1)
            Out << ParamAttr.StrideOrArg;
          break;
        case Uniform:
          Out << 'u';
          break;
        case Vector:
          Out << 'v';
          break;
        }
        if (!!ParamAttr.Alignment)
          Out << 'a' << ParamAttr.Alignment;
      }
      Out << '_' << Fn->getName();
      Fn->addFnAttr(Out.str());
    }
  }
}

// This are the Functions that are needed to mangle the name of the
// vector functions generated by the compiler, according to the rules
// defined in the "Vector Function ABI specifications for AArch64",
// available at
// https://developer.arm.com/products/software-development-tools/hpc/arm-compiler-for-hpc/vector-function-abi.

/// Maps To Vector (MTV), as defined in 3.1.1 of the AAVFABI.
///
/// TODO: Need to implement the behavior for reference marked with a
/// var or no linear modifiers (1.b in the section). For this, we
/// need to extend ParamKindTy to support the linear modifiers.
static bool getAArch64MTV(QualType QT, ParamKindTy Kind) {
  QT = QT.getCanonicalType();

  if (QT->isVoidType())
    return false;

  if (Kind == ParamKindTy::Uniform)
    return false;

  if (Kind == ParamKindTy::Linear)
    return false;

  // TODO: Handle linear references with modifiers

  if (Kind == ParamKindTy::LinearWithVarStride)
    return false;

  return true;
}

/// Pass By Value (PBV), as defined in 3.1.2 of the AAVFABI.
static bool getAArch64PBV(QualType QT, ASTContext &C) {
  QT = QT.getCanonicalType();
  unsigned Size = C.getTypeSize(QT);

  // Only scalars and complex within 16 bytes wide set PVB to true.
  if (Size != 8 && Size != 16 && Size != 32 && Size != 64 && Size != 128)
    return false;

  if (QT->isFloatingType())
    return true;

  if (QT->isIntegerType())
    return true;

  if (QT->isPointerType())
    return true;

  // TODO: Add support for complex types (section 3.1.2, item 2).

  return false;
}

/// Computes the lane size (LS) of a return type or of an input parameter,
/// as defined by `LS(P)` in 3.2.1 of the AAVFABI.
/// TODO: Add support for references, section 3.2.1, item 1.
static unsigned getAArch64LS(QualType QT, ParamKindTy Kind, ASTContext &C) {
  if (!getAArch64MTV(QT, Kind) && QT.getCanonicalType()->isPointerType()) {
    QualType PTy = QT.getCanonicalType()->getPointeeType();
    if (getAArch64PBV(PTy, C))
      return C.getTypeSize(PTy);
  }
  if (getAArch64PBV(QT, C))
    return C.getTypeSize(QT);

  return C.getTypeSize(C.getUIntPtrType());
}

// Get Narrowest Data Size (NDS) and Widest Data Size (WDS) from the
// signature of the scalar function, as defined in 3.2.2 of the
// AAVFABI.
static std::tuple<unsigned, unsigned, bool>
getNDSWDS(const FunctionDecl *FD, ArrayRef<ParamAttrTy> ParamAttrs) {
  QualType RetType = FD->getReturnType().getCanonicalType();

  ASTContext &C = FD->getASTContext();

  bool OutputBecomesInput = false;

  llvm::SmallVector<unsigned, 8> Sizes;
  if (!RetType->isVoidType()) {
    Sizes.push_back(getAArch64LS(RetType, ParamKindTy::Vector, C));
    if (!getAArch64PBV(RetType, C) && getAArch64MTV(RetType, {}))
      OutputBecomesInput = true;
  }
  for (unsigned I = 0, E = FD->getNumParams(); I < E; ++I) {
    QualType QT = FD->getParamDecl(I)->getType().getCanonicalType();
    Sizes.push_back(getAArch64LS(QT, ParamAttrs[I].Kind, C));
  }

  assert(!Sizes.empty() && "Unable to determine NDS and WDS.");
  // The LS of a function parameter / return value can only be a power
  // of 2, starting from 8 bits, up to 128.
  assert(llvm::all_of(Sizes,
                      [](unsigned Size) {
                        return Size == 8 || Size == 16 || Size == 32 ||
                               Size == 64 || Size == 128;
                      }) &&
         "Invalid size");

  return std::make_tuple(*std::min_element(std::begin(Sizes), std::end(Sizes)),
                         *std::max_element(std::begin(Sizes), std::end(Sizes)),
                         OutputBecomesInput);
}

/// Mangle the parameter part of the vector function name according to
/// their OpenMP classification. The mangling function is defined in
/// section 3.5 of the AAVFABI.
static std::string mangleVectorParameters(ArrayRef<ParamAttrTy> ParamAttrs) {
  SmallString<256> Buffer;
  llvm::raw_svector_ostream Out(Buffer);
  for (const auto &ParamAttr : ParamAttrs) {
    switch (ParamAttr.Kind) {
    case LinearWithVarStride:
      Out << "ls" << ParamAttr.StrideOrArg;
      break;
    case Linear:
      Out << 'l';
      // Don't print the step value if it is not present or if it is
      // equal to 1.
      if (ParamAttr.StrideOrArg != 1)
        Out << ParamAttr.StrideOrArg;
      break;
    case Uniform:
      Out << 'u';
      break;
    case Vector:
      Out << 'v';
      break;
    }

    if (!!ParamAttr.Alignment)
      Out << 'a' << ParamAttr.Alignment;
  }

  return std::string(Out.str());
}

// Function used to add the attribute. The parameter `VLEN` is
// templated to allow the use of "x" when targeting scalable functions
// for SVE.
template <typename T>
static void addAArch64VectorName(T VLEN, StringRef LMask, StringRef Prefix,
                                 char ISA, StringRef ParSeq,
                                 StringRef MangledName, bool OutputBecomesInput,
                                 llvm::Function *Fn) {
  SmallString<256> Buffer;
  llvm::raw_svector_ostream Out(Buffer);
  Out << Prefix << ISA << LMask << VLEN;
  if (OutputBecomesInput)
    Out << "v";
  Out << ParSeq << "_" << MangledName;
  Fn->addFnAttr(Out.str());
}

// Helper function to generate the Advanced SIMD names depending on
// the value of the NDS when simdlen is not present.
static void addAArch64AdvSIMDNDSNames(unsigned NDS, StringRef Mask,
                                      StringRef Prefix, char ISA,
                                      StringRef ParSeq, StringRef MangledName,
                                      bool OutputBecomesInput,
                                      llvm::Function *Fn) {
  switch (NDS) {
  case 8:
    addAArch64VectorName(8, Mask, Prefix, ISA, ParSeq, MangledName,
                         OutputBecomesInput, Fn);
    addAArch64VectorName(16, Mask, Prefix, ISA, ParSeq, MangledName,
                         OutputBecomesInput, Fn);
    break;
  case 16:
    addAArch64VectorName(4, Mask, Prefix, ISA, ParSeq, MangledName,
                         OutputBecomesInput, Fn);
    addAArch64VectorName(8, Mask, Prefix, ISA, ParSeq, MangledName,
                         OutputBecomesInput, Fn);
    break;
  case 32:
    addAArch64VectorName(2, Mask, Prefix, ISA, ParSeq, MangledName,
                         OutputBecomesInput, Fn);
    addAArch64VectorName(4, Mask, Prefix, ISA, ParSeq, MangledName,
                         OutputBecomesInput, Fn);
    break;
  case 64:
  case 128:
    addAArch64VectorName(2, Mask, Prefix, ISA, ParSeq, MangledName,
                         OutputBecomesInput, Fn);
    break;
  default:
    llvm_unreachable("Scalar type is too wide.");
  }
}

/// Emit vector function attributes for AArch64, as defined in the AAVFABI.
static void emitAArch64DeclareSimdFunction(
    CodeGenModule &CGM, const FunctionDecl *FD, unsigned UserVLEN,
    ArrayRef<ParamAttrTy> ParamAttrs,
    OMPDeclareSimdDeclAttr::BranchStateTy State, StringRef MangledName,
    char ISA, unsigned VecRegSize, llvm::Function *Fn, SourceLocation SLoc) {

  // Get basic data for building the vector signature.
  const auto Data = getNDSWDS(FD, ParamAttrs);
  const unsigned NDS = std::get<0>(Data);
  const unsigned WDS = std::get<1>(Data);
  const bool OutputBecomesInput = std::get<2>(Data);

  // Check the values provided via `simdlen` by the user.
  // 1. A `simdlen(1)` doesn't produce vector signatures,
  if (UserVLEN == 1) {
    unsigned DiagID = CGM.getDiags().getCustomDiagID(
        DiagnosticsEngine::Warning,
        "The clause simdlen(1) has no effect when targeting aarch64.");
    CGM.getDiags().Report(SLoc, DiagID);
    return;
  }

  // 2. Section 3.3.1, item 1: user input must be a power of 2 for
  // Advanced SIMD output.
  if (ISA == 'n' && UserVLEN && !llvm::isPowerOf2_32(UserVLEN)) {
    unsigned DiagID = CGM.getDiags().getCustomDiagID(
        DiagnosticsEngine::Warning, "The value specified in simdlen must be a "
                                    "power of 2 when targeting Advanced SIMD.");
    CGM.getDiags().Report(SLoc, DiagID);
    return;
  }

  // 3. Section 3.4.1. SVE fixed lengh must obey the architectural
  // limits.
  if (ISA == 's' && UserVLEN != 0) {
    if ((UserVLEN * WDS > 2048) || (UserVLEN * WDS % 128 != 0)) {
      unsigned DiagID = CGM.getDiags().getCustomDiagID(
          DiagnosticsEngine::Warning, "The clause simdlen must fit the %0-bit "
                                      "lanes in the architectural constraints "
                                      "for SVE (min is 128-bit, max is "
                                      "2048-bit, by steps of 128-bit)");
      CGM.getDiags().Report(SLoc, DiagID) << WDS;
      return;
    }
  }

  // Sort out parameter sequence.
  const std::string ParSeq = mangleVectorParameters(ParamAttrs);
  StringRef Prefix = "_ZGV";
  // Generate simdlen from user input (if any).
  if (UserVLEN) {
    if (ISA == 's') {
      // SVE generates only a masked function.
      addAArch64VectorName(UserVLEN, "M", Prefix, ISA, ParSeq, MangledName,
                           OutputBecomesInput, Fn);
    } else {
      assert(ISA == 'n' && "Expected ISA either 's' or 'n'.");
      // Advanced SIMD generates one or two functions, depending on
      // the `[not]inbranch` clause.
      switch (State) {
      case OMPDeclareSimdDeclAttr::BS_Undefined:
        addAArch64VectorName(UserVLEN, "N", Prefix, ISA, ParSeq, MangledName,
                             OutputBecomesInput, Fn);
        addAArch64VectorName(UserVLEN, "M", Prefix, ISA, ParSeq, MangledName,
                             OutputBecomesInput, Fn);
        break;
      case OMPDeclareSimdDeclAttr::BS_Notinbranch:
        addAArch64VectorName(UserVLEN, "N", Prefix, ISA, ParSeq, MangledName,
                             OutputBecomesInput, Fn);
        break;
      case OMPDeclareSimdDeclAttr::BS_Inbranch:
        addAArch64VectorName(UserVLEN, "M", Prefix, ISA, ParSeq, MangledName,
                             OutputBecomesInput, Fn);
        break;
      }
    }
  } else {
    // If no user simdlen is provided, follow the AAVFABI rules for
    // generating the vector length.
    if (ISA == 's') {
      // SVE, section 3.4.1, item 1.
      addAArch64VectorName("x", "M", Prefix, ISA, ParSeq, MangledName,
                           OutputBecomesInput, Fn);
    } else {
      assert(ISA == 'n' && "Expected ISA either 's' or 'n'.");
      // Advanced SIMD, Section 3.3.1 of the AAVFABI, generates one or
      // two vector names depending on the use of the clause
      // `[not]inbranch`.
      switch (State) {
      case OMPDeclareSimdDeclAttr::BS_Undefined:
        addAArch64AdvSIMDNDSNames(NDS, "N", Prefix, ISA, ParSeq, MangledName,
                                  OutputBecomesInput, Fn);
        addAArch64AdvSIMDNDSNames(NDS, "M", Prefix, ISA, ParSeq, MangledName,
                                  OutputBecomesInput, Fn);
        break;
      case OMPDeclareSimdDeclAttr::BS_Notinbranch:
        addAArch64AdvSIMDNDSNames(NDS, "N", Prefix, ISA, ParSeq, MangledName,
                                  OutputBecomesInput, Fn);
        break;
      case OMPDeclareSimdDeclAttr::BS_Inbranch:
        addAArch64AdvSIMDNDSNames(NDS, "M", Prefix, ISA, ParSeq, MangledName,
                                  OutputBecomesInput, Fn);
        break;
      }
    }
  }
}

void CGOpenMPRuntime::emitDeclareSimdFunction(const FunctionDecl *FD,
                                              llvm::Function *Fn) {
  ASTContext &C = CGM.getContext();
  FD = FD->getMostRecentDecl();
  // Map params to their positions in function decl.
  llvm::DenseMap<const Decl *, unsigned> ParamPositions;
  if (isa<CXXMethodDecl>(FD))
    ParamPositions.try_emplace(FD, 0);
  unsigned ParamPos = ParamPositions.size();
  for (const ParmVarDecl *P : FD->parameters()) {
    ParamPositions.try_emplace(P->getCanonicalDecl(), ParamPos);
    ++ParamPos;
  }
  while (FD) {
    for (const auto *Attr : FD->specific_attrs<OMPDeclareSimdDeclAttr>()) {
      llvm::SmallVector<ParamAttrTy, 8> ParamAttrs(ParamPositions.size());
      // Mark uniform parameters.
      for (const Expr *E : Attr->uniforms()) {
        E = E->IgnoreParenImpCasts();
        unsigned Pos;
        if (isa<CXXThisExpr>(E)) {
          Pos = ParamPositions[FD];
        } else {
          const auto *PVD = cast<ParmVarDecl>(cast<DeclRefExpr>(E)->getDecl())
                                ->getCanonicalDecl();
          Pos = ParamPositions[PVD];
        }
        ParamAttrs[Pos].Kind = Uniform;
      }
      // Get alignment info.
      auto NI = Attr->alignments_begin();
      for (const Expr *E : Attr->aligneds()) {
        E = E->IgnoreParenImpCasts();
        unsigned Pos;
        QualType ParmTy;
        if (isa<CXXThisExpr>(E)) {
          Pos = ParamPositions[FD];
          ParmTy = E->getType();
        } else {
          const auto *PVD = cast<ParmVarDecl>(cast<DeclRefExpr>(E)->getDecl())
                                ->getCanonicalDecl();
          Pos = ParamPositions[PVD];
          ParmTy = PVD->getType();
        }
        ParamAttrs[Pos].Alignment =
            (*NI)
                ? (*NI)->EvaluateKnownConstInt(C)
                : llvm::APSInt::getUnsigned(
                      C.toCharUnitsFromBits(C.getOpenMPDefaultSimdAlign(ParmTy))
                          .getQuantity());
        ++NI;
      }
      // Mark linear parameters.
      auto SI = Attr->steps_begin();
      auto MI = Attr->modifiers_begin();
      for (const Expr *E : Attr->linears()) {
        E = E->IgnoreParenImpCasts();
        unsigned Pos;
        // Rescaling factor needed to compute the linear parameter
        // value in the mangled name.
        unsigned PtrRescalingFactor = 1;
        if (isa<CXXThisExpr>(E)) {
          Pos = ParamPositions[FD];
        } else {
          const auto *PVD = cast<ParmVarDecl>(cast<DeclRefExpr>(E)->getDecl())
                                ->getCanonicalDecl();
          Pos = ParamPositions[PVD];
          if (auto *P = dyn_cast<PointerType>(PVD->getType()))
            PtrRescalingFactor = CGM.getContext()
                                     .getTypeSizeInChars(P->getPointeeType())
                                     .getQuantity();
        }
        ParamAttrTy &ParamAttr = ParamAttrs[Pos];
        ParamAttr.Kind = Linear;
        // Assuming a stride of 1, for `linear` without modifiers.
        ParamAttr.StrideOrArg = llvm::APSInt::getUnsigned(1);
        if (*SI) {
          Expr::EvalResult Result;
          if (!(*SI)->EvaluateAsInt(Result, C, Expr::SE_AllowSideEffects)) {
            if (const auto *DRE =
                    cast<DeclRefExpr>((*SI)->IgnoreParenImpCasts())) {
              if (const auto *StridePVD = cast<ParmVarDecl>(DRE->getDecl())) {
                ParamAttr.Kind = LinearWithVarStride;
                ParamAttr.StrideOrArg = llvm::APSInt::getUnsigned(
                    ParamPositions[StridePVD->getCanonicalDecl()]);
              }
            }
          } else {
            ParamAttr.StrideOrArg = Result.Val.getInt();
          }
        }
        // If we are using a linear clause on a pointer, we need to
        // rescale the value of linear_step with the byte size of the
        // pointee type.
        if (Linear == ParamAttr.Kind)
          ParamAttr.StrideOrArg = ParamAttr.StrideOrArg * PtrRescalingFactor;
        ++SI;
        ++MI;
      }
      llvm::APSInt VLENVal;
      SourceLocation ExprLoc;
      const Expr *VLENExpr = Attr->getSimdlen();
      if (VLENExpr) {
        VLENVal = VLENExpr->EvaluateKnownConstInt(C);
        ExprLoc = VLENExpr->getExprLoc();
      }
      OMPDeclareSimdDeclAttr::BranchStateTy State = Attr->getBranchState();
      if (CGM.getTriple().isX86()) {
        emitX86DeclareSimdFunction(FD, Fn, VLENVal, ParamAttrs, State);
      } else if (CGM.getTriple().getArch() == llvm::Triple::aarch64) {
        unsigned VLEN = VLENVal.getExtValue();
        StringRef MangledName = Fn->getName();
        if (CGM.getTarget().hasFeature("sve"))
          emitAArch64DeclareSimdFunction(CGM, FD, VLEN, ParamAttrs, State,
                                         MangledName, 's', 128, Fn, ExprLoc);
        if (CGM.getTarget().hasFeature("neon"))
          emitAArch64DeclareSimdFunction(CGM, FD, VLEN, ParamAttrs, State,
                                         MangledName, 'n', 128, Fn, ExprLoc);
      }
    }
    FD = FD->getPreviousDecl();
  }
}

namespace {
/// Cleanup action for doacross support.
class DoacrossCleanupTy final : public EHScopeStack::Cleanup {
public:
  static const int DoacrossFinArgs = 2;

private:
  llvm::FunctionCallee RTLFn;
  llvm::Value *Args[DoacrossFinArgs];

public:
  DoacrossCleanupTy(llvm::FunctionCallee RTLFn,
                    ArrayRef<llvm::Value *> CallArgs)
      : RTLFn(RTLFn) {
    assert(CallArgs.size() == DoacrossFinArgs);
    std::copy(CallArgs.begin(), CallArgs.end(), std::begin(Args));
  }
  void Emit(CodeGenFunction &CGF, Flags /*flags*/) override {
    if (!CGF.HaveInsertPoint())
      return;
    CGF.EmitRuntimeCall(RTLFn, Args);
  }
};
} // namespace

void CGOpenMPRuntime::emitDoacrossInit(CodeGenFunction &CGF,
                                       const OMPLoopDirective &D,
                                       ArrayRef<Expr *> NumIterations) {
  if (!CGF.HaveInsertPoint())
    return;

  ASTContext &C = CGM.getContext();
  QualType Int64Ty = C.getIntTypeForBitwidth(/*DestWidth=*/64, /*Signed=*/true);
  RecordDecl *RD;
  if (KmpDimTy.isNull()) {
    // Build struct kmp_dim {  // loop bounds info casted to kmp_int64
    //  kmp_int64 lo; // lower
    //  kmp_int64 up; // upper
    //  kmp_int64 st; // stride
    // };
    RD = C.buildImplicitRecord("kmp_dim");
    RD->startDefinition();
    addFieldToRecordDecl(C, RD, Int64Ty);
    addFieldToRecordDecl(C, RD, Int64Ty);
    addFieldToRecordDecl(C, RD, Int64Ty);
    RD->completeDefinition();
    KmpDimTy = C.getRecordType(RD);
  } else {
    RD = cast<RecordDecl>(KmpDimTy->getAsTagDecl());
  }
  llvm::APInt Size(/*numBits=*/32, NumIterations.size());
  QualType ArrayTy =
      C.getConstantArrayType(KmpDimTy, Size, nullptr, ArrayType::Normal, 0);

  Address DimsAddr = CGF.CreateMemTemp(ArrayTy, "dims");
  CGF.EmitNullInitialization(DimsAddr, ArrayTy);
  enum { LowerFD = 0, UpperFD, StrideFD };
  // Fill dims with data.
  for (unsigned I = 0, E = NumIterations.size(); I < E; ++I) {
    LValue DimsLVal = CGF.MakeAddrLValue(
        CGF.Builder.CreateConstArrayGEP(DimsAddr, I), KmpDimTy);
    // dims.upper = num_iterations;
    LValue UpperLVal = CGF.EmitLValueForField(
        DimsLVal, *std::next(RD->field_begin(), UpperFD));
    llvm::Value *NumIterVal = CGF.EmitScalarConversion(
        CGF.EmitScalarExpr(NumIterations[I]), NumIterations[I]->getType(),
        Int64Ty, NumIterations[I]->getExprLoc());
    CGF.EmitStoreOfScalar(NumIterVal, UpperLVal);
    // dims.stride = 1;
    LValue StrideLVal = CGF.EmitLValueForField(
        DimsLVal, *std::next(RD->field_begin(), StrideFD));
    CGF.EmitStoreOfScalar(llvm::ConstantInt::getSigned(CGM.Int64Ty, /*V=*/1),
                          StrideLVal);
  }

  // Build call void __kmpc_doacross_init(ident_t *loc, kmp_int32 gtid,
  // kmp_int32 num_dims, struct kmp_dim * dims);
  llvm::Value *Args[] = {
      emitUpdateLocation(CGF, D.getBeginLoc()),
      getThreadID(CGF, D.getBeginLoc()),
      llvm::ConstantInt::getSigned(CGM.Int32Ty, NumIterations.size()),
      CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
          CGF.Builder.CreateConstArrayGEP(DimsAddr, 0).getPointer(),
          CGM.VoidPtrTy)};

  llvm::FunctionCallee RTLFn = OMPBuilder.getOrCreateRuntimeFunction(
      CGM.getModule(), OMPRTL___kmpc_doacross_init);
  CGF.EmitRuntimeCall(RTLFn, Args);
  llvm::Value *FiniArgs[DoacrossCleanupTy::DoacrossFinArgs] = {
      emitUpdateLocation(CGF, D.getEndLoc()), getThreadID(CGF, D.getEndLoc())};
  llvm::FunctionCallee FiniRTLFn = OMPBuilder.getOrCreateRuntimeFunction(
      CGM.getModule(), OMPRTL___kmpc_doacross_fini);
  CGF.EHStack.pushCleanup<DoacrossCleanupTy>(NormalAndEHCleanup, FiniRTLFn,
                                             llvm::makeArrayRef(FiniArgs));
}

void CGOpenMPRuntime::emitDoacrossOrdered(CodeGenFunction &CGF,
                                          const OMPDependClause *C) {
  QualType Int64Ty =
      CGM.getContext().getIntTypeForBitwidth(/*DestWidth=*/64, /*Signed=*/1);
  llvm::APInt Size(/*numBits=*/32, C->getNumLoops());
  QualType ArrayTy = CGM.getContext().getConstantArrayType(
      Int64Ty, Size, nullptr, ArrayType::Normal, 0);
  Address CntAddr = CGF.CreateMemTemp(ArrayTy, ".cnt.addr");
  for (unsigned I = 0, E = C->getNumLoops(); I < E; ++I) {
    const Expr *CounterVal = C->getLoopData(I);
    assert(CounterVal);
    llvm::Value *CntVal = CGF.EmitScalarConversion(
        CGF.EmitScalarExpr(CounterVal), CounterVal->getType(), Int64Ty,
        CounterVal->getExprLoc());
    CGF.EmitStoreOfScalar(CntVal, CGF.Builder.CreateConstArrayGEP(CntAddr, I),
                          /*Volatile=*/false, Int64Ty);
  }
  llvm::Value *Args[] = {
      emitUpdateLocation(CGF, C->getBeginLoc()),
      getThreadID(CGF, C->getBeginLoc()),
      CGF.Builder.CreateConstArrayGEP(CntAddr, 0).getPointer()};
  llvm::FunctionCallee RTLFn;
  if (C->getDependencyKind() == OMPC_DEPEND_source) {
    RTLFn = OMPBuilder.getOrCreateRuntimeFunction(CGM.getModule(),
                                                  OMPRTL___kmpc_doacross_post);
  } else {
    assert(C->getDependencyKind() == OMPC_DEPEND_sink);
    RTLFn = OMPBuilder.getOrCreateRuntimeFunction(CGM.getModule(),
                                                  OMPRTL___kmpc_doacross_wait);
  }
  CGF.EmitRuntimeCall(RTLFn, Args);
}

void CGOpenMPRuntime::emitCall(CodeGenFunction &CGF, SourceLocation Loc,
                               llvm::FunctionCallee Callee,
                               ArrayRef<llvm::Value *> Args) const {
  assert(Loc.isValid() && "Outlined function call location must be valid.");
  auto DL = ApplyDebugLocation::CreateDefaultArtificial(CGF, Loc);

  if (auto *Fn = dyn_cast<llvm::Function>(Callee.getCallee())) {
    if (Fn->doesNotThrow()) {
      CGF.EmitNounwindRuntimeCall(Fn, Args);
      return;
    }
  }
  CGF.EmitRuntimeCall(Callee, Args);
}

void CGOpenMPRuntime::emitOutlinedFunctionCall(
    CodeGenFunction &CGF, SourceLocation Loc, llvm::FunctionCallee OutlinedFn,
    ArrayRef<llvm::Value *> Args) const {
  emitCall(CGF, Loc, OutlinedFn, Args);
}

void CGOpenMPRuntime::emitFunctionProlog(CodeGenFunction &CGF, const Decl *D) {
  if (const auto *FD = dyn_cast<FunctionDecl>(D))
    if (OMPDeclareTargetDeclAttr::isDeclareTargetDeclaration(FD))
      HasEmittedDeclareTargetRegion = true;
}

Address CGOpenMPRuntime::getParameterAddress(CodeGenFunction &CGF,
                                             const VarDecl *NativeParam,
                                             const VarDecl *TargetParam) const {
  return CGF.GetAddrOfLocalVar(NativeParam);
}

/// Return allocator value from expression, or return a null allocator (default
/// when no allocator specified).
static llvm::Value *getAllocatorVal(CodeGenFunction &CGF,
                                    const Expr *Allocator) {
  llvm::Value *AllocVal;
  if (Allocator) {
    AllocVal = CGF.EmitScalarExpr(Allocator);
    // According to the standard, the original allocator type is a enum
    // (integer). Convert to pointer type, if required.
    AllocVal = CGF.EmitScalarConversion(AllocVal, Allocator->getType(),
                                        CGF.getContext().VoidPtrTy,
                                        Allocator->getExprLoc());
  } else {
    // If no allocator specified, it defaults to the null allocator.
    AllocVal = llvm::Constant::getNullValue(
        CGF.CGM.getTypes().ConvertType(CGF.getContext().VoidPtrTy));
  }
  return AllocVal;
}

Address CGOpenMPRuntime::getAddressOfLocalVariable(CodeGenFunction &CGF,
                                                   const VarDecl *VD) {
  if (!VD)
    return Address::invalid();
  Address UntiedAddr = Address::invalid();
  Address UntiedRealAddr = Address::invalid();
  auto It = FunctionToUntiedTaskStackMap.find(CGF.CurFn);
  if (It != FunctionToUntiedTaskStackMap.end()) {
    const UntiedLocalVarsAddressesMap &UntiedData =
        UntiedLocalVarsStack[It->second];
    auto I = UntiedData.find(VD);
    if (I != UntiedData.end()) {
      UntiedAddr = I->second.first;
      UntiedRealAddr = I->second.second;
    }
  }
  const VarDecl *CVD = VD->getCanonicalDecl();
  if (CVD->hasAttr<OMPAllocateDeclAttr>()) {
    // Use the default allocation.
    if (!isAllocatableDecl(VD))
      return UntiedAddr;
    llvm::Value *Size;
    CharUnits Align = CGM.getContext().getDeclAlign(CVD);
    if (CVD->getType()->isVariablyModifiedType()) {
      Size = CGF.getTypeSize(CVD->getType());
      // Align the size: ((size + align - 1) / align) * align
      Size = CGF.Builder.CreateNUWAdd(
          Size, CGM.getSize(Align - CharUnits::fromQuantity(1)));
      Size = CGF.Builder.CreateUDiv(Size, CGM.getSize(Align));
      Size = CGF.Builder.CreateNUWMul(Size, CGM.getSize(Align));
    } else {
      CharUnits Sz = CGM.getContext().getTypeSizeInChars(CVD->getType());
      Size = CGM.getSize(Sz.alignTo(Align));
    }
    llvm::Value *ThreadID = getThreadID(CGF, CVD->getBeginLoc());
    const auto *AA = CVD->getAttr<OMPAllocateDeclAttr>();
    const Expr *Allocator = AA->getAllocator();
    llvm::Value *AllocVal = getAllocatorVal(CGF, Allocator);
    llvm::Value *Alignment =
        AA->getAlignment()
            ? CGF.Builder.CreateIntCast(CGF.EmitScalarExpr(AA->getAlignment()),
                                        CGM.SizeTy, /*isSigned=*/false)
            : nullptr;
    SmallVector<llvm::Value *, 4> Args;
    Args.push_back(ThreadID);
    if (Alignment)
      Args.push_back(Alignment);
    Args.push_back(Size);
    Args.push_back(AllocVal);
    llvm::omp::RuntimeFunction FnID =
        Alignment ? OMPRTL___kmpc_aligned_alloc : OMPRTL___kmpc_alloc;
    llvm::Value *Addr = CGF.EmitRuntimeCall(
        OMPBuilder.getOrCreateRuntimeFunction(CGM.getModule(), FnID), Args,
        getName({CVD->getName(), ".void.addr"}));
    llvm::FunctionCallee FiniRTLFn = OMPBuilder.getOrCreateRuntimeFunction(
        CGM.getModule(), OMPRTL___kmpc_free);
    QualType Ty = CGM.getContext().getPointerType(CVD->getType());
    Addr = CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
        Addr, CGF.ConvertTypeForMem(Ty), getName({CVD->getName(), ".addr"}));
    if (UntiedAddr.isValid())
      CGF.EmitStoreOfScalar(Addr, UntiedAddr, /*Volatile=*/false, Ty);

    // Cleanup action for allocate support.
    class OMPAllocateCleanupTy final : public EHScopeStack::Cleanup {
      llvm::FunctionCallee RTLFn;
      SourceLocation::UIntTy LocEncoding;
      Address Addr;
      const Expr *AllocExpr;

    public:
      OMPAllocateCleanupTy(llvm::FunctionCallee RTLFn,
                           SourceLocation::UIntTy LocEncoding, Address Addr,
                           const Expr *AllocExpr)
          : RTLFn(RTLFn), LocEncoding(LocEncoding), Addr(Addr),
            AllocExpr(AllocExpr) {}
      void Emit(CodeGenFunction &CGF, Flags /*flags*/) override {
        if (!CGF.HaveInsertPoint())
          return;
        llvm::Value *Args[3];
        Args[0] = CGF.CGM.getOpenMPRuntime().getThreadID(
            CGF, SourceLocation::getFromRawEncoding(LocEncoding));
        Args[1] = CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
            Addr.getPointer(), CGF.VoidPtrTy);
        llvm::Value *AllocVal = getAllocatorVal(CGF, AllocExpr);
        Args[2] = AllocVal;
        CGF.EmitRuntimeCall(RTLFn, Args);
      }
    };
    Address VDAddr =
        UntiedRealAddr.isValid() ? UntiedRealAddr : Address(Addr, Align);
    CGF.EHStack.pushCleanup<OMPAllocateCleanupTy>(
        NormalAndEHCleanup, FiniRTLFn, CVD->getLocation().getRawEncoding(),
        VDAddr, Allocator);
    if (UntiedRealAddr.isValid())
      if (auto *Region =
              dyn_cast_or_null<CGOpenMPRegionInfo>(CGF.CapturedStmtInfo))
        Region->emitUntiedSwitch(CGF);
    return VDAddr;
  }
  return UntiedAddr;
}

bool CGOpenMPRuntime::isLocalVarInUntiedTask(CodeGenFunction &CGF,
                                             const VarDecl *VD) const {
  auto It = FunctionToUntiedTaskStackMap.find(CGF.CurFn);
  if (It == FunctionToUntiedTaskStackMap.end())
    return false;
  return UntiedLocalVarsStack[It->second].count(VD) > 0;
}

CGOpenMPRuntime::NontemporalDeclsRAII::NontemporalDeclsRAII(
    CodeGenModule &CGM, const OMPLoopDirective &S)
    : CGM(CGM), NeedToPush(S.hasClausesOfKind<OMPNontemporalClause>()) {
  assert(CGM.getLangOpts().OpenMP && "Not in OpenMP mode.");
  if (!NeedToPush)
    return;
  NontemporalDeclsSet &DS =
      CGM.getOpenMPRuntime().NontemporalDeclsStack.emplace_back();
  for (const auto *C : S.getClausesOfKind<OMPNontemporalClause>()) {
    for (const Stmt *Ref : C->private_refs()) {
      const auto *SimpleRefExpr = cast<Expr>(Ref)->IgnoreParenImpCasts();
      const ValueDecl *VD;
      if (const auto *DRE = dyn_cast<DeclRefExpr>(SimpleRefExpr)) {
        VD = DRE->getDecl();
      } else {
        const auto *ME = cast<MemberExpr>(SimpleRefExpr);
        assert((ME->isImplicitCXXThis() ||
                isa<CXXThisExpr>(ME->getBase()->IgnoreParenImpCasts())) &&
               "Expected member of current class.");
        VD = ME->getMemberDecl();
      }
      DS.insert(VD);
    }
  }
}

CGOpenMPRuntime::NontemporalDeclsRAII::~NontemporalDeclsRAII() {
  if (!NeedToPush)
    return;
  CGM.getOpenMPRuntime().NontemporalDeclsStack.pop_back();
}

CGOpenMPRuntime::UntiedTaskLocalDeclsRAII::UntiedTaskLocalDeclsRAII(
    CodeGenFunction &CGF,
    const llvm::MapVector<CanonicalDeclPtr<const VarDecl>,
                          std::pair<Address, Address>> &LocalVars)
    : CGM(CGF.CGM), NeedToPush(!LocalVars.empty()) {
  if (!NeedToPush)
    return;
  CGM.getOpenMPRuntime().FunctionToUntiedTaskStackMap.try_emplace(
      CGF.CurFn, CGM.getOpenMPRuntime().UntiedLocalVarsStack.size());
  CGM.getOpenMPRuntime().UntiedLocalVarsStack.push_back(LocalVars);
}

CGOpenMPRuntime::UntiedTaskLocalDeclsRAII::~UntiedTaskLocalDeclsRAII() {
  if (!NeedToPush)
    return;
  CGM.getOpenMPRuntime().UntiedLocalVarsStack.pop_back();
}

bool CGOpenMPRuntime::isNontemporalDecl(const ValueDecl *VD) const {
  assert(CGM.getLangOpts().OpenMP && "Not in OpenMP mode.");

  return llvm::any_of(
      CGM.getOpenMPRuntime().NontemporalDeclsStack,
      [VD](const NontemporalDeclsSet &Set) { return Set.contains(VD); });
}

void CGOpenMPRuntime::LastprivateConditionalRAII::tryToDisableInnerAnalysis(
    const OMPExecutableDirective &S,
    llvm::DenseSet<CanonicalDeclPtr<const Decl>> &NeedToAddForLPCsAsDisabled)
    const {
  llvm::DenseSet<CanonicalDeclPtr<const Decl>> NeedToCheckForLPCs;
  // Vars in target/task regions must be excluded completely.
  if (isOpenMPTargetExecutionDirective(S.getDirectiveKind()) ||
      isOpenMPTaskingDirective(S.getDirectiveKind())) {
    SmallVector<OpenMPDirectiveKind, 4> CaptureRegions;
    getOpenMPCaptureRegions(CaptureRegions, S.getDirectiveKind());
    const CapturedStmt *CS = S.getCapturedStmt(CaptureRegions.front());
    for (const CapturedStmt::Capture &Cap : CS->captures()) {
      if (Cap.capturesVariable() || Cap.capturesVariableByCopy())
        NeedToCheckForLPCs.insert(Cap.getCapturedVar());
    }
  }
  // Exclude vars in private clauses.
  for (const auto *C : S.getClausesOfKind<OMPPrivateClause>()) {
    for (const Expr *Ref : C->varlists()) {
      if (!Ref->getType()->isScalarType())
        continue;
      const auto *DRE = dyn_cast<DeclRefExpr>(Ref->IgnoreParenImpCasts());
      if (!DRE)
        continue;
      NeedToCheckForLPCs.insert(DRE->getDecl());
    }
  }
  for (const auto *C : S.getClausesOfKind<OMPFirstprivateClause>()) {
    for (const Expr *Ref : C->varlists()) {
      if (!Ref->getType()->isScalarType())
        continue;
      const auto *DRE = dyn_cast<DeclRefExpr>(Ref->IgnoreParenImpCasts());
      if (!DRE)
        continue;
      NeedToCheckForLPCs.insert(DRE->getDecl());
    }
  }
  for (const auto *C : S.getClausesOfKind<OMPLastprivateClause>()) {
    for (const Expr *Ref : C->varlists()) {
      if (!Ref->getType()->isScalarType())
        continue;
      const auto *DRE = dyn_cast<DeclRefExpr>(Ref->IgnoreParenImpCasts());
      if (!DRE)
        continue;
      NeedToCheckForLPCs.insert(DRE->getDecl());
    }
  }
  for (const auto *C : S.getClausesOfKind<OMPReductionClause>()) {
    for (const Expr *Ref : C->varlists()) {
      if (!Ref->getType()->isScalarType())
        continue;
      const auto *DRE = dyn_cast<DeclRefExpr>(Ref->IgnoreParenImpCasts());
      if (!DRE)
        continue;
      NeedToCheckForLPCs.insert(DRE->getDecl());
    }
  }
  for (const auto *C : S.getClausesOfKind<OMPLinearClause>()) {
    for (const Expr *Ref : C->varlists()) {
      if (!Ref->getType()->isScalarType())
        continue;
      const auto *DRE = dyn_cast<DeclRefExpr>(Ref->IgnoreParenImpCasts());
      if (!DRE)
        continue;
      NeedToCheckForLPCs.insert(DRE->getDecl());
    }
  }
  for (const Decl *VD : NeedToCheckForLPCs) {
    for (const LastprivateConditionalData &Data :
         llvm::reverse(CGM.getOpenMPRuntime().LastprivateConditionalStack)) {
      if (Data.DeclToUniqueName.count(VD) > 0) {
        if (!Data.Disabled)
          NeedToAddForLPCsAsDisabled.insert(VD);
        break;
      }
    }
  }
}

CGOpenMPRuntime::LastprivateConditionalRAII::LastprivateConditionalRAII(
    CodeGenFunction &CGF, const OMPExecutableDirective &S, LValue IVLVal)
    : CGM(CGF.CGM),
      Action((CGM.getLangOpts().OpenMP >= 50 &&
              llvm::any_of(S.getClausesOfKind<OMPLastprivateClause>(),
                           [](const OMPLastprivateClause *C) {
                             return C->getKind() ==
                                    OMPC_LASTPRIVATE_conditional;
                           }))
                 ? ActionToDo::PushAsLastprivateConditional
                 : ActionToDo::DoNotPush) {
  assert(CGM.getLangOpts().OpenMP && "Not in OpenMP mode.");
  if (CGM.getLangOpts().OpenMP < 50 || Action == ActionToDo::DoNotPush)
    return;
  assert(Action == ActionToDo::PushAsLastprivateConditional &&
         "Expected a push action.");
  LastprivateConditionalData &Data =
      CGM.getOpenMPRuntime().LastprivateConditionalStack.emplace_back();
  for (const auto *C : S.getClausesOfKind<OMPLastprivateClause>()) {
    if (C->getKind() != OMPC_LASTPRIVATE_conditional)
      continue;

    for (const Expr *Ref : C->varlists()) {
      Data.DeclToUniqueName.insert(std::make_pair(
          cast<DeclRefExpr>(Ref->IgnoreParenImpCasts())->getDecl(),
          SmallString<16>(generateUniqueName(CGM, "pl_cond", Ref))));
    }
  }
  Data.IVLVal = IVLVal;
  Data.Fn = CGF.CurFn;
}

CGOpenMPRuntime::LastprivateConditionalRAII::LastprivateConditionalRAII(
    CodeGenFunction &CGF, const OMPExecutableDirective &S)
    : CGM(CGF.CGM), Action(ActionToDo::DoNotPush) {
  assert(CGM.getLangOpts().OpenMP && "Not in OpenMP mode.");
  if (CGM.getLangOpts().OpenMP < 50)
    return;
  llvm::DenseSet<CanonicalDeclPtr<const Decl>> NeedToAddForLPCsAsDisabled;
  tryToDisableInnerAnalysis(S, NeedToAddForLPCsAsDisabled);
  if (!NeedToAddForLPCsAsDisabled.empty()) {
    Action = ActionToDo::DisableLastprivateConditional;
    LastprivateConditionalData &Data =
        CGM.getOpenMPRuntime().LastprivateConditionalStack.emplace_back();
    for (const Decl *VD : NeedToAddForLPCsAsDisabled)
      Data.DeclToUniqueName.insert(std::make_pair(VD, SmallString<16>()));
    Data.Fn = CGF.CurFn;
    Data.Disabled = true;
  }
}

CGOpenMPRuntime::LastprivateConditionalRAII
CGOpenMPRuntime::LastprivateConditionalRAII::disable(
    CodeGenFunction &CGF, const OMPExecutableDirective &S) {
  return LastprivateConditionalRAII(CGF, S);
}

CGOpenMPRuntime::LastprivateConditionalRAII::~LastprivateConditionalRAII() {
  if (CGM.getLangOpts().OpenMP < 50)
    return;
  if (Action == ActionToDo::DisableLastprivateConditional) {
    assert(CGM.getOpenMPRuntime().LastprivateConditionalStack.back().Disabled &&
           "Expected list of disabled private vars.");
    CGM.getOpenMPRuntime().LastprivateConditionalStack.pop_back();
  }
  if (Action == ActionToDo::PushAsLastprivateConditional) {
    assert(
        !CGM.getOpenMPRuntime().LastprivateConditionalStack.back().Disabled &&
        "Expected list of lastprivate conditional vars.");
    CGM.getOpenMPRuntime().LastprivateConditionalStack.pop_back();
  }
}

Address CGOpenMPRuntime::emitLastprivateConditionalInit(CodeGenFunction &CGF,
                                                        const VarDecl *VD) {
  ASTContext &C = CGM.getContext();
  auto I = LastprivateConditionalToTypes.find(CGF.CurFn);
  if (I == LastprivateConditionalToTypes.end())
    I = LastprivateConditionalToTypes.try_emplace(CGF.CurFn).first;
  QualType NewType;
  const FieldDecl *VDField;
  const FieldDecl *FiredField;
  LValue BaseLVal;
  auto VI = I->getSecond().find(VD);
  if (VI == I->getSecond().end()) {
    RecordDecl *RD = C.buildImplicitRecord("lasprivate.conditional");
    RD->startDefinition();
    VDField = addFieldToRecordDecl(C, RD, VD->getType().getNonReferenceType());
    FiredField = addFieldToRecordDecl(C, RD, C.CharTy);
    RD->completeDefinition();
    NewType = C.getRecordType(RD);
    Address Addr = CGF.CreateMemTemp(NewType, C.getDeclAlign(VD), VD->getName());
    BaseLVal = CGF.MakeAddrLValue(Addr, NewType, AlignmentSource::Decl);
    I->getSecond().try_emplace(VD, NewType, VDField, FiredField, BaseLVal);
  } else {
    NewType = std::get<0>(VI->getSecond());
    VDField = std::get<1>(VI->getSecond());
    FiredField = std::get<2>(VI->getSecond());
    BaseLVal = std::get<3>(VI->getSecond());
  }
  LValue FiredLVal =
      CGF.EmitLValueForField(BaseLVal, FiredField);
  CGF.EmitStoreOfScalar(
      llvm::ConstantInt::getNullValue(CGF.ConvertTypeForMem(C.CharTy)),
      FiredLVal);
  return CGF.EmitLValueForField(BaseLVal, VDField).getAddress(CGF);
}

namespace {
/// Checks if the lastprivate conditional variable is referenced in LHS.
class LastprivateConditionalRefChecker final
    : public ConstStmtVisitor<LastprivateConditionalRefChecker, bool> {
  ArrayRef<CGOpenMPRuntime::LastprivateConditionalData> LPM;
  const Expr *FoundE = nullptr;
  const Decl *FoundD = nullptr;
  StringRef UniqueDeclName;
  LValue IVLVal;
  llvm::Function *FoundFn = nullptr;
  SourceLocation Loc;

public:
  bool VisitDeclRefExpr(const DeclRefExpr *E) {
    for (const CGOpenMPRuntime::LastprivateConditionalData &D :
         llvm::reverse(LPM)) {
      auto It = D.DeclToUniqueName.find(E->getDecl());
      if (It == D.DeclToUniqueName.end())
        continue;
      if (D.Disabled)
        return false;
      FoundE = E;
      FoundD = E->getDecl()->getCanonicalDecl();
      UniqueDeclName = It->second;
      IVLVal = D.IVLVal;
      FoundFn = D.Fn;
      break;
    }
    return FoundE == E;
  }
  bool VisitMemberExpr(const MemberExpr *E) {
    if (!CodeGenFunction::IsWrappedCXXThis(E->getBase()))
      return false;
    for (const CGOpenMPRuntime::LastprivateConditionalData &D :
         llvm::reverse(LPM)) {
      auto It = D.DeclToUniqueName.find(E->getMemberDecl());
      if (It == D.DeclToUniqueName.end())
        continue;
      if (D.Disabled)
        return false;
      FoundE = E;
      FoundD = E->getMemberDecl()->getCanonicalDecl();
      UniqueDeclName = It->second;
      IVLVal = D.IVLVal;
      FoundFn = D.Fn;
      break;
    }
    return FoundE == E;
  }
  bool VisitStmt(const Stmt *S) {
    for (const Stmt *Child : S->children()) {
      if (!Child)
        continue;
      if (const auto *E = dyn_cast<Expr>(Child))
        if (!E->isGLValue())
          continue;
      if (Visit(Child))
        return true;
    }
    return false;
  }
  explicit LastprivateConditionalRefChecker(
      ArrayRef<CGOpenMPRuntime::LastprivateConditionalData> LPM)
      : LPM(LPM) {}
  std::tuple<const Expr *, const Decl *, StringRef, LValue, llvm::Function *>
  getFoundData() const {
    return std::make_tuple(FoundE, FoundD, UniqueDeclName, IVLVal, FoundFn);
  }
};
} // namespace

void CGOpenMPRuntime::emitLastprivateConditionalUpdate(CodeGenFunction &CGF,
                                                       LValue IVLVal,
                                                       StringRef UniqueDeclName,
                                                       LValue LVal,
                                                       SourceLocation Loc) {
  // Last updated loop counter for the lastprivate conditional var.
  // int<xx> last_iv = 0;
  llvm::Type *LLIVTy = CGF.ConvertTypeForMem(IVLVal.getType());
  llvm::Constant *LastIV =
      getOrCreateInternalVariable(LLIVTy, getName({UniqueDeclName, "iv"}));
  cast<llvm::GlobalVariable>(LastIV)->setAlignment(
      IVLVal.getAlignment().getAsAlign());
  LValue LastIVLVal = CGF.MakeNaturalAlignAddrLValue(LastIV, IVLVal.getType());

  // Last value of the lastprivate conditional.
  // decltype(priv_a) last_a;
  llvm::GlobalVariable *Last = getOrCreateInternalVariable(
      CGF.ConvertTypeForMem(LVal.getType()), UniqueDeclName);
  Last->setAlignment(LVal.getAlignment().getAsAlign());
  LValue LastLVal = CGF.MakeAddrLValue(
      Address(Last, Last->getValueType(), LVal.getAlignment()), LVal.getType());

  // Global loop counter. Required to handle inner parallel-for regions.
  // iv
  llvm::Value *IVVal = CGF.EmitLoadOfScalar(IVLVal, Loc);

  // #pragma omp critical(a)
  // if (last_iv <= iv) {
  //   last_iv = iv;
  //   last_a = priv_a;
  // }
  auto &&CodeGen = [&LastIVLVal, &IVLVal, IVVal, &LVal, &LastLVal,
                    Loc](CodeGenFunction &CGF, PrePostActionTy &Action) {
    Action.Enter(CGF);
    llvm::Value *LastIVVal = CGF.EmitLoadOfScalar(LastIVLVal, Loc);
    // (last_iv <= iv) ? Check if the variable is updated and store new
    // value in global var.
    llvm::Value *CmpRes;
    if (IVLVal.getType()->isSignedIntegerType()) {
      CmpRes = CGF.Builder.CreateICmpSLE(LastIVVal, IVVal);
    } else {
      assert(IVLVal.getType()->isUnsignedIntegerType() &&
             "Loop iteration variable must be integer.");
      CmpRes = CGF.Builder.CreateICmpULE(LastIVVal, IVVal);
    }
    llvm::BasicBlock *ThenBB = CGF.createBasicBlock("lp_cond_then");
    llvm::BasicBlock *ExitBB = CGF.createBasicBlock("lp_cond_exit");
    CGF.Builder.CreateCondBr(CmpRes, ThenBB, ExitBB);
    // {
    CGF.EmitBlock(ThenBB);

    //   last_iv = iv;
    CGF.EmitStoreOfScalar(IVVal, LastIVLVal);

    //   last_a = priv_a;
    switch (CGF.getEvaluationKind(LVal.getType())) {
    case TEK_Scalar: {
      llvm::Value *PrivVal = CGF.EmitLoadOfScalar(LVal, Loc);
      CGF.EmitStoreOfScalar(PrivVal, LastLVal);
      break;
    }
    case TEK_Complex: {
      CodeGenFunction::ComplexPairTy PrivVal = CGF.EmitLoadOfComplex(LVal, Loc);
      CGF.EmitStoreOfComplex(PrivVal, LastLVal, /*isInit=*/false);
      break;
    }
    case TEK_Aggregate:
      llvm_unreachable(
          "Aggregates are not supported in lastprivate conditional.");
    }
    // }
    CGF.EmitBranch(ExitBB);
    // There is no need to emit line number for unconditional branch.
    (void)ApplyDebugLocation::CreateEmpty(CGF);
    CGF.EmitBlock(ExitBB, /*IsFinished=*/true);
  };

  if (CGM.getLangOpts().OpenMPSimd) {
    // Do not emit as a critical region as no parallel region could be emitted.
    RegionCodeGenTy ThenRCG(CodeGen);
    ThenRCG(CGF);
  } else {
    emitCriticalRegion(CGF, UniqueDeclName, CodeGen, Loc);
  }
}

void CGOpenMPRuntime::checkAndEmitLastprivateConditional(CodeGenFunction &CGF,
                                                         const Expr *LHS) {
  if (CGF.getLangOpts().OpenMP < 50 || LastprivateConditionalStack.empty())
    return;
  LastprivateConditionalRefChecker Checker(LastprivateConditionalStack);
  if (!Checker.Visit(LHS))
    return;
  const Expr *FoundE;
  const Decl *FoundD;
  StringRef UniqueDeclName;
  LValue IVLVal;
  llvm::Function *FoundFn;
  std::tie(FoundE, FoundD, UniqueDeclName, IVLVal, FoundFn) =
      Checker.getFoundData();
  if (FoundFn != CGF.CurFn) {
    // Special codegen for inner parallel regions.
    // ((struct.lastprivate.conditional*)&priv_a)->Fired = 1;
    auto It = LastprivateConditionalToTypes[FoundFn].find(FoundD);
    assert(It != LastprivateConditionalToTypes[FoundFn].end() &&
           "Lastprivate conditional is not found in outer region.");
    QualType StructTy = std::get<0>(It->getSecond());
    const FieldDecl* FiredDecl = std::get<2>(It->getSecond());
    LValue PrivLVal = CGF.EmitLValue(FoundE);
    Address StructAddr = CGF.Builder.CreatePointerBitCastOrAddrSpaceCast(
        PrivLVal.getAddress(CGF),
        CGF.ConvertTypeForMem(CGF.getContext().getPointerType(StructTy)));
    LValue BaseLVal =
        CGF.MakeAddrLValue(StructAddr, StructTy, AlignmentSource::Decl);
    LValue FiredLVal = CGF.EmitLValueForField(BaseLVal, FiredDecl);
    CGF.EmitAtomicStore(RValue::get(llvm::ConstantInt::get(
                            CGF.ConvertTypeForMem(FiredDecl->getType()), 1)),
                        FiredLVal, llvm::AtomicOrdering::Unordered,
                        /*IsVolatile=*/true, /*isInit=*/false);
    return;
  }

  // Private address of the lastprivate conditional in the current context.
  // priv_a
  LValue LVal = CGF.EmitLValue(FoundE);
  emitLastprivateConditionalUpdate(CGF, IVLVal, UniqueDeclName, LVal,
                                   FoundE->getExprLoc());
}

void CGOpenMPRuntime::checkAndEmitSharedLastprivateConditional(
    CodeGenFunction &CGF, const OMPExecutableDirective &D,
    const llvm::DenseSet<CanonicalDeclPtr<const VarDecl>> &IgnoredDecls) {
  if (CGF.getLangOpts().OpenMP < 50 || LastprivateConditionalStack.empty())
    return;
  auto Range = llvm::reverse(LastprivateConditionalStack);
  auto It = llvm::find_if(
      Range, [](const LastprivateConditionalData &D) { return !D.Disabled; });
  if (It == Range.end() || It->Fn != CGF.CurFn)
    return;
  auto LPCI = LastprivateConditionalToTypes.find(It->Fn);
  assert(LPCI != LastprivateConditionalToTypes.end() &&
         "Lastprivates must be registered already.");
  SmallVector<OpenMPDirectiveKind, 4> CaptureRegions;
  getOpenMPCaptureRegions(CaptureRegions, D.getDirectiveKind());
  const CapturedStmt *CS = D.getCapturedStmt(CaptureRegions.back());
  for (const auto &Pair : It->DeclToUniqueName) {
    const auto *VD = cast<VarDecl>(Pair.first->getCanonicalDecl());
    if (!CS->capturesVariable(VD) || IgnoredDecls.contains(VD))
      continue;
    auto I = LPCI->getSecond().find(Pair.first);
    assert(I != LPCI->getSecond().end() &&
           "Lastprivate must be rehistered already.");
    // bool Cmp = priv_a.Fired != 0;
    LValue BaseLVal = std::get<3>(I->getSecond());
    LValue FiredLVal =
        CGF.EmitLValueForField(BaseLVal, std::get<2>(I->getSecond()));
    llvm::Value *Res = CGF.EmitLoadOfScalar(FiredLVal, D.getBeginLoc());
    llvm::Value *Cmp = CGF.Builder.CreateIsNotNull(Res);
    llvm::BasicBlock *ThenBB = CGF.createBasicBlock("lpc.then");
    llvm::BasicBlock *DoneBB = CGF.createBasicBlock("lpc.done");
    // if (Cmp) {
    CGF.Builder.CreateCondBr(Cmp, ThenBB, DoneBB);
    CGF.EmitBlock(ThenBB);
    Address Addr = CGF.GetAddrOfLocalVar(VD);
    LValue LVal;
    if (VD->getType()->isReferenceType())
      LVal = CGF.EmitLoadOfReferenceLValue(Addr, VD->getType(),
                                           AlignmentSource::Decl);
    else
      LVal = CGF.MakeAddrLValue(Addr, VD->getType().getNonReferenceType(),
                                AlignmentSource::Decl);
    emitLastprivateConditionalUpdate(CGF, It->IVLVal, Pair.second, LVal,
                                     D.getBeginLoc());
    auto AL = ApplyDebugLocation::CreateArtificial(CGF);
    CGF.EmitBlock(DoneBB, /*IsFinal=*/true);
    // }
  }
}

void CGOpenMPRuntime::emitLastprivateConditionalFinalUpdate(
    CodeGenFunction &CGF, LValue PrivLVal, const VarDecl *VD,
    SourceLocation Loc) {
  if (CGF.getLangOpts().OpenMP < 50)
    return;
  auto It = LastprivateConditionalStack.back().DeclToUniqueName.find(VD);
  assert(It != LastprivateConditionalStack.back().DeclToUniqueName.end() &&
         "Unknown lastprivate conditional variable.");
  StringRef UniqueName = It->second;
  llvm::GlobalVariable *GV = CGM.getModule().getNamedGlobal(UniqueName);
  // The variable was not updated in the region - exit.
  if (!GV)
    return;
  LValue LPLVal = CGF.MakeAddrLValue(
      Address(GV, GV->getValueType(), PrivLVal.getAlignment()),
      PrivLVal.getType().getNonReferenceType());
  llvm::Value *Res = CGF.EmitLoadOfScalar(LPLVal, Loc);
  CGF.EmitStoreOfScalar(Res, PrivLVal);
}

llvm::Function *CGOpenMPSIMDRuntime::emitParallelOutlinedFunction(
    const OMPExecutableDirective &D, const VarDecl *ThreadIDVar,
    OpenMPDirectiveKind InnermostKind, const RegionCodeGenTy &CodeGen) {
  llvm_unreachable("Not supported in SIMD-only mode");
}

llvm::Function *CGOpenMPSIMDRuntime::emitTeamsOutlinedFunction(
    const OMPExecutableDirective &D, const VarDecl *ThreadIDVar,
    OpenMPDirectiveKind InnermostKind, const RegionCodeGenTy &CodeGen) {
  llvm_unreachable("Not supported in SIMD-only mode");
}

llvm::Function *CGOpenMPSIMDRuntime::emitTaskOutlinedFunction(
    const OMPExecutableDirective &D, const VarDecl *ThreadIDVar,
    const VarDecl *PartIDVar, const VarDecl *TaskTVar,
    OpenMPDirectiveKind InnermostKind, const RegionCodeGenTy &CodeGen,
    bool Tied, unsigned &NumberOfParts) {
  llvm_unreachable("Not supported in SIMD-only mode");
}

void CGOpenMPSIMDRuntime::emitParallelCall(CodeGenFunction &CGF,
                                           SourceLocation Loc,
                                           llvm::Function *OutlinedFn,
                                           ArrayRef<llvm::Value *> CapturedVars,
                                           const Expr *IfCond,
                                           llvm::Value *NumThreads) {
  llvm_unreachable("Not supported in SIMD-only mode");
}

void CGOpenMPSIMDRuntime::emitCriticalRegion(
    CodeGenFunction &CGF, StringRef CriticalName,
    const RegionCodeGenTy &CriticalOpGen, SourceLocation Loc,
    const Expr *Hint) {
  llvm_unreachable("Not supported in SIMD-only mode");
}

void CGOpenMPSIMDRuntime::emitMasterRegion(CodeGenFunction &CGF,
                                           const RegionCodeGenTy &MasterOpGen,
                                           SourceLocation Loc) {
  llvm_unreachable("Not supported in SIMD-only mode");
}

void CGOpenMPSIMDRuntime::emitMaskedRegion(CodeGenFunction &CGF,
                                           const RegionCodeGenTy &MasterOpGen,
                                           SourceLocation Loc,
                                           const Expr *Filter) {
  llvm_unreachable("Not supported in SIMD-only mode");
}

void CGOpenMPSIMDRuntime::emitTaskyieldCall(CodeGenFunction &CGF,
                                            SourceLocation Loc) {
  llvm_unreachable("Not supported in SIMD-only mode");
}

void CGOpenMPSIMDRuntime::emitTaskgroupRegion(
    CodeGenFunction &CGF, const RegionCodeGenTy &TaskgroupOpGen,
    SourceLocation Loc) {
  llvm_unreachable("Not supported in SIMD-only mode");
}

void CGOpenMPSIMDRuntime::emitSingleRegion(
    CodeGenFunction &CGF, const RegionCodeGenTy &SingleOpGen,
    SourceLocation Loc, ArrayRef<const Expr *> CopyprivateVars,
    ArrayRef<const Expr *> DestExprs, ArrayRef<const Expr *> SrcExprs,
    ArrayRef<const Expr *> AssignmentOps) {
  llvm_unreachable("Not supported in SIMD-only mode");
}

void CGOpenMPSIMDRuntime::emitOrderedRegion(CodeGenFunction &CGF,
                                            const RegionCodeGenTy &OrderedOpGen,
                                            SourceLocation Loc,
                                            bool IsThreads) {
  llvm_unreachable("Not supported in SIMD-only mode");
}

void CGOpenMPSIMDRuntime::emitBarrierCall(CodeGenFunction &CGF,
                                          SourceLocation Loc,
                                          OpenMPDirectiveKind Kind,
                                          bool EmitChecks,
                                          bool ForceSimpleCall) {
  llvm_unreachable("Not supported in SIMD-only mode");
}

void CGOpenMPSIMDRuntime::emitForDispatchInit(
    CodeGenFunction &CGF, SourceLocation Loc,
    const OpenMPScheduleTy &ScheduleKind, unsigned IVSize, bool IVSigned,
    bool Ordered, const DispatchRTInput &DispatchValues) {
  llvm_unreachable("Not supported in SIMD-only mode");
}

void CGOpenMPSIMDRuntime::emitForStaticInit(
    CodeGenFunction &CGF, SourceLocation Loc, OpenMPDirectiveKind DKind,
    const OpenMPScheduleTy &ScheduleKind, const StaticRTInput &Values) {
  llvm_unreachable("Not supported in SIMD-only mode");
}

void CGOpenMPSIMDRuntime::emitDistributeStaticInit(
    CodeGenFunction &CGF, SourceLocation Loc,
    OpenMPDistScheduleClauseKind SchedKind, const StaticRTInput &Values) {
  llvm_unreachable("Not supported in SIMD-only mode");
}

void CGOpenMPSIMDRuntime::emitForOrderedIterationEnd(CodeGenFunction &CGF,
                                                     SourceLocation Loc,
                                                     unsigned IVSize,
                                                     bool IVSigned) {
  llvm_unreachable("Not supported in SIMD-only mode");
}

void CGOpenMPSIMDRuntime::emitForStaticFinish(CodeGenFunction &CGF,
                                              SourceLocation Loc,
                                              OpenMPDirectiveKind DKind) {
  llvm_unreachable("Not supported in SIMD-only mode");
}

llvm::Value *CGOpenMPSIMDRuntime::emitForNext(CodeGenFunction &CGF,
                                              SourceLocation Loc,
                                              unsigned IVSize, bool IVSigned,
                                              Address IL, Address LB,
                                              Address UB, Address ST) {
  llvm_unreachable("Not supported in SIMD-only mode");
}

void CGOpenMPSIMDRuntime::emitNumThreadsClause(CodeGenFunction &CGF,
                                               llvm::Value *NumThreads,
                                               SourceLocation Loc) {
  llvm_unreachable("Not supported in SIMD-only mode");
}

void CGOpenMPSIMDRuntime::emitProcBindClause(CodeGenFunction &CGF,
                                             ProcBindKind ProcBind,
                                             SourceLocation Loc) {
  llvm_unreachable("Not supported in SIMD-only mode");
}

Address CGOpenMPSIMDRuntime::getAddrOfThreadPrivate(CodeGenFunction &CGF,
                                                    const VarDecl *VD,
                                                    Address VDAddr,
                                                    SourceLocation Loc) {
  llvm_unreachable("Not supported in SIMD-only mode");
}

llvm::Function *CGOpenMPSIMDRuntime::emitThreadPrivateVarDefinition(
    const VarDecl *VD, Address VDAddr, SourceLocation Loc, bool PerformInit,
    CodeGenFunction *CGF) {
  llvm_unreachable("Not supported in SIMD-only mode");
}

Address CGOpenMPSIMDRuntime::getAddrOfArtificialThreadPrivate(
    CodeGenFunction &CGF, QualType VarType, StringRef Name) {
  llvm_unreachable("Not supported in SIMD-only mode");
}

void CGOpenMPSIMDRuntime::emitFlush(CodeGenFunction &CGF,
                                    ArrayRef<const Expr *> Vars,
                                    SourceLocation Loc,
                                    llvm::AtomicOrdering AO) {
  llvm_unreachable("Not supported in SIMD-only mode");
}

void CGOpenMPSIMDRuntime::emitTaskCall(CodeGenFunction &CGF, SourceLocation Loc,
                                       const OMPExecutableDirective &D,
                                       llvm::Function *TaskFunction,
                                       QualType SharedsTy, Address Shareds,
                                       const Expr *IfCond,
                                       const OMPTaskDataTy &Data) {
  llvm_unreachable("Not supported in SIMD-only mode");
}

void CGOpenMPSIMDRuntime::emitTaskLoopCall(
    CodeGenFunction &CGF, SourceLocation Loc, const OMPLoopDirective &D,
    llvm::Function *TaskFunction, QualType SharedsTy, Address Shareds,
    const Expr *IfCond, const OMPTaskDataTy &Data) {
  llvm_unreachable("Not supported in SIMD-only mode");
}

void CGOpenMPSIMDRuntime::emitReduction(
    CodeGenFunction &CGF, SourceLocation Loc, ArrayRef<const Expr *> Privates,
    ArrayRef<const Expr *> LHSExprs, ArrayRef<const Expr *> RHSExprs,
    ArrayRef<const Expr *> ReductionOps, ReductionOptionsTy Options) {
  assert(Options.SimpleReduction && "Only simple reduction is expected.");
  CGOpenMPRuntime::emitReduction(CGF, Loc, Privates, LHSExprs, RHSExprs,
                                 ReductionOps, Options);
}

llvm::Value *CGOpenMPSIMDRuntime::emitTaskReductionInit(
    CodeGenFunction &CGF, SourceLocation Loc, ArrayRef<const Expr *> LHSExprs,
    ArrayRef<const Expr *> RHSExprs, const OMPTaskDataTy &Data) {
  llvm_unreachable("Not supported in SIMD-only mode");
}

void CGOpenMPSIMDRuntime::emitTaskReductionFini(CodeGenFunction &CGF,
                                                SourceLocation Loc,
                                                bool IsWorksharingReduction) {
  llvm_unreachable("Not supported in SIMD-only mode");
}

void CGOpenMPSIMDRuntime::emitTaskReductionFixups(CodeGenFunction &CGF,
                                                  SourceLocation Loc,
                                                  ReductionCodeGen &RCG,
                                                  unsigned N) {
  llvm_unreachable("Not supported in SIMD-only mode");
}

Address CGOpenMPSIMDRuntime::getTaskReductionItem(CodeGenFunction &CGF,
                                                  SourceLocation Loc,
                                                  llvm::Value *ReductionsPtr,
                                                  LValue SharedLVal) {
  llvm_unreachable("Not supported in SIMD-only mode");
}

void CGOpenMPSIMDRuntime::emitTaskwaitCall(CodeGenFunction &CGF,
                                           SourceLocation Loc,
                                           const OMPTaskDataTy &Data) {
  llvm_unreachable("Not supported in SIMD-only mode");
}

void CGOpenMPSIMDRuntime::emitCancellationPointCall(
    CodeGenFunction &CGF, SourceLocation Loc,
    OpenMPDirectiveKind CancelRegion) {
  llvm_unreachable("Not supported in SIMD-only mode");
}

void CGOpenMPSIMDRuntime::emitCancelCall(CodeGenFunction &CGF,
                                         SourceLocation Loc, const Expr *IfCond,
                                         OpenMPDirectiveKind CancelRegion) {
  llvm_unreachable("Not supported in SIMD-only mode");
}

void CGOpenMPSIMDRuntime::emitTargetOutlinedFunction(
    const OMPExecutableDirective &D, StringRef ParentName,
    llvm::Function *&OutlinedFn, llvm::Constant *&OutlinedFnID,
    bool IsOffloadEntry, const RegionCodeGenTy &CodeGen) {
  llvm_unreachable("Not supported in SIMD-only mode");
}

void CGOpenMPSIMDRuntime::emitTargetCall(
    CodeGenFunction &CGF, const OMPExecutableDirective &D,
    llvm::Function *OutlinedFn, llvm::Value *OutlinedFnID, const Expr *IfCond,
    llvm::PointerIntPair<const Expr *, 2, OpenMPDeviceClauseModifier> Device,
    llvm::function_ref<llvm::Value *(CodeGenFunction &CGF,
                                     const OMPLoopDirective &D)>
        SizeEmitter) {
  llvm_unreachable("Not supported in SIMD-only mode");
}

bool CGOpenMPSIMDRuntime::emitTargetFunctions(GlobalDecl GD) {
  llvm_unreachable("Not supported in SIMD-only mode");
}

bool CGOpenMPSIMDRuntime::emitTargetGlobalVariable(GlobalDecl GD) {
  llvm_unreachable("Not supported in SIMD-only mode");
}

bool CGOpenMPSIMDRuntime::emitTargetGlobal(GlobalDecl GD) {
  return false;
}

void CGOpenMPSIMDRuntime::emitTeamsCall(CodeGenFunction &CGF,
                                        const OMPExecutableDirective &D,
                                        SourceLocation Loc,
                                        llvm::Function *OutlinedFn,
                                        ArrayRef<llvm::Value *> CapturedVars) {
  llvm_unreachable("Not supported in SIMD-only mode");
}

void CGOpenMPSIMDRuntime::emitNumTeamsClause(CodeGenFunction &CGF,
                                             const Expr *NumTeams,
                                             const Expr *ThreadLimit,
                                             SourceLocation Loc) {
  llvm_unreachable("Not supported in SIMD-only mode");
}

void CGOpenMPSIMDRuntime::emitTargetDataCalls(
    CodeGenFunction &CGF, const OMPExecutableDirective &D, const Expr *IfCond,
    const Expr *Device, const RegionCodeGenTy &CodeGen, TargetDataInfo &Info) {
  llvm_unreachable("Not supported in SIMD-only mode");
}

void CGOpenMPSIMDRuntime::emitTargetDataStandAloneCall(
    CodeGenFunction &CGF, const OMPExecutableDirective &D, const Expr *IfCond,
    const Expr *Device) {
  llvm_unreachable("Not supported in SIMD-only mode");
}

void CGOpenMPSIMDRuntime::emitDoacrossInit(CodeGenFunction &CGF,
                                           const OMPLoopDirective &D,
                                           ArrayRef<Expr *> NumIterations) {
  llvm_unreachable("Not supported in SIMD-only mode");
}

void CGOpenMPSIMDRuntime::emitDoacrossOrdered(CodeGenFunction &CGF,
                                              const OMPDependClause *C) {
  llvm_unreachable("Not supported in SIMD-only mode");
}

const VarDecl *
CGOpenMPSIMDRuntime::translateParameter(const FieldDecl *FD,
                                        const VarDecl *NativeParam) const {
  llvm_unreachable("Not supported in SIMD-only mode");
}

Address
CGOpenMPSIMDRuntime::getParameterAddress(CodeGenFunction &CGF,
                                         const VarDecl *NativeParam,
                                         const VarDecl *TargetParam) const {
  llvm_unreachable("Not supported in SIMD-only mode");
}
