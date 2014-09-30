//===--- SemaCUDA.cpp - Semantic Analysis for CUDA constructs -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// \brief This file implements semantic analysis for CUDA constructs.
///
//===----------------------------------------------------------------------===//

#include "clang/Sema/Sema.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/Sema/SemaDiagnostic.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
using namespace clang;

ExprResult Sema::ActOnCUDAExecConfigExpr(Scope *S, SourceLocation LLLLoc,
                                         MultiExprArg ExecConfig,
                                         SourceLocation GGGLoc) {
  FunctionDecl *ConfigDecl = Context.getcudaConfigureCallDecl();
  if (!ConfigDecl)
    return ExprError(Diag(LLLLoc, diag::err_undeclared_var_use)
                     << "cudaConfigureCall");
  QualType ConfigQTy = ConfigDecl->getType();

  DeclRefExpr *ConfigDR = new (Context)
      DeclRefExpr(ConfigDecl, false, ConfigQTy, VK_LValue, LLLLoc);
  MarkFunctionReferenced(LLLLoc, ConfigDecl);

  return ActOnCallExpr(S, ConfigDR, LLLLoc, ExecConfig, GGGLoc, nullptr,
                       /*IsExecConfig=*/true);
}

/// IdentifyCUDATarget - Determine the CUDA compilation target for this function
Sema::CUDAFunctionTarget Sema::IdentifyCUDATarget(const FunctionDecl *D) {
  if (D->hasAttr<CUDAInvalidTargetAttr>())
    return CFT_InvalidTarget;

  if (D->hasAttr<CUDAGlobalAttr>())
    return CFT_Global;

  if (D->hasAttr<CUDADeviceAttr>()) {
    if (D->hasAttr<CUDAHostAttr>())
      return CFT_HostDevice;
    return CFT_Device;
  } else if (D->hasAttr<CUDAHostAttr>()) {
    return CFT_Host;
  } else if (D->isImplicit()) {
    // Some implicit declarations (like intrinsic functions) are not marked.
    // Set the most lenient target on them for maximal flexibility.
    return CFT_HostDevice;
  }

  return CFT_Host;
}

bool Sema::CheckCUDATarget(const FunctionDecl *Caller,
                           const FunctionDecl *Callee) {
  return CheckCUDATarget(IdentifyCUDATarget(Caller),
                         IdentifyCUDATarget(Callee));
}

bool Sema::CheckCUDATarget(CUDAFunctionTarget CallerTarget,
                           CUDAFunctionTarget CalleeTarget) {
  // If one of the targets is invalid, the check always fails, no matter what
  // the other target is.
  if (CallerTarget == CFT_InvalidTarget || CalleeTarget == CFT_InvalidTarget)
    return true;

  // CUDA B.1.1 "The __device__ qualifier declares a function that is...
  // Callable from the device only."
  if (CallerTarget == CFT_Host && CalleeTarget == CFT_Device)
    return true;

  // CUDA B.1.2 "The __global__ qualifier declares a function that is...
  // Callable from the host only."
  // CUDA B.1.3 "The __host__ qualifier declares a function that is...
  // Callable from the host only."
  if ((CallerTarget == CFT_Device || CallerTarget == CFT_Global) &&
      (CalleeTarget == CFT_Host || CalleeTarget == CFT_Global))
    return true;

  if (CallerTarget == CFT_HostDevice && CalleeTarget != CFT_HostDevice)
    return true;

  return false;
}

/// When an implicitly-declared special member has to invoke more than one
/// base/field special member, conflicts may occur in the targets of these
/// members. For example, if one base's member __host__ and another's is
/// __device__, it's a conflict.
/// This function figures out if the given targets \param Target1 and
/// \param Target2 conflict, and if they do not it fills in
/// \param ResolvedTarget with a target that resolves for both calls.
/// \return true if there's a conflict, false otherwise.
static bool
resolveCalleeCUDATargetConflict(Sema::CUDAFunctionTarget Target1,
                                Sema::CUDAFunctionTarget Target2,
                                Sema::CUDAFunctionTarget *ResolvedTarget) {
  if (Target1 == Sema::CFT_Global && Target2 == Sema::CFT_Global) {
    // TODO: this shouldn't happen, really. Methods cannot be marked __global__.
    // Clang should detect this earlier and produce an error. Then this
    // condition can be changed to an assertion.
    return true;
  }

  if (Target1 == Sema::CFT_HostDevice) {
    *ResolvedTarget = Target2;
  } else if (Target2 == Sema::CFT_HostDevice) {
    *ResolvedTarget = Target1;
  } else if (Target1 != Target2) {
    return true;
  } else {
    *ResolvedTarget = Target1;
  }

  return false;
}

bool Sema::inferCUDATargetForImplicitSpecialMember(CXXRecordDecl *ClassDecl,
                                                   CXXSpecialMember CSM,
                                                   CXXMethodDecl *MemberDecl,
                                                   bool ConstRHS,
                                                   bool Diagnose) {
  llvm::Optional<CUDAFunctionTarget> InferredTarget;

  // We're going to invoke special member lookup; mark that these special
  // members are called from this one, and not from its caller.
  ContextRAII MethodContext(*this, MemberDecl);

  // Look for special members in base classes that should be invoked from here.
  // Infer the target of this member base on the ones it should call.
  // Skip direct and indirect virtual bases for abstract classes.
  llvm::SmallVector<const CXXBaseSpecifier *, 16> Bases;
  for (const auto &B : ClassDecl->bases()) {
    if (!B.isVirtual()) {
      Bases.push_back(&B);
    }
  }

  if (!ClassDecl->isAbstract()) {
    for (const auto &VB : ClassDecl->vbases()) {
      Bases.push_back(&VB);
    }
  }

  for (const auto *B : Bases) {
    const RecordType *BaseType = B->getType()->getAs<RecordType>();
    if (!BaseType) {
      continue;
    }

    CXXRecordDecl *BaseClassDecl = cast<CXXRecordDecl>(BaseType->getDecl());
    Sema::SpecialMemberOverloadResult *SMOR =
        LookupSpecialMember(BaseClassDecl, CSM,
                            /* ConstArg */ ConstRHS,
                            /* VolatileArg */ false,
                            /* RValueThis */ false,
                            /* ConstThis */ false,
                            /* VolatileThis */ false);

    if (!SMOR || !SMOR->getMethod()) {
      continue;
    }

    CUDAFunctionTarget BaseMethodTarget = IdentifyCUDATarget(SMOR->getMethod());
    if (!InferredTarget.hasValue()) {
      InferredTarget = BaseMethodTarget;
    } else {
      bool ResolutionError = resolveCalleeCUDATargetConflict(
          InferredTarget.getValue(), BaseMethodTarget,
          InferredTarget.getPointer());
      if (ResolutionError) {
        if (Diagnose) {
          Diag(ClassDecl->getLocation(),
               diag::note_implicit_member_target_infer_collision)
              << (unsigned)CSM << InferredTarget.getValue() << BaseMethodTarget;
        }
        MemberDecl->addAttr(CUDAInvalidTargetAttr::CreateImplicit(Context));
        return true;
      }
    }
  }

  // Same as for bases, but now for special members of fields.
  for (const auto *F : ClassDecl->fields()) {
    if (F->isInvalidDecl()) {
      continue;
    }

    const RecordType *FieldType =
        Context.getBaseElementType(F->getType())->getAs<RecordType>();
    if (!FieldType) {
      continue;
    }

    CXXRecordDecl *FieldRecDecl = cast<CXXRecordDecl>(FieldType->getDecl());
    Sema::SpecialMemberOverloadResult *SMOR =
        LookupSpecialMember(FieldRecDecl, CSM,
                            /* ConstArg */ ConstRHS && !F->isMutable(),
                            /* VolatileArg */ false,
                            /* RValueThis */ false,
                            /* ConstThis */ false,
                            /* VolatileThis */ false);

    if (!SMOR || !SMOR->getMethod()) {
      continue;
    }

    CUDAFunctionTarget FieldMethodTarget =
        IdentifyCUDATarget(SMOR->getMethod());
    if (!InferredTarget.hasValue()) {
      InferredTarget = FieldMethodTarget;
    } else {
      bool ResolutionError = resolveCalleeCUDATargetConflict(
          InferredTarget.getValue(), FieldMethodTarget,
          InferredTarget.getPointer());
      if (ResolutionError) {
        if (Diagnose) {
          Diag(ClassDecl->getLocation(),
               diag::note_implicit_member_target_infer_collision)
              << (unsigned)CSM << InferredTarget.getValue()
              << FieldMethodTarget;
        }
        MemberDecl->addAttr(CUDAInvalidTargetAttr::CreateImplicit(Context));
        return true;
      }
    }
  }

  if (InferredTarget.hasValue()) {
    if (InferredTarget.getValue() == CFT_Device) {
      MemberDecl->addAttr(CUDADeviceAttr::CreateImplicit(Context));
    } else if (InferredTarget.getValue() == CFT_Host) {
      MemberDecl->addAttr(CUDAHostAttr::CreateImplicit(Context));
    } else {
      MemberDecl->addAttr(CUDADeviceAttr::CreateImplicit(Context));
      MemberDecl->addAttr(CUDAHostAttr::CreateImplicit(Context));
    }
  } else {
    // If no target was inferred, mark this member as __host__ __device__;
    // it's the least restrictive option that can be invoked from any target.
    MemberDecl->addAttr(CUDADeviceAttr::CreateImplicit(Context));
    MemberDecl->addAttr(CUDAHostAttr::CreateImplicit(Context));
  }

  return false;
}
