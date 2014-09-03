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
  // Implicitly declared functions (e.g. copy constructors) are
  // __host__ __device__
  if (D->isImplicit())
    return CFT_HostDevice;

  if (D->hasAttr<CUDAGlobalAttr>())
    return CFT_Global;

  if (D->hasAttr<CUDADeviceAttr>()) {
    if (D->hasAttr<CUDAHostAttr>())
      return CFT_HostDevice;
    return CFT_Device;
  }

  return CFT_Host;
}

bool Sema::CheckCUDATarget(CUDAFunctionTarget CallerTarget,
                           CUDAFunctionTarget CalleeTarget) {
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

