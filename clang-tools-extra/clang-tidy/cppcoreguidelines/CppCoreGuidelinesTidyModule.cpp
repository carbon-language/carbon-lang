//===--- CppCoreGuidelinesModule.cpp - clang-tidy -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "../ClangTidy.h"
#include "../ClangTidyModule.h"
#include "../ClangTidyModuleRegistry.h"
#include "../misc/AssignOperatorSignatureCheck.h"
#include "ProBoundsPointerArithmeticCheck.h"
#include "ProTypeConstCastCheck.h"
#include "ProTypeReinterpretCastCheck.h"
#include "ProTypeStaticCastDowncastCheck.h"

namespace clang {
namespace tidy {
namespace cppcoreguidelines {

/// A module containing checks of the C++ Core Guidelines
class CppCoreGuidelinesModule : public ClangTidyModule {
public:
  void addCheckFactories(ClangTidyCheckFactories &CheckFactories) override {
    CheckFactories.registerCheck<ProBoundsPointerArithmeticCheck>(
        "cppcoreguidelines-pro-bounds-pointer-arithmetic");
    CheckFactories.registerCheck<ProTypeConstCastCheck>(
        "cppcoreguidelines-pro-type-const-cast");
    CheckFactories.registerCheck<ProTypeReinterpretCastCheck>(
        "cppcoreguidelines-pro-type-reinterpret-cast");
    CheckFactories.registerCheck<ProTypeStaticCastDowncastCheck>(
        "cppcoreguidelines-pro-type-static-cast-downcast");
    CheckFactories.registerCheck<misc::AssignOperatorSignatureCheck>(
        "cppcoreguidelines-c-copy-assignment-signature");
  }
};

// Register the LLVMTidyModule using this statically initialized variable.
static ClangTidyModuleRegistry::Add<CppCoreGuidelinesModule>
    X("cppcoreguidelines-module", "Adds checks for the C++ Core Guidelines.");

} // namespace cppcoreguidelines

// This anchor is used to force the linker to link in the generated object file
// and thus register the CppCoreGuidelinesModule.
volatile int CppCoreGuidelinesModuleAnchorSource = 0;

} // namespace tidy
} // namespace clang
