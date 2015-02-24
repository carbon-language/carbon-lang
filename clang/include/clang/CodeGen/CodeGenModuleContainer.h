//===--- CodeGen/ModuleContainerGenerator.h - Emit .pcm files ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CODEGEN_MODULE_CONTAINER_H
#define LLVM_CLANG_CODEGEN_MODULE_CONTAINER_H

#include "ModuleBuilder.h"
#include <string>

namespace llvm {
class raw_ostream;
}

namespace clang {

class PCHGenerator;
class TargetOptions;

/// \brief Create a CodeGenerator instance.
/// It is the responsibility of the caller to call delete on
/// the allocated CodeGenerator instance.
CodeGenerator *CreateModuleContainerGenerator(
    DiagnosticsEngine &Diags, const std::string &ModuleName,
    const CodeGenOptions &CGO, const TargetOptions &TO, const LangOptions &LO,
    llvm::raw_ostream *OS, PCHGenerator *PCHGen);
}

#endif
