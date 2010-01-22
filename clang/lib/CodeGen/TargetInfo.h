//===---- TargetInfo.h - Encapsulate target details -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// These classes wrap the information about a call or function
// definition used to handle ABI compliancy.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CODEGEN_TARGETINFO_H
#define CLANG_CODEGEN_TARGETINFO_H

namespace llvm {
  class GlobalValue;
}

namespace clang {
  class ABIInfo;
  class Decl;

  namespace CodeGen {
    class CodeGenModule;
  }

  /// TargetCodeGenInfo - This class organizes various target-specific
  /// codegeneration issues, like target-specific attributes, builtins and so
  /// on.
  class TargetCodeGenInfo {
    ABIInfo *Info;
  public:
    // WARNING: Acquires the ownership of ABIInfo.
    TargetCodeGenInfo(ABIInfo *info = 0):Info(info) { }
    virtual ~TargetCodeGenInfo();

    /// getABIInfo() - Returns ABI info helper for the target.
    const ABIInfo& getABIInfo() const { return *Info; }

    /// SetTargetAttributes - Provides a convenient hook to handle extra
    /// target-specific attributes for the given global.
    virtual void SetTargetAttributes(const Decl *D, llvm::GlobalValue *GV,
                                     CodeGen::CodeGenModule &M) const { }
  };
}

#endif // CLANG_CODEGEN_TARGETINFO_H
