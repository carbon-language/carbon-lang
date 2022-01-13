//===--- SystemZ.h - Declare SystemZ target feature support -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares SystemZ TargetInfo objects.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_BASIC_TARGETS_SYSTEMZ_H
#define LLVM_CLANG_LIB_BASIC_TARGETS_SYSTEMZ_H

#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/Compiler.h"

namespace clang {
namespace targets {

class LLVM_LIBRARY_VISIBILITY SystemZTargetInfo : public TargetInfo {

  static const Builtin::Info BuiltinInfo[];
  static const char *const GCCRegNames[];
  std::string CPU;
  int ISARevision;
  bool HasTransactionalExecution;
  bool HasVector;
  bool SoftFloat;

public:
  SystemZTargetInfo(const llvm::Triple &Triple, const TargetOptions &)
      : TargetInfo(Triple), CPU("z10"), ISARevision(8),
        HasTransactionalExecution(false), HasVector(false), SoftFloat(false) {
    IntMaxType = SignedLong;
    Int64Type = SignedLong;
    TLSSupported = true;
    IntWidth = IntAlign = 32;
    LongWidth = LongLongWidth = LongAlign = LongLongAlign = 64;
    PointerWidth = PointerAlign = 64;
    LongDoubleWidth = 128;
    LongDoubleAlign = 64;
    LongDoubleFormat = &llvm::APFloat::IEEEquad();
    DefaultAlignForAttributeAligned = 64;
    MinGlobalAlign = 16;
    resetDataLayout("E-m:e-i1:8:16-i8:8:16-i64:64-f128:64-a:8:16-n32:64");
    MaxAtomicPromoteWidth = MaxAtomicInlineWidth = 64;
    HasStrictFP = true;
  }

  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override;

  ArrayRef<Builtin::Info> getTargetBuiltins() const override;

  ArrayRef<const char *> getGCCRegNames() const override;

  ArrayRef<TargetInfo::GCCRegAlias> getGCCRegAliases() const override {
    // No aliases.
    return None;
  }

  ArrayRef<TargetInfo::AddlRegName> getGCCAddlRegNames() const override;

  bool isSPRegName(StringRef RegName) const override {
    return RegName.equals("r15");
  }

  bool validateAsmConstraint(const char *&Name,
                             TargetInfo::ConstraintInfo &info) const override;

  const char *getClobbers() const override {
    // FIXME: Is this really right?
    return "";
  }

  BuiltinVaListKind getBuiltinVaListKind() const override {
    return TargetInfo::SystemZBuiltinVaList;
  }

  int getISARevision(StringRef Name) const;

  bool isValidCPUName(StringRef Name) const override {
    return getISARevision(Name) != -1;
  }

  void fillValidCPUList(SmallVectorImpl<StringRef> &Values) const override;

  bool setCPU(const std::string &Name) override {
    CPU = Name;
    ISARevision = getISARevision(CPU);
    return ISARevision != -1;
  }

  bool
  initFeatureMap(llvm::StringMap<bool> &Features, DiagnosticsEngine &Diags,
                 StringRef CPU,
                 const std::vector<std::string> &FeaturesVec) const override {
    int ISARevision = getISARevision(CPU);
    if (ISARevision >= 10)
      Features["transactional-execution"] = true;
    if (ISARevision >= 11)
      Features["vector"] = true;
    if (ISARevision >= 12)
      Features["vector-enhancements-1"] = true;
    if (ISARevision >= 13)
      Features["vector-enhancements-2"] = true;
    if (ISARevision >= 14)
      Features["nnp-assist"] = true;
    return TargetInfo::initFeatureMap(Features, Diags, CPU, FeaturesVec);
  }

  bool handleTargetFeatures(std::vector<std::string> &Features,
                            DiagnosticsEngine &Diags) override {
    HasTransactionalExecution = false;
    HasVector = false;
    SoftFloat = false;
    for (const auto &Feature : Features) {
      if (Feature == "+transactional-execution")
        HasTransactionalExecution = true;
      else if (Feature == "+vector")
        HasVector = true;
      else if (Feature == "+soft-float")
        SoftFloat = true;
    }
    HasVector &= !SoftFloat;

    // If we use the vector ABI, vector types are 64-bit aligned.
    if (HasVector) {
      MaxVectorAlign = 64;
      resetDataLayout("E-m:e-i1:8:16-i8:8:16-i64:64-f128:64"
                      "-v128:64-a:8:16-n32:64");
    }
    return true;
  }

  bool hasFeature(StringRef Feature) const override;

  CallingConvCheckResult checkCallingConvention(CallingConv CC) const override {
    switch (CC) {
    case CC_C:
    case CC_Swift:
    case CC_OpenCLKernel:
      return CCCR_OK;
    case CC_SwiftAsync:
      return CCCR_Error;
    default:
      return CCCR_Warning;
    }
  }

  StringRef getABI() const override {
    if (HasVector)
      return "vector";
    return "";
  }

  const char *getLongDoubleMangling() const override { return "g"; }

  bool hasExtIntType() const override { return true; }

  int getEHDataRegisterNumber(unsigned RegNo) const override {
    return RegNo < 4 ? 6 + RegNo : -1;
  }
};
} // namespace targets
} // namespace clang
#endif // LLVM_CLANG_LIB_BASIC_TARGETS_SYSTEMZ_H
