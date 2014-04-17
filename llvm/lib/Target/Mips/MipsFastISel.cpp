//===-- MipsastISel.cpp - Mips FastISel implementation
//---------------------===//

#include "llvm/CodeGen/FunctionLoweringInfo.h"
#include "llvm/CodeGen/FastISel.h"
#include "llvm/Target/TargetLibraryInfo.h"
#include "MipsISelLowering.h"

using namespace llvm;

namespace {

class MipsFastISel final : public FastISel {

public:
  explicit MipsFastISel(FunctionLoweringInfo &funcInfo,
                        const TargetLibraryInfo *libInfo)
      : FastISel(funcInfo, libInfo) {}
  bool TargetSelectInstruction(const Instruction *I) override { return false; }
};
}

namespace llvm {
FastISel *Mips::createFastISel(FunctionLoweringInfo &funcInfo,
                               const TargetLibraryInfo *libInfo) {
  return new MipsFastISel(funcInfo, libInfo);
}
}
