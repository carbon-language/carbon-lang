#include "gtest/gtest.h"

#include <memory>

#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"

#include "MCTargetDesc/HexagonMCInst.h"
#include "MCTargetDesc/HexagonMCTargetDesc.h"

namespace {
class TestEmitter {
public:
  TestEmitter() : Triple("hexagon-unknown-elf") {
    LLVMInitializeHexagonTargetInfo();
    LLVMInitializeHexagonTarget();
    LLVMInitializeHexagonTargetMC();
    std::string error;
    Target = llvm::TargetRegistry::lookupTarget("hexagon", error);
    assert(Target != nullptr && "Expected to find target");
    assert(error.empty() && "Error should be empty if we have a target");
    RegisterInfo = Target->createMCRegInfo(Triple);
    assert(RegisterInfo != nullptr && "Expecting to find register info");
    AsmInfo = Target->createMCAsmInfo(*RegisterInfo, Triple);
    assert(AsmInfo != nullptr && "Expecting to find asm info");
    Context = new llvm::MCContext(AsmInfo, RegisterInfo, nullptr);
    assert(Context != nullptr && "Expecting to create a context");
    Subtarget = Target->createMCSubtargetInfo(Triple, "hexagonv4", "");
    assert(Subtarget != nullptr && "Expecting to find a subtarget");
    InstrInfo = Target->createMCInstrInfo();
    assert(InstrInfo != nullptr && "Expecting to find instr info");
    Emitter = Target->createMCCodeEmitter(*InstrInfo, *RegisterInfo, *Subtarget,
                                          *Context);
    assert(Emitter != nullptr);
  }
  std::string Triple;
  llvm::Target const *Target;
  llvm::MCRegisterInfo *RegisterInfo;
  llvm::MCAsmInfo *AsmInfo;
  llvm::MCContext *Context;
  llvm::MCSubtargetInfo *Subtarget;
  llvm::MCInstrInfo *InstrInfo;
  llvm::MCCodeEmitter *Emitter;
};
TestEmitter Emitter;
}

TEST(HexagonMCCodeEmitter, emitter_creation) {
  ASSERT_NE(nullptr, Emitter.Emitter);
}
