#include "llvm/ADT/Triple.h"
#include "llvm/ExecutionEngine/Orc/OrcTargetSupport.h"
#include <array>

using namespace llvm::orc;

namespace {

std::array<const char *, 12> X86GPRsToSave = {{
    "rbp", "rbx", "r12", "r13", "r14", "r15", // Callee saved.
    "rdi", "rsi", "rdx", "rcx", "r8", "r9",   // Int args.
}};

std::array<const char *, 8> X86XMMsToSave = {{
    "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7" // FP args
}};

template <typename OStream> unsigned saveX86Regs(OStream &OS) {
  for (const auto &GPR : X86GPRsToSave)
    OS << "  pushq   %" << GPR << "\n";

  OS << "  subq    $" << (16 * X86XMMsToSave.size()) << ", %rsp\n";

  for (unsigned i = 0; i < X86XMMsToSave.size(); ++i)
    OS << "  movdqu  %" << X86XMMsToSave[i] << ", "
       << (16 * (X86XMMsToSave.size() - i - 1)) << "(%rsp)\n";

  return (8 * X86GPRsToSave.size()) + (16 * X86XMMsToSave.size());
}

template <typename OStream> void restoreX86Regs(OStream &OS) {
  for (unsigned i = 0; i < X86XMMsToSave.size(); ++i)
    OS << "  movdqu  " << (16 * i) << "(%rsp), %"
       << X86XMMsToSave[(X86XMMsToSave.size() - i - 1)] << "\n";
  OS << "  addq    $" << (16 * X86XMMsToSave.size()) << ", %rsp\n";

  for (unsigned i = 0; i < X86GPRsToSave.size(); ++i)
    OS << "  popq    %" << X86GPRsToSave[X86GPRsToSave.size() - i - 1] << "\n";
}

template <typename TargetT>
uint64_t executeCompileCallback(JITCompileCallbackManagerBase<TargetT> *JCBM,
                                TargetAddress CallbackID) {
  return JCBM->executeCompileCallback(CallbackID);
}

}

namespace llvm {
namespace orc {

const char* OrcX86_64::ResolverBlockName = "orc_resolver_block";

void OrcX86_64::insertResolverBlock(
    Module &M, JITCompileCallbackManagerBase<OrcX86_64> &JCBM) {
  auto CallbackPtr = executeCompileCallback<OrcX86_64>;
  uint64_t CallbackAddr =
      static_cast<uint64_t>(reinterpret_cast<uintptr_t>(CallbackPtr));

  std::ostringstream AsmStream;
  Triple TT(M.getTargetTriple());

  if (TT.getOS() == Triple::Darwin)
    AsmStream << ".section __TEXT,__text,regular,pure_instructions\n"
              << ".align 4, 0x90\n";
  else
    AsmStream << ".text\n"
              << ".align 16, 0x90\n";

  AsmStream << "jit_callback_manager_addr:\n"
            << "  .quad " << &JCBM << "\n"
            << ResolverBlockName << ":\n";

  uint64_t ReturnAddrOffset = saveX86Regs(AsmStream);

  // Compute index, load object address, and call JIT.
  AsmStream << "  leaq    jit_callback_manager_addr(%rip), %rdi\n"
            << "  movq    (%rdi), %rdi\n"
            << "  movq    " << ReturnAddrOffset << "(%rsp), %rsi\n"
            << "  movabsq $" << CallbackAddr << ", %rax\n"
            << "  callq   *%rax\n"
            << "  movq    %rax, " << ReturnAddrOffset << "(%rsp)\n";

  restoreX86Regs(AsmStream);

  AsmStream << "  retq\n";

  M.appendModuleInlineAsm(AsmStream.str());
}

OrcX86_64::LabelNameFtor
OrcX86_64::insertCompileCallbackTrampolines(Module &M,
                                            TargetAddress ResolverBlockAddr,
                                            unsigned NumCalls,
                                            unsigned StartIndex) {
  const char *ResolverBlockPtrName = "Lorc_resolve_block_addr";

  std::ostringstream AsmStream;
  Triple TT(M.getTargetTriple());

  if (TT.getOS() == Triple::Darwin)
    AsmStream << ".section __TEXT,__text,regular,pure_instructions\n"
              << ".align 4, 0x90\n";
  else
    AsmStream << ".text\n"
              << ".align 16, 0x90\n";

  AsmStream << ResolverBlockPtrName << ":\n"
            << "  .quad " << ResolverBlockAddr << "\n";

  auto GetLabelName =
    [=](unsigned I) {
      std::ostringstream LabelStream;
      LabelStream << "orc_jcc_" << (StartIndex + I);
      return LabelStream.str();
  };

  for (unsigned I = 0; I < NumCalls; ++I)
    AsmStream << GetLabelName(I) << ":\n"
              << "  callq *" << ResolverBlockPtrName << "(%rip)\n";

  M.appendModuleInlineAsm(AsmStream.str());

  return GetLabelName;
}

} // End namespace orc.
} // End namespace llvm.
