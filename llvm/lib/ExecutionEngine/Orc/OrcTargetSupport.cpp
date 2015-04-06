#include "llvm/ADT/Triple.h"
#include "llvm/ExecutionEngine/Orc/OrcTargetSupport.h"
#include <array>

using namespace llvm::orc;

namespace {

uint64_t executeCompileCallback(JITCompileCallbackManagerBase *JCBM,
                                TargetAddress CallbackID) {
  return JCBM->executeCompileCallback(CallbackID);
}

}

namespace llvm {
namespace orc {

const char* OrcX86_64::ResolverBlockName = "orc_resolver_block";

void OrcX86_64::insertResolverBlock(
    Module &M, JITCompileCallbackManagerBase &JCBM) {

  // Trampoline code-sequence length, used to get trampoline address from return
  // address.
  const unsigned X86_64_TrampolineLength = 6;

  // List of x86-64 GPRs to save.
  std::array<const char *, 11> GPRs = {{
      "rbx", "r12", "r13", "r14", "r15", // Callee saved (rbp preserved below).
      "rdi", "rsi", "rdx", "rcx", "r8", "r9",   // Int args.
    }};

  // Address of the executeCompileCallback function.
  uint64_t CallbackAddr =
      static_cast<uint64_t>(
        reinterpret_cast<uintptr_t>(executeCompileCallback));

  std::ostringstream AsmStream;
  Triple TT(M.getTargetTriple());

  // Switch to text section.
  if (TT.getOS() == Triple::Darwin)
    AsmStream << ".section __TEXT,__text,regular,pure_instructions\n"
              << ".align 4, 0x90\n";
  else
    AsmStream << ".text\n"
              << ".align 16, 0x90\n";

  // Bake in a pointer to the callback manager immediately before the
  // start of the resolver function.
  AsmStream << "jit_callback_manager_addr:\n"
            << "  .quad " << &JCBM << "\n";

  // Start the resolver function.
  AsmStream << ResolverBlockName << ":\n"
            << "  pushq   %rbp\n"
            << "  movq    %rsp, %rbp\n";

  // Store the GPRs.
  for (const auto &GPR : GPRs)
    AsmStream << "  pushq   %" << GPR << "\n";

  // Store floating-point state with FXSAVE.
  AsmStream << "  subq    $512, %rsp\n"
            << "  fxsave  (%rsp)\n"

  // Load callback manager address, compute trampoline address, call JIT.
            << "  lea     jit_callback_manager_addr(%rip), %rdi\n"
            << "  movq    (%rdi), %rdi\n"
            << "  movq    0x8(%rbp), %rsi\n"
            << "  subq    $" << X86_64_TrampolineLength << ", %rsi\n"
            << "  movabsq $" << CallbackAddr << ", %rax\n"
            << "  callq   *%rax\n"

  // Replace the return to the trampoline with the return address of the
  // compiled function body.
            << "  movq    %rax, 0x8(%rbp)\n"

  // Restore the floating point state.
            << "  fxrstor (%rsp)\n"
            << "  addq    $512, %rsp\n";

  for (const auto &GPR : make_range(GPRs.rbegin(), GPRs.rend()))
    AsmStream << "  popq    %" << GPR << "\n";

  // Restore original RBP and return to compiled function body.
  AsmStream << "  popq    %rbp\n"
            << "  retq\n";

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
