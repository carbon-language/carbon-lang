#include "llvm/ADT/Triple.h"
#include "llvm/ExecutionEngine/Orc/IndirectionUtils.h"

#include <array>

using namespace llvm;

namespace {

const char *JITCallbackFuncName = "call_jit_for_lazy_compile";
const char *JITCallbackIndexLabelPrefix = "jit_resolve_";

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

uint64_t call_jit_for_fn(JITResolveCallbackHandler *J, uint64_t FuncIdx) {
  return J->resolve(FuncIdx);
}
}

namespace llvm {

std::string getJITResolveCallbackIndexLabel(unsigned I) {
  std::ostringstream LabelStream;
  LabelStream << JITCallbackIndexLabelPrefix << I;
  return LabelStream.str();
}

void insertX86CallbackAsm(Module &M, JITResolveCallbackHandler &J) {
  uint64_t CallbackAddr =
      static_cast<uint64_t>(reinterpret_cast<uintptr_t>(call_jit_for_fn));

  std::ostringstream JITCallbackAsm;
  Triple TT(M.getTargetTriple());

  if (TT.getOS() == Triple::Darwin)
    JITCallbackAsm << ".section __TEXT,__text,regular,pure_instructions\n"
                   << ".align 4, 0x90\n";
  else
    JITCallbackAsm << ".text\n"
                   << ".align 16, 0x90\n";

  JITCallbackAsm << "jit_object_addr:\n"
                 << "  .quad " << &J << "\n" << JITCallbackFuncName << ":\n";

  uint64_t ReturnAddrOffset = saveX86Regs(JITCallbackAsm);

  // Compute index, load object address, and call JIT.
  JITCallbackAsm << "  movq    " << ReturnAddrOffset << "(%rsp), %rax\n"
                 << "  leaq    (jit_indices_start+5)(%rip), %rbx\n"
                 << "  subq    %rbx, %rax\n"
                 << "  xorq    %rdx, %rdx\n"
                 << "  movq    $5, %rbx\n"
                 << "  divq    %rbx\n"
                 << "  movq    %rax, %rsi\n"
                 << "  leaq    jit_object_addr(%rip), %rdi\n"
                 << "  movq    (%rdi), %rdi\n"
                 << "  movabsq $" << CallbackAddr << ", %rax\n"
                 << "  callq   *%rax\n"
                 << "  movq    %rax, " << ReturnAddrOffset << "(%rsp)\n";

  restoreX86Regs(JITCallbackAsm);

  JITCallbackAsm << "  retq\n"
                 << "jit_indices_start:\n";

  for (JITResolveCallbackHandler::StubIndex I = 0; I < J.getNumFuncs(); ++I)
    JITCallbackAsm << getJITResolveCallbackIndexLabel(I) << ":\n"
                   << "  callq " << JITCallbackFuncName << "\n";

  M.appendModuleInlineAsm(JITCallbackAsm.str());
}
}
