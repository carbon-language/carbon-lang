#include "llvm/ADT/Triple.h"
#include "llvm/ExecutionEngine/Orc/OrcTargetSupport.h"
#include "llvm/Support/Process.h"
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

  // List of x86-64 GPRs to save. Note - RBP saved separately below.
  std::array<const char *, 14> GPRs = {{
      "rax", "rbx", "rcx", "rdx",
      "rsi", "rdi", "r8", "r9",
      "r10", "r11", "r12", "r13",
      "r14", "r15"
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
            << "  pushq     %rbp\n"
            << "  movq      %rsp, %rbp\n";

  // Store the GPRs.
  for (const auto &GPR : GPRs)
    AsmStream << "  pushq     %" << GPR << "\n";

  // Store floating-point state with FXSAVE.
  // Note: We need to keep the stack 16-byte aligned, so if we've emitted an odd
  //       number of 64-bit pushes so far (GPRs.size() plus 1 for RBP) then add
  //       an extra 64 bits of padding to the FXSave area.
  unsigned Padding = (GPRs.size() + 1) % 2 ? 8 : 0;
  unsigned FXSaveSize = 512 + Padding;
  AsmStream << "  subq      $" << FXSaveSize << ", %rsp\n"
            << "  fxsave64  (%rsp)\n"

  // Load callback manager address, compute trampoline address, call JIT.
            << "  lea       jit_callback_manager_addr(%rip), %rdi\n"
            << "  movq      (%rdi), %rdi\n"
            << "  movq      0x8(%rbp), %rsi\n"
            << "  subq      $" << X86_64_TrampolineLength << ", %rsi\n"
            << "  movabsq   $" << CallbackAddr << ", %rax\n"
            << "  callq     *%rax\n"

  // Replace the return to the trampoline with the return address of the
  // compiled function body.
            << "  movq      %rax, 0x8(%rbp)\n"

  // Restore the floating point state.
            << "  fxrstor64 (%rsp)\n"
            << "  addq      $" << FXSaveSize << ", %rsp\n";

  for (const auto &GPR : make_range(GPRs.rbegin(), GPRs.rend()))
    AsmStream << "  popq      %" << GPR << "\n";

  // Restore original RBP and return to compiled function body.
  AsmStream << "  popq      %rbp\n"
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

OrcX86_64::IndirectStubsInfo::~IndirectStubsInfo() {
  sys::Memory::releaseMappedMemory(StubsBlock);
  sys::Memory::releaseMappedMemory(PtrsBlock);
}

std::error_code OrcX86_64::emitIndirectStubsBlock(IndirectStubsInfo &StubsInfo,
                                                  unsigned MinStubs,
                                                  void *InitialPtrVal) {
  // Stub format is:
  //
  // .section __orc_stubs
  // stub1:
  //                 jmpq    *ptr1(%rip)
  //                 .byte   0xC4         ; <- Invalid opcode padding.
  //                 .byte   0xF1
  // stub2:
  //                 jmpq    *ptr2(%rip)
  //
  // ...
  //
  // .section __orc_ptrs
  // ptr1:
  //                 .quad 0x0
  // ptr2:
  //                 .quad 0x0
  //
  // ...

  const unsigned StubSize = IndirectStubsInfo::StubSize;

  // Emit at least MinStubs, rounded up to fill the pages allocated.
  unsigned PageSize = sys::Process::getPageSize();
  unsigned NumPages = ((MinStubs * StubSize) + (PageSize - 1)) / PageSize;
  unsigned NumStubs = (NumPages * PageSize) / StubSize;

  // Allocate memory for stubs and pointers in one call.
  std::error_code EC;
  auto InitialBlock = sys::Memory::allocateMappedMemory(2 * NumPages * PageSize,
                                                        nullptr,
                                                        sys::Memory::MF_READ |
                                                        sys::Memory::MF_WRITE,
                                                        EC);

  if (EC)
    return EC;

  // Create separate MemoryBlocks representing the stubs and pointers.
  sys::MemoryBlock StubsBlock(InitialBlock.base(), NumPages * PageSize);
  sys::MemoryBlock PtrsBlock(static_cast<char*>(InitialBlock.base()) +
                             NumPages * PageSize,
                             NumPages * PageSize);

  // Populate the stubs page stubs and mark it executable.
  uint64_t *Stub = reinterpret_cast<uint64_t*>(StubsBlock.base());
  uint64_t PtrOffsetField =
    static_cast<uint64_t>(NumPages * PageSize - 6) << 16;
  for (unsigned I = 0; I < NumStubs; ++I)
    Stub[I] = 0xF1C40000000025ff | PtrOffsetField;

  if (auto EC = sys::Memory::protectMappedMemory(StubsBlock,
                                                 sys::Memory::MF_READ |
                                                 sys::Memory::MF_EXEC))
    return EC;

  // Initialize all pointers to point at FailureAddress.
  void **Ptr = reinterpret_cast<void**>(PtrsBlock.base());
  for (unsigned I = 0; I < NumStubs; ++I)
    Ptr[I] = InitialPtrVal;

  StubsInfo.NumStubs = NumStubs;
  StubsInfo.StubsBlock = std::move(StubsBlock);
  StubsInfo.PtrsBlock = std::move(PtrsBlock);

  return std::error_code();
}

} // End namespace orc.
} // End namespace llvm.
