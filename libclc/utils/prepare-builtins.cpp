#include "llvm/ADT/OwningPtr.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/system_error.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace llvm;

static cl::opt<std::string>
InputFilename(cl::Positional, cl::desc("<input bitcode>"), cl::init("-"));

static cl::opt<std::string>
OutputFilename("o", cl::desc("Output filename"),
               cl::value_desc("filename"));

int main(int argc, char **argv) {
  LLVMContext &Context = getGlobalContext();
  llvm_shutdown_obj Y;  // Call llvm_shutdown() on exit.

  cl::ParseCommandLineOptions(argc, argv, "libclc builtin preparation tool\n");

  std::string ErrorMessage;
  std::auto_ptr<Module> M;

  {
    OwningPtr<MemoryBuffer> BufferPtr;
    if (error_code ec = MemoryBuffer::getFileOrSTDIN(InputFilename, BufferPtr))
      ErrorMessage = ec.message();
    else
      M.reset(ParseBitcodeFile(BufferPtr.get(), Context, &ErrorMessage));
  }

  if (M.get() == 0) {
    errs() << argv[0] << ": ";
    if (ErrorMessage.size())
      errs() << ErrorMessage << "\n";
    else
      errs() << "bitcode didn't read correctly.\n";
    return 1;
  }

  // Set linkage of every external definition to linkonce_odr.
  for (Module::iterator i = M->begin(), e = M->end(); i != e; ++i) {
    if (!i->isDeclaration() && i->getLinkage() == GlobalValue::ExternalLinkage)
      i->setLinkage(GlobalValue::LinkOnceODRLinkage);
  }

  for (Module::global_iterator i = M->global_begin(), e = M->global_end();
       i != e; ++i) {
    if (!i->isDeclaration() && i->getLinkage() == GlobalValue::ExternalLinkage)
      i->setLinkage(GlobalValue::LinkOnceODRLinkage);
  }

  if (OutputFilename.empty()) {
    errs() << "no output file\n";
    return 1;
  }

  std::string ErrorInfo;
  OwningPtr<tool_output_file> Out
  (new tool_output_file(OutputFilename.c_str(), ErrorInfo,
                        raw_fd_ostream::F_Binary));
  if (!ErrorInfo.empty()) {
    errs() << ErrorInfo << '\n';
    exit(1);
  }

  WriteBitcodeToFile(M.get(), Out->os());

  // Declare success.
  Out->keep();
  return 0;
}

