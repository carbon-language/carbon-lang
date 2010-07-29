#include <string>
#include <utility>

#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/DenseMap.h>

// Required to parse .ll files.
#include <llvm/Support/SourceMgr.h>
#include <llvm/Assembly/Parser.h>

// Required to parse .bc files.
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Bitcode/ReaderWriter.h>

#include <llvm/Support/raw_ostream.h>
#include <llvm/LLVMContext.h>
#include <llvm/Module.h>
#include <llvm/Type.h>
#include <llvm/Instructions.h>

#include "DifferenceEngine.h"

using namespace llvm;

/// Reads a module from a file.  If the filename ends in .ll, it is
/// interpreted as an assembly file;  otherwise, it is interpreted as
/// bitcode.  On error, messages are written to stderr and null is
/// returned.
static Module *ReadModule(LLVMContext &Context, StringRef Name) {
  // LLVM assembly path.
  if (Name.endswith(".ll")) {
    SMDiagnostic Diag;
    Module *M = ParseAssemblyFile(Name, Diag, Context);
    if (M) return M;

    Diag.Print("llvmdiff", errs());
    return 0;
  }

  // Bitcode path.
  MemoryBuffer *Buffer = MemoryBuffer::getFile(Name);

  // ParseBitcodeFile takes ownership of the buffer if it succeeds.
  std::string Error;
  Module *M = ParseBitcodeFile(Buffer, Context, &Error);
  if (M) return M;

  errs() << "error parsing " << Name << ": " << Error;
  delete Buffer;
  return 0;
}

static int usage() {
  errs() << "expected usage:\n";
  errs() << "  llvm-diff oldmodule.ll newmodule.ll [function list]\n";
  errs() << "Assembly or bitcode modules may be used interchangeably.\n";
  errs() << "If no functions are provided, all functions will be compared.\n";
  return 1;
}

namespace {
struct DiffContext {
  DiffContext(Value *L, Value *R)
    : L(L), R(R), Differences(false), IsFunction(isa<Function>(L)) {}
  Value *L;
  Value *R;
  bool Differences;
  bool IsFunction;
  DenseMap<Value*,unsigned> LNumbering;
  DenseMap<Value*,unsigned> RNumbering;
};

void ComputeNumbering(Function *F, DenseMap<Value*,unsigned> &Numbering) {
  unsigned BBN = 0;
  unsigned IN = 0;

  // Arguments get the first numbers.
  for (Function::arg_iterator
         AI = F->arg_begin(), AE = F->arg_end(); AI != AE; ++AI)
    if (!AI->hasName())
      Numbering[&*AI] = IN++;

  // Walk the basic blocks in order.
  for (Function::iterator FI = F->begin(), FE = F->end(); FI != FE; ++FI) {
    // Basic blocks have their own 'namespace'.
    if (!FI->hasName())
      Numbering[&*FI] = BBN++;

    // Walk the instructions in order.
    for (BasicBlock::iterator BI = FI->begin(), BE = FI->end(); BI != BE; ++BI)
      // void instructions don't get numbers.
      if (!BI->hasName() && !BI->getType()->isVoidTy())
        Numbering[&*BI] = IN++;
  }

  assert(!Numbering.empty() && "asked for numbering but numbering was no-op");
}

class DiffConsumer : public DifferenceEngine::Consumer {
private:
  Module *LModule;
  Module *RModule;
  SmallVector<DiffContext, 5> contexts;
  bool Differences;
  unsigned Indent;

  void printValue(Value *V, bool isL) {
    if (V->hasName()) {
      errs() << (isa<GlobalValue>(V) ? '@' : '%') << V->getName();
      return;
    }
    if (V->getType()->isVoidTy()) {
      if (isa<StoreInst>(V)) {
        errs() << "store to ";
        printValue(cast<StoreInst>(V)->getPointerOperand(), isL);
      } else if (isa<CallInst>(V)) {
        errs() << "call to ";
        printValue(cast<CallInst>(V)->getCalledValue(), isL);
      } else if (isa<InvokeInst>(V)) {
        errs() << "invoke to ";
        printValue(cast<InvokeInst>(V)->getCalledValue(), isL);
      } else {
        errs() << *V;
      }
      return;
    }

    unsigned N = contexts.size();
    while (N > 0) {
      --N;
      DiffContext &ctxt = contexts[N];
      if (!ctxt.IsFunction) continue;
      if (isL) {
        if (ctxt.LNumbering.empty())
          ComputeNumbering(cast<Function>(ctxt.L), ctxt.LNumbering);
        errs() << '%' << ctxt.LNumbering[V];
        return;
      } else {
        if (ctxt.RNumbering.empty())
          ComputeNumbering(cast<Function>(ctxt.R), ctxt.RNumbering);
        errs() << '%' << ctxt.RNumbering[V];
        return;
      }
    }

    errs() << "<anonymous>";
  }

  void header() {
    if (contexts.empty()) return;
    for (SmallVectorImpl<DiffContext>::iterator
           I = contexts.begin(), E = contexts.end(); I != E; ++I) {
      if (I->Differences) continue;
      if (isa<Function>(I->L)) {
        // Extra newline between functions.
        if (Differences) errs() << "\n";

        Function *L = cast<Function>(I->L);
        Function *R = cast<Function>(I->R);
        if (L->getName() != R->getName())
          errs() << "in function " << L->getName() << " / " << R->getName() << ":\n";
        else
          errs() << "in function " << L->getName() << ":\n";
      } else if (isa<BasicBlock>(I->L)) {
        BasicBlock *L = cast<BasicBlock>(I->L);
        BasicBlock *R = cast<BasicBlock>(I->R);
        errs() << "  in block ";
        printValue(L, true);
        errs() << " / ";
        printValue(R, false);
        errs() << ":\n";
      } else if (isa<Instruction>(I->L)) {
        errs() << "    in instruction ";
        printValue(I->L, true);
        errs() << " / ";
        printValue(I->R, false);
        errs() << ":\n";
      }

      I->Differences = true;
    }
  }

  void indent() {
    unsigned N = Indent;
    while (N--) errs() << ' ';
  }

public:
  DiffConsumer(Module *L, Module *R)
    : LModule(L), RModule(R), Differences(false), Indent(0) {}

  bool hadDifferences() const { return Differences; }

  void enterContext(Value *L, Value *R) {
    contexts.push_back(DiffContext(L, R));
    Indent += 2;
  }
  void exitContext() {
    Differences |= contexts.back().Differences;
    contexts.pop_back();
    Indent -= 2;
  }

  void log(StringRef text) {
    header();
    indent();
    errs() << text << "\n";
  }

  void logf(const DifferenceEngine::LogBuilder &Log) {
    header();
    indent();

    // FIXME: we don't know whether these are l-values or r-values (ha!)
    // Print them in some saner way!
    errs() << Log.getFormat() << "\n";
    for (unsigned I = 0, E = Log.getNumArguments(); I != E; ++I)
      Log.getArgument(I)->dump();
  }

  void logd(const DifferenceEngine::DiffLogBuilder &Log) {
    header();

    for (unsigned I = 0, E = Log.getNumLines(); I != E; ++I) {
      indent();
      switch (Log.getLineKind(I)) {
      case DifferenceEngine::DC_match:
        errs() << "  ";
        Log.getLeft(I)->dump();
        //printValue(Log.getLeft(I), true);
        break;
      case DifferenceEngine::DC_left:
        errs() << "< ";
        Log.getLeft(I)->dump();
        //printValue(Log.getLeft(I), true);
        break;
      case DifferenceEngine::DC_right:
        errs() << "> ";
        Log.getRight(I)->dump();
        //printValue(Log.getRight(I), false);
        break;
      }
      //errs() << "\n";
    }
  }
  
};
}

int main(int argc, const char **argv) {
  if (argc < 3) return usage();

  // Don't make StringRef locals like this at home.
  StringRef LModuleFile = argv[1];
  StringRef RModuleFile = argv[2];

  LLVMContext Context;
  
  // Load both modules.  Die if that fails.
  Module *LModule = ReadModule(Context, LModuleFile);
  Module *RModule = ReadModule(Context, RModuleFile);
  if (!LModule || !RModule) return 1;

  DiffConsumer Consumer(LModule, RModule);
  DifferenceEngine Engine(Context, Consumer);

  // If any function names were given, just diff those.
  const char **FnNames = argv + 3;
  unsigned NumFnNames = argc - 3;
  if (NumFnNames) {
    for (unsigned I = 0; I != NumFnNames; ++I) {
      StringRef FnName = FnNames[I];

      // Drop leading sigils from the function name.
      if (FnName.startswith("@")) FnName = FnName.substr(1);

      Function *LFn = LModule->getFunction(FnName);
      Function *RFn = RModule->getFunction(FnName);
      if (LFn && RFn)
        Engine.diff(LFn, RFn);
      else {
        if (!LFn && !RFn)
          errs() << "No function named @" << FnName << " in either module\n";
        else if (!LFn)
          errs() << "No function named @" << FnName << " in left module\n";
        else
          errs() << "No function named @" << FnName << " in right module\n";
      }
    }
  } else {
    // Otherwise, diff all functions in the modules.
    Engine.diff(LModule, RModule);
  }

  delete LModule;
  delete RModule;

  return Consumer.hadDifferences();
}
