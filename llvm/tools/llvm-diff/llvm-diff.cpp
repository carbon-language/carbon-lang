//===-- llvm-diff.cpp - Module comparator command-line driver ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the command-line driver for the difference engine.
//
//===----------------------------------------------------------------------===//

#include "DifferenceEngine.h"

#include "llvm/Instructions.h"
#include "llvm/LLVMContext.h"
#include "llvm/Module.h"
#include "llvm/Type.h"
#include "llvm/Assembly/Parser.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/SourceMgr.h"

#include <string>
#include <utility>


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
  unsigned IN = 0;

  // Arguments get the first numbers.
  for (Function::arg_iterator
         AI = F->arg_begin(), AE = F->arg_end(); AI != AE; ++AI)
    if (!AI->hasName())
      Numbering[&*AI] = IN++;

  // Walk the basic blocks in order.
  for (Function::iterator FI = F->begin(), FE = F->end(); FI != FE; ++FI) {
    if (!FI->hasName())
      Numbering[&*FI] = IN++;

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
  raw_ostream &out;
  Module *LModule;
  Module *RModule;
  SmallVector<DiffContext, 5> contexts;
  bool Differences;
  unsigned Indent;

  void printValue(Value *V, bool isL) {
    if (V->hasName()) {
      out << (isa<GlobalValue>(V) ? '@' : '%') << V->getName();
      return;
    }
    if (V->getType()->isVoidTy()) {
      if (isa<StoreInst>(V)) {
        out << "store to ";
        printValue(cast<StoreInst>(V)->getPointerOperand(), isL);
      } else if (isa<CallInst>(V)) {
        out << "call to ";
        printValue(cast<CallInst>(V)->getCalledValue(), isL);
      } else if (isa<InvokeInst>(V)) {
        out << "invoke to ";
        printValue(cast<InvokeInst>(V)->getCalledValue(), isL);
      } else {
        out << *V;
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
        out << '%' << ctxt.LNumbering[V];
        return;
      } else {
        if (ctxt.RNumbering.empty())
          ComputeNumbering(cast<Function>(ctxt.R), ctxt.RNumbering);
        out << '%' << ctxt.RNumbering[V];
        return;
      }
    }

    out << "<anonymous>";
  }

  void header() {
    if (contexts.empty()) return;
    for (SmallVectorImpl<DiffContext>::iterator
           I = contexts.begin(), E = contexts.end(); I != E; ++I) {
      if (I->Differences) continue;
      if (isa<Function>(I->L)) {
        // Extra newline between functions.
        if (Differences) out << "\n";

        Function *L = cast<Function>(I->L);
        Function *R = cast<Function>(I->R);
        if (L->getName() != R->getName())
          out << "in function " << L->getName()
              << " / " << R->getName() << ":\n";
        else
          out << "in function " << L->getName() << ":\n";
      } else if (isa<BasicBlock>(I->L)) {
        BasicBlock *L = cast<BasicBlock>(I->L);
        BasicBlock *R = cast<BasicBlock>(I->R);
        if (L->hasName() && R->hasName() && L->getName() == R->getName())
          out << "  in block %" << L->getName() << ":\n";
        else {
          out << "  in block ";
          printValue(L, true);
          out << " / ";
          printValue(R, false);
          out << ":\n";
        }
      } else if (isa<Instruction>(I->L)) {
        out << "    in instruction ";
        printValue(I->L, true);
        out << " / ";
        printValue(I->R, false);
        out << ":\n";
      }

      I->Differences = true;
    }
  }

  void indent() {
    unsigned N = Indent;
    while (N--) out << ' ';
  }

public:
  DiffConsumer(Module *L, Module *R)
    : out(errs()), LModule(L), RModule(R), Differences(false), Indent(0) {}

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
    out << text << '\n';
  }

  void logf(const DifferenceEngine::LogBuilder &Log) {
    header();
    indent();

    unsigned arg = 0;

    StringRef format = Log.getFormat();
    while (true) {
      size_t percent = format.find('%');
      if (percent == StringRef::npos) {
        out << format;
        break;
      }
      assert(format[percent] == '%');

      if (percent > 0) out << format.substr(0, percent);

      switch (format[percent+1]) {
      case '%': out << '%'; break;
      case 'l': printValue(Log.getArgument(arg++), true); break;
      case 'r': printValue(Log.getArgument(arg++), false); break;
      default: llvm_unreachable("unknown format character");
      }

      format = format.substr(percent+2);
    }

    out << '\n';
  }

  void logd(const DifferenceEngine::DiffLogBuilder &Log) {
    header();

    for (unsigned I = 0, E = Log.getNumLines(); I != E; ++I) {
      indent();
      switch (Log.getLineKind(I)) {
      case DifferenceEngine::DC_match:
        out << "  ";
        Log.getLeft(I)->dump();
        //printValue(Log.getLeft(I), true);
        break;
      case DifferenceEngine::DC_left:
        out << "< ";
        Log.getLeft(I)->dump();
        //printValue(Log.getLeft(I), true);
        break;
      case DifferenceEngine::DC_right:
        out << "> ";
        Log.getRight(I)->dump();
        //printValue(Log.getRight(I), false);
        break;
      }
      //out << "\n";
    }
  }
  
};
}

static void diffGlobal(DifferenceEngine &Engine, Module *L, Module *R,
                       StringRef Name) {
  // Drop leading sigils from the global name.
  if (Name.startswith("@")) Name = Name.substr(1);

  Function *LFn = L->getFunction(Name);
  Function *RFn = R->getFunction(Name);
  if (LFn && RFn)
    Engine.diff(LFn, RFn);
  else if (!LFn && !RFn)
    errs() << "No function named @" << Name << " in either module\n";
  else if (!LFn)
    errs() << "No function named @" << Name << " in left module\n";
  else
    errs() << "No function named @" << Name << " in right module\n";
}

cl::opt<std::string> LeftFilename(cl::Positional,
                                  cl::desc("<first file>"),
                                  cl::Required);
cl::opt<std::string> RightFilename(cl::Positional,
                                   cl::desc("<second file>"),
                                   cl::Required);
cl::list<std::string> GlobalsToCompare(cl::Positional,
                                       cl::desc("<globals to compare>"));

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv);

  LLVMContext Context;
  
  // Load both modules.  Die if that fails.
  Module *LModule = ReadModule(Context, LeftFilename);
  Module *RModule = ReadModule(Context, RightFilename);
  if (!LModule || !RModule) return 1;

  DiffConsumer Consumer(LModule, RModule);
  DifferenceEngine Engine(Context, Consumer);

  // If any global names were given, just diff those.
  if (!GlobalsToCompare.empty()) {
    for (unsigned I = 0, E = GlobalsToCompare.size(); I != E; ++I)
      diffGlobal(Engine, LModule, RModule, GlobalsToCompare[I]);

  // Otherwise, diff everything in the module.
  } else {
    Engine.diff(LModule, RModule);
  }

  delete LModule;
  delete RModule;

  return Consumer.hadDifferences();
}
