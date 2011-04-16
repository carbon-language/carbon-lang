//===- GCOVProfiling.cpp - Insert edge counters for gcov profiling --------===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass implements GCOV-style profiling. When this pass is run it emits
// "gcno" files next to the existing source, and instruments the code that runs
// to records the edges between blocks that run and emit a complementary "gcda"
// file on exit.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "insert-gcov-profiling"

#include "ProfilingUtils.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Analysis/DebugInfo.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/Instructions.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLoc.h"
#include "llvm/Support/InstIterator.h"
#include "llvm/Support/IRBuilder.h"
#include "llvm/Support/PathV2.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/UniqueVector.h"
#include <string>
#include <utility>
using namespace llvm;

namespace {
  class GCOVProfiler : public ModulePass {
    bool runOnModule(Module &M);
  public:
    static char ID;
    GCOVProfiler() : ModulePass(ID) {
      initializeGCOVProfilerPass(*PassRegistry::getPassRegistry());
    }
    virtual const char *getPassName() const {
      return "GCOV Profiler";
    }

  private:
    // Create the GCNO files for the Module based on DebugInfo.
    void EmitGCNO(DebugInfoFinder &DIF);

    // Get pointers to the functions in the runtime library.
    Constant *getStartFileFunc();
    Constant *getEmitFunctionFunc();
    Constant *getEmitArcsFunc();
    Constant *getEndFileFunc();

    // Add the function to write out all our counters to the global destructor
    // list.
    void InsertCounterWriteout(DebugInfoFinder &,
                               SmallVector<std::pair<GlobalVariable *,
                                                     uint32_t>, 8> &);

    Module *Mod;
    LLVMContext *Ctx;
  };
}

char GCOVProfiler::ID = 0;
INITIALIZE_PASS(GCOVProfiler, "insert-gcov-profiling",
                "Insert instrumentation for GCOV profiling", false, false)

ModulePass *llvm::createGCOVProfilerPass() { return new GCOVProfiler(); }

static DISubprogram FindSubprogram(DIScope scope) {
  while (!scope.isSubprogram()) {
    assert(scope.isLexicalBlock() &&
           "Debug location not lexical block or subprogram");
    scope = DILexicalBlock(scope).getContext();
  }
  return DISubprogram(scope);
}

namespace {
  class GCOVRecord {
   protected:
    static const char *lines_tag;
    static const char *function_tag;
    static const char *block_tag;
    static const char *edge_tag;

    GCOVRecord() {}

    void WriteBytes(const char *b, int size) {
      os->write(b, size);
    }

    void Write(uint32_t i) {
      WriteBytes(reinterpret_cast<char*>(&i), 4);
    }

    // Returns the length measured in 4-byte blocks that will be used to
    // represent this string in a GCOV file
    unsigned LengthOfGCOVString(StringRef s) {
      // A GCOV string is a length, followed by a NUL, then between 0 and 3 NULs
      // padding out to the next 4-byte word. The length is measured in 4-byte words
      // including padding, not bytes of actual string.
      return (s.size() + 5) / 4;
    }

    void WriteGCOVString(StringRef s) {
      uint32_t len = LengthOfGCOVString(s);
      Write(len);
      WriteBytes(s.data(), s.size());

      // Write 1 to 4 bytes of NUL padding.
      assert((unsigned)(5 - ((s.size() + 1) % 4)) > 0);
      assert((unsigned)(5 - ((s.size() + 1) % 4)) <= 4);
      WriteBytes("\0\0\0\0", 5 - ((s.size() + 1) % 4));
    }

    raw_ostream *os;
  };
  const char *GCOVRecord::lines_tag = "\0\0\x45\x01";
  const char *GCOVRecord::function_tag = "\0\0\0\1";
  const char *GCOVRecord::block_tag = "\0\0\x41\x01";
  const char *GCOVRecord::edge_tag = "\0\0\x43\x01";

  class GCOVFunction;
  class GCOVBlock;

  // Constructed only by requesting it from a GCOVBlock, this object stores a
  // list of line numbers and a single filename, representing lines that belong
  // to the block.
  class GCOVLines : public GCOVRecord {
   public:
    void AddLine(uint32_t line) {
      lines.push_back(line);
    }

    uint32_t Length() {
      return LengthOfGCOVString(filename) + 2 + lines.size();
    }

   private:
    friend class GCOVBlock;

    GCOVLines(std::string filename, raw_ostream *os)
        : filename(filename) {
      this->os = os;
    }

    std::string filename;
    SmallVector<uint32_t, 32> lines;
  };

  // Represent a basic block in GCOV. Each block has a unique number in the
  // function, number of lines belonging to each block, and a set of edges to
  // other blocks.
  class GCOVBlock : public GCOVRecord {
   public:
    GCOVLines &GetFile(std::string filename) {
      GCOVLines *&lines = lines_by_file[filename];
      if (!lines) {
        lines = new GCOVLines(filename, os);
      }
      return *lines;
    }

    void AddEdge(GCOVBlock &successor) {
      out_edges.push_back(&successor);
    }

    void WriteOut() {
      uint32_t len = 3;
      for (StringMap<GCOVLines *>::iterator I = lines_by_file.begin(),
               E = lines_by_file.end(); I != E; ++I) {
        len += I->second->Length();
      }

      WriteBytes(lines_tag, 4);
      Write(len);
      Write(number);
      for (StringMap<GCOVLines *>::iterator I = lines_by_file.begin(),
               E = lines_by_file.end(); I != E; ++I) {
        Write(0);
        WriteGCOVString(I->second->filename);
        for (int i = 0, e = I->second->lines.size(); i != e; ++i) {
          Write(I->second->lines[i]);
        }
      }
      Write(0);
      Write(0);
    }

    ~GCOVBlock() {
      DeleteContainerSeconds(lines_by_file);
    }

   private:
    friend class GCOVFunction;

    GCOVBlock(uint32_t number, raw_ostream *os)
        : number(number) {
      this->os = os;
    }

    uint32_t number;
    BasicBlock *block;
    StringMap<GCOVLines *> lines_by_file;
    SmallVector<GCOVBlock *, 4> out_edges;
  };

  // A function has a unique identifier, a checksum (we leave as zero) and a
  // set of blocks and a map of edges between blocks. This is the only GCOV
  // object users can construct, the blocks and lines will be rooted here.
  class GCOVFunction : public GCOVRecord {
   public:
    GCOVFunction(DISubprogram SP, raw_ostream *os) {
      this->os = os;

      Function *F = SP.getFunction();
      uint32_t i = 0;
      for (Function::iterator BB = F->begin(), E = F->end(); BB != E; ++BB) {
        blocks[BB] = new GCOVBlock(i++, os);
      }

      WriteBytes(function_tag, 4);
      uint32_t block_len = 1 + 1 + 1 + LengthOfGCOVString(SP.getName()) +
          1 + LengthOfGCOVString(SP.getFilename()) + 1;
      Write(block_len);
      uint32_t ident = reinterpret_cast<intptr_t>((MDNode*)SP);
      Write(ident);
      Write(0); // checksum
      WriteGCOVString(SP.getName());
      WriteGCOVString(SP.getFilename());
      Write(SP.getLineNumber());
    }

    ~GCOVFunction() {
      DeleteContainerSeconds(blocks);
    }

    GCOVBlock &GetBlock(BasicBlock *BB) {
      return *blocks[BB];
    }

    void WriteOut() {
      // Emit count of blocks.
      WriteBytes(block_tag, 4);
      Write(blocks.size());
      for (int i = 0, e = blocks.size(); i != e; ++i) {
        Write(0);  // No flags on our blocks.
      }

      // Emit edges between blocks.
      for (DenseMap<BasicBlock *, GCOVBlock *>::iterator I = blocks.begin(),
               E = blocks.end(); I != E; ++I) {
        GCOVBlock &block = *I->second;
        if (block.out_edges.empty()) continue;

        WriteBytes(edge_tag, 4);
        Write(block.out_edges.size() * 2 + 1);
        Write(block.number);
        for (int i = 0, e = block.out_edges.size(); i != e; ++i) {
          Write(block.out_edges[i]->number);
          Write(0);  // no flags
        }
      }

      // Emit lines for each block.
      for (DenseMap<BasicBlock *, GCOVBlock *>::iterator I = blocks.begin(),
               E = blocks.end(); I != E; ++I) {
        I->second->WriteOut();
      }
    }

   private:
    DenseMap<BasicBlock *, GCOVBlock *> blocks;
  };
}

void GCOVProfiler::EmitGCNO(DebugInfoFinder &DIF) {
  DenseMap<const MDNode *, raw_fd_ostream *> gcno_files;
  for (DebugInfoFinder::iterator I = DIF.compile_unit_begin(),
           E = DIF.compile_unit_end(); I != E; ++I) {
    // Each compile unit gets its own .gcno file. This means that whether we run
    // this pass over the original .o's as they're produced, or run it after
    // LTO, we'll generate the same .gcno files.

    DICompileUnit CU(*I);
    raw_fd_ostream *&Out = gcno_files[CU];
    std::string ErrorInfo;
    Out = new raw_fd_ostream(
        (sys::path::stem(CU.getFilename()) + ".gcno").str().c_str(),
        ErrorInfo, raw_fd_ostream::F_Binary);
    Out->write("oncg*404MVLL", 12);
  }

  for (DebugInfoFinder::iterator SPI = DIF.subprogram_begin(),
           SPE = DIF.subprogram_end(); SPI != SPE; ++SPI) {
    DISubprogram SP(*SPI);
    raw_fd_ostream *&os = gcno_files[SP.getCompileUnit()];

    GCOVFunction function(SP, os);
    Function *F = SP.getFunction();
    for (Function::iterator BB = F->begin(), E = F->end(); BB != E; ++BB) {
      GCOVBlock &block = function.GetBlock(BB);
      TerminatorInst *TI = BB->getTerminator();
      if (int successors = TI->getNumSuccessors()) {
        for (int i = 0; i != successors; ++i) {
          block.AddEdge(function.GetBlock(TI->getSuccessor(i)));
        }
      }

      uint32_t line = 0;
      for (BasicBlock::iterator I = BB->begin(), IE = BB->end(); I != IE; ++I) {
        const DebugLoc &loc = I->getDebugLoc();
        if (loc.isUnknown()) continue;
        if (line == loc.getLine()) continue;
        line = loc.getLine();
        if (SP != FindSubprogram(DIScope(loc.getScope(*Ctx)))) continue;

        GCOVLines &lines = block.GetFile(SP.getFilename());
        lines.AddLine(loc.getLine());
      }
    }
    function.WriteOut();
  }

  for (DenseMap<const MDNode *, raw_fd_ostream *>::iterator
           I = gcno_files.begin(), E = gcno_files.end(); I != E; ++I) {
    raw_fd_ostream *&Out = I->second;
    Out->write("\0\0\0\0\0\0\0\0", 4); // EOF
    Out->close();
    delete Out;
  }
}

bool GCOVProfiler::runOnModule(Module &M) {
  Mod = &M;
  Ctx = &M.getContext();

  DebugInfoFinder DIF;
  DIF.processModule(*Mod);

  EmitGCNO(DIF);

  SmallVector<std::pair<GlobalVariable *, uint32_t>, 8> counters_by_ident;
  for (DebugInfoFinder::iterator SPI = DIF.subprogram_begin(),
           SPE = DIF.subprogram_end(); SPI != SPE; ++SPI) {
    DISubprogram SP(*SPI);
    Function *F = SP.getFunction();

    // TODO: GCOV format requires a distinct unified exit block.
    unsigned edges = 0;
    for (Function::iterator BB = F->begin(), E = F->end(); BB != E; ++BB) {
      TerminatorInst *TI = BB->getTerminator();
      edges += TI->getNumSuccessors();
    }

    const ArrayType *counter_type =
        ArrayType::get(Type::getInt64Ty(*Ctx), edges);
    GlobalVariable *counter =
        new GlobalVariable(*Mod, counter_type, false,
                           GlobalValue::InternalLinkage,
                           Constant::getNullValue(counter_type),
                           "__llvm_gcov_ctr", 0, false, 0);
    counters_by_ident.push_back(
        std::make_pair(counter, reinterpret_cast<intptr_t>((MDNode*)SP)));

    UniqueVector<BasicBlock *> complex_edge_preds;
    UniqueVector<BasicBlock *> complex_edge_succs;

    unsigned edge_num = 0;
    for (Function::iterator BB = F->begin(), E = F->end(); BB != E; ++BB) {
      TerminatorInst *TI = BB->getTerminator();
      if (int successors = TI->getNumSuccessors()) {
        IRBuilder<> builder(TI);

        if (successors == 1) {
          Value *ctr = builder.CreateConstInBoundsGEP2_64(counter, 0, edge_num);
          Value *count = builder.CreateLoad(ctr);
          count = builder.CreateAdd(count,
                                    ConstantInt::get(Type::getInt64Ty(*Ctx),1));
          builder.CreateStore(count, ctr);
        } else if (BranchInst *BI = dyn_cast<BranchInst>(TI)) {
          Value *sel = builder.CreateSelect(
              BI->getCondition(),
              ConstantInt::get(Type::getInt64Ty(*Ctx), edge_num),
              ConstantInt::get(Type::getInt64Ty(*Ctx), edge_num + 1));
          SmallVector<Value *, 2> idx;
          idx.push_back(Constant::getNullValue(Type::getInt64Ty(*Ctx)));
          idx.push_back(sel);
          Value *ctr = builder.CreateInBoundsGEP(counter,
                                                 idx.begin(), idx.end());
          Value *count = builder.CreateLoad(ctr);
          count = builder.CreateAdd(count,
                                    ConstantInt::get(Type::getInt64Ty(*Ctx),1));
          builder.CreateStore(count, ctr);
        } else {
          complex_edge_preds.insert(BB);
          for (int i = 0; i != successors; ++i) {
            complex_edge_succs.insert(TI->getSuccessor(i));
          }
        }
        edge_num += successors;
      }
    }

    // TODO: support switch, invoke, indirectbr
    if (!complex_edge_preds.empty()) {
      // emit a [preds x [succs x i64*]].
      for (int i = 0, e = complex_edge_preds.size(); i != e; ++i) {
        // call runtime to state save
      }
      for (int i = 0, e = complex_edge_succs.size(); i != e; ++i) {
        // call runtime to perform increment
      }
    }
  }

  InsertCounterWriteout(DIF, counters_by_ident);

  return true;
}

Constant *GCOVProfiler::getStartFileFunc() {
  const Type *Args[1] = { Type::getInt8PtrTy(*Ctx) };
  const FunctionType *FTy = FunctionType::get(Type::getVoidTy(*Ctx),
                                              Args, false);
  return Mod->getOrInsertFunction("llvm_gcda_start_file", FTy);
}

Constant *GCOVProfiler::getEmitFunctionFunc() {
  const Type *Args[1] = { Type::getInt32Ty(*Ctx) };
  const FunctionType *FTy = FunctionType::get(Type::getVoidTy(*Ctx),
                                              Args, false);
  return Mod->getOrInsertFunction("llvm_gcda_emit_function", FTy);
}

Constant *GCOVProfiler::getEmitArcsFunc() {
  const Type *Args[] = {
    Type::getInt32Ty(*Ctx),     // uint32_t num_counters
    Type::getInt64PtrTy(*Ctx),  // uint64_t *counters
  };
  const FunctionType *FTy = FunctionType::get(Type::getVoidTy(*Ctx),
                                              Args, false);
  return Mod->getOrInsertFunction("llvm_gcda_emit_arcs", FTy);
}

Constant *GCOVProfiler::getEndFileFunc() {
  const FunctionType *FTy = FunctionType::get(Type::getVoidTy(*Ctx), false);
  return Mod->getOrInsertFunction("llvm_gcda_end_file", FTy);
}

static std::string ReplaceStem(std::string orig_filename, std::string new_stem){
  return (sys::path::stem(orig_filename) + "." + new_stem).str();
}

void GCOVProfiler::InsertCounterWriteout(
    DebugInfoFinder &DIF,
    SmallVector<std::pair<GlobalVariable *, uint32_t>, 8> &counters_by_ident) {

  const FunctionType *WriteoutFTy =
      FunctionType::get(Type::getVoidTy(*Ctx), false);
  Function *WriteoutF = Function::Create(WriteoutFTy,
                                         GlobalValue::InternalLinkage,
                                         "__llvm_gcda_writeout", Mod);
  WriteoutF->setUnnamedAddr(true);
  BasicBlock *BB = BasicBlock::Create(*Ctx, "", WriteoutF);
  IRBuilder<> builder(BB);

  Constant *StartFile = getStartFileFunc();
  Constant *EmitFunction = getEmitFunctionFunc();
  Constant *EmitArcs = getEmitArcsFunc();
  Constant *EndFile = getEndFileFunc();

  for (DebugInfoFinder::iterator CUI = DIF.compile_unit_begin(),
           CUE = DIF.compile_unit_end(); CUI != CUE; ++CUI) {
    DICompileUnit compile_unit(*CUI);
    std::string filename_gcda = ReplaceStem(compile_unit.getFilename(), "gcda");
    builder.CreateCall(StartFile,
                       builder.CreateGlobalStringPtr(filename_gcda));
    for (SmallVector<std::pair<GlobalVariable *, uint32_t>, 8>::iterator
             I = counters_by_ident.begin(), E = counters_by_ident.end();
         I != E; ++I) {
      builder.CreateCall(EmitFunction, ConstantInt::get(Type::getInt32Ty(*Ctx),
                                                        I->second));
      GlobalVariable *GV = I->first;
      unsigned num_arcs =
          cast<ArrayType>(GV->getType()->getElementType())->getNumElements();
      builder.CreateCall2(
          EmitArcs,
          ConstantInt::get(Type::getInt32Ty(*Ctx), num_arcs),
          builder.CreateConstGEP2_64(GV, 0, 0));
    }
    builder.CreateCall(EndFile);
  }
  builder.CreateRetVoid();

  InsertProfilingShutdownCall(WriteoutF, Mod);
}
