//===--- CodeGenPGO.cpp - PGO Instrumentation for LLVM CodeGen --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Instrumentation-based profile-guided optimization
//
//===----------------------------------------------------------------------===//

#include "CodeGenPGO.h"
#include "CodeGenFunction.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/StmtVisitor.h"
#include "llvm/Config/config.h" // for strtoull()/strtoll() define
#include "llvm/IR/MDBuilder.h"
#include "llvm/Support/FileSystem.h"

using namespace clang;
using namespace CodeGen;

static void ReportBadPGOData(CodeGenModule &CGM, const char *Message) {
  DiagnosticsEngine &Diags = CGM.getDiags();
  unsigned diagID = Diags.getCustomDiagID(DiagnosticsEngine::Error, "%0");
  Diags.Report(diagID) << Message;
}

PGOProfileData::PGOProfileData(CodeGenModule &CGM, std::string Path)
  : CGM(CGM) {
  if (llvm::MemoryBuffer::getFile(Path, DataBuffer)) {
    ReportBadPGOData(CGM, "failed to open pgo data file");
    return;
  }

  if (DataBuffer->getBufferSize() > std::numeric_limits<unsigned>::max()) {
    ReportBadPGOData(CGM, "pgo data file too big");
    return;
  }

  // Scan through the data file and map each function to the corresponding
  // file offset where its counts are stored.
  const char *BufferStart = DataBuffer->getBufferStart();
  const char *BufferEnd = DataBuffer->getBufferEnd();
  const char *CurPtr = BufferStart;
  uint64_t MaxCount = 0;
  while (CurPtr < BufferEnd) {
    // Read the mangled function name.
    const char *FuncName = CurPtr;
    // FIXME: Something will need to be added to distinguish static functions.
    CurPtr = strchr(CurPtr, ' ');
    if (!CurPtr) {
      ReportBadPGOData(CGM, "pgo data file has malformed function entry");
      return;
    }
    StringRef MangledName(FuncName, CurPtr - FuncName);

    // Read the number of counters.
    char *EndPtr;
    unsigned NumCounters = strtol(++CurPtr, &EndPtr, 10);
    if (EndPtr == CurPtr || *EndPtr != '\n' || NumCounters <= 0) {
      ReportBadPGOData(CGM, "pgo data file has unexpected number of counters");
      return;
    }
    CurPtr = EndPtr;

    // Read function count.
    uint64_t Count = strtoll(CurPtr, &EndPtr, 10);
    if (EndPtr == CurPtr || *EndPtr != '\n') {
      ReportBadPGOData(CGM, "pgo-data file has bad count value");
      return;
    }
    CurPtr = EndPtr + 1;
    FunctionCounts[MangledName] = Count;
    MaxCount = Count > MaxCount ? Count : MaxCount;

    // There is one line for each counter; skip over those lines.
    // Since function count is already read, we start the loop from 1.
    for (unsigned N = 1; N < NumCounters; ++N) {
      CurPtr = strchr(++CurPtr, '\n');
      if (!CurPtr) {
        ReportBadPGOData(CGM, "pgo data file is missing some counter info");
        return;
      }
    }

    // Skip over the blank line separating functions.
    CurPtr += 2;

    DataOffsets[MangledName] = FuncName - BufferStart;
  }
  MaxFunctionCount = MaxCount;
}

/// Return true if a function is hot. If we know nothing about the function,
/// return false.
bool PGOProfileData::isHotFunction(StringRef MangledName) {
  llvm::StringMap<uint64_t>::const_iterator CountIter =
    FunctionCounts.find(MangledName);
  // If we know nothing about the function, return false.
  if (CountIter == FunctionCounts.end())
    return false;
  // FIXME: functions with >= 30% of the maximal function count are
  // treated as hot. This number is from preliminary tuning on SPEC.
  return CountIter->getValue() >= (uint64_t)(0.3 * (double)MaxFunctionCount);
}

/// Return true if a function is cold. If we know nothing about the function,
/// return false.
bool PGOProfileData::isColdFunction(StringRef MangledName) {
  llvm::StringMap<uint64_t>::const_iterator CountIter =
    FunctionCounts.find(MangledName);
  // If we know nothing about the function, return false.
  if (CountIter == FunctionCounts.end())
    return false;
  // FIXME: functions with <= 1% of the maximal function count are treated as
  // cold. This number is from preliminary tuning on SPEC.
  return CountIter->getValue() <= (uint64_t)(0.01 * (double)MaxFunctionCount);
}

bool PGOProfileData::getFunctionCounts(StringRef MangledName,
                                       std::vector<uint64_t> &Counts) {
  // Find the relevant section of the pgo-data file.
  llvm::StringMap<unsigned>::const_iterator OffsetIter =
    DataOffsets.find(MangledName);
  if (OffsetIter == DataOffsets.end())
    return true;
  const char *CurPtr = DataBuffer->getBufferStart() + OffsetIter->getValue();

  // Skip over the function name.
  CurPtr = strchr(CurPtr, ' ');
  assert(CurPtr && "pgo-data has corrupted function entry");

  // Read the number of counters.
  char *EndPtr;
  unsigned NumCounters = strtol(++CurPtr, &EndPtr, 10);
  assert(EndPtr != CurPtr && *EndPtr == '\n' && NumCounters > 0 &&
         "pgo-data file has corrupted number of counters");
  CurPtr = EndPtr;

  Counts.reserve(NumCounters);

  for (unsigned N = 0; N < NumCounters; ++N) {
    // Read the count value.
    uint64_t Count = strtoll(CurPtr, &EndPtr, 10);
    if (EndPtr == CurPtr || *EndPtr != '\n') {
      ReportBadPGOData(CGM, "pgo-data file has bad count value");
      return true;
    }
    Counts.push_back(Count);
    CurPtr = EndPtr + 1;
  }

  // Make sure the number of counters matches up.
  if (Counts.size() != NumCounters) {
    ReportBadPGOData(CGM, "pgo-data file has inconsistent counters");
    return true;
  }

  return false;
}

void CodeGenPGO::emitWriteoutFunction(GlobalDecl &GD) {
  if (!CGM.getCodeGenOpts().ProfileInstrGenerate)
    return;

  llvm::LLVMContext &Ctx = CGM.getLLVMContext();

  llvm::Type *Int32Ty = llvm::Type::getInt32Ty(Ctx);
  llvm::Type *Int8PtrTy = llvm::Type::getInt8PtrTy(Ctx);

  llvm::Function *WriteoutF =
    CGM.getModule().getFunction("__llvm_pgo_writeout");
  if (!WriteoutF) {
    llvm::FunctionType *WriteoutFTy =
      llvm::FunctionType::get(llvm::Type::getVoidTy(Ctx), false);
    WriteoutF = llvm::Function::Create(WriteoutFTy,
                                       llvm::GlobalValue::InternalLinkage,
                                       "__llvm_pgo_writeout", &CGM.getModule());
  }
  WriteoutF->setUnnamedAddr(true);
  WriteoutF->addFnAttr(llvm::Attribute::NoInline);
  if (CGM.getCodeGenOpts().DisableRedZone)
    WriteoutF->addFnAttr(llvm::Attribute::NoRedZone);

  llvm::BasicBlock *BB = WriteoutF->empty() ?
    llvm::BasicBlock::Create(Ctx, "", WriteoutF) : &WriteoutF->getEntryBlock();

  CGBuilderTy PGOBuilder(BB);

  llvm::Instruction *I = BB->getTerminator();
  if (!I)
    I = PGOBuilder.CreateRetVoid();
  PGOBuilder.SetInsertPoint(I);

  llvm::Type *Int64PtrTy = llvm::Type::getInt64PtrTy(Ctx);
  llvm::Type *Args[] = {
    Int8PtrTy,                       // const char *MangledName
    Int32Ty,                         // uint32_t NumCounters
    Int64PtrTy                       // uint64_t *Counters
  };
  llvm::FunctionType *FTy =
    llvm::FunctionType::get(PGOBuilder.getVoidTy(), Args, false);
  llvm::Constant *EmitFunc =
    CGM.getModule().getOrInsertFunction("llvm_pgo_emit", FTy);

  llvm::Constant *MangledName =
    CGM.GetAddrOfConstantCString(CGM.getMangledName(GD), "__llvm_pgo_name");
  MangledName = llvm::ConstantExpr::getBitCast(MangledName, Int8PtrTy);
  PGOBuilder.CreateCall3(EmitFunc, MangledName,
                         PGOBuilder.getInt32(NumRegionCounters),
                         PGOBuilder.CreateBitCast(RegionCounters, Int64PtrTy));
}

llvm::Function *CodeGenPGO::emitInitialization(CodeGenModule &CGM) {
  llvm::Function *WriteoutF =
    CGM.getModule().getFunction("__llvm_pgo_writeout");
  if (!WriteoutF)
    return NULL;

  // Create a small bit of code that registers the "__llvm_pgo_writeout" to
  // be executed at exit.
  llvm::Function *F = CGM.getModule().getFunction("__llvm_pgo_init");
  if (F)
    return NULL;

  llvm::LLVMContext &Ctx = CGM.getLLVMContext();
  llvm::FunctionType *FTy = llvm::FunctionType::get(llvm::Type::getVoidTy(Ctx),
                                                    false);
  F = llvm::Function::Create(FTy, llvm::GlobalValue::InternalLinkage,
                             "__llvm_pgo_init", &CGM.getModule());
  F->setUnnamedAddr(true);
  F->setLinkage(llvm::GlobalValue::InternalLinkage);
  F->addFnAttr(llvm::Attribute::NoInline);
  if (CGM.getCodeGenOpts().DisableRedZone)
    F->addFnAttr(llvm::Attribute::NoRedZone);

  llvm::BasicBlock *BB = llvm::BasicBlock::Create(CGM.getLLVMContext(), "", F);
  CGBuilderTy PGOBuilder(BB);

  FTy = llvm::FunctionType::get(PGOBuilder.getVoidTy(), false);
  llvm::Type *Params[] = {
    llvm::PointerType::get(FTy, 0)
  };
  FTy = llvm::FunctionType::get(PGOBuilder.getVoidTy(), Params, false);

  // Inialize the environment and register the local writeout function.
  llvm::Constant *PGOInit =
    CGM.getModule().getOrInsertFunction("llvm_pgo_init", FTy);
  PGOBuilder.CreateCall(PGOInit, WriteoutF);
  PGOBuilder.CreateRetVoid();

  return F;
}

namespace {
  /// A StmtVisitor that fills a map of statements to PGO counters.
  struct MapRegionCounters : public ConstStmtVisitor<MapRegionCounters> {
    /// The next counter value to assign.
    unsigned NextCounter;
    /// The map of statements to counters.
    llvm::DenseMap<const Stmt*, unsigned> *CounterMap;

    MapRegionCounters(llvm::DenseMap<const Stmt*, unsigned> *CounterMap) :
      NextCounter(0), CounterMap(CounterMap) {
    }

    void VisitChildren(const Stmt *S) {
      for (Stmt::const_child_range I = S->children(); I; ++I)
        if (*I)
         this->Visit(*I);
    }
    void VisitStmt(const Stmt *S) { VisitChildren(S); }

    /// Assign a counter to track entry to the function body.
    void VisitFunctionDecl(const FunctionDecl *S) {
      (*CounterMap)[S->getBody()] = NextCounter++;
      Visit(S->getBody());
    }
    /// Assign a counter to track the block following a label.
    void VisitLabelStmt(const LabelStmt *S) {
      (*CounterMap)[S] = NextCounter++;
      Visit(S->getSubStmt());
    }
    /// Assign three counters - one for the body of the loop, one for breaks
    /// from the loop, and one for continues.
    ///
    /// The break and continue counters cover all such statements in this loop,
    /// and are used in calculations to find the number of times the condition
    /// and exit of the loop occur. They are needed so we can differentiate
    /// these statements from non-local exits like return and goto.
    void VisitWhileStmt(const WhileStmt *S) {
      (*CounterMap)[S] = NextCounter;
      NextCounter += 3;
      Visit(S->getCond());
      Visit(S->getBody());
    }
    /// Assign counters for the body of the loop, and for breaks and
    /// continues. See VisitWhileStmt.
    void VisitDoStmt(const DoStmt *S) {
      (*CounterMap)[S] = NextCounter;
      NextCounter += 3;
      Visit(S->getBody());
      Visit(S->getCond());
    }
    /// Assign counters for the body of the loop, and for breaks and
    /// continues. See VisitWhileStmt.
    void VisitForStmt(const ForStmt *S) {
      (*CounterMap)[S] = NextCounter;
      NextCounter += 3;
      const Expr *E;
      if ((E = S->getCond()))
        Visit(E);
      Visit(S->getBody());
      if ((E = S->getInc()))
        Visit(E);
    }
    /// Assign counters for the body of the loop, and for breaks and
    /// continues. See VisitWhileStmt.
    void VisitCXXForRangeStmt(const CXXForRangeStmt *S) {
      (*CounterMap)[S] = NextCounter;
      NextCounter += 3;
      const Expr *E;
      if ((E = S->getCond()))
        Visit(E);
      Visit(S->getBody());
      if ((E = S->getInc()))
        Visit(E);
    }
    /// Assign counters for the body of the loop, and for breaks and
    /// continues. See VisitWhileStmt.
    void VisitObjCForCollectionStmt(const ObjCForCollectionStmt *S) {
      (*CounterMap)[S] = NextCounter;
      NextCounter += 3;
      Visit(S->getElement());
      Visit(S->getBody());
    }
    /// Assign a counter for the exit block of the switch statement.
    void VisitSwitchStmt(const SwitchStmt *S) {
      (*CounterMap)[S] = NextCounter++;
      Visit(S->getCond());
      Visit(S->getBody());
    }
    /// Assign a counter for a particular case in a switch. This counts jumps
    /// from the switch header as well as fallthrough from the case before this
    /// one.
    void VisitCaseStmt(const CaseStmt *S) {
      (*CounterMap)[S] = NextCounter++;
      Visit(S->getSubStmt());
    }
    /// Assign a counter for the default case of a switch statement. The count
    /// is the number of branches from the loop header to the default, and does
    /// not include fallthrough from previous cases. If we have multiple
    /// conditional branch blocks from the switch instruction to the default
    /// block, as with large GNU case ranges, this is the counter for the last
    /// edge in that series, rather than the first.
    void VisitDefaultStmt(const DefaultStmt *S) {
      (*CounterMap)[S] = NextCounter++;
      Visit(S->getSubStmt());
    }
    /// Assign a counter for the "then" part of an if statement. The count for
    /// the "else" part, if it exists, will be calculated from this counter.
    void VisitIfStmt(const IfStmt *S) {
      (*CounterMap)[S] = NextCounter++;
      Visit(S->getCond());
      Visit(S->getThen());
      if (S->getElse())
        Visit(S->getElse());
    }
    /// Assign a counter for the continuation block of a C++ try statement.
    void VisitCXXTryStmt(const CXXTryStmt *S) {
      (*CounterMap)[S] = NextCounter++;
      Visit(S->getTryBlock());
      for (unsigned I = 0, E = S->getNumHandlers(); I < E; ++I)
        Visit(S->getHandler(I));
    }
    /// Assign a counter for a catch statement's handler block.
    void VisitCXXCatchStmt(const CXXCatchStmt *S) {
      (*CounterMap)[S] = NextCounter++;
      Visit(S->getHandlerBlock());
    }
    /// Assign a counter for the "true" part of a conditional operator. The
    /// count in the "false" part will be calculated from this counter.
    void VisitConditionalOperator(const ConditionalOperator *E) {
      (*CounterMap)[E] = NextCounter++;
      Visit(E->getCond());
      Visit(E->getTrueExpr());
      Visit(E->getFalseExpr());
    }
    /// Assign a counter for the right hand side of a logical and operator.
    void VisitBinLAnd(const BinaryOperator *E) {
      (*CounterMap)[E] = NextCounter++;
      Visit(E->getLHS());
      Visit(E->getRHS());
    }
    /// Assign a counter for the right hand side of a logical or operator.
    void VisitBinLOr(const BinaryOperator *E) {
      (*CounterMap)[E] = NextCounter++;
      Visit(E->getLHS());
      Visit(E->getRHS());
    }
  };
}

void CodeGenPGO::assignRegionCounters(GlobalDecl &GD) {
  bool InstrumentRegions = CGM.getCodeGenOpts().ProfileInstrGenerate;
  PGOProfileData *PGOData = CGM.getPGOData();
  if (!InstrumentRegions && !PGOData)
    return;
  const Decl *D = GD.getDecl();
  if (!D)
    return;
  mapRegionCounters(D);
  if (InstrumentRegions)
    emitCounterVariables();
  if (PGOData)
    loadRegionCounts(GD, PGOData);
}

void CodeGenPGO::mapRegionCounters(const Decl *D) {
  RegionCounterMap = new llvm::DenseMap<const Stmt*, unsigned>();
  MapRegionCounters Walker(RegionCounterMap);
  if (const FunctionDecl *FD = dyn_cast_or_null<FunctionDecl>(D))
    Walker.VisitFunctionDecl(FD);
  NumRegionCounters = Walker.NextCounter;
}

void CodeGenPGO::emitCounterVariables() {
  llvm::LLVMContext &Ctx = CGM.getLLVMContext();
  llvm::ArrayType *CounterTy = llvm::ArrayType::get(llvm::Type::getInt64Ty(Ctx),
                                                    NumRegionCounters);
  RegionCounters =
    new llvm::GlobalVariable(CGM.getModule(), CounterTy, false,
                             llvm::GlobalVariable::PrivateLinkage,
                             llvm::Constant::getNullValue(CounterTy),
                             "__llvm_pgo_ctr");
}

void CodeGenPGO::emitCounterIncrement(CGBuilderTy &Builder, unsigned Counter) {
  if (!CGM.getCodeGenOpts().ProfileInstrGenerate)
    return;
  llvm::Value *Addr =
    Builder.CreateConstInBoundsGEP2_64(RegionCounters, 0, Counter);
  llvm::Value *Count = Builder.CreateLoad(Addr, "pgocount");
  Count = Builder.CreateAdd(Count, Builder.getInt64(1));
  Builder.CreateStore(Count, Addr);
}

void CodeGenPGO::loadRegionCounts(GlobalDecl &GD, PGOProfileData *PGOData) {
  // For now, ignore the counts from the PGO data file only if the number of
  // counters does not match. This could be tightened down in the future to
  // ignore counts when the input changes in various ways, e.g., by comparing a
  // hash value based on some characteristics of the input.
  RegionCounts = new std::vector<uint64_t>();
  if (PGOData->getFunctionCounts(CGM.getMangledName(GD), *RegionCounts) ||
      RegionCounts->size() != NumRegionCounters) {
    delete RegionCounts;
    RegionCounts = 0;
  }
}

void CodeGenPGO::destroyRegionCounters() {
  if (RegionCounterMap != 0)
    delete RegionCounterMap;
  if (RegionCounts != 0)
    delete RegionCounts;
}

llvm::MDNode *CodeGenPGO::createBranchWeights(uint64_t TrueCount,
                                              uint64_t FalseCount) {
  if (!TrueCount && !FalseCount)
    return 0;

  llvm::MDBuilder MDHelper(CGM.getLLVMContext());
  // TODO: need to scale down to 32-bits
  // According to Laplace's Rule of Succession, it is better to compute the
  // weight based on the count plus 1.
  return MDHelper.createBranchWeights(TrueCount + 1, FalseCount + 1);
}

llvm::MDNode *
CodeGenPGO::createBranchWeights(ArrayRef<uint64_t> Weights) {
  llvm::MDBuilder MDHelper(CGM.getLLVMContext());
  // TODO: need to scale down to 32-bits, instead of just truncating.
  // According to Laplace's Rule of Succession, it is better to compute the
  // weight based on the count plus 1.
  SmallVector<uint32_t, 16> ScaledWeights;
  ScaledWeights.reserve(Weights.size());
  for (ArrayRef<uint64_t>::iterator WI = Weights.begin(), WE = Weights.end();
       WI != WE; ++WI) {
    ScaledWeights.push_back(*WI + 1);
  }
  return MDHelper.createBranchWeights(ScaledWeights);
}
