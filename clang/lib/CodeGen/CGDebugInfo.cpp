//===--- CGDebugInfo.cpp - Emit Debug Information for a Module ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This coordinates the debug information generation while generating code.
//
//===----------------------------------------------------------------------===//

#include "CGDebugInfo.h"
#include "CodeGenModule.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Instructions.h"
#include "llvm/Intrinsics.h"
#include "llvm/Module.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/IRBuilder.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/FileManager.h"
#include "clang/AST/ASTContext.h"
using namespace clang;
using namespace clang::CodeGen;

CGDebugInfo::CGDebugInfo(CodeGenModule *m)
: M(m)
, CurLoc()
, PrevLoc()
, CompileUnitCache()
, StopPointFn(NULL)
, RegionStartFn(NULL)
, RegionEndFn(NULL)
, RegionStack()
{
  SR = new llvm::DISerializer();
  SR->setModule (&M->getModule());
}

CGDebugInfo::~CGDebugInfo()
{
  delete SR;
}


/// getCastValueFor - Return a llvm representation for a given debug information
/// descriptor cast to an empty struct pointer.
llvm::Value *CGDebugInfo::getCastValueFor(llvm::DebugInfoDesc *DD) {
  return llvm::ConstantExpr::getBitCast(SR->Serialize(DD), 
					SR->getEmptyStructPtrType());
}

/// getOrCreateCompileUnit - Get the compile unit from the cache or create a new
/// one if necessary.
llvm::CompileUnitDesc 
*CGDebugInfo::getOrCreateCompileUnit(const SourceLocation Loc) {

  // See if this compile unit has been used before.
  llvm::CompileUnitDesc *&Slot = CompileUnitCache[Loc.getFileID()];
  if (Slot) return Slot;

  // Create new compile unit.
  // FIXME: Where to free these?
  // One way is to iterate over the CompileUnitCache in ~CGDebugInfo.
  llvm::CompileUnitDesc *Unit = new llvm::CompileUnitDesc();

  // Make sure we have an anchor.
  if (!CompileUnitAnchor) {
    CompileUnitAnchor = new llvm::AnchorDesc(Unit);
  }

  // Get source file information.
  SourceManager &SM = M->getContext().getSourceManager();
  const FileEntry *FE = SM.getFileEntryForLoc(Loc);
  const char *FileName = FE->getName();
  const char *DirName = FE->getDir()->getName();

  Unit->setAnchor(CompileUnitAnchor);
  Unit->setFileName(FileName);
  Unit->setDirectory(DirName);

  // Set up producer name.
  // FIXME: Do not know how to get clang version yet.
  Unit->setProducer("clang");

  // Set up Language number.
  // FIXME: Handle other languages as well.
  Unit->setLanguage(llvm::dwarf::DW_LANG_C89);

  // Update cache.
  Slot = Unit;

  return Unit;
}


void 
CGDebugInfo::EmitStopPoint(llvm::Function *Fn, llvm::IRBuilder &Builder) {

  // Don't bother if things are the same as last time.
  SourceManager &SM = M->getContext().getSourceManager();
  if (CurLoc == PrevLoc 
       || (SM.getLineNumber(CurLoc) == SM.getLineNumber(PrevLoc)
           && SM.isFromSameFile(CurLoc, PrevLoc)))
    return;
  if (CurLoc.isInvalid()) return;

  // Update last state.
  PrevLoc = CurLoc;

  // Get the appropriate compile unit.
  llvm::CompileUnitDesc *Unit = getOrCreateCompileUnit(CurLoc);

  // Lazily construct llvm.dbg.stoppoint function.
  if (!StopPointFn)
    StopPointFn = llvm::Intrinsic::getDeclaration(&M->getModule(), 
					llvm::Intrinsic::dbg_stoppoint);

  uint64_t CurLineNo = SM.getLogicalLineNumber(CurLoc);
  uint64_t ColumnNo = SM.getLogicalColumnNumber(CurLoc);

  // Invoke llvm.dbg.stoppoint
  Builder.CreateCall3(StopPointFn, 
		       llvm::ConstantInt::get(llvm::Type::Int32Ty, CurLineNo),
		       llvm::ConstantInt::get(llvm::Type::Int32Ty, ColumnNo),
		       getCastValueFor(Unit), "");
}

/// EmitRegionStart- Constructs the debug code for entering a declarative
/// region - "llvm.dbg.region.start.".
void CGDebugInfo::EmitRegionStart(llvm::Function *Fn, llvm::IRBuilder &Builder) 
{
  llvm::BlockDesc *Block = new llvm::BlockDesc();
  if (RegionStack.size() > 0)
    Block->setContext(RegionStack.back());
  RegionStack.push_back(Block);

  // Lazily construct llvm.dbg.region.start function.
  if (!RegionStartFn)
    RegionStartFn = llvm::Intrinsic::getDeclaration(&M->getModule(), 
				llvm::Intrinsic::dbg_region_start);

  // Call llvm.dbg.func.start.
  Builder.CreateCall(RegionStartFn, getCastValueFor(Block), "");
}

/// EmitRegionEnd - Constructs the debug code for exiting a declarative
/// region - "llvm.dbg.region.end."
void CGDebugInfo::EmitRegionEnd(llvm::Function *Fn, llvm::IRBuilder &Builder) 
{
  // Lazily construct llvm.dbg.region.end function.
  if (!RegionEndFn)
    RegionEndFn =llvm::Intrinsic::getDeclaration(&M->getModule(), 
				llvm::Intrinsic::dbg_region_end);

  // Provide an region stop point.
  EmitStopPoint(Fn, Builder);
  
  // Call llvm.dbg.func.end.
  Builder.CreateCall(RegionEndFn, getCastValueFor(RegionStack.back()), "");
  RegionStack.pop_back();
  // FIXME: Free here the memory created for BlockDesc in RegionStart?
}

