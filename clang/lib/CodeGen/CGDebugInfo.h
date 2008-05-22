//===--- CGDebugInfo.h - DebugInfo for LLVM CodeGen -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is the source level debug info generator for llvm translation. 
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CODEGEN_CGDEBUGINFO_H
#define CLANG_CODEGEN_CGDEBUGINFO_H

#include "clang/Basic/SourceLocation.h"
#include <map>
#include <vector>


namespace llvm {
  class Function;
  class IRBuilder;
  class DISerializer;
  class CompileUnitDesc;
  class BasicBlock;
  class AnchorDesc;
  class DebugInfoDesc;
  class Value;
}

namespace clang {
namespace CodeGen {
  class CodeGenModule;

/// DebugInfo - This class gathers all debug information during compilation and
/// is responsible for emitting to llvm globals or pass directly to the backend.
class CGDebugInfo {
private:
  CodeGenModule *M;
  llvm::DISerializer *SR;
  SourceLocation CurLoc;
  SourceLocation PrevLoc;

  /// CompileUnitCache - Cache of previously constructed CompileUnits.
  std::map<unsigned, llvm::CompileUnitDesc *> CompileUnitCache;
  std::vector<llvm::DebugInfoDesc*> DebugAllocationList;
  
  llvm::Function *StopPointFn;
  llvm::AnchorDesc *CompileUnitAnchor;
  llvm::AnchorDesc *SubProgramAnchor;
  llvm::Function *RegionStartFn;
  llvm::Function *RegionEndFn;
  llvm::Function *FuncStartFn;
  llvm::Value *CurFuncDesc;  

  /// getOrCreateCompileUnit - Get the compile unit from the cache or create a
  /// new one if necessary.
  llvm::CompileUnitDesc *getOrCreateCompileUnit(SourceLocation loc);

  /// getCastValueFor - Return a llvm representation for a given debug
  /// information descriptor cast to an empty struct pointer.
  llvm::Value *getCastValueFor(llvm::DebugInfoDesc *DD);

public:
  CGDebugInfo(CodeGenModule *m);
  ~CGDebugInfo();

  void setLocation(SourceLocation loc) { CurLoc = loc; };

  /// EmitStopPoint - Emit a call to llvm.dbg.stoppoint to indicate a change of
  /// source line.
  void EmitStopPoint(llvm::Function *Fn, llvm::IRBuilder &Builder);
  
  /// EmitFunctionStart - Emit a call to llvm.dbg.func.start to indicate start
  /// of a new block.  
  void EmitFunctionStart(llvm::Function *Fn, llvm::IRBuilder &Builder);
  
  /// EmitFunctionEnd - Emit call to llvm.dbg.region.end to indicate end of a 
  /// block.
  void EmitFunctionEnd(llvm::Function *Fn, llvm::IRBuilder &Builder);
};
} // namespace CodeGen
} // namespace clang

#endif
