//===--- CGDebugInfo.h - DebugInfo for LLVM CodeGen -------------*- C++ -*-===//
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

#include "clang/AST/Type.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Analysis/DebugInfo.h"
#include <map>

#include "CGBuilder.h"

namespace clang {
  class VarDecl;
  class ObjCInterfaceDecl;

namespace CodeGen {
  class CodeGenModule;

/// CGDebugInfo - This class gathers all debug information during compilation 
/// and is responsible for emitting to llvm globals or pass directly to 
/// the backend.
class CGDebugInfo {
  CodeGenModule *M;
  bool isMainCompileUnitCreated;
  llvm::DIFactory DebugFactory;
  
  SourceLocation CurLoc, PrevLoc;

  /// CompileUnitCache - Cache of previously constructed CompileUnits.
  llvm::DenseMap<unsigned, llvm::DICompileUnit> CompileUnitCache;

  /// TypeCache - Cache of previously constructed Types.
  // FIXME: Eliminate this map.  Be careful of iterator invalidation.
  std::map<void *, llvm::DIType> TypeCache;
  
  std::vector<llvm::DIDescriptor> RegionStack;

  /// Helper functions for getOrCreateType.
  llvm::DIType CreateType(const BuiltinType *Ty, llvm::DICompileUnit U);
  llvm::DIType CreateType(const ComplexType *Ty, llvm::DICompileUnit U);
  llvm::DIType CreateCVRType(QualType Ty, llvm::DICompileUnit U);
  llvm::DIType CreateType(const TypedefType *Ty, llvm::DICompileUnit U);
  llvm::DIType CreateType(const PointerType *Ty, llvm::DICompileUnit U);
  llvm::DIType CreateType(const FunctionType *Ty, llvm::DICompileUnit U);
  llvm::DIType CreateType(const TagType *Ty, llvm::DICompileUnit U);
  llvm::DIType CreateType(const RecordType *Ty, llvm::DICompileUnit U);
  llvm::DIType CreateType(const ObjCInterfaceType *Ty, llvm::DICompileUnit U);
  llvm::DIType CreateType(const EnumType *Ty, llvm::DICompileUnit U);
  llvm::DIType CreateType(const ArrayType *Ty, llvm::DICompileUnit U);

public:
  CGDebugInfo(CodeGenModule *m);
  ~CGDebugInfo();

  /// setLocation - Update the current source location. If \arg loc is
  /// invalid it is ignored.
  void setLocation(SourceLocation Loc);

  /// EmitStopPoint - Emit a call to llvm.dbg.stoppoint to indicate a change of
  /// source line.
  void EmitStopPoint(llvm::Function *Fn, CGBuilderTy &Builder);

  /// EmitFunctionStart - Emit a call to llvm.dbg.function.start to indicate
  /// start of a new function.
  void EmitFunctionStart(const char *Name, QualType ReturnType,
                         llvm::Function *Fn, CGBuilderTy &Builder);
  
  /// EmitRegionStart - Emit a call to llvm.dbg.region.start to indicate start
  /// of a new block.  
  void EmitRegionStart(llvm::Function *Fn, CGBuilderTy &Builder);
  
  /// EmitRegionEnd - Emit call to llvm.dbg.region.end to indicate end of a 
  /// block.
  void EmitRegionEnd(llvm::Function *Fn, CGBuilderTy &Builder);

  /// EmitDeclareOfAutoVariable - Emit call to llvm.dbg.declare for an automatic
  /// variable declaration.
  void EmitDeclareOfAutoVariable(const VarDecl *Decl, llvm::Value *AI,
                                 CGBuilderTy &Builder);

  /// EmitDeclareOfArgVariable - Emit call to llvm.dbg.declare for an argument
  /// variable declaration.
  void EmitDeclareOfArgVariable(const VarDecl *Decl, llvm::Value *AI,
                                CGBuilderTy &Builder);
  
  /// EmitGlobalVariable - Emit information about a global variable.
  void EmitGlobalVariable(llvm::GlobalVariable *GV, const VarDecl *Decl);

  /// EmitGlobalVariable - Emit information about an objective-c interface.
  void EmitGlobalVariable(llvm::GlobalVariable *GV, ObjCInterfaceDecl *Decl);
   
private:
  /// EmitDeclare - Emit call to llvm.dbg.declare for a variable declaration.
  void EmitDeclare(const VarDecl *decl, unsigned Tag, llvm::Value *AI,
                   CGBuilderTy &Builder);
  
  
  /// getOrCreateCompileUnit - Get the compile unit from the cache or create a
  /// new one if necessary.
  llvm::DICompileUnit getOrCreateCompileUnit(SourceLocation Loc);

  /// getOrCreateType - Get the type from the cache or create a new type if
  /// necessary.
  llvm::DIType getOrCreateType(QualType Ty, llvm::DICompileUnit Unit);
};
} // namespace CodeGen
} // namespace clang

#endif
