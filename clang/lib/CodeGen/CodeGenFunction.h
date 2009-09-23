//===-- CodeGenFunction.h - Per-Function state for LLVM CodeGen -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is the internal per-function state used for llvm translation.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CODEGEN_CODEGENFUNCTION_H
#define CLANG_CODEGEN_CODEGENFUNCTION_H

#include "clang/AST/Type.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ExprObjC.h"
#include "clang/Basic/TargetInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ValueHandle.h"
#include <map>
#include "CodeGenModule.h"
#include "CGBlocks.h"
#include "CGBuilder.h"
#include "CGCall.h"
#include "CGCXX.h"
#include "CGValue.h"

namespace llvm {
  class BasicBlock;
  class LLVMContext;
  class Module;
  class SwitchInst;
  class Value;
}

namespace clang {
  class ASTContext;
  class CXXDestructorDecl;
  class Decl;
  class EnumConstantDecl;
  class FunctionDecl;
  class FunctionProtoType;
  class LabelStmt;
  class ObjCContainerDecl;
  class ObjCInterfaceDecl;
  class ObjCIvarDecl;
  class ObjCMethodDecl;
  class ObjCImplementationDecl;
  class ObjCPropertyImplDecl;
  class TargetInfo;
  class VarDecl;
  class ObjCForCollectionStmt;
  class ObjCAtTryStmt;
  class ObjCAtThrowStmt;
  class ObjCAtSynchronizedStmt;

namespace CodeGen {
  class CodeGenModule;
  class CodeGenTypes;
  class CGDebugInfo;
  class CGFunctionInfo;
  class CGRecordLayout;

/// CodeGenFunction - This class organizes the per-function state that is used
/// while generating LLVM code.
class CodeGenFunction : public BlockFunction {
  CodeGenFunction(const CodeGenFunction&); // DO NOT IMPLEMENT
  void operator=(const CodeGenFunction&);  // DO NOT IMPLEMENT
public:
  CodeGenModule &CGM;  // Per-module state.
  TargetInfo &Target;

  typedef std::pair<llvm::Value *, llvm::Value *> ComplexPairTy;
  CGBuilderTy Builder;

  /// CurFuncDecl - Holds the Decl for the current function or ObjC method.
  /// This excludes BlockDecls.
  const Decl *CurFuncDecl;
  /// CurCodeDecl - This is the inner-most code context, which includes blocks.
  const Decl *CurCodeDecl;
  const CGFunctionInfo *CurFnInfo;
  QualType FnRetTy;
  llvm::Function *CurFn;

  /// ReturnBlock - Unified return block.
  llvm::BasicBlock *ReturnBlock;
  /// ReturnValue - The temporary alloca to hold the return value. This is null
  /// iff the function has no return value.
  llvm::Instruction *ReturnValue;

  /// AllocaInsertPoint - This is an instruction in the entry block before which
  /// we prefer to insert allocas.
  llvm::AssertingVH<llvm::Instruction> AllocaInsertPt;

  const llvm::Type *LLVMIntTy;
  uint32_t LLVMPointerWidth;

public:
  /// ObjCEHValueStack - Stack of Objective-C exception values, used for
  /// rethrows.
  llvm::SmallVector<llvm::Value*, 8> ObjCEHValueStack;

  /// PushCleanupBlock - Push a new cleanup entry on the stack and set the
  /// passed in block as the cleanup block.
  void PushCleanupBlock(llvm::BasicBlock *CleanupBlock);

  /// CleanupBlockInfo - A struct representing a popped cleanup block.
  struct CleanupBlockInfo {
    /// CleanupBlock - the cleanup block
    llvm::BasicBlock *CleanupBlock;

    /// SwitchBlock - the block (if any) containing the switch instruction used
    /// for jumping to the final destination.
    llvm::BasicBlock *SwitchBlock;

    /// EndBlock - the default destination for the switch instruction.
    llvm::BasicBlock *EndBlock;

    CleanupBlockInfo(llvm::BasicBlock *cb, llvm::BasicBlock *sb,
                     llvm::BasicBlock *eb)
      : CleanupBlock(cb), SwitchBlock(sb), EndBlock(eb) {}
  };

  /// PopCleanupBlock - Will pop the cleanup entry on the stack, process all
  /// branch fixups and return a block info struct with the switch block and end
  /// block.
  CleanupBlockInfo PopCleanupBlock();

  /// CleanupScope - RAII object that will create a cleanup block and set the
  /// insert point to that block. When destructed, it sets the insert point to
  /// the previous block and pushes a new cleanup entry on the stack.
  class CleanupScope {
    CodeGenFunction& CGF;
    llvm::BasicBlock *CurBB;
    llvm::BasicBlock *CleanupBB;

  public:
    CleanupScope(CodeGenFunction &cgf)
      : CGF(cgf), CurBB(CGF.Builder.GetInsertBlock()) {
      CleanupBB = CGF.createBasicBlock("cleanup");
      CGF.Builder.SetInsertPoint(CleanupBB);
    }

    ~CleanupScope() {
      CGF.PushCleanupBlock(CleanupBB);
      // FIXME: This is silly, move this into the builder.
      if (CurBB)
        CGF.Builder.SetInsertPoint(CurBB);
      else
        CGF.Builder.ClearInsertionPoint();
    }
  };

  /// EmitCleanupBlocks - Takes the old cleanup stack size and emits the cleanup
  /// blocks that have been added.
  void EmitCleanupBlocks(size_t OldCleanupStackSize);

  /// EmitBranchThroughCleanup - Emit a branch from the current insert block
  /// through the cleanup handling code (if any) and then on to \arg Dest.
  ///
  /// FIXME: Maybe this should really be in EmitBranch? Don't we always want
  /// this behavior for branches?
  void EmitBranchThroughCleanup(llvm::BasicBlock *Dest);

  /// PushConditionalTempDestruction - Should be called before a conditional
  /// part of an expression is emitted. For example, before the RHS of the
  /// expression below is emitted:
  ///
  /// b && f(T());
  ///
  /// This is used to make sure that any temporaryes created in the conditional
  /// branch are only destroyed if the branch is taken.
  void PushConditionalTempDestruction();

  /// PopConditionalTempDestruction - Should be called after a conditional
  /// part of an expression has been emitted.
  void PopConditionalTempDestruction();

private:
  CGDebugInfo* DebugInfo;

  /// LabelIDs - Track arbitrary ids assigned to labels for use in implementing
  /// the GCC address-of-label extension and indirect goto. IDs are assigned to
  /// labels inside getIDForAddrOfLabel().
  std::map<const LabelStmt*, unsigned> LabelIDs;

  /// IndirectSwitches - Record the list of switches for indirect
  /// gotos. Emission of the actual switching code needs to be delayed until all
  /// AddrLabelExprs have been seen.
  std::vector<llvm::SwitchInst*> IndirectSwitches;

  /// LocalDeclMap - This keeps track of the LLVM allocas or globals for local C
  /// decls.
  llvm::DenseMap<const Decl*, llvm::Value*> LocalDeclMap;

  /// LabelMap - This keeps track of the LLVM basic block for each C label.
  llvm::DenseMap<const LabelStmt*, llvm::BasicBlock*> LabelMap;

  // BreakContinueStack - This keeps track of where break and continue
  // statements should jump to.
  struct BreakContinue {
    BreakContinue(llvm::BasicBlock *bb, llvm::BasicBlock *cb)
      : BreakBlock(bb), ContinueBlock(cb) {}

    llvm::BasicBlock *BreakBlock;
    llvm::BasicBlock *ContinueBlock;
  };
  llvm::SmallVector<BreakContinue, 8> BreakContinueStack;

  /// SwitchInsn - This is nearest current switch instruction. It is null if if
  /// current context is not in a switch.
  llvm::SwitchInst *SwitchInsn;

  /// CaseRangeBlock - This block holds if condition check for last case
  /// statement range in current switch instruction.
  llvm::BasicBlock *CaseRangeBlock;

  /// InvokeDest - This is the nearest exception target for calls
  /// which can unwind, when exceptions are being used.
  llvm::BasicBlock *InvokeDest;

  // VLASizeMap - This keeps track of the associated size for each VLA type.
  // We track this by the size expression rather than the type itself because
  // in certain situations, like a const qualifier applied to an VLA typedef,
  // multiple VLA types can share the same size expression.
  // FIXME: Maybe this could be a stack of maps that is pushed/popped as we
  // enter/leave scopes.
  llvm::DenseMap<const Expr*, llvm::Value*> VLASizeMap;

  /// DidCallStackSave - Whether llvm.stacksave has been called. Used to avoid
  /// calling llvm.stacksave for multiple VLAs in the same scope.
  bool DidCallStackSave;

  struct CleanupEntry {
    /// CleanupBlock - The block of code that does the actual cleanup.
    llvm::BasicBlock *CleanupBlock;

    /// Blocks - Basic blocks that were emitted in the current cleanup scope.
    std::vector<llvm::BasicBlock *> Blocks;

    /// BranchFixups - Branch instructions to basic blocks that haven't been
    /// inserted into the current function yet.
    std::vector<llvm::BranchInst *> BranchFixups;

    explicit CleanupEntry(llvm::BasicBlock *cb)
      : CleanupBlock(cb) {}
  };

  /// CleanupEntries - Stack of cleanup entries.
  llvm::SmallVector<CleanupEntry, 8> CleanupEntries;

  typedef llvm::DenseMap<llvm::BasicBlock*, size_t> BlockScopeMap;

  /// BlockScopes - Map of which "cleanup scope" scope basic blocks have.
  BlockScopeMap BlockScopes;

  /// CXXThisDecl - When parsing an C++ function, this will hold the implicit
  /// 'this' declaration.
  ImplicitParamDecl *CXXThisDecl;

  /// CXXLiveTemporaryInfo - Holds information about a live C++ temporary.
  struct CXXLiveTemporaryInfo {
    /// Temporary - The live temporary.
    const CXXTemporary *Temporary;

    /// ThisPtr - The pointer to the temporary.
    llvm::Value *ThisPtr;

    /// DtorBlock - The destructor block.
    llvm::BasicBlock *DtorBlock;

    /// CondPtr - If this is a conditional temporary, this is the pointer to
    /// the condition variable that states whether the destructor should be
    /// called or not.
    llvm::Value *CondPtr;

    CXXLiveTemporaryInfo(const CXXTemporary *temporary,
                         llvm::Value *thisptr, llvm::BasicBlock *dtorblock,
                         llvm::Value *condptr)
      : Temporary(temporary), ThisPtr(thisptr), DtorBlock(dtorblock),
      CondPtr(condptr) { }
  };

  llvm::SmallVector<CXXLiveTemporaryInfo, 4> LiveTemporaries;

  /// ConditionalTempDestructionStack - Contains the number of live temporaries
  /// when PushConditionalTempDestruction was called. This is used so that
  /// we know how many temporaries were created by a certain expression.
  llvm::SmallVector<size_t, 4> ConditionalTempDestructionStack;


  /// ByrefValueInfoMap - For each __block variable, contains a pair of the LLVM
  /// type as well as the field number that contains the actual data.
  llvm::DenseMap<const ValueDecl *, std::pair<const llvm::Type *, 
                                              unsigned> > ByRefValueInfo;
  
  /// getByrefValueFieldNumber - Given a declaration, returns the LLVM field
  /// number that holds the value.
  unsigned getByRefValueLLVMField(const ValueDecl *VD) const;
  
public:
  CodeGenFunction(CodeGenModule &cgm);

  ASTContext &getContext() const;
  CGDebugInfo *getDebugInfo() { return DebugInfo; }

  llvm::BasicBlock *getInvokeDest() { return InvokeDest; }
  void setInvokeDest(llvm::BasicBlock *B) { InvokeDest = B; }

  llvm::LLVMContext &getLLVMContext() { return VMContext; }

  //===--------------------------------------------------------------------===//
  //                                  Objective-C
  //===--------------------------------------------------------------------===//

  void GenerateObjCMethod(const ObjCMethodDecl *OMD);

  void StartObjCMethod(const ObjCMethodDecl *MD,
                       const ObjCContainerDecl *CD);

  /// GenerateObjCGetter - Synthesize an Objective-C property getter function.
  void GenerateObjCGetter(ObjCImplementationDecl *IMP,
                          const ObjCPropertyImplDecl *PID);

  /// GenerateObjCSetter - Synthesize an Objective-C property setter function
  /// for the given property.
  void GenerateObjCSetter(ObjCImplementationDecl *IMP,
                          const ObjCPropertyImplDecl *PID);

  //===--------------------------------------------------------------------===//
  //                                  Block Bits
  //===--------------------------------------------------------------------===//

  llvm::Value *BuildBlockLiteralTmp(const BlockExpr *);
  llvm::Constant *BuildDescriptorBlockDecl(bool BlockHasCopyDispose,
                                           uint64_t Size,
                                           const llvm::StructType *,
                                           std::vector<HelperInfo> *);

  llvm::Function *GenerateBlockFunction(const BlockExpr *BExpr,
                                        const BlockInfo& Info,
                                        const Decl *OuterFuncDecl,
                                  llvm::DenseMap<const Decl*, llvm::Value*> ldm,
                                        uint64_t &Size, uint64_t &Align,
                      llvm::SmallVector<const Expr *, 8> &subBlockDeclRefDecls,
                                        bool &subBlockHasCopyDispose);

  void BlockForwardSelf();
  llvm::Value *LoadBlockStruct();

  llvm::Value *GetAddrOfBlockDecl(const BlockDeclRefExpr *E);
  const llvm::Type *BuildByRefType(const ValueDecl *D);

  void GenerateCode(GlobalDecl GD, llvm::Function *Fn);
  void StartFunction(GlobalDecl GD, QualType RetTy,
                     llvm::Function *Fn,
                     const FunctionArgList &Args,
                     SourceLocation StartLoc);

  /// EmitReturnBlock - Emit the unified return block, trying to avoid its
  /// emission when possible.
  void EmitReturnBlock();

  /// FinishFunction - Complete IR generation of the current function. It is
  /// legal to call this function even if there is no current insertion point.
  void FinishFunction(SourceLocation EndLoc=SourceLocation());

  /// GenerateVtable - Generate the vtable for the given type.
  llvm::Value *GenerateVtable(const CXXRecordDecl *RD);

  /// GenerateThunk - Generate a thunk for the given method
  llvm::Constant *GenerateThunk(llvm::Function *Fn, const CXXMethodDecl *MD,
                                bool Extern, int64_t nv, int64_t v);
  llvm::Constant *GenerateCovariantThunk(llvm::Function *Fn,
                                         const CXXMethodDecl *MD, bool Extern,
                                         int64_t nv_t, int64_t v_t,
                                         int64_t nv_r, int64_t v_r);

  void EmitCtorPrologue(const CXXConstructorDecl *CD, CXXCtorType Type);

  void SynthesizeCXXCopyConstructor(const CXXConstructorDecl *Ctor,
                                    CXXCtorType Type,
                                    llvm::Function *Fn,
                                    const FunctionArgList &Args);

  void SynthesizeCXXCopyAssignment(const CXXMethodDecl *CD,
                                   llvm::Function *Fn,
                                   const FunctionArgList &Args);

  void SynthesizeDefaultConstructor(const CXXConstructorDecl *Ctor,
                                    CXXCtorType Type,
                                    llvm::Function *Fn,
                                    const FunctionArgList &Args);

  void SynthesizeDefaultDestructor(const CXXDestructorDecl *Dtor,
                                   CXXDtorType Type,
                                   llvm::Function *Fn,
                                   const FunctionArgList &Args);

  /// EmitDtorEpilogue - Emit all code that comes at the end of class's
  /// destructor. This is to call destructors on members and base classes
  /// in reverse order of their construction.
  void EmitDtorEpilogue(const CXXDestructorDecl *Dtor,
                        CXXDtorType Type);

  /// EmitFunctionProlog - Emit the target specific LLVM code to load the
  /// arguments for the given function. This is also responsible for naming the
  /// LLVM function arguments.
  void EmitFunctionProlog(const CGFunctionInfo &FI,
                          llvm::Function *Fn,
                          const FunctionArgList &Args);

  /// EmitFunctionEpilog - Emit the target specific LLVM code to return the
  /// given temporary.
  void EmitFunctionEpilog(const CGFunctionInfo &FI, llvm::Value *ReturnValue);

  const llvm::Type *ConvertTypeForMem(QualType T);
  const llvm::Type *ConvertType(QualType T);

  /// LoadObjCSelf - Load the value of self. This function is only valid while
  /// generating code for an Objective-C method.
  llvm::Value *LoadObjCSelf();

  /// TypeOfSelfObject - Return type of object that this self represents.
  QualType TypeOfSelfObject();

  /// hasAggregateLLVMType - Return true if the specified AST type will map into
  /// an aggregate LLVM type or is void.
  static bool hasAggregateLLVMType(QualType T);

  /// createBasicBlock - Create an LLVM basic block.
  llvm::BasicBlock *createBasicBlock(const char *Name="",
                                     llvm::Function *Parent=0,
                                     llvm::BasicBlock *InsertBefore=0) {
#ifdef NDEBUG
    return llvm::BasicBlock::Create(VMContext, "", Parent, InsertBefore);
#else
    return llvm::BasicBlock::Create(VMContext, Name, Parent, InsertBefore);
#endif
  }

  /// getBasicBlockForLabel - Return the LLVM basicblock that the specified
  /// label maps to.
  llvm::BasicBlock *getBasicBlockForLabel(const LabelStmt *S);

  /// SimplifyForwardingBlocks - If the given basic block is only a
  /// branch to another basic block, simplify it. This assumes that no
  /// other code could potentially reference the basic block.
  void SimplifyForwardingBlocks(llvm::BasicBlock *BB);

  /// EmitBlock - Emit the given block \arg BB and set it as the insert point,
  /// adding a fall-through branch from the current insert block if
  /// necessary. It is legal to call this function even if there is no current
  /// insertion point.
  ///
  /// IsFinished - If true, indicates that the caller has finished emitting
  /// branches to the given block and does not expect to emit code into it. This
  /// means the block can be ignored if it is unreachable.
  void EmitBlock(llvm::BasicBlock *BB, bool IsFinished=false);

  /// EmitBranch - Emit a branch to the specified basic block from the current
  /// insert block, taking care to avoid creation of branches from dummy
  /// blocks. It is legal to call this function even if there is no current
  /// insertion point.
  ///
  /// This function clears the current insertion point. The caller should follow
  /// calls to this function with calls to Emit*Block prior to generation new
  /// code.
  void EmitBranch(llvm::BasicBlock *Block);

  /// HaveInsertPoint - True if an insertion point is defined. If not, this
  /// indicates that the current code being emitted is unreachable.
  bool HaveInsertPoint() const {
    return Builder.GetInsertBlock() != 0;
  }

  /// EnsureInsertPoint - Ensure that an insertion point is defined so that
  /// emitted IR has a place to go. Note that by definition, if this function
  /// creates a block then that block is unreachable; callers may do better to
  /// detect when no insertion point is defined and simply skip IR generation.
  void EnsureInsertPoint() {
    if (!HaveInsertPoint())
      EmitBlock(createBasicBlock());
  }

  /// ErrorUnsupported - Print out an error that codegen doesn't support the
  /// specified stmt yet.
  void ErrorUnsupported(const Stmt *S, const char *Type,
                        bool OmitOnError=false);

  //===--------------------------------------------------------------------===//
  //                                  Helpers
  //===--------------------------------------------------------------------===//

  /// CreateTempAlloca - This creates a alloca and inserts it into the entry
  /// block.
  llvm::AllocaInst *CreateTempAlloca(const llvm::Type *Ty,
                                     const char *Name = "tmp");

  /// EvaluateExprAsBool - Perform the usual unary conversions on the specified
  /// expression and compare the result against zero, returning an Int1Ty value.
  llvm::Value *EvaluateExprAsBool(const Expr *E);

  /// EmitAnyExpr - Emit code to compute the specified expression which can have
  /// any type.  The result is returned as an RValue struct.  If this is an
  /// aggregate expression, the aggloc/agglocvolatile arguments indicate where
  /// the result should be returned.
  ///
  /// \param IgnoreResult - True if the resulting value isn't used.
  RValue EmitAnyExpr(const Expr *E, llvm::Value *AggLoc = 0,
                     bool IsAggLocVolatile = false, bool IgnoreResult = false,
                     bool IsInitializer = false);

  // EmitVAListRef - Emit a "reference" to a va_list; this is either the address
  // or the value of the expression, depending on how va_list is defined.
  llvm::Value *EmitVAListRef(const Expr *E);

  /// EmitAnyExprToTemp - Similary to EmitAnyExpr(), however, the result will
  /// always be accessible even if no aggregate location is provided.
  RValue EmitAnyExprToTemp(const Expr *E, bool IsAggLocVolatile = false,
                           bool IsInitializer = false);

  /// EmitAggregateCopy - Emit an aggrate copy.
  ///
  /// \param isVolatile - True iff either the source or the destination is
  /// volatile.
  void EmitAggregateCopy(llvm::Value *DestPtr, llvm::Value *SrcPtr,
                         QualType EltTy, bool isVolatile=false);

  void EmitAggregateClear(llvm::Value *DestPtr, QualType Ty);

  /// StartBlock - Start new block named N. If insert block is a dummy block
  /// then reuse it.
  void StartBlock(const char *N);

  /// GetAddrOfStaticLocalVar - Return the address of a static local variable.
  llvm::Constant *GetAddrOfStaticLocalVar(const VarDecl *BVD);

  /// GetAddrOfLocalVar - Return the address of a local variable.
  llvm::Value *GetAddrOfLocalVar(const VarDecl *VD);

  /// getAccessedFieldNo - Given an encoded value and a result number, return
  /// the input field number being accessed.
  static unsigned getAccessedFieldNo(unsigned Idx, const llvm::Constant *Elts);

  unsigned GetIDForAddrOfLabel(const LabelStmt *L);

  /// EmitMemSetToZero - Generate code to memset a value of the given type to 0.
  void EmitMemSetToZero(llvm::Value *DestPtr, QualType Ty);

  // EmitVAArg - Generate code to get an argument from the passed in pointer
  // and update it accordingly. The return value is a pointer to the argument.
  // FIXME: We should be able to get rid of this method and use the va_arg
  // instruction in LLVM instead once it works well enough.
  llvm::Value *EmitVAArg(llvm::Value *VAListAddr, QualType Ty);

  // EmitVLASize - Generate code for any VLA size expressions that might occur
  // in a variably modified type. If Ty is a VLA, will return the value that
  // corresponds to the size in bytes of the VLA type. Will return 0 otherwise.
  ///
  /// This function can be called with a null (unreachable) insert point.
  llvm::Value *EmitVLASize(QualType Ty);

  // GetVLASize - Returns an LLVM value that corresponds to the size in bytes
  // of a variable length array type.
  llvm::Value *GetVLASize(const VariableArrayType *);

  /// LoadCXXThis - Load the value of 'this'. This function is only valid while
  /// generating code for an C++ member function.
  llvm::Value *LoadCXXThis();

  /// GetAddressCXXOfBaseClass - This function will add the necessary delta
  /// to the load of 'this' and returns address of the base class.
  // FIXME. This currently only does a derived to non-virtual base conversion.
  // Other kinds of conversions will come later.
  llvm::Value *GetAddressCXXOfBaseClass(llvm::Value *BaseValue,
                                        const CXXRecordDecl *ClassDecl,
                                        const CXXRecordDecl *BaseClassDecl,
                                        bool NullCheckValue);

  void EmitClassAggrMemberwiseCopy(llvm::Value *DestValue,
                                   llvm::Value *SrcValue,
                                   const ArrayType *Array,
                                   const CXXRecordDecl *BaseClassDecl,
                                   QualType Ty);

  void EmitClassAggrCopyAssignment(llvm::Value *DestValue,
                                   llvm::Value *SrcValue,
                                   const ArrayType *Array,
                                   const CXXRecordDecl *BaseClassDecl,
                                   QualType Ty);

  void EmitClassMemberwiseCopy(llvm::Value *DestValue, llvm::Value *SrcValue,
                               const CXXRecordDecl *ClassDecl,
                               const CXXRecordDecl *BaseClassDecl,
                               QualType Ty);

  void EmitClassCopyAssignment(llvm::Value *DestValue, llvm::Value *SrcValue,
                               const CXXRecordDecl *ClassDecl,
                               const CXXRecordDecl *BaseClassDecl,
                               QualType Ty);

  void EmitCXXConstructorCall(const CXXConstructorDecl *D, CXXCtorType Type,
                              llvm::Value *This,
                              CallExpr::const_arg_iterator ArgBeg,
                              CallExpr::const_arg_iterator ArgEnd);

  void EmitCXXAggrConstructorCall(const CXXConstructorDecl *D,
                                  const ConstantArrayType *ArrayTy,
                                  llvm::Value *ArrayPtr);
  void EmitCXXAggrConstructorCall(const CXXConstructorDecl *D,
                                  llvm::Value *NumElements,
                                  llvm::Value *ArrayPtr);

  void EmitCXXAggrDestructorCall(const CXXDestructorDecl *D,
                                 const ArrayType *Array,
                                 llvm::Value *This);

  void EmitCXXDestructorCall(const CXXDestructorDecl *D, CXXDtorType Type,
                             llvm::Value *This);

  void PushCXXTemporary(const CXXTemporary *Temporary, llvm::Value *Ptr);
  void PopCXXTemporary();

  llvm::Value *EmitCXXNewExpr(const CXXNewExpr *E);
  void EmitCXXDeleteExpr(const CXXDeleteExpr *E);

  //===--------------------------------------------------------------------===//
  //                            Declaration Emission
  //===--------------------------------------------------------------------===//

  /// EmitDecl - Emit a declaration.
  ///
  /// This function can be called with a null (unreachable) insert point.
  void EmitDecl(const Decl &D);

  /// EmitBlockVarDecl - Emit a block variable declaration.
  ///
  /// This function can be called with a null (unreachable) insert point.
  void EmitBlockVarDecl(const VarDecl &D);

  /// EmitLocalBlockVarDecl - Emit a local block variable declaration.
  ///
  /// This function can be called with a null (unreachable) insert point.
  void EmitLocalBlockVarDecl(const VarDecl &D);

  void EmitStaticBlockVarDecl(const VarDecl &D);

  /// EmitParmDecl - Emit a ParmVarDecl or an ImplicitParamDecl.
  void EmitParmDecl(const VarDecl &D, llvm::Value *Arg);

  //===--------------------------------------------------------------------===//
  //                             Statement Emission
  //===--------------------------------------------------------------------===//

  /// EmitStopPoint - Emit a debug stoppoint if we are emitting debug info.
  void EmitStopPoint(const Stmt *S);

  /// EmitStmt - Emit the code for the statement \arg S. It is legal to call
  /// this function even if there is no current insertion point.
  ///
  /// This function may clear the current insertion point; callers should use
  /// EnsureInsertPoint if they wish to subsequently generate code without first
  /// calling EmitBlock, EmitBranch, or EmitStmt.
  void EmitStmt(const Stmt *S);

  /// EmitSimpleStmt - Try to emit a "simple" statement which does not
  /// necessarily require an insertion point or debug information; typically
  /// because the statement amounts to a jump or a container of other
  /// statements.
  ///
  /// \return True if the statement was handled.
  bool EmitSimpleStmt(const Stmt *S);

  RValue EmitCompoundStmt(const CompoundStmt &S, bool GetLast = false,
                          llvm::Value *AggLoc = 0, bool isAggVol = false);

  /// EmitLabel - Emit the block for the given label. It is legal to call this
  /// function even if there is no current insertion point.
  void EmitLabel(const LabelStmt &S); // helper for EmitLabelStmt.

  void EmitLabelStmt(const LabelStmt &S);
  void EmitGotoStmt(const GotoStmt &S);
  void EmitIndirectGotoStmt(const IndirectGotoStmt &S);
  void EmitIfStmt(const IfStmt &S);
  void EmitWhileStmt(const WhileStmt &S);
  void EmitDoStmt(const DoStmt &S);
  void EmitForStmt(const ForStmt &S);
  void EmitReturnStmt(const ReturnStmt &S);
  void EmitDeclStmt(const DeclStmt &S);
  void EmitBreakStmt(const BreakStmt &S);
  void EmitContinueStmt(const ContinueStmt &S);
  void EmitSwitchStmt(const SwitchStmt &S);
  void EmitDefaultStmt(const DefaultStmt &S);
  void EmitCaseStmt(const CaseStmt &S);
  void EmitCaseStmtRange(const CaseStmt &S);
  void EmitAsmStmt(const AsmStmt &S);

  void EmitObjCForCollectionStmt(const ObjCForCollectionStmt &S);
  void EmitObjCAtTryStmt(const ObjCAtTryStmt &S);
  void EmitObjCAtThrowStmt(const ObjCAtThrowStmt &S);
  void EmitObjCAtSynchronizedStmt(const ObjCAtSynchronizedStmt &S);

  //===--------------------------------------------------------------------===//
  //                         LValue Expression Emission
  //===--------------------------------------------------------------------===//

  /// GetUndefRValue - Get an appropriate 'undef' rvalue for the given type.
  RValue GetUndefRValue(QualType Ty);

  /// EmitUnsupportedRValue - Emit a dummy r-value using the type of E
  /// and issue an ErrorUnsupported style diagnostic (using the
  /// provided Name).
  RValue EmitUnsupportedRValue(const Expr *E,
                               const char *Name);

  /// EmitUnsupportedLValue - Emit a dummy l-value using the type of E and issue
  /// an ErrorUnsupported style diagnostic (using the provided Name).
  LValue EmitUnsupportedLValue(const Expr *E,
                               const char *Name);

  /// EmitLValue - Emit code to compute a designator that specifies the location
  /// of the expression.
  ///
  /// This can return one of two things: a simple address or a bitfield
  /// reference.  In either case, the LLVM Value* in the LValue structure is
  /// guaranteed to be an LLVM pointer type.
  ///
  /// If this returns a bitfield reference, nothing about the pointee type of
  /// the LLVM value is known: For example, it may not be a pointer to an
  /// integer.
  ///
  /// If this returns a normal address, and if the lvalue's C type is fixed
  /// size, this method guarantees that the returned pointer type will point to
  /// an LLVM type of the same size of the lvalue's type.  If the lvalue has a
  /// variable length type, this is not possible.
  ///
  LValue EmitLValue(const Expr *E);

  /// EmitLoadOfScalar - Load a scalar value from an address, taking
  /// care to appropriately convert from the memory representation to
  /// the LLVM value representation.
  llvm::Value *EmitLoadOfScalar(llvm::Value *Addr, bool Volatile,
                                QualType Ty);

  /// EmitStoreOfScalar - Store a scalar value to an address, taking
  /// care to appropriately convert from the memory representation to
  /// the LLVM value representation.
  void EmitStoreOfScalar(llvm::Value *Value, llvm::Value *Addr,
                         bool Volatile, QualType Ty);

  /// EmitLoadOfLValue - Given an expression that represents a value lvalue,
  /// this method emits the address of the lvalue, then loads the result as an
  /// rvalue, returning the rvalue.
  RValue EmitLoadOfLValue(LValue V, QualType LVType);
  RValue EmitLoadOfExtVectorElementLValue(LValue V, QualType LVType);
  RValue EmitLoadOfBitfieldLValue(LValue LV, QualType ExprType);
  RValue EmitLoadOfPropertyRefLValue(LValue LV, QualType ExprType);
  RValue EmitLoadOfKVCRefLValue(LValue LV, QualType ExprType);


  /// EmitStoreThroughLValue - Store the specified rvalue into the specified
  /// lvalue, where both are guaranteed to the have the same type, and that type
  /// is 'Ty'.
  void EmitStoreThroughLValue(RValue Src, LValue Dst, QualType Ty);
  void EmitStoreThroughExtVectorComponentLValue(RValue Src, LValue Dst,
                                                QualType Ty);
  void EmitStoreThroughPropertyRefLValue(RValue Src, LValue Dst, QualType Ty);
  void EmitStoreThroughKVCRefLValue(RValue Src, LValue Dst, QualType Ty);

  /// EmitStoreThroughLValue - Store Src into Dst with same constraints as
  /// EmitStoreThroughLValue.
  ///
  /// \param Result [out] - If non-null, this will be set to a Value* for the
  /// bit-field contents after the store, appropriate for use as the result of
  /// an assignment to the bit-field.
  void EmitStoreThroughBitfieldLValue(RValue Src, LValue Dst, QualType Ty,
                                      llvm::Value **Result=0);

  // Note: only availabe for agg return types
  LValue EmitBinaryOperatorLValue(const BinaryOperator *E);
  // Note: only available for agg return types
  LValue EmitCallExprLValue(const CallExpr *E);
  // Note: only available for agg return types
  LValue EmitVAArgExprLValue(const VAArgExpr *E);
  LValue EmitDeclRefLValue(const DeclRefExpr *E);
  LValue EmitStringLiteralLValue(const StringLiteral *E);
  LValue EmitObjCEncodeExprLValue(const ObjCEncodeExpr *E);
  LValue EmitPredefinedFunctionName(unsigned Type);
  LValue EmitPredefinedLValue(const PredefinedExpr *E);
  LValue EmitUnaryOpLValue(const UnaryOperator *E);
  LValue EmitArraySubscriptExpr(const ArraySubscriptExpr *E);
  LValue EmitExtVectorElementExpr(const ExtVectorElementExpr *E);
  LValue EmitMemberExpr(const MemberExpr *E);
  LValue EmitCompoundLiteralLValue(const CompoundLiteralExpr *E);
  LValue EmitConditionalOperatorLValue(const ConditionalOperator *E);
  LValue EmitCastLValue(const CastExpr *E);

  llvm::Value *EmitIvarOffset(const ObjCInterfaceDecl *Interface,
                              const ObjCIvarDecl *Ivar);
  LValue EmitLValueForField(llvm::Value* Base, FieldDecl* Field,
                            bool isUnion, unsigned CVRQualifiers);
  LValue EmitLValueForIvar(QualType ObjectTy,
                           llvm::Value* Base, const ObjCIvarDecl *Ivar,
                           unsigned CVRQualifiers);

  LValue EmitLValueForBitfield(llvm::Value* Base, FieldDecl* Field,
                                unsigned CVRQualifiers);

  LValue EmitBlockDeclRefLValue(const BlockDeclRefExpr *E);

  LValue EmitCXXConditionDeclLValue(const CXXConditionDeclExpr *E);
  LValue EmitCXXConstructLValue(const CXXConstructExpr *E);
  LValue EmitCXXBindTemporaryLValue(const CXXBindTemporaryExpr *E);
  LValue EmitCXXExprWithTemporariesLValue(const CXXExprWithTemporaries *E);
  
  LValue EmitObjCMessageExprLValue(const ObjCMessageExpr *E);
  LValue EmitObjCIvarRefLValue(const ObjCIvarRefExpr *E);
  LValue EmitObjCPropertyRefLValue(const ObjCPropertyRefExpr *E);
  LValue EmitObjCKVCRefLValue(const ObjCImplicitSetterGetterRefExpr *E);
  LValue EmitObjCSuperExprLValue(const ObjCSuperExpr *E);
  LValue EmitStmtExprLValue(const StmtExpr *E);

  //===--------------------------------------------------------------------===//
  //                         Scalar Expression Emission
  //===--------------------------------------------------------------------===//

  /// EmitCall - Generate a call of the given function, expecting the given
  /// result type, and using the given argument list which specifies both the
  /// LLVM arguments and the types they were derived from.
  ///
  /// \param TargetDecl - If given, the decl of the function in a
  /// direct call; used to set attributes on the call (noreturn,
  /// etc.).
  RValue EmitCall(const CGFunctionInfo &FnInfo,
                  llvm::Value *Callee,
                  const CallArgList &Args,
                  const Decl *TargetDecl = 0);

  RValue EmitCall(llvm::Value *Callee, QualType FnType,
                  CallExpr::const_arg_iterator ArgBeg,
                  CallExpr::const_arg_iterator ArgEnd,
                  const Decl *TargetDecl = 0);
  RValue EmitCallExpr(const CallExpr *E);

  llvm::Value *BuildVirtualCall(const CXXMethodDecl *MD, llvm::Value *&This,
                                const llvm::Type *Ty);
  RValue EmitCXXMemberCall(const CXXMethodDecl *MD,
                           llvm::Value *Callee,
                           llvm::Value *This,
                           CallExpr::const_arg_iterator ArgBeg,
                           CallExpr::const_arg_iterator ArgEnd);
  RValue EmitCXXMemberCallExpr(const CXXMemberCallExpr *E);

  RValue EmitCXXOperatorMemberCallExpr(const CXXOperatorCallExpr *E,
                                       const CXXMethodDecl *MD);

  RValue EmitBuiltinExpr(const FunctionDecl *FD,
                         unsigned BuiltinID, const CallExpr *E);

  RValue EmitBlockCallExpr(const CallExpr *E);

  /// EmitTargetBuiltinExpr - Emit the given builtin call. Returns 0 if the call
  /// is unhandled by the current target.
  llvm::Value *EmitTargetBuiltinExpr(unsigned BuiltinID, const CallExpr *E);

  llvm::Value *EmitX86BuiltinExpr(unsigned BuiltinID, const CallExpr *E);
  llvm::Value *EmitPPCBuiltinExpr(unsigned BuiltinID, const CallExpr *E);

  llvm::Value *EmitShuffleVector(llvm::Value* V1, llvm::Value *V2, ...);
  llvm::Value *EmitVector(llvm::Value * const *Vals, unsigned NumVals,
                          bool isSplat = false);

  llvm::Value *EmitObjCProtocolExpr(const ObjCProtocolExpr *E);
  llvm::Value *EmitObjCStringLiteral(const ObjCStringLiteral *E);
  llvm::Value *EmitObjCSelectorExpr(const ObjCSelectorExpr *E);
  RValue EmitObjCMessageExpr(const ObjCMessageExpr *E);
  RValue EmitObjCPropertyGet(const Expr *E);
  RValue EmitObjCSuperPropertyGet(const Expr *Exp, const Selector &S);
  void EmitObjCPropertySet(const Expr *E, RValue Src);
  void EmitObjCSuperPropertySet(const Expr *E, const Selector &S, RValue Src);


  /// EmitReferenceBindingToExpr - Emits a reference binding to the passed in
  /// expression. Will emit a temporary variable if E is not an LValue.
  RValue EmitReferenceBindingToExpr(const Expr* E, QualType DestType,
                                    bool IsInitializer = false);

  //===--------------------------------------------------------------------===//
  //                           Expression Emission
  //===--------------------------------------------------------------------===//

  // Expressions are broken into three classes: scalar, complex, aggregate.

  /// EmitScalarExpr - Emit the computation of the specified expression of LLVM
  /// scalar type, returning the result.
  llvm::Value *EmitScalarExpr(const Expr *E , bool IgnoreResultAssign = false);

  /// EmitScalarConversion - Emit a conversion from the specified type to the
  /// specified destination type, both of which are LLVM scalar types.
  llvm::Value *EmitScalarConversion(llvm::Value *Src, QualType SrcTy,
                                    QualType DstTy);

  /// EmitComplexToScalarConversion - Emit a conversion from the specified
  /// complex type to the specified destination type, where the destination type
  /// is an LLVM scalar type.
  llvm::Value *EmitComplexToScalarConversion(ComplexPairTy Src, QualType SrcTy,
                                             QualType DstTy);


  /// EmitAggExpr - Emit the computation of the specified expression of
  /// aggregate type.  The result is computed into DestPtr.  Note that if
  /// DestPtr is null, the value of the aggregate expression is not needed.
  void EmitAggExpr(const Expr *E, llvm::Value *DestPtr, bool VolatileDest,
                   bool IgnoreResult = false, bool IsInitializer = false,
                   bool RequiresGCollection = false);

  /// EmitGCMemmoveCollectable - Emit special API for structs with object
  /// pointers.
  void EmitGCMemmoveCollectable(llvm::Value *DestPtr, llvm::Value *SrcPtr,
                                QualType Ty);

  /// EmitComplexExpr - Emit the computation of the specified expression of
  /// complex type, returning the result.
  ComplexPairTy EmitComplexExpr(const Expr *E, bool IgnoreReal = false,
                                bool IgnoreImag = false,
                                bool IgnoreRealAssign = false,
                                bool IgnoreImagAssign = false);

  /// EmitComplexExprIntoAddr - Emit the computation of the specified expression
  /// of complex type, storing into the specified Value*.
  void EmitComplexExprIntoAddr(const Expr *E, llvm::Value *DestAddr,
                               bool DestIsVolatile);

  /// StoreComplexToAddr - Store a complex number into the specified address.
  void StoreComplexToAddr(ComplexPairTy V, llvm::Value *DestAddr,
                          bool DestIsVolatile);
  /// LoadComplexFromAddr - Load a complex number from the specified address.
  ComplexPairTy LoadComplexFromAddr(llvm::Value *SrcAddr, bool SrcIsVolatile);

  /// CreateStaticBlockVarDecl - Create a zero-initialized LLVM global
  /// for a static block var decl.
  llvm::GlobalVariable * CreateStaticBlockVarDecl(const VarDecl &D,
                                                  const char *Separator,
                                                  llvm::GlobalValue::LinkageTypes
                                                  Linkage);

  /// EmitStaticCXXBlockVarDeclInit - Create the initializer for a C++
  /// runtime initialized static block var decl.
  void EmitStaticCXXBlockVarDeclInit(const VarDecl &D,
                                     llvm::GlobalVariable *GV);

  /// EmitCXXGlobalVarDeclInit - Create the initializer for a C++
  /// variable with global storage.
  void EmitCXXGlobalVarDeclInit(const VarDecl &D, llvm::Constant *DeclPtr);

  /// EmitCXXGlobalDtorRegistration - Emits a call to register the global ptr
  /// with the C++ runtime so that its destructor will be called at exit.
  void EmitCXXGlobalDtorRegistration(const CXXDestructorDecl *Dtor,
                                     llvm::Constant *DeclPtr);

  /// GenerateCXXGlobalInitFunc - Generates code for initializing global
  /// variables.
  void GenerateCXXGlobalInitFunc(llvm::Function *Fn,
                                 const VarDecl **Decls,
                                 unsigned NumDecls);

  void EmitCXXConstructExpr(llvm::Value *Dest, const CXXConstructExpr *E);

  RValue EmitCXXExprWithTemporaries(const CXXExprWithTemporaries *E,
                                    llvm::Value *AggLoc = 0,
                                    bool IsAggLocVolatile = false,
                                    bool IsInitializer = false);

  //===--------------------------------------------------------------------===//
  //                             Internal Helpers
  //===--------------------------------------------------------------------===//

  /// ContainsLabel - Return true if the statement contains a label in it.  If
  /// this statement is not executed normally, it not containing a label means
  /// that we can just remove the code.
  static bool ContainsLabel(const Stmt *S, bool IgnoreCaseStmts = false);

  /// ConstantFoldsToSimpleInteger - If the specified expression does not fold
  /// to a constant, or if it does but contains a label, return 0.  If it
  /// constant folds to 'true' and does not contain a label, return 1, if it
  /// constant folds to 'false' and does not contain a label, return -1.
  int ConstantFoldsToSimpleInteger(const Expr *Cond);

  /// EmitBranchOnBoolExpr - Emit a branch on a boolean condition (e.g. for an
  /// if statement) to the specified blocks.  Based on the condition, this might
  /// try to simplify the codegen of the conditional based on the branch.
  void EmitBranchOnBoolExpr(const Expr *Cond, llvm::BasicBlock *TrueBlock,
                            llvm::BasicBlock *FalseBlock);
private:

  /// EmitIndirectSwitches - Emit code for all of the switch
  /// instructions in IndirectSwitches.
  void EmitIndirectSwitches();

  void EmitReturnOfRValue(RValue RV, QualType Ty);

  /// ExpandTypeFromArgs - Reconstruct a structure of type \arg Ty
  /// from function arguments into \arg Dst. See ABIArgInfo::Expand.
  ///
  /// \param AI - The first function argument of the expansion.
  /// \return The argument following the last expanded function
  /// argument.
  llvm::Function::arg_iterator
  ExpandTypeFromArgs(QualType Ty, LValue Dst,
                     llvm::Function::arg_iterator AI);

  /// ExpandTypeToArgs - Expand an RValue \arg Src, with the LLVM type for \arg
  /// Ty, into individual arguments on the provided vector \arg Args. See
  /// ABIArgInfo::Expand.
  void ExpandTypeToArgs(QualType Ty, RValue Src,
                        llvm::SmallVector<llvm::Value*, 16> &Args);

  llvm::Value* EmitAsmInput(const AsmStmt &S,
                            const TargetInfo::ConstraintInfo &Info,
                            const Expr *InputExpr, std::string &ConstraintStr);

  /// EmitCleanupBlock - emits a single cleanup block.
  void EmitCleanupBlock();

  /// AddBranchFixup - adds a branch instruction to the list of fixups for the
  /// current cleanup scope.
  void AddBranchFixup(llvm::BranchInst *BI);

  /// EmitCallArg - Emit a single call argument.
  RValue EmitCallArg(const Expr *E, QualType ArgType);

  /// EmitCallArgs - Emit call arguments for a function.
  /// The CallArgTypeInfo parameter is used for iterating over the known
  /// argument types of the function being called.
  template<typename T>
  void EmitCallArgs(CallArgList& Args, const T* CallArgTypeInfo,
                    CallExpr::const_arg_iterator ArgBeg,
                    CallExpr::const_arg_iterator ArgEnd) {
      CallExpr::const_arg_iterator Arg = ArgBeg;

    // First, use the argument types that the type info knows about
    if (CallArgTypeInfo) {
      for (typename T::arg_type_iterator I = CallArgTypeInfo->arg_type_begin(),
           E = CallArgTypeInfo->arg_type_end(); I != E; ++I, ++Arg) {
        QualType ArgType = *I;

        assert(getContext().getCanonicalType(ArgType.getNonReferenceType()).
               getTypePtr() ==
               getContext().getCanonicalType(Arg->getType()).getTypePtr() &&
               "type mismatch in call argument!");

        Args.push_back(std::make_pair(EmitCallArg(*Arg, ArgType),
                                      ArgType));
      }

      // Either we've emitted all the call args, or we have a call to a
      // variadic function.
      assert((Arg == ArgEnd || CallArgTypeInfo->isVariadic()) &&
             "Extra arguments in non-variadic function!");

    }

    // If we still have any arguments, emit them using the type of the argument.
    for (; Arg != ArgEnd; ++Arg) {
      QualType ArgType = Arg->getType();
      Args.push_back(std::make_pair(EmitCallArg(*Arg, ArgType),
                                    ArgType));
    }
  }
};


}  // end namespace CodeGen
}  // end namespace clang

#endif
