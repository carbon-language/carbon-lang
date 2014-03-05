//===-- Verifier.cpp - Implement the Module Verifier -----------------------==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the function verifier interface, that can be used for some
// sanity checking of input to the system.
//
// Note that this does not provide full `Java style' security and verifications,
// instead it just tries to ensure that code is well-formed.
//
//  * Both of a binary operator's parameters are of the same type
//  * Verify that the indices of mem access instructions match other operands
//  * Verify that arithmetic and other things are only performed on first-class
//    types.  Verify that shifts & logicals only happen on integrals f.e.
//  * All of the constants in a switch statement are of the correct type
//  * The code is in valid SSA form
//  * It should be illegal to put a label into any other type (like a structure)
//    or to return one. [except constant arrays!]
//  * Only phi nodes can be self referential: 'add i32 %0, %0 ; <int>:0' is bad
//  * PHI nodes must have an entry for each predecessor, with no extras.
//  * PHI nodes must be the first thing in a basic block, all grouped together
//  * PHI nodes must have at least one entry
//  * All basic blocks should only end with terminator insts, not contain them
//  * The entry node to a function must not have predecessors
//  * All Instructions must be embedded into a basic block
//  * Functions cannot take a void-typed parameter
//  * Verify that a function's argument list agrees with it's declared type.
//  * It is illegal to specify a name for a void value.
//  * It is illegal to have a internal global value with no initializer
//  * It is illegal to have a ret instruction that returns a value that does not
//    agree with the function return value type.
//  * Function call argument types match the function prototype
//  * A landing pad is defined by a landingpad instruction, and can be jumped to
//    only by the unwind edge of an invoke instruction.
//  * A landingpad instruction must be the first non-PHI instruction in the
//    block.
//  * All landingpad instructions must use the same personality function with
//    the same function.
//  * All other things that are tested by asserts spread about the code...
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Verifier.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/DebugInfo.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/ConstantRange.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/InstVisitor.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstdarg>
using namespace llvm;

static cl::opt<bool> DisableDebugInfoVerifier("disable-debug-info-verifier",
                                              cl::init(true));

namespace {
class Verifier : public InstVisitor<Verifier> {
  friend class InstVisitor<Verifier>;

  raw_ostream &OS;
  const Module *M;
  LLVMContext *Context;
  const DataLayout *DL;
  DominatorTree DT;

  /// \brief When verifying a basic block, keep track of all of the
  /// instructions we have seen so far.
  ///
  /// This allows us to do efficient dominance checks for the case when an
  /// instruction has an operand that is an instruction in the same block.
  SmallPtrSet<Instruction *, 16> InstsInThisBlock;

  /// \brief Keep track of the metadata nodes that have been checked already.
  SmallPtrSet<MDNode *, 32> MDNodes;

  /// \brief The personality function referenced by the LandingPadInsts.
  /// All LandingPadInsts within the same function must use the same
  /// personality function.
  const Value *PersonalityFn;

  /// \brief Finder keeps track of all debug info MDNodes in a Module.
  DebugInfoFinder Finder;

  /// \brief Track the brokenness of the module while recursively visiting.
  bool Broken;

public:
  explicit Verifier(raw_ostream &OS = dbgs())
      : OS(OS), M(0), Context(0), DL(0), PersonalityFn(0), Broken(false) {}

  bool verify(const Function &F) {
    M = F.getParent();
    Context = &M->getContext();

    // First ensure the function is well-enough formed to compute dominance
    // information.
    if (F.empty()) {
      OS << "Function '" << F.getName()
         << "' does not contain an entry block!\n";
      return false;
    }
    for (Function::const_iterator I = F.begin(), E = F.end(); I != E; ++I) {
      if (I->empty() || !I->back().isTerminator()) {
        OS << "Basic Block in function '" << F.getName()
           << "' does not have terminator!\n";
        I->printAsOperand(OS, true);
        OS << "\n";
        return false;
      }
    }

    // Now directly compute a dominance tree. We don't rely on the pass
    // manager to provide this as it isolates us from a potentially
    // out-of-date dominator tree and makes it significantly more complex to
    // run this code outside of a pass manager.
    // FIXME: It's really gross that we have to cast away constness here.
    DT.recalculate(const_cast<Function &>(F));

    Finder.reset();
    Broken = false;
    // FIXME: We strip const here because the inst visitor strips const.
    visit(const_cast<Function &>(F));
    InstsInThisBlock.clear();
    PersonalityFn = 0;

    if (!DisableDebugInfoVerifier)
      // Verify Debug Info.
      verifyDebugInfo();

    return !Broken;
  }

  bool verify(const Module &M) {
    this->M = &M;
    Context = &M.getContext();
    Finder.reset();
    Broken = false;

    // Scan through, checking all of the external function's linkage now...
    for (Module::const_iterator I = M.begin(), E = M.end(); I != E; ++I) {
      visitGlobalValue(*I);

      // Check to make sure function prototypes are okay.
      if (I->isDeclaration())
        visitFunction(*I);
    }

    for (Module::const_global_iterator I = M.global_begin(), E = M.global_end();
         I != E; ++I)
      visitGlobalVariable(*I);

    for (Module::const_alias_iterator I = M.alias_begin(), E = M.alias_end();
         I != E; ++I)
      visitGlobalAlias(*I);

    for (Module::const_named_metadata_iterator I = M.named_metadata_begin(),
                                               E = M.named_metadata_end();
         I != E; ++I)
      visitNamedMDNode(*I);

    visitModuleFlags(M);
    visitModuleIdents(M);

    if (!DisableDebugInfoVerifier) {
      Finder.reset();
      Finder.processModule(M);
      // Verify Debug Info.
      verifyDebugInfo();
    }

    return !Broken;
  }

private:
  // Verification methods...
  void visitGlobalValue(const GlobalValue &GV);
  void visitGlobalVariable(const GlobalVariable &GV);
  void visitGlobalAlias(const GlobalAlias &GA);
  void visitNamedMDNode(const NamedMDNode &NMD);
  void visitMDNode(MDNode &MD, Function *F);
  void visitModuleIdents(const Module &M);
  void visitModuleFlags(const Module &M);
  void visitModuleFlag(const MDNode *Op,
                       DenseMap<const MDString *, const MDNode *> &SeenIDs,
                       SmallVectorImpl<const MDNode *> &Requirements);
  void visitFunction(const Function &F);
  void visitBasicBlock(BasicBlock &BB);

  // InstVisitor overrides...
  using InstVisitor<Verifier>::visit;
  void visit(Instruction &I);

  void visitTruncInst(TruncInst &I);
  void visitZExtInst(ZExtInst &I);
  void visitSExtInst(SExtInst &I);
  void visitFPTruncInst(FPTruncInst &I);
  void visitFPExtInst(FPExtInst &I);
  void visitFPToUIInst(FPToUIInst &I);
  void visitFPToSIInst(FPToSIInst &I);
  void visitUIToFPInst(UIToFPInst &I);
  void visitSIToFPInst(SIToFPInst &I);
  void visitIntToPtrInst(IntToPtrInst &I);
  void visitPtrToIntInst(PtrToIntInst &I);
  void visitBitCastInst(BitCastInst &I);
  void visitAddrSpaceCastInst(AddrSpaceCastInst &I);
  void visitPHINode(PHINode &PN);
  void visitBinaryOperator(BinaryOperator &B);
  void visitICmpInst(ICmpInst &IC);
  void visitFCmpInst(FCmpInst &FC);
  void visitExtractElementInst(ExtractElementInst &EI);
  void visitInsertElementInst(InsertElementInst &EI);
  void visitShuffleVectorInst(ShuffleVectorInst &EI);
  void visitVAArgInst(VAArgInst &VAA) { visitInstruction(VAA); }
  void visitCallInst(CallInst &CI);
  void visitInvokeInst(InvokeInst &II);
  void visitGetElementPtrInst(GetElementPtrInst &GEP);
  void visitLoadInst(LoadInst &LI);
  void visitStoreInst(StoreInst &SI);
  void verifyDominatesUse(Instruction &I, unsigned i);
  void visitInstruction(Instruction &I);
  void visitTerminatorInst(TerminatorInst &I);
  void visitBranchInst(BranchInst &BI);
  void visitReturnInst(ReturnInst &RI);
  void visitSwitchInst(SwitchInst &SI);
  void visitIndirectBrInst(IndirectBrInst &BI);
  void visitSelectInst(SelectInst &SI);
  void visitUserOp1(Instruction &I);
  void visitUserOp2(Instruction &I) { visitUserOp1(I); }
  void visitIntrinsicFunctionCall(Intrinsic::ID ID, CallInst &CI);
  void visitAtomicCmpXchgInst(AtomicCmpXchgInst &CXI);
  void visitAtomicRMWInst(AtomicRMWInst &RMWI);
  void visitFenceInst(FenceInst &FI);
  void visitAllocaInst(AllocaInst &AI);
  void visitExtractValueInst(ExtractValueInst &EVI);
  void visitInsertValueInst(InsertValueInst &IVI);
  void visitLandingPadInst(LandingPadInst &LPI);

  void VerifyCallSite(CallSite CS);
  bool PerformTypeCheck(Intrinsic::ID ID, Function *F, Type *Ty, int VT,
                        unsigned ArgNo, std::string &Suffix);
  bool VerifyIntrinsicType(Type *Ty, ArrayRef<Intrinsic::IITDescriptor> &Infos,
                           SmallVectorImpl<Type *> &ArgTys);
  bool VerifyIntrinsicIsVarArg(bool isVarArg,
                               ArrayRef<Intrinsic::IITDescriptor> &Infos);
  bool VerifyAttributeCount(AttributeSet Attrs, unsigned Params);
  void VerifyAttributeTypes(AttributeSet Attrs, unsigned Idx, bool isFunction,
                            const Value *V);
  void VerifyParameterAttrs(AttributeSet Attrs, unsigned Idx, Type *Ty,
                            bool isReturnValue, const Value *V);
  void VerifyFunctionAttrs(FunctionType *FT, AttributeSet Attrs,
                           const Value *V);

  void VerifyBitcastType(const Value *V, Type *DestTy, Type *SrcTy);
  void VerifyConstantExprBitcastType(const ConstantExpr *CE);

  void verifyDebugInfo();

  void WriteValue(const Value *V) {
    if (!V)
      return;
    if (isa<Instruction>(V)) {
      OS << *V << '\n';
    } else {
      V->printAsOperand(OS, true, M);
      OS << '\n';
    }
  }

  void WriteType(Type *T) {
    if (!T)
      return;
    OS << ' ' << *T;
  }

  // CheckFailed - A check failed, so print out the condition and the message
  // that failed.  This provides a nice place to put a breakpoint if you want
  // to see why something is not correct.
  void CheckFailed(const Twine &Message, const Value *V1 = 0,
                   const Value *V2 = 0, const Value *V3 = 0,
                   const Value *V4 = 0) {
    OS << Message.str() << "\n";
    WriteValue(V1);
    WriteValue(V2);
    WriteValue(V3);
    WriteValue(V4);
    Broken = true;
  }

  void CheckFailed(const Twine &Message, const Value *V1, Type *T2,
                   const Value *V3 = 0) {
    OS << Message.str() << "\n";
    WriteValue(V1);
    WriteType(T2);
    WriteValue(V3);
    Broken = true;
  }

  void CheckFailed(const Twine &Message, Type *T1, Type *T2 = 0, Type *T3 = 0) {
    OS << Message.str() << "\n";
    WriteType(T1);
    WriteType(T2);
    WriteType(T3);
    Broken = true;
  }
};
} // End anonymous namespace

// Assert - We know that cond should be true, if not print an error message.
#define Assert(C, M) \
  do { if (!(C)) { CheckFailed(M); return; } } while (0)
#define Assert1(C, M, V1) \
  do { if (!(C)) { CheckFailed(M, V1); return; } } while (0)
#define Assert2(C, M, V1, V2) \
  do { if (!(C)) { CheckFailed(M, V1, V2); return; } } while (0)
#define Assert3(C, M, V1, V2, V3) \
  do { if (!(C)) { CheckFailed(M, V1, V2, V3); return; } } while (0)
#define Assert4(C, M, V1, V2, V3, V4) \
  do { if (!(C)) { CheckFailed(M, V1, V2, V3, V4); return; } } while (0)

void Verifier::visit(Instruction &I) {
  for (unsigned i = 0, e = I.getNumOperands(); i != e; ++i)
    Assert1(I.getOperand(i) != 0, "Operand is null", &I);
  InstVisitor<Verifier>::visit(I);
}


void Verifier::visitGlobalValue(const GlobalValue &GV) {
  Assert1(!GV.isDeclaration() ||
          GV.isMaterializable() ||
          GV.hasExternalLinkage() ||
          GV.hasExternalWeakLinkage() ||
          (isa<GlobalAlias>(GV) &&
           (GV.hasLocalLinkage() || GV.hasWeakLinkage())),
          "Global is external, but doesn't have external or weak linkage!",
          &GV);

  Assert1(!GV.hasAppendingLinkage() || isa<GlobalVariable>(GV),
          "Only global variables can have appending linkage!", &GV);

  if (GV.hasAppendingLinkage()) {
    const GlobalVariable *GVar = dyn_cast<GlobalVariable>(&GV);
    Assert1(GVar && GVar->getType()->getElementType()->isArrayTy(),
            "Only global arrays can have appending linkage!", GVar);
  }
}

void Verifier::visitGlobalVariable(const GlobalVariable &GV) {
  if (GV.hasInitializer()) {
    Assert1(GV.getInitializer()->getType() == GV.getType()->getElementType(),
            "Global variable initializer type does not match global "
            "variable type!", &GV);

    // If the global has common linkage, it must have a zero initializer and
    // cannot be constant.
    if (GV.hasCommonLinkage()) {
      Assert1(GV.getInitializer()->isNullValue(),
              "'common' global must have a zero initializer!", &GV);
      Assert1(!GV.isConstant(), "'common' global may not be marked constant!",
              &GV);
    }
  } else {
    Assert1(GV.hasExternalLinkage() || GV.hasExternalWeakLinkage(),
            "invalid linkage type for global declaration", &GV);
  }

  if (GV.hasName() && (GV.getName() == "llvm.global_ctors" ||
                       GV.getName() == "llvm.global_dtors")) {
    Assert1(!GV.hasInitializer() || GV.hasAppendingLinkage(),
            "invalid linkage for intrinsic global variable", &GV);
    // Don't worry about emitting an error for it not being an array,
    // visitGlobalValue will complain on appending non-array.
    if (ArrayType *ATy = dyn_cast<ArrayType>(GV.getType())) {
      StructType *STy = dyn_cast<StructType>(ATy->getElementType());
      PointerType *FuncPtrTy =
          FunctionType::get(Type::getVoidTy(*Context), false)->getPointerTo();
      Assert1(STy && STy->getNumElements() == 2 &&
              STy->getTypeAtIndex(0u)->isIntegerTy(32) &&
              STy->getTypeAtIndex(1) == FuncPtrTy,
              "wrong type for intrinsic global variable", &GV);
    }
  }

  if (GV.hasName() && (GV.getName() == "llvm.used" ||
                       GV.getName() == "llvm.compiler.used")) {
    Assert1(!GV.hasInitializer() || GV.hasAppendingLinkage(),
            "invalid linkage for intrinsic global variable", &GV);
    Type *GVType = GV.getType()->getElementType();
    if (ArrayType *ATy = dyn_cast<ArrayType>(GVType)) {
      PointerType *PTy = dyn_cast<PointerType>(ATy->getElementType());
      Assert1(PTy, "wrong type for intrinsic global variable", &GV);
      if (GV.hasInitializer()) {
        const Constant *Init = GV.getInitializer();
        const ConstantArray *InitArray = dyn_cast<ConstantArray>(Init);
        Assert1(InitArray, "wrong initalizer for intrinsic global variable",
                Init);
        for (unsigned i = 0, e = InitArray->getNumOperands(); i != e; ++i) {
          Value *V = Init->getOperand(i)->stripPointerCastsNoFollowAliases();
          Assert1(
              isa<GlobalVariable>(V) || isa<Function>(V) || isa<GlobalAlias>(V),
              "invalid llvm.used member", V);
          Assert1(V->hasName(), "members of llvm.used must be named", V);
        }
      }
    }
  }

  Assert1(!GV.hasDLLImportStorageClass() ||
          (GV.isDeclaration() && GV.hasExternalLinkage()) ||
          GV.hasAvailableExternallyLinkage(),
          "Global is marked as dllimport, but not external", &GV);

  if (!GV.hasInitializer()) {
    visitGlobalValue(GV);
    return;
  }

  // Walk any aggregate initializers looking for bitcasts between address spaces
  SmallPtrSet<const Value *, 4> Visited;
  SmallVector<const Value *, 4> WorkStack;
  WorkStack.push_back(cast<Value>(GV.getInitializer()));

  while (!WorkStack.empty()) {
    const Value *V = WorkStack.pop_back_val();
    if (!Visited.insert(V))
      continue;

    if (const User *U = dyn_cast<User>(V)) {
      for (unsigned I = 0, N = U->getNumOperands(); I != N; ++I)
        WorkStack.push_back(U->getOperand(I));
    }

    if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(V)) {
      VerifyConstantExprBitcastType(CE);
      if (Broken)
        return;
    }
  }

  visitGlobalValue(GV);
}

void Verifier::visitGlobalAlias(const GlobalAlias &GA) {
  Assert1(!GA.getName().empty(),
          "Alias name cannot be empty!", &GA);
  Assert1(GlobalAlias::isValidLinkage(GA.getLinkage()),
          "Alias should have external or external weak linkage!", &GA);
  Assert1(GA.getAliasee(),
          "Aliasee cannot be NULL!", &GA);
  Assert1(GA.getType() == GA.getAliasee()->getType(),
          "Alias and aliasee types should match!", &GA);
  Assert1(!GA.hasUnnamedAddr(), "Alias cannot have unnamed_addr!", &GA);
  Assert1(!GA.hasSection(), "Alias cannot have a section!", &GA);
  Assert1(!GA.getAlignment(), "Alias connot have an alignment", &GA);

  const Constant *Aliasee = GA.getAliasee();

  if (!isa<GlobalValue>(Aliasee)) {
    const ConstantExpr *CE = dyn_cast<ConstantExpr>(Aliasee);
    Assert1(CE &&
            (CE->getOpcode() == Instruction::BitCast ||
             CE->getOpcode() == Instruction::AddrSpaceCast ||
             CE->getOpcode() == Instruction::GetElementPtr) &&
            isa<GlobalValue>(CE->getOperand(0)),
            "Aliasee should be either GlobalValue, bitcast or "
             "addrspacecast of GlobalValue",
            &GA);

    if (CE->getOpcode() == Instruction::BitCast) {
      unsigned SrcAS = CE->getOperand(0)->getType()->getPointerAddressSpace();
      unsigned DstAS = CE->getType()->getPointerAddressSpace();

      Assert1(SrcAS == DstAS,
              "Alias bitcasts cannot be between different address spaces",
              &GA);
    }
  }

  const GlobalValue* Resolved = GA.resolveAliasedGlobal(/*stopOnWeak*/ false);
  Assert1(Resolved,
          "Aliasing chain should end with function or global variable", &GA);

  visitGlobalValue(GA);
}

void Verifier::visitNamedMDNode(const NamedMDNode &NMD) {
  for (unsigned i = 0, e = NMD.getNumOperands(); i != e; ++i) {
    MDNode *MD = NMD.getOperand(i);
    if (!MD)
      continue;

    Assert1(!MD->isFunctionLocal(),
            "Named metadata operand cannot be function local!", MD);
    visitMDNode(*MD, 0);
  }
}

void Verifier::visitMDNode(MDNode &MD, Function *F) {
  // Only visit each node once.  Metadata can be mutually recursive, so this
  // avoids infinite recursion here, as well as being an optimization.
  if (!MDNodes.insert(&MD))
    return;

  for (unsigned i = 0, e = MD.getNumOperands(); i != e; ++i) {
    Value *Op = MD.getOperand(i);
    if (!Op)
      continue;
    if (isa<Constant>(Op) || isa<MDString>(Op))
      continue;
    if (MDNode *N = dyn_cast<MDNode>(Op)) {
      Assert2(MD.isFunctionLocal() || !N->isFunctionLocal(),
              "Global metadata operand cannot be function local!", &MD, N);
      visitMDNode(*N, F);
      continue;
    }
    Assert2(MD.isFunctionLocal(), "Invalid operand for global metadata!", &MD, Op);

    // If this was an instruction, bb, or argument, verify that it is in the
    // function that we expect.
    Function *ActualF = 0;
    if (Instruction *I = dyn_cast<Instruction>(Op))
      ActualF = I->getParent()->getParent();
    else if (BasicBlock *BB = dyn_cast<BasicBlock>(Op))
      ActualF = BB->getParent();
    else if (Argument *A = dyn_cast<Argument>(Op))
      ActualF = A->getParent();
    assert(ActualF && "Unimplemented function local metadata case!");

    Assert2(ActualF == F, "function-local metadata used in wrong function",
            &MD, Op);
  }
}

void Verifier::visitModuleIdents(const Module &M) {
  const NamedMDNode *Idents = M.getNamedMetadata("llvm.ident");
  if (!Idents) 
    return;
  
  // llvm.ident takes a list of metadata entry. Each entry has only one string.
  // Scan each llvm.ident entry and make sure that this requirement is met.
  for (unsigned i = 0, e = Idents->getNumOperands(); i != e; ++i) {
    const MDNode *N = Idents->getOperand(i);
    Assert1(N->getNumOperands() == 1,
            "incorrect number of operands in llvm.ident metadata", N);
    Assert1(isa<MDString>(N->getOperand(0)),
            ("invalid value for llvm.ident metadata entry operand"
             "(the operand should be a string)"),
            N->getOperand(0));
  } 
}

void Verifier::visitModuleFlags(const Module &M) {
  const NamedMDNode *Flags = M.getModuleFlagsMetadata();
  if (!Flags) return;

  // Scan each flag, and track the flags and requirements.
  DenseMap<const MDString*, const MDNode*> SeenIDs;
  SmallVector<const MDNode*, 16> Requirements;
  for (unsigned I = 0, E = Flags->getNumOperands(); I != E; ++I) {
    visitModuleFlag(Flags->getOperand(I), SeenIDs, Requirements);
  }

  // Validate that the requirements in the module are valid.
  for (unsigned I = 0, E = Requirements.size(); I != E; ++I) {
    const MDNode *Requirement = Requirements[I];
    const MDString *Flag = cast<MDString>(Requirement->getOperand(0));
    const Value *ReqValue = Requirement->getOperand(1);

    const MDNode *Op = SeenIDs.lookup(Flag);
    if (!Op) {
      CheckFailed("invalid requirement on flag, flag is not present in module",
                  Flag);
      continue;
    }

    if (Op->getOperand(2) != ReqValue) {
      CheckFailed(("invalid requirement on flag, "
                   "flag does not have the required value"),
                  Flag);
      continue;
    }
  }
}

void
Verifier::visitModuleFlag(const MDNode *Op,
                          DenseMap<const MDString *, const MDNode *> &SeenIDs,
                          SmallVectorImpl<const MDNode *> &Requirements) {
  // Each module flag should have three arguments, the merge behavior (a
  // constant int), the flag ID (an MDString), and the value.
  Assert1(Op->getNumOperands() == 3,
          "incorrect number of operands in module flag", Op);
  ConstantInt *Behavior = dyn_cast<ConstantInt>(Op->getOperand(0));
  MDString *ID = dyn_cast<MDString>(Op->getOperand(1));
  Assert1(Behavior,
          "invalid behavior operand in module flag (expected constant integer)",
          Op->getOperand(0));
  unsigned BehaviorValue = Behavior->getZExtValue();
  Assert1(ID,
          "invalid ID operand in module flag (expected metadata string)",
          Op->getOperand(1));

  // Sanity check the values for behaviors with additional requirements.
  switch (BehaviorValue) {
  default:
    Assert1(false,
            "invalid behavior operand in module flag (unexpected constant)",
            Op->getOperand(0));
    break;

  case Module::Error:
  case Module::Warning:
  case Module::Override:
    // These behavior types accept any value.
    break;

  case Module::Require: {
    // The value should itself be an MDNode with two operands, a flag ID (an
    // MDString), and a value.
    MDNode *Value = dyn_cast<MDNode>(Op->getOperand(2));
    Assert1(Value && Value->getNumOperands() == 2,
            "invalid value for 'require' module flag (expected metadata pair)",
            Op->getOperand(2));
    Assert1(isa<MDString>(Value->getOperand(0)),
            ("invalid value for 'require' module flag "
             "(first value operand should be a string)"),
            Value->getOperand(0));

    // Append it to the list of requirements, to check once all module flags are
    // scanned.
    Requirements.push_back(Value);
    break;
  }

  case Module::Append:
  case Module::AppendUnique: {
    // These behavior types require the operand be an MDNode.
    Assert1(isa<MDNode>(Op->getOperand(2)),
            "invalid value for 'append'-type module flag "
            "(expected a metadata node)", Op->getOperand(2));
    break;
  }
  }

  // Unless this is a "requires" flag, check the ID is unique.
  if (BehaviorValue != Module::Require) {
    bool Inserted = SeenIDs.insert(std::make_pair(ID, Op)).second;
    Assert1(Inserted,
            "module flag identifiers must be unique (or of 'require' type)",
            ID);
  }
}

void Verifier::VerifyAttributeTypes(AttributeSet Attrs, unsigned Idx,
                                    bool isFunction, const Value *V) {
  unsigned Slot = ~0U;
  for (unsigned I = 0, E = Attrs.getNumSlots(); I != E; ++I)
    if (Attrs.getSlotIndex(I) == Idx) {
      Slot = I;
      break;
    }

  assert(Slot != ~0U && "Attribute set inconsistency!");

  for (AttributeSet::iterator I = Attrs.begin(Slot), E = Attrs.end(Slot);
         I != E; ++I) {
    if (I->isStringAttribute())
      continue;

    if (I->getKindAsEnum() == Attribute::NoReturn ||
        I->getKindAsEnum() == Attribute::NoUnwind ||
        I->getKindAsEnum() == Attribute::NoInline ||
        I->getKindAsEnum() == Attribute::AlwaysInline ||
        I->getKindAsEnum() == Attribute::OptimizeForSize ||
        I->getKindAsEnum() == Attribute::StackProtect ||
        I->getKindAsEnum() == Attribute::StackProtectReq ||
        I->getKindAsEnum() == Attribute::StackProtectStrong ||
        I->getKindAsEnum() == Attribute::NoRedZone ||
        I->getKindAsEnum() == Attribute::NoImplicitFloat ||
        I->getKindAsEnum() == Attribute::Naked ||
        I->getKindAsEnum() == Attribute::InlineHint ||
        I->getKindAsEnum() == Attribute::StackAlignment ||
        I->getKindAsEnum() == Attribute::UWTable ||
        I->getKindAsEnum() == Attribute::NonLazyBind ||
        I->getKindAsEnum() == Attribute::ReturnsTwice ||
        I->getKindAsEnum() == Attribute::SanitizeAddress ||
        I->getKindAsEnum() == Attribute::SanitizeThread ||
        I->getKindAsEnum() == Attribute::SanitizeMemory ||
        I->getKindAsEnum() == Attribute::MinSize ||
        I->getKindAsEnum() == Attribute::NoDuplicate ||
        I->getKindAsEnum() == Attribute::Builtin ||
        I->getKindAsEnum() == Attribute::NoBuiltin ||
        I->getKindAsEnum() == Attribute::Cold ||
        I->getKindAsEnum() == Attribute::OptimizeNone) {
      if (!isFunction) {
        CheckFailed("Attribute '" + I->getAsString() +
                    "' only applies to functions!", V);
        return;
      }
    } else if (I->getKindAsEnum() == Attribute::ReadOnly ||
               I->getKindAsEnum() == Attribute::ReadNone) {
      if (Idx == 0) {
        CheckFailed("Attribute '" + I->getAsString() +
                    "' does not apply to function returns");
        return;
      }
    } else if (isFunction) {
      CheckFailed("Attribute '" + I->getAsString() +
                  "' does not apply to functions!", V);
      return;
    }
  }
}

// VerifyParameterAttrs - Check the given attributes for an argument or return
// value of the specified type.  The value V is printed in error messages.
void Verifier::VerifyParameterAttrs(AttributeSet Attrs, unsigned Idx, Type *Ty,
                                    bool isReturnValue, const Value *V) {
  if (!Attrs.hasAttributes(Idx))
    return;

  VerifyAttributeTypes(Attrs, Idx, false, V);

  if (isReturnValue)
    Assert1(!Attrs.hasAttribute(Idx, Attribute::ByVal) &&
            !Attrs.hasAttribute(Idx, Attribute::Nest) &&
            !Attrs.hasAttribute(Idx, Attribute::StructRet) &&
            !Attrs.hasAttribute(Idx, Attribute::NoCapture) &&
            !Attrs.hasAttribute(Idx, Attribute::Returned) &&
            !Attrs.hasAttribute(Idx, Attribute::InAlloca),
            "Attributes 'byval', 'inalloca', 'nest', 'sret', 'nocapture', and "
            "'returned' do not apply to return values!", V);

  // Check for mutually incompatible attributes.  Only inreg is compatible with
  // sret.
  unsigned AttrCount = 0;
  AttrCount += Attrs.hasAttribute(Idx, Attribute::ByVal);
  AttrCount += Attrs.hasAttribute(Idx, Attribute::InAlloca);
  AttrCount += Attrs.hasAttribute(Idx, Attribute::StructRet) ||
               Attrs.hasAttribute(Idx, Attribute::InReg);
  AttrCount += Attrs.hasAttribute(Idx, Attribute::Nest);
  Assert1(AttrCount <= 1, "Attributes 'byval', 'inalloca', 'inreg', 'nest', "
                          "and 'sret' are incompatible!", V);

  Assert1(!(Attrs.hasAttribute(Idx, Attribute::InAlloca) &&
            Attrs.hasAttribute(Idx, Attribute::ReadOnly)), "Attributes "
          "'inalloca and readonly' are incompatible!", V);

  Assert1(!(Attrs.hasAttribute(Idx, Attribute::StructRet) &&
            Attrs.hasAttribute(Idx, Attribute::Returned)), "Attributes "
          "'sret and returned' are incompatible!", V);

  Assert1(!(Attrs.hasAttribute(Idx, Attribute::ZExt) &&
            Attrs.hasAttribute(Idx, Attribute::SExt)), "Attributes "
          "'zeroext and signext' are incompatible!", V);

  Assert1(!(Attrs.hasAttribute(Idx, Attribute::ReadNone) &&
            Attrs.hasAttribute(Idx, Attribute::ReadOnly)), "Attributes "
          "'readnone and readonly' are incompatible!", V);

  Assert1(!(Attrs.hasAttribute(Idx, Attribute::NoInline) &&
            Attrs.hasAttribute(Idx, Attribute::AlwaysInline)), "Attributes "
          "'noinline and alwaysinline' are incompatible!", V);

  Assert1(!AttrBuilder(Attrs, Idx).
            hasAttributes(AttributeFuncs::typeIncompatible(Ty, Idx), Idx),
          "Wrong types for attribute: " +
          AttributeFuncs::typeIncompatible(Ty, Idx).getAsString(Idx), V);

  if (PointerType *PTy = dyn_cast<PointerType>(Ty)) {
    if (!PTy->getElementType()->isSized()) {
      Assert1(!Attrs.hasAttribute(Idx, Attribute::ByVal) &&
              !Attrs.hasAttribute(Idx, Attribute::InAlloca),
              "Attributes 'byval' and 'inalloca' do not support unsized types!",
              V);
    }
  } else {
    Assert1(!Attrs.hasAttribute(Idx, Attribute::ByVal),
            "Attribute 'byval' only applies to parameters with pointer type!",
            V);
  }
}

// VerifyFunctionAttrs - Check parameter attributes against a function type.
// The value V is printed in error messages.
void Verifier::VerifyFunctionAttrs(FunctionType *FT, AttributeSet Attrs,
                                   const Value *V) {
  if (Attrs.isEmpty())
    return;

  bool SawNest = false;
  bool SawReturned = false;

  for (unsigned i = 0, e = Attrs.getNumSlots(); i != e; ++i) {
    unsigned Idx = Attrs.getSlotIndex(i);

    Type *Ty;
    if (Idx == 0)
      Ty = FT->getReturnType();
    else if (Idx-1 < FT->getNumParams())
      Ty = FT->getParamType(Idx-1);
    else
      break;  // VarArgs attributes, verified elsewhere.

    VerifyParameterAttrs(Attrs, Idx, Ty, Idx == 0, V);

    if (Idx == 0)
      continue;

    if (Attrs.hasAttribute(Idx, Attribute::Nest)) {
      Assert1(!SawNest, "More than one parameter has attribute nest!", V);
      SawNest = true;
    }

    if (Attrs.hasAttribute(Idx, Attribute::Returned)) {
      Assert1(!SawReturned, "More than one parameter has attribute returned!",
              V);
      Assert1(Ty->canLosslesslyBitCastTo(FT->getReturnType()), "Incompatible "
              "argument and return types for 'returned' attribute", V);
      SawReturned = true;
    }

    if (Attrs.hasAttribute(Idx, Attribute::StructRet))
      Assert1(Idx == 1, "Attribute sret is not on first parameter!", V);

    if (Attrs.hasAttribute(Idx, Attribute::InAlloca)) {
      Assert1(Idx == FT->getNumParams(),
              "inalloca isn't on the last parameter!", V);
    }
  }

  if (!Attrs.hasAttributes(AttributeSet::FunctionIndex))
    return;

  VerifyAttributeTypes(Attrs, AttributeSet::FunctionIndex, true, V);

  Assert1(!(Attrs.hasAttribute(AttributeSet::FunctionIndex,
                               Attribute::ReadNone) &&
            Attrs.hasAttribute(AttributeSet::FunctionIndex,
                               Attribute::ReadOnly)),
          "Attributes 'readnone and readonly' are incompatible!", V);

  Assert1(!(Attrs.hasAttribute(AttributeSet::FunctionIndex,
                               Attribute::NoInline) &&
            Attrs.hasAttribute(AttributeSet::FunctionIndex,
                               Attribute::AlwaysInline)),
          "Attributes 'noinline and alwaysinline' are incompatible!", V);

  if (Attrs.hasAttribute(AttributeSet::FunctionIndex, 
                         Attribute::OptimizeNone)) {
    Assert1(Attrs.hasAttribute(AttributeSet::FunctionIndex,
                               Attribute::NoInline),
            "Attribute 'optnone' requires 'noinline'!", V);

    Assert1(!Attrs.hasAttribute(AttributeSet::FunctionIndex,
                                Attribute::OptimizeForSize),
            "Attributes 'optsize and optnone' are incompatible!", V);

    Assert1(!Attrs.hasAttribute(AttributeSet::FunctionIndex,
                                Attribute::MinSize),
            "Attributes 'minsize and optnone' are incompatible!", V);
  }
}

void Verifier::VerifyBitcastType(const Value *V, Type *DestTy, Type *SrcTy) {
  // Get the size of the types in bits, we'll need this later
  unsigned SrcBitSize = SrcTy->getPrimitiveSizeInBits();
  unsigned DestBitSize = DestTy->getPrimitiveSizeInBits();

  // BitCast implies a no-op cast of type only. No bits change.
  // However, you can't cast pointers to anything but pointers.
  Assert1(SrcTy->isPointerTy() == DestTy->isPointerTy(),
          "Bitcast requires both operands to be pointer or neither", V);
  Assert1(SrcBitSize == DestBitSize,
          "Bitcast requires types of same width", V);

  // Disallow aggregates.
  Assert1(!SrcTy->isAggregateType(),
          "Bitcast operand must not be aggregate", V);
  Assert1(!DestTy->isAggregateType(),
          "Bitcast type must not be aggregate", V);

  // Without datalayout, assume all address spaces are the same size.
  // Don't check if both types are not pointers.
  // Skip casts between scalars and vectors.
  if (!DL ||
      !SrcTy->isPtrOrPtrVectorTy() ||
      !DestTy->isPtrOrPtrVectorTy() ||
      SrcTy->isVectorTy() != DestTy->isVectorTy()) {
    return;
  }

  unsigned SrcAS = SrcTy->getPointerAddressSpace();
  unsigned DstAS = DestTy->getPointerAddressSpace();

  Assert1(SrcAS == DstAS,
          "Bitcasts between pointers of different address spaces is not legal."
          "Use AddrSpaceCast instead.", V);
}

void Verifier::VerifyConstantExprBitcastType(const ConstantExpr *CE) {
  if (CE->getOpcode() == Instruction::BitCast) {
    Type *SrcTy = CE->getOperand(0)->getType();
    Type *DstTy = CE->getType();
    VerifyBitcastType(CE, DstTy, SrcTy);
  }
}

bool Verifier::VerifyAttributeCount(AttributeSet Attrs, unsigned Params) {
  if (Attrs.getNumSlots() == 0)
    return true;

  unsigned LastSlot = Attrs.getNumSlots() - 1;
  unsigned LastIndex = Attrs.getSlotIndex(LastSlot);
  if (LastIndex <= Params
      || (LastIndex == AttributeSet::FunctionIndex
          && (LastSlot == 0 || Attrs.getSlotIndex(LastSlot - 1) <= Params)))
    return true;

  return false;
}

// visitFunction - Verify that a function is ok.
//
void Verifier::visitFunction(const Function &F) {
  // Check function arguments.
  FunctionType *FT = F.getFunctionType();
  unsigned NumArgs = F.arg_size();

  Assert1(Context == &F.getContext(),
          "Function context does not match Module context!", &F);

  Assert1(!F.hasCommonLinkage(), "Functions may not have common linkage", &F);
  Assert2(FT->getNumParams() == NumArgs,
          "# formal arguments must match # of arguments for function type!",
          &F, FT);
  Assert1(F.getReturnType()->isFirstClassType() ||
          F.getReturnType()->isVoidTy() ||
          F.getReturnType()->isStructTy(),
          "Functions cannot return aggregate values!", &F);

  Assert1(!F.hasStructRetAttr() || F.getReturnType()->isVoidTy(),
          "Invalid struct return type!", &F);

  AttributeSet Attrs = F.getAttributes();

  Assert1(VerifyAttributeCount(Attrs, FT->getNumParams()),
          "Attribute after last parameter!", &F);

  // Check function attributes.
  VerifyFunctionAttrs(FT, Attrs, &F);

  // On function declarations/definitions, we do not support the builtin
  // attribute. We do not check this in VerifyFunctionAttrs since that is
  // checking for Attributes that can/can not ever be on functions.
  Assert1(!Attrs.hasAttribute(AttributeSet::FunctionIndex,
                              Attribute::Builtin),
          "Attribute 'builtin' can only be applied to a callsite.", &F);

  // Check that this function meets the restrictions on this calling convention.
  switch (F.getCallingConv()) {
  default:
    break;
  case CallingConv::C:
    break;
  case CallingConv::Fast:
  case CallingConv::Cold:
  case CallingConv::X86_FastCall:
  case CallingConv::X86_ThisCall:
  case CallingConv::Intel_OCL_BI:
  case CallingConv::PTX_Kernel:
  case CallingConv::PTX_Device:
    Assert1(!F.isVarArg(),
            "Varargs functions must have C calling conventions!", &F);
    break;
  }

  bool isLLVMdotName = F.getName().size() >= 5 &&
                       F.getName().substr(0, 5) == "llvm.";

  // Check that the argument values match the function type for this function...
  unsigned i = 0;
  for (Function::const_arg_iterator I = F.arg_begin(), E = F.arg_end(); I != E;
       ++I, ++i) {
    Assert2(I->getType() == FT->getParamType(i),
            "Argument value does not match function argument type!",
            I, FT->getParamType(i));
    Assert1(I->getType()->isFirstClassType(),
            "Function arguments must have first-class types!", I);
    if (!isLLVMdotName)
      Assert2(!I->getType()->isMetadataTy(),
              "Function takes metadata but isn't an intrinsic", I, &F);
  }

  if (F.isMaterializable()) {
    // Function has a body somewhere we can't see.
  } else if (F.isDeclaration()) {
    Assert1(F.hasExternalLinkage() || F.hasExternalWeakLinkage(),
            "invalid linkage type for function declaration", &F);
  } else {
    // Verify that this function (which has a body) is not named "llvm.*".  It
    // is not legal to define intrinsics.
    Assert1(!isLLVMdotName, "llvm intrinsics cannot be defined!", &F);

    // Check the entry node
    const BasicBlock *Entry = &F.getEntryBlock();
    Assert1(pred_begin(Entry) == pred_end(Entry),
            "Entry block to function must not have predecessors!", Entry);

    // The address of the entry block cannot be taken, unless it is dead.
    if (Entry->hasAddressTaken()) {
      Assert1(!BlockAddress::lookup(Entry)->isConstantUsed(),
              "blockaddress may not be used with the entry block!", Entry);
    }
  }

  // If this function is actually an intrinsic, verify that it is only used in
  // direct call/invokes, never having its "address taken".
  if (F.getIntrinsicID()) {
    const User *U;
    if (F.hasAddressTaken(&U))
      Assert1(0, "Invalid user of intrinsic instruction!", U);
  }

  Assert1(!F.hasDLLImportStorageClass() ||
          (F.isDeclaration() && F.hasExternalLinkage()) ||
          F.hasAvailableExternallyLinkage(),
          "Function is marked as dllimport, but not external.", &F);
}

// verifyBasicBlock - Verify that a basic block is well formed...
//
void Verifier::visitBasicBlock(BasicBlock &BB) {
  InstsInThisBlock.clear();

  // Ensure that basic blocks have terminators!
  Assert1(BB.getTerminator(), "Basic Block does not have terminator!", &BB);

  // Check constraints that this basic block imposes on all of the PHI nodes in
  // it.
  if (isa<PHINode>(BB.front())) {
    SmallVector<BasicBlock*, 8> Preds(pred_begin(&BB), pred_end(&BB));
    SmallVector<std::pair<BasicBlock*, Value*>, 8> Values;
    std::sort(Preds.begin(), Preds.end());
    PHINode *PN;
    for (BasicBlock::iterator I = BB.begin(); (PN = dyn_cast<PHINode>(I));++I) {
      // Ensure that PHI nodes have at least one entry!
      Assert1(PN->getNumIncomingValues() != 0,
              "PHI nodes must have at least one entry.  If the block is dead, "
              "the PHI should be removed!", PN);
      Assert1(PN->getNumIncomingValues() == Preds.size(),
              "PHINode should have one entry for each predecessor of its "
              "parent basic block!", PN);

      // Get and sort all incoming values in the PHI node...
      Values.clear();
      Values.reserve(PN->getNumIncomingValues());
      for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i)
        Values.push_back(std::make_pair(PN->getIncomingBlock(i),
                                        PN->getIncomingValue(i)));
      std::sort(Values.begin(), Values.end());

      for (unsigned i = 0, e = Values.size(); i != e; ++i) {
        // Check to make sure that if there is more than one entry for a
        // particular basic block in this PHI node, that the incoming values are
        // all identical.
        //
        Assert4(i == 0 || Values[i].first  != Values[i-1].first ||
                Values[i].second == Values[i-1].second,
                "PHI node has multiple entries for the same basic block with "
                "different incoming values!", PN, Values[i].first,
                Values[i].second, Values[i-1].second);

        // Check to make sure that the predecessors and PHI node entries are
        // matched up.
        Assert3(Values[i].first == Preds[i],
                "PHI node entries do not match predecessors!", PN,
                Values[i].first, Preds[i]);
      }
    }
  }
}

void Verifier::visitTerminatorInst(TerminatorInst &I) {
  // Ensure that terminators only exist at the end of the basic block.
  Assert1(&I == I.getParent()->getTerminator(),
          "Terminator found in the middle of a basic block!", I.getParent());
  visitInstruction(I);
}

void Verifier::visitBranchInst(BranchInst &BI) {
  if (BI.isConditional()) {
    Assert2(BI.getCondition()->getType()->isIntegerTy(1),
            "Branch condition is not 'i1' type!", &BI, BI.getCondition());
  }
  visitTerminatorInst(BI);
}

void Verifier::visitReturnInst(ReturnInst &RI) {
  Function *F = RI.getParent()->getParent();
  unsigned N = RI.getNumOperands();
  if (F->getReturnType()->isVoidTy())
    Assert2(N == 0,
            "Found return instr that returns non-void in Function of void "
            "return type!", &RI, F->getReturnType());
  else
    Assert2(N == 1 && F->getReturnType() == RI.getOperand(0)->getType(),
            "Function return type does not match operand "
            "type of return inst!", &RI, F->getReturnType());

  // Check to make sure that the return value has necessary properties for
  // terminators...
  visitTerminatorInst(RI);
}

void Verifier::visitSwitchInst(SwitchInst &SI) {
  // Check to make sure that all of the constants in the switch instruction
  // have the same type as the switched-on value.
  Type *SwitchTy = SI.getCondition()->getType();
  SmallPtrSet<ConstantInt*, 32> Constants;
  for (SwitchInst::CaseIt i = SI.case_begin(), e = SI.case_end(); i != e; ++i) {
    Assert1(i.getCaseValue()->getType() == SwitchTy,
            "Switch constants must all be same type as switch value!", &SI);
    Assert2(Constants.insert(i.getCaseValue()),
            "Duplicate integer as switch case", &SI, i.getCaseValue());
  }

  visitTerminatorInst(SI);
}

void Verifier::visitIndirectBrInst(IndirectBrInst &BI) {
  Assert1(BI.getAddress()->getType()->isPointerTy(),
          "Indirectbr operand must have pointer type!", &BI);
  for (unsigned i = 0, e = BI.getNumDestinations(); i != e; ++i)
    Assert1(BI.getDestination(i)->getType()->isLabelTy(),
            "Indirectbr destinations must all have pointer type!", &BI);

  visitTerminatorInst(BI);
}

void Verifier::visitSelectInst(SelectInst &SI) {
  Assert1(!SelectInst::areInvalidOperands(SI.getOperand(0), SI.getOperand(1),
                                          SI.getOperand(2)),
          "Invalid operands for select instruction!", &SI);

  Assert1(SI.getTrueValue()->getType() == SI.getType(),
          "Select values must have same type as select instruction!", &SI);
  visitInstruction(SI);
}

/// visitUserOp1 - User defined operators shouldn't live beyond the lifetime of
/// a pass, if any exist, it's an error.
///
void Verifier::visitUserOp1(Instruction &I) {
  Assert1(0, "User-defined operators should not live outside of a pass!", &I);
}

void Verifier::visitTruncInst(TruncInst &I) {
  // Get the source and destination types
  Type *SrcTy = I.getOperand(0)->getType();
  Type *DestTy = I.getType();

  // Get the size of the types in bits, we'll need this later
  unsigned SrcBitSize = SrcTy->getScalarSizeInBits();
  unsigned DestBitSize = DestTy->getScalarSizeInBits();

  Assert1(SrcTy->isIntOrIntVectorTy(), "Trunc only operates on integer", &I);
  Assert1(DestTy->isIntOrIntVectorTy(), "Trunc only produces integer", &I);
  Assert1(SrcTy->isVectorTy() == DestTy->isVectorTy(),
          "trunc source and destination must both be a vector or neither", &I);
  Assert1(SrcBitSize > DestBitSize,"DestTy too big for Trunc", &I);

  visitInstruction(I);
}

void Verifier::visitZExtInst(ZExtInst &I) {
  // Get the source and destination types
  Type *SrcTy = I.getOperand(0)->getType();
  Type *DestTy = I.getType();

  // Get the size of the types in bits, we'll need this later
  Assert1(SrcTy->isIntOrIntVectorTy(), "ZExt only operates on integer", &I);
  Assert1(DestTy->isIntOrIntVectorTy(), "ZExt only produces an integer", &I);
  Assert1(SrcTy->isVectorTy() == DestTy->isVectorTy(),
          "zext source and destination must both be a vector or neither", &I);
  unsigned SrcBitSize = SrcTy->getScalarSizeInBits();
  unsigned DestBitSize = DestTy->getScalarSizeInBits();

  Assert1(SrcBitSize < DestBitSize,"Type too small for ZExt", &I);

  visitInstruction(I);
}

void Verifier::visitSExtInst(SExtInst &I) {
  // Get the source and destination types
  Type *SrcTy = I.getOperand(0)->getType();
  Type *DestTy = I.getType();

  // Get the size of the types in bits, we'll need this later
  unsigned SrcBitSize = SrcTy->getScalarSizeInBits();
  unsigned DestBitSize = DestTy->getScalarSizeInBits();

  Assert1(SrcTy->isIntOrIntVectorTy(), "SExt only operates on integer", &I);
  Assert1(DestTy->isIntOrIntVectorTy(), "SExt only produces an integer", &I);
  Assert1(SrcTy->isVectorTy() == DestTy->isVectorTy(),
          "sext source and destination must both be a vector or neither", &I);
  Assert1(SrcBitSize < DestBitSize,"Type too small for SExt", &I);

  visitInstruction(I);
}

void Verifier::visitFPTruncInst(FPTruncInst &I) {
  // Get the source and destination types
  Type *SrcTy = I.getOperand(0)->getType();
  Type *DestTy = I.getType();
  // Get the size of the types in bits, we'll need this later
  unsigned SrcBitSize = SrcTy->getScalarSizeInBits();
  unsigned DestBitSize = DestTy->getScalarSizeInBits();

  Assert1(SrcTy->isFPOrFPVectorTy(),"FPTrunc only operates on FP", &I);
  Assert1(DestTy->isFPOrFPVectorTy(),"FPTrunc only produces an FP", &I);
  Assert1(SrcTy->isVectorTy() == DestTy->isVectorTy(),
          "fptrunc source and destination must both be a vector or neither",&I);
  Assert1(SrcBitSize > DestBitSize,"DestTy too big for FPTrunc", &I);

  visitInstruction(I);
}

void Verifier::visitFPExtInst(FPExtInst &I) {
  // Get the source and destination types
  Type *SrcTy = I.getOperand(0)->getType();
  Type *DestTy = I.getType();

  // Get the size of the types in bits, we'll need this later
  unsigned SrcBitSize = SrcTy->getScalarSizeInBits();
  unsigned DestBitSize = DestTy->getScalarSizeInBits();

  Assert1(SrcTy->isFPOrFPVectorTy(),"FPExt only operates on FP", &I);
  Assert1(DestTy->isFPOrFPVectorTy(),"FPExt only produces an FP", &I);
  Assert1(SrcTy->isVectorTy() == DestTy->isVectorTy(),
          "fpext source and destination must both be a vector or neither", &I);
  Assert1(SrcBitSize < DestBitSize,"DestTy too small for FPExt", &I);

  visitInstruction(I);
}

void Verifier::visitUIToFPInst(UIToFPInst &I) {
  // Get the source and destination types
  Type *SrcTy = I.getOperand(0)->getType();
  Type *DestTy = I.getType();

  bool SrcVec = SrcTy->isVectorTy();
  bool DstVec = DestTy->isVectorTy();

  Assert1(SrcVec == DstVec,
          "UIToFP source and dest must both be vector or scalar", &I);
  Assert1(SrcTy->isIntOrIntVectorTy(),
          "UIToFP source must be integer or integer vector", &I);
  Assert1(DestTy->isFPOrFPVectorTy(),
          "UIToFP result must be FP or FP vector", &I);

  if (SrcVec && DstVec)
    Assert1(cast<VectorType>(SrcTy)->getNumElements() ==
            cast<VectorType>(DestTy)->getNumElements(),
            "UIToFP source and dest vector length mismatch", &I);

  visitInstruction(I);
}

void Verifier::visitSIToFPInst(SIToFPInst &I) {
  // Get the source and destination types
  Type *SrcTy = I.getOperand(0)->getType();
  Type *DestTy = I.getType();

  bool SrcVec = SrcTy->isVectorTy();
  bool DstVec = DestTy->isVectorTy();

  Assert1(SrcVec == DstVec,
          "SIToFP source and dest must both be vector or scalar", &I);
  Assert1(SrcTy->isIntOrIntVectorTy(),
          "SIToFP source must be integer or integer vector", &I);
  Assert1(DestTy->isFPOrFPVectorTy(),
          "SIToFP result must be FP or FP vector", &I);

  if (SrcVec && DstVec)
    Assert1(cast<VectorType>(SrcTy)->getNumElements() ==
            cast<VectorType>(DestTy)->getNumElements(),
            "SIToFP source and dest vector length mismatch", &I);

  visitInstruction(I);
}

void Verifier::visitFPToUIInst(FPToUIInst &I) {
  // Get the source and destination types
  Type *SrcTy = I.getOperand(0)->getType();
  Type *DestTy = I.getType();

  bool SrcVec = SrcTy->isVectorTy();
  bool DstVec = DestTy->isVectorTy();

  Assert1(SrcVec == DstVec,
          "FPToUI source and dest must both be vector or scalar", &I);
  Assert1(SrcTy->isFPOrFPVectorTy(), "FPToUI source must be FP or FP vector",
          &I);
  Assert1(DestTy->isIntOrIntVectorTy(),
          "FPToUI result must be integer or integer vector", &I);

  if (SrcVec && DstVec)
    Assert1(cast<VectorType>(SrcTy)->getNumElements() ==
            cast<VectorType>(DestTy)->getNumElements(),
            "FPToUI source and dest vector length mismatch", &I);

  visitInstruction(I);
}

void Verifier::visitFPToSIInst(FPToSIInst &I) {
  // Get the source and destination types
  Type *SrcTy = I.getOperand(0)->getType();
  Type *DestTy = I.getType();

  bool SrcVec = SrcTy->isVectorTy();
  bool DstVec = DestTy->isVectorTy();

  Assert1(SrcVec == DstVec,
          "FPToSI source and dest must both be vector or scalar", &I);
  Assert1(SrcTy->isFPOrFPVectorTy(),
          "FPToSI source must be FP or FP vector", &I);
  Assert1(DestTy->isIntOrIntVectorTy(),
          "FPToSI result must be integer or integer vector", &I);

  if (SrcVec && DstVec)
    Assert1(cast<VectorType>(SrcTy)->getNumElements() ==
            cast<VectorType>(DestTy)->getNumElements(),
            "FPToSI source and dest vector length mismatch", &I);

  visitInstruction(I);
}

void Verifier::visitPtrToIntInst(PtrToIntInst &I) {
  // Get the source and destination types
  Type *SrcTy = I.getOperand(0)->getType();
  Type *DestTy = I.getType();

  Assert1(SrcTy->getScalarType()->isPointerTy(),
          "PtrToInt source must be pointer", &I);
  Assert1(DestTy->getScalarType()->isIntegerTy(),
          "PtrToInt result must be integral", &I);
  Assert1(SrcTy->isVectorTy() == DestTy->isVectorTy(),
          "PtrToInt type mismatch", &I);

  if (SrcTy->isVectorTy()) {
    VectorType *VSrc = dyn_cast<VectorType>(SrcTy);
    VectorType *VDest = dyn_cast<VectorType>(DestTy);
    Assert1(VSrc->getNumElements() == VDest->getNumElements(),
          "PtrToInt Vector width mismatch", &I);
  }

  visitInstruction(I);
}

void Verifier::visitIntToPtrInst(IntToPtrInst &I) {
  // Get the source and destination types
  Type *SrcTy = I.getOperand(0)->getType();
  Type *DestTy = I.getType();

  Assert1(SrcTy->getScalarType()->isIntegerTy(),
          "IntToPtr source must be an integral", &I);
  Assert1(DestTy->getScalarType()->isPointerTy(),
          "IntToPtr result must be a pointer",&I);
  Assert1(SrcTy->isVectorTy() == DestTy->isVectorTy(),
          "IntToPtr type mismatch", &I);
  if (SrcTy->isVectorTy()) {
    VectorType *VSrc = dyn_cast<VectorType>(SrcTy);
    VectorType *VDest = dyn_cast<VectorType>(DestTy);
    Assert1(VSrc->getNumElements() == VDest->getNumElements(),
          "IntToPtr Vector width mismatch", &I);
  }
  visitInstruction(I);
}

void Verifier::visitBitCastInst(BitCastInst &I) {
  Type *SrcTy = I.getOperand(0)->getType();
  Type *DestTy = I.getType();
  VerifyBitcastType(&I, DestTy, SrcTy);
  visitInstruction(I);
}

void Verifier::visitAddrSpaceCastInst(AddrSpaceCastInst &I) {
  Type *SrcTy = I.getOperand(0)->getType();
  Type *DestTy = I.getType();

  Assert1(SrcTy->isPtrOrPtrVectorTy(),
          "AddrSpaceCast source must be a pointer", &I);
  Assert1(DestTy->isPtrOrPtrVectorTy(),
          "AddrSpaceCast result must be a pointer", &I);
  Assert1(SrcTy->getPointerAddressSpace() != DestTy->getPointerAddressSpace(),
          "AddrSpaceCast must be between different address spaces", &I);
  if (SrcTy->isVectorTy())
    Assert1(SrcTy->getVectorNumElements() == DestTy->getVectorNumElements(),
            "AddrSpaceCast vector pointer number of elements mismatch", &I);
  visitInstruction(I);
}

/// visitPHINode - Ensure that a PHI node is well formed.
///
void Verifier::visitPHINode(PHINode &PN) {
  // Ensure that the PHI nodes are all grouped together at the top of the block.
  // This can be tested by checking whether the instruction before this is
  // either nonexistent (because this is begin()) or is a PHI node.  If not,
  // then there is some other instruction before a PHI.
  Assert2(&PN == &PN.getParent()->front() ||
          isa<PHINode>(--BasicBlock::iterator(&PN)),
          "PHI nodes not grouped at top of basic block!",
          &PN, PN.getParent());

  // Check that all of the values of the PHI node have the same type as the
  // result, and that the incoming blocks are really basic blocks.
  for (unsigned i = 0, e = PN.getNumIncomingValues(); i != e; ++i) {
    Assert1(PN.getType() == PN.getIncomingValue(i)->getType(),
            "PHI node operands are not the same type as the result!", &PN);
  }

  // All other PHI node constraints are checked in the visitBasicBlock method.

  visitInstruction(PN);
}

void Verifier::VerifyCallSite(CallSite CS) {
  Instruction *I = CS.getInstruction();

  Assert1(CS.getCalledValue()->getType()->isPointerTy(),
          "Called function must be a pointer!", I);
  PointerType *FPTy = cast<PointerType>(CS.getCalledValue()->getType());

  Assert1(FPTy->getElementType()->isFunctionTy(),
          "Called function is not pointer to function type!", I);
  FunctionType *FTy = cast<FunctionType>(FPTy->getElementType());

  // Verify that the correct number of arguments are being passed
  if (FTy->isVarArg())
    Assert1(CS.arg_size() >= FTy->getNumParams(),
            "Called function requires more parameters than were provided!",I);
  else
    Assert1(CS.arg_size() == FTy->getNumParams(),
            "Incorrect number of arguments passed to called function!", I);

  // Verify that all arguments to the call match the function type.
  for (unsigned i = 0, e = FTy->getNumParams(); i != e; ++i)
    Assert3(CS.getArgument(i)->getType() == FTy->getParamType(i),
            "Call parameter type does not match function signature!",
            CS.getArgument(i), FTy->getParamType(i), I);

  AttributeSet Attrs = CS.getAttributes();

  Assert1(VerifyAttributeCount(Attrs, CS.arg_size()),
          "Attribute after last parameter!", I);

  // Verify call attributes.
  VerifyFunctionAttrs(FTy, Attrs, I);

  if (FTy->isVarArg()) {
    // FIXME? is 'nest' even legal here?
    bool SawNest = false;
    bool SawReturned = false;

    for (unsigned Idx = 1; Idx < 1 + FTy->getNumParams(); ++Idx) {
      if (Attrs.hasAttribute(Idx, Attribute::Nest))
        SawNest = true;
      if (Attrs.hasAttribute(Idx, Attribute::Returned))
        SawReturned = true;
    }

    // Check attributes on the varargs part.
    for (unsigned Idx = 1 + FTy->getNumParams(); Idx <= CS.arg_size(); ++Idx) {
      Type *Ty = CS.getArgument(Idx-1)->getType();
      VerifyParameterAttrs(Attrs, Idx, Ty, false, I);

      if (Attrs.hasAttribute(Idx, Attribute::Nest)) {
        Assert1(!SawNest, "More than one parameter has attribute nest!", I);
        SawNest = true;
      }

      if (Attrs.hasAttribute(Idx, Attribute::Returned)) {
        Assert1(!SawReturned, "More than one parameter has attribute returned!",
                I);
        Assert1(Ty->canLosslesslyBitCastTo(FTy->getReturnType()),
                "Incompatible argument and return types for 'returned' "
                "attribute", I);
        SawReturned = true;
      }

      Assert1(!Attrs.hasAttribute(Idx, Attribute::StructRet),
              "Attribute 'sret' cannot be used for vararg call arguments!", I);

      if (Attrs.hasAttribute(Idx, Attribute::InAlloca))
        Assert1(Idx == CS.arg_size(), "inalloca isn't on the last argument!",
                I);
    }
  }

  // Verify that there's no metadata unless it's a direct call to an intrinsic.
  if (CS.getCalledFunction() == 0 ||
      !CS.getCalledFunction()->getName().startswith("llvm.")) {
    for (FunctionType::param_iterator PI = FTy->param_begin(),
           PE = FTy->param_end(); PI != PE; ++PI)
      Assert1(!(*PI)->isMetadataTy(),
              "Function has metadata parameter but isn't an intrinsic", I);
  }

  visitInstruction(*I);
}

void Verifier::visitCallInst(CallInst &CI) {
  VerifyCallSite(&CI);

  if (Function *F = CI.getCalledFunction())
    if (Intrinsic::ID ID = (Intrinsic::ID)F->getIntrinsicID())
      visitIntrinsicFunctionCall(ID, CI);
}

void Verifier::visitInvokeInst(InvokeInst &II) {
  VerifyCallSite(&II);

  // Verify that there is a landingpad instruction as the first non-PHI
  // instruction of the 'unwind' destination.
  Assert1(II.getUnwindDest()->isLandingPad(),
          "The unwind destination does not have a landingpad instruction!",&II);

  visitTerminatorInst(II);
}

/// visitBinaryOperator - Check that both arguments to the binary operator are
/// of the same type!
///
void Verifier::visitBinaryOperator(BinaryOperator &B) {
  Assert1(B.getOperand(0)->getType() == B.getOperand(1)->getType(),
          "Both operands to a binary operator are not of the same type!", &B);

  switch (B.getOpcode()) {
  // Check that integer arithmetic operators are only used with
  // integral operands.
  case Instruction::Add:
  case Instruction::Sub:
  case Instruction::Mul:
  case Instruction::SDiv:
  case Instruction::UDiv:
  case Instruction::SRem:
  case Instruction::URem:
    Assert1(B.getType()->isIntOrIntVectorTy(),
            "Integer arithmetic operators only work with integral types!", &B);
    Assert1(B.getType() == B.getOperand(0)->getType(),
            "Integer arithmetic operators must have same type "
            "for operands and result!", &B);
    break;
  // Check that floating-point arithmetic operators are only used with
  // floating-point operands.
  case Instruction::FAdd:
  case Instruction::FSub:
  case Instruction::FMul:
  case Instruction::FDiv:
  case Instruction::FRem:
    Assert1(B.getType()->isFPOrFPVectorTy(),
            "Floating-point arithmetic operators only work with "
            "floating-point types!", &B);
    Assert1(B.getType() == B.getOperand(0)->getType(),
            "Floating-point arithmetic operators must have same type "
            "for operands and result!", &B);
    break;
  // Check that logical operators are only used with integral operands.
  case Instruction::And:
  case Instruction::Or:
  case Instruction::Xor:
    Assert1(B.getType()->isIntOrIntVectorTy(),
            "Logical operators only work with integral types!", &B);
    Assert1(B.getType() == B.getOperand(0)->getType(),
            "Logical operators must have same type for operands and result!",
            &B);
    break;
  case Instruction::Shl:
  case Instruction::LShr:
  case Instruction::AShr:
    Assert1(B.getType()->isIntOrIntVectorTy(),
            "Shifts only work with integral types!", &B);
    Assert1(B.getType() == B.getOperand(0)->getType(),
            "Shift return type must be same as operands!", &B);
    break;
  default:
    llvm_unreachable("Unknown BinaryOperator opcode!");
  }

  visitInstruction(B);
}

void Verifier::visitICmpInst(ICmpInst &IC) {
  // Check that the operands are the same type
  Type *Op0Ty = IC.getOperand(0)->getType();
  Type *Op1Ty = IC.getOperand(1)->getType();
  Assert1(Op0Ty == Op1Ty,
          "Both operands to ICmp instruction are not of the same type!", &IC);
  // Check that the operands are the right type
  Assert1(Op0Ty->isIntOrIntVectorTy() || Op0Ty->getScalarType()->isPointerTy(),
          "Invalid operand types for ICmp instruction", &IC);
  // Check that the predicate is valid.
  Assert1(IC.getPredicate() >= CmpInst::FIRST_ICMP_PREDICATE &&
          IC.getPredicate() <= CmpInst::LAST_ICMP_PREDICATE,
          "Invalid predicate in ICmp instruction!", &IC);

  visitInstruction(IC);
}

void Verifier::visitFCmpInst(FCmpInst &FC) {
  // Check that the operands are the same type
  Type *Op0Ty = FC.getOperand(0)->getType();
  Type *Op1Ty = FC.getOperand(1)->getType();
  Assert1(Op0Ty == Op1Ty,
          "Both operands to FCmp instruction are not of the same type!", &FC);
  // Check that the operands are the right type
  Assert1(Op0Ty->isFPOrFPVectorTy(),
          "Invalid operand types for FCmp instruction", &FC);
  // Check that the predicate is valid.
  Assert1(FC.getPredicate() >= CmpInst::FIRST_FCMP_PREDICATE &&
          FC.getPredicate() <= CmpInst::LAST_FCMP_PREDICATE,
          "Invalid predicate in FCmp instruction!", &FC);

  visitInstruction(FC);
}

void Verifier::visitExtractElementInst(ExtractElementInst &EI) {
  Assert1(ExtractElementInst::isValidOperands(EI.getOperand(0),
                                              EI.getOperand(1)),
          "Invalid extractelement operands!", &EI);
  visitInstruction(EI);
}

void Verifier::visitInsertElementInst(InsertElementInst &IE) {
  Assert1(InsertElementInst::isValidOperands(IE.getOperand(0),
                                             IE.getOperand(1),
                                             IE.getOperand(2)),
          "Invalid insertelement operands!", &IE);
  visitInstruction(IE);
}

void Verifier::visitShuffleVectorInst(ShuffleVectorInst &SV) {
  Assert1(ShuffleVectorInst::isValidOperands(SV.getOperand(0), SV.getOperand(1),
                                             SV.getOperand(2)),
          "Invalid shufflevector operands!", &SV);
  visitInstruction(SV);
}

void Verifier::visitGetElementPtrInst(GetElementPtrInst &GEP) {
  Type *TargetTy = GEP.getPointerOperandType()->getScalarType();

  Assert1(isa<PointerType>(TargetTy),
    "GEP base pointer is not a vector or a vector of pointers", &GEP);
  Assert1(cast<PointerType>(TargetTy)->getElementType()->isSized(),
          "GEP into unsized type!", &GEP);
  Assert1(GEP.getPointerOperandType()->isVectorTy() ==
          GEP.getType()->isVectorTy(), "Vector GEP must return a vector value",
          &GEP);

  SmallVector<Value*, 16> Idxs(GEP.idx_begin(), GEP.idx_end());
  Type *ElTy =
    GetElementPtrInst::getIndexedType(GEP.getPointerOperandType(), Idxs);
  Assert1(ElTy, "Invalid indices for GEP pointer type!", &GEP);

  Assert2(GEP.getType()->getScalarType()->isPointerTy() &&
          cast<PointerType>(GEP.getType()->getScalarType())->getElementType()
          == ElTy, "GEP is not of right type for indices!", &GEP, ElTy);

  if (GEP.getPointerOperandType()->isVectorTy()) {
    // Additional checks for vector GEPs.
    unsigned GepWidth = GEP.getPointerOperandType()->getVectorNumElements();
    Assert1(GepWidth == GEP.getType()->getVectorNumElements(),
            "Vector GEP result width doesn't match operand's", &GEP);
    for (unsigned i = 0, e = Idxs.size(); i != e; ++i) {
      Type *IndexTy = Idxs[i]->getType();
      Assert1(IndexTy->isVectorTy(),
              "Vector GEP must have vector indices!", &GEP);
      unsigned IndexWidth = IndexTy->getVectorNumElements();
      Assert1(IndexWidth == GepWidth, "Invalid GEP index vector width", &GEP);
    }
  }
  visitInstruction(GEP);
}

static bool isContiguous(const ConstantRange &A, const ConstantRange &B) {
  return A.getUpper() == B.getLower() || A.getLower() == B.getUpper();
}

void Verifier::visitLoadInst(LoadInst &LI) {
  PointerType *PTy = dyn_cast<PointerType>(LI.getOperand(0)->getType());
  Assert1(PTy, "Load operand must be a pointer.", &LI);
  Type *ElTy = PTy->getElementType();
  Assert2(ElTy == LI.getType(),
          "Load result type does not match pointer operand type!", &LI, ElTy);
  if (LI.isAtomic()) {
    Assert1(LI.getOrdering() != Release && LI.getOrdering() != AcquireRelease,
            "Load cannot have Release ordering", &LI);
    Assert1(LI.getAlignment() != 0,
            "Atomic load must specify explicit alignment", &LI);
    if (!ElTy->isPointerTy()) {
      Assert2(ElTy->isIntegerTy(),
              "atomic store operand must have integer type!",
              &LI, ElTy);
      unsigned Size = ElTy->getPrimitiveSizeInBits();
      Assert2(Size >= 8 && !(Size & (Size - 1)),
              "atomic store operand must be power-of-two byte-sized integer",
              &LI, ElTy);
    }
  } else {
    Assert1(LI.getSynchScope() == CrossThread,
            "Non-atomic load cannot have SynchronizationScope specified", &LI);
  }

  if (MDNode *Range = LI.getMetadata(LLVMContext::MD_range)) {
    unsigned NumOperands = Range->getNumOperands();
    Assert1(NumOperands % 2 == 0, "Unfinished range!", Range);
    unsigned NumRanges = NumOperands / 2;
    Assert1(NumRanges >= 1, "It should have at least one range!", Range);

    ConstantRange LastRange(1); // Dummy initial value
    for (unsigned i = 0; i < NumRanges; ++i) {
      ConstantInt *Low = dyn_cast<ConstantInt>(Range->getOperand(2*i));
      Assert1(Low, "The lower limit must be an integer!", Low);
      ConstantInt *High = dyn_cast<ConstantInt>(Range->getOperand(2*i + 1));
      Assert1(High, "The upper limit must be an integer!", High);
      Assert1(High->getType() == Low->getType() &&
              High->getType() == ElTy, "Range types must match load type!",
              &LI);

      APInt HighV = High->getValue();
      APInt LowV = Low->getValue();
      ConstantRange CurRange(LowV, HighV);
      Assert1(!CurRange.isEmptySet() && !CurRange.isFullSet(),
              "Range must not be empty!", Range);
      if (i != 0) {
        Assert1(CurRange.intersectWith(LastRange).isEmptySet(),
                "Intervals are overlapping", Range);
        Assert1(LowV.sgt(LastRange.getLower()), "Intervals are not in order",
                Range);
        Assert1(!isContiguous(CurRange, LastRange), "Intervals are contiguous",
                Range);
      }
      LastRange = ConstantRange(LowV, HighV);
    }
    if (NumRanges > 2) {
      APInt FirstLow =
        dyn_cast<ConstantInt>(Range->getOperand(0))->getValue();
      APInt FirstHigh =
        dyn_cast<ConstantInt>(Range->getOperand(1))->getValue();
      ConstantRange FirstRange(FirstLow, FirstHigh);
      Assert1(FirstRange.intersectWith(LastRange).isEmptySet(),
              "Intervals are overlapping", Range);
      Assert1(!isContiguous(FirstRange, LastRange), "Intervals are contiguous",
              Range);
    }


  }

  visitInstruction(LI);
}

void Verifier::visitStoreInst(StoreInst &SI) {
  PointerType *PTy = dyn_cast<PointerType>(SI.getOperand(1)->getType());
  Assert1(PTy, "Store operand must be a pointer.", &SI);
  Type *ElTy = PTy->getElementType();
  Assert2(ElTy == SI.getOperand(0)->getType(),
          "Stored value type does not match pointer operand type!",
          &SI, ElTy);
  if (SI.isAtomic()) {
    Assert1(SI.getOrdering() != Acquire && SI.getOrdering() != AcquireRelease,
            "Store cannot have Acquire ordering", &SI);
    Assert1(SI.getAlignment() != 0,
            "Atomic store must specify explicit alignment", &SI);
    if (!ElTy->isPointerTy()) {
      Assert2(ElTy->isIntegerTy(),
              "atomic store operand must have integer type!",
              &SI, ElTy);
      unsigned Size = ElTy->getPrimitiveSizeInBits();
      Assert2(Size >= 8 && !(Size & (Size - 1)),
              "atomic store operand must be power-of-two byte-sized integer",
              &SI, ElTy);
    }
  } else {
    Assert1(SI.getSynchScope() == CrossThread,
            "Non-atomic store cannot have SynchronizationScope specified", &SI);
  }
  visitInstruction(SI);
}

void Verifier::visitAllocaInst(AllocaInst &AI) {
  SmallPtrSet<const Type*, 4> Visited;
  PointerType *PTy = AI.getType();
  Assert1(PTy->getAddressSpace() == 0,
          "Allocation instruction pointer not in the generic address space!",
          &AI);
  Assert1(PTy->getElementType()->isSized(&Visited), "Cannot allocate unsized type",
          &AI);
  Assert1(AI.getArraySize()->getType()->isIntegerTy(),
          "Alloca array size must have integer type", &AI);

  visitInstruction(AI);
}

void Verifier::visitAtomicCmpXchgInst(AtomicCmpXchgInst &CXI) {
  Assert1(CXI.getOrdering() != NotAtomic,
          "cmpxchg instructions must be atomic.", &CXI);
  Assert1(CXI.getOrdering() != Unordered,
          "cmpxchg instructions cannot be unordered.", &CXI);
  PointerType *PTy = dyn_cast<PointerType>(CXI.getOperand(0)->getType());
  Assert1(PTy, "First cmpxchg operand must be a pointer.", &CXI);
  Type *ElTy = PTy->getElementType();
  Assert2(ElTy->isIntegerTy(),
          "cmpxchg operand must have integer type!",
          &CXI, ElTy);
  unsigned Size = ElTy->getPrimitiveSizeInBits();
  Assert2(Size >= 8 && !(Size & (Size - 1)),
          "cmpxchg operand must be power-of-two byte-sized integer",
          &CXI, ElTy);
  Assert2(ElTy == CXI.getOperand(1)->getType(),
          "Expected value type does not match pointer operand type!",
          &CXI, ElTy);
  Assert2(ElTy == CXI.getOperand(2)->getType(),
          "Stored value type does not match pointer operand type!",
          &CXI, ElTy);
  visitInstruction(CXI);
}

void Verifier::visitAtomicRMWInst(AtomicRMWInst &RMWI) {
  Assert1(RMWI.getOrdering() != NotAtomic,
          "atomicrmw instructions must be atomic.", &RMWI);
  Assert1(RMWI.getOrdering() != Unordered,
          "atomicrmw instructions cannot be unordered.", &RMWI);
  PointerType *PTy = dyn_cast<PointerType>(RMWI.getOperand(0)->getType());
  Assert1(PTy, "First atomicrmw operand must be a pointer.", &RMWI);
  Type *ElTy = PTy->getElementType();
  Assert2(ElTy->isIntegerTy(),
          "atomicrmw operand must have integer type!",
          &RMWI, ElTy);
  unsigned Size = ElTy->getPrimitiveSizeInBits();
  Assert2(Size >= 8 && !(Size & (Size - 1)),
          "atomicrmw operand must be power-of-two byte-sized integer",
          &RMWI, ElTy);
  Assert2(ElTy == RMWI.getOperand(1)->getType(),
          "Argument value type does not match pointer operand type!",
          &RMWI, ElTy);
  Assert1(AtomicRMWInst::FIRST_BINOP <= RMWI.getOperation() &&
          RMWI.getOperation() <= AtomicRMWInst::LAST_BINOP,
          "Invalid binary operation!", &RMWI);
  visitInstruction(RMWI);
}

void Verifier::visitFenceInst(FenceInst &FI) {
  const AtomicOrdering Ordering = FI.getOrdering();
  Assert1(Ordering == Acquire || Ordering == Release ||
          Ordering == AcquireRelease || Ordering == SequentiallyConsistent,
          "fence instructions may only have "
          "acquire, release, acq_rel, or seq_cst ordering.", &FI);
  visitInstruction(FI);
}

void Verifier::visitExtractValueInst(ExtractValueInst &EVI) {
  Assert1(ExtractValueInst::getIndexedType(EVI.getAggregateOperand()->getType(),
                                           EVI.getIndices()) ==
          EVI.getType(),
          "Invalid ExtractValueInst operands!", &EVI);

  visitInstruction(EVI);
}

void Verifier::visitInsertValueInst(InsertValueInst &IVI) {
  Assert1(ExtractValueInst::getIndexedType(IVI.getAggregateOperand()->getType(),
                                           IVI.getIndices()) ==
          IVI.getOperand(1)->getType(),
          "Invalid InsertValueInst operands!", &IVI);

  visitInstruction(IVI);
}

void Verifier::visitLandingPadInst(LandingPadInst &LPI) {
  BasicBlock *BB = LPI.getParent();

  // The landingpad instruction is ill-formed if it doesn't have any clauses and
  // isn't a cleanup.
  Assert1(LPI.getNumClauses() > 0 || LPI.isCleanup(),
          "LandingPadInst needs at least one clause or to be a cleanup.", &LPI);

  // The landingpad instruction defines its parent as a landing pad block. The
  // landing pad block may be branched to only by the unwind edge of an invoke.
  for (pred_iterator I = pred_begin(BB), E = pred_end(BB); I != E; ++I) {
    const InvokeInst *II = dyn_cast<InvokeInst>((*I)->getTerminator());
    Assert1(II && II->getUnwindDest() == BB && II->getNormalDest() != BB,
            "Block containing LandingPadInst must be jumped to "
            "only by the unwind edge of an invoke.", &LPI);
  }

  // The landingpad instruction must be the first non-PHI instruction in the
  // block.
  Assert1(LPI.getParent()->getLandingPadInst() == &LPI,
          "LandingPadInst not the first non-PHI instruction in the block.",
          &LPI);

  // The personality functions for all landingpad instructions within the same
  // function should match.
  if (PersonalityFn)
    Assert1(LPI.getPersonalityFn() == PersonalityFn,
            "Personality function doesn't match others in function", &LPI);
  PersonalityFn = LPI.getPersonalityFn();

  // All operands must be constants.
  Assert1(isa<Constant>(PersonalityFn), "Personality function is not constant!",
          &LPI);
  for (unsigned i = 0, e = LPI.getNumClauses(); i < e; ++i) {
    Value *Clause = LPI.getClause(i);
    Assert1(isa<Constant>(Clause), "Clause is not constant!", &LPI);
    if (LPI.isCatch(i)) {
      Assert1(isa<PointerType>(Clause->getType()),
              "Catch operand does not have pointer type!", &LPI);
    } else {
      Assert1(LPI.isFilter(i), "Clause is neither catch nor filter!", &LPI);
      Assert1(isa<ConstantArray>(Clause) || isa<ConstantAggregateZero>(Clause),
              "Filter operand is not an array of constants!", &LPI);
    }
  }

  visitInstruction(LPI);
}

void Verifier::verifyDominatesUse(Instruction &I, unsigned i) {
  Instruction *Op = cast<Instruction>(I.getOperand(i));
  // If the we have an invalid invoke, don't try to compute the dominance.
  // We already reject it in the invoke specific checks and the dominance
  // computation doesn't handle multiple edges.
  if (InvokeInst *II = dyn_cast<InvokeInst>(Op)) {
    if (II->getNormalDest() == II->getUnwindDest())
      return;
  }

  const Use &U = I.getOperandUse(i);
  Assert2(InstsInThisBlock.count(Op) || DT.dominates(Op, U),
          "Instruction does not dominate all uses!", Op, &I);
}

/// verifyInstruction - Verify that an instruction is well formed.
///
void Verifier::visitInstruction(Instruction &I) {
  BasicBlock *BB = I.getParent();
  Assert1(BB, "Instruction not embedded in basic block!", &I);

  if (!isa<PHINode>(I)) {   // Check that non-phi nodes are not self referential
    for (Value::use_iterator UI = I.use_begin(), UE = I.use_end();
         UI != UE; ++UI)
      Assert1(*UI != (User*)&I || !DT.isReachableFromEntry(BB),
              "Only PHI nodes may reference their own value!", &I);
  }

  // Check that void typed values don't have names
  Assert1(!I.getType()->isVoidTy() || !I.hasName(),
          "Instruction has a name, but provides a void value!", &I);

  // Check that the return value of the instruction is either void or a legal
  // value type.
  Assert1(I.getType()->isVoidTy() ||
          I.getType()->isFirstClassType(),
          "Instruction returns a non-scalar type!", &I);

  // Check that the instruction doesn't produce metadata. Calls are already
  // checked against the callee type.
  Assert1(!I.getType()->isMetadataTy() ||
          isa<CallInst>(I) || isa<InvokeInst>(I),
          "Invalid use of metadata!", &I);

  // Check that all uses of the instruction, if they are instructions
  // themselves, actually have parent basic blocks.  If the use is not an
  // instruction, it is an error!
  for (User::use_iterator UI = I.use_begin(), UE = I.use_end();
       UI != UE; ++UI) {
    if (Instruction *Used = dyn_cast<Instruction>(*UI))
      Assert2(Used->getParent() != 0, "Instruction referencing instruction not"
              " embedded in a basic block!", &I, Used);
    else {
      CheckFailed("Use of instruction is not an instruction!", *UI);
      return;
    }
  }

  for (unsigned i = 0, e = I.getNumOperands(); i != e; ++i) {
    Assert1(I.getOperand(i) != 0, "Instruction has null operand!", &I);

    // Check to make sure that only first-class-values are operands to
    // instructions.
    if (!I.getOperand(i)->getType()->isFirstClassType()) {
      Assert1(0, "Instruction operands must be first-class values!", &I);
    }

    if (Function *F = dyn_cast<Function>(I.getOperand(i))) {
      // Check to make sure that the "address of" an intrinsic function is never
      // taken.
      Assert1(!F->isIntrinsic() || i == (isa<CallInst>(I) ? e-1 : 0),
              "Cannot take the address of an intrinsic!", &I);
      Assert1(!F->isIntrinsic() || isa<CallInst>(I) ||
              F->getIntrinsicID() == Intrinsic::donothing,
              "Cannot invoke an intrinsinc other than donothing", &I);
      Assert1(F->getParent() == M, "Referencing function in another module!",
              &I);
    } else if (BasicBlock *OpBB = dyn_cast<BasicBlock>(I.getOperand(i))) {
      Assert1(OpBB->getParent() == BB->getParent(),
              "Referring to a basic block in another function!", &I);
    } else if (Argument *OpArg = dyn_cast<Argument>(I.getOperand(i))) {
      Assert1(OpArg->getParent() == BB->getParent(),
              "Referring to an argument in another function!", &I);
    } else if (GlobalValue *GV = dyn_cast<GlobalValue>(I.getOperand(i))) {
      Assert1(GV->getParent() == M, "Referencing global in another module!",
              &I);
    } else if (isa<Instruction>(I.getOperand(i))) {
      verifyDominatesUse(I, i);
    } else if (isa<InlineAsm>(I.getOperand(i))) {
      Assert1((i + 1 == e && isa<CallInst>(I)) ||
              (i + 3 == e && isa<InvokeInst>(I)),
              "Cannot take the address of an inline asm!", &I);
    } else if (ConstantExpr *CE = dyn_cast<ConstantExpr>(I.getOperand(i))) {
      if (CE->getType()->isPtrOrPtrVectorTy()) {
        // If we have a ConstantExpr pointer, we need to see if it came from an
        // illegal bitcast (inttoptr <constant int> )
        SmallVector<const ConstantExpr *, 4> Stack;
        SmallPtrSet<const ConstantExpr *, 4> Visited;
        Stack.push_back(CE);

        while (!Stack.empty()) {
          const ConstantExpr *V = Stack.pop_back_val();
          if (!Visited.insert(V))
            continue;

          VerifyConstantExprBitcastType(V);

          for (unsigned I = 0, N = V->getNumOperands(); I != N; ++I) {
            if (ConstantExpr *Op = dyn_cast<ConstantExpr>(V->getOperand(I)))
              Stack.push_back(Op);
          }
        }
      }
    }
  }

  if (MDNode *MD = I.getMetadata(LLVMContext::MD_fpmath)) {
    Assert1(I.getType()->isFPOrFPVectorTy(),
            "fpmath requires a floating point result!", &I);
    Assert1(MD->getNumOperands() == 1, "fpmath takes one operand!", &I);
    Value *Op0 = MD->getOperand(0);
    if (ConstantFP *CFP0 = dyn_cast_or_null<ConstantFP>(Op0)) {
      APFloat Accuracy = CFP0->getValueAPF();
      Assert1(Accuracy.isFiniteNonZero() && !Accuracy.isNegative(),
              "fpmath accuracy not a positive number!", &I);
    } else {
      Assert1(false, "invalid fpmath accuracy!", &I);
    }
  }

  MDNode *MD = I.getMetadata(LLVMContext::MD_range);
  Assert1(!MD || isa<LoadInst>(I), "Ranges are only for loads!", &I);

  if (!DisableDebugInfoVerifier) {
    MD = I.getMetadata(LLVMContext::MD_dbg);
    Finder.processLocation(*M, DILocation(MD));
  }

  InstsInThisBlock.insert(&I);
}

/// VerifyIntrinsicType - Verify that the specified type (which comes from an
/// intrinsic argument or return value) matches the type constraints specified
/// by the .td file (e.g. an "any integer" argument really is an integer).
///
/// This return true on error but does not print a message.
bool Verifier::VerifyIntrinsicType(Type *Ty,
                                   ArrayRef<Intrinsic::IITDescriptor> &Infos,
                                   SmallVectorImpl<Type*> &ArgTys) {
  using namespace Intrinsic;

  // If we ran out of descriptors, there are too many arguments.
  if (Infos.empty()) return true;
  IITDescriptor D = Infos.front();
  Infos = Infos.slice(1);

  switch (D.Kind) {
  case IITDescriptor::Void: return !Ty->isVoidTy();
  case IITDescriptor::VarArg: return true;
  case IITDescriptor::MMX:  return !Ty->isX86_MMXTy();
  case IITDescriptor::Metadata: return !Ty->isMetadataTy();
  case IITDescriptor::Half: return !Ty->isHalfTy();
  case IITDescriptor::Float: return !Ty->isFloatTy();
  case IITDescriptor::Double: return !Ty->isDoubleTy();
  case IITDescriptor::Integer: return !Ty->isIntegerTy(D.Integer_Width);
  case IITDescriptor::Vector: {
    VectorType *VT = dyn_cast<VectorType>(Ty);
    return VT == 0 || VT->getNumElements() != D.Vector_Width ||
           VerifyIntrinsicType(VT->getElementType(), Infos, ArgTys);
  }
  case IITDescriptor::Pointer: {
    PointerType *PT = dyn_cast<PointerType>(Ty);
    return PT == 0 || PT->getAddressSpace() != D.Pointer_AddressSpace ||
           VerifyIntrinsicType(PT->getElementType(), Infos, ArgTys);
  }

  case IITDescriptor::Struct: {
    StructType *ST = dyn_cast<StructType>(Ty);
    if (ST == 0 || ST->getNumElements() != D.Struct_NumElements)
      return true;

    for (unsigned i = 0, e = D.Struct_NumElements; i != e; ++i)
      if (VerifyIntrinsicType(ST->getElementType(i), Infos, ArgTys))
        return true;
    return false;
  }

  case IITDescriptor::Argument:
    // Two cases here - If this is the second occurrence of an argument, verify
    // that the later instance matches the previous instance.
    if (D.getArgumentNumber() < ArgTys.size())
      return Ty != ArgTys[D.getArgumentNumber()];

    // Otherwise, if this is the first instance of an argument, record it and
    // verify the "Any" kind.
    assert(D.getArgumentNumber() == ArgTys.size() && "Table consistency error");
    ArgTys.push_back(Ty);

    switch (D.getArgumentKind()) {
    case IITDescriptor::AK_AnyInteger: return !Ty->isIntOrIntVectorTy();
    case IITDescriptor::AK_AnyFloat:   return !Ty->isFPOrFPVectorTy();
    case IITDescriptor::AK_AnyVector:  return !isa<VectorType>(Ty);
    case IITDescriptor::AK_AnyPointer: return !isa<PointerType>(Ty);
    }
    llvm_unreachable("all argument kinds not covered");

  case IITDescriptor::ExtendVecArgument:
    // This may only be used when referring to a previous vector argument.
    return D.getArgumentNumber() >= ArgTys.size() ||
           !isa<VectorType>(ArgTys[D.getArgumentNumber()]) ||
           VectorType::getExtendedElementVectorType(
                       cast<VectorType>(ArgTys[D.getArgumentNumber()])) != Ty;

  case IITDescriptor::TruncVecArgument:
    // This may only be used when referring to a previous vector argument.
    return D.getArgumentNumber() >= ArgTys.size() ||
           !isa<VectorType>(ArgTys[D.getArgumentNumber()]) ||
           VectorType::getTruncatedElementVectorType(
                         cast<VectorType>(ArgTys[D.getArgumentNumber()])) != Ty;
  }
  llvm_unreachable("unhandled");
}

/// \brief Verify if the intrinsic has variable arguments.
/// This method is intended to be called after all the fixed arguments have been
/// verified first.
///
/// This method returns true on error and does not print an error message.
bool
Verifier::VerifyIntrinsicIsVarArg(bool isVarArg,
                                  ArrayRef<Intrinsic::IITDescriptor> &Infos) {
  using namespace Intrinsic;

  // If there are no descriptors left, then it can't be a vararg.
  if (Infos.empty())
    return isVarArg ? true : false;

  // There should be only one descriptor remaining at this point.
  if (Infos.size() != 1)
    return true;

  // Check and verify the descriptor.
  IITDescriptor D = Infos.front();
  Infos = Infos.slice(1);
  if (D.Kind == IITDescriptor::VarArg)
    return isVarArg ? false : true;

  return true;
}

/// visitIntrinsicFunction - Allow intrinsics to be verified in different ways.
///
void Verifier::visitIntrinsicFunctionCall(Intrinsic::ID ID, CallInst &CI) {
  Function *IF = CI.getCalledFunction();
  Assert1(IF->isDeclaration(), "Intrinsic functions should never be defined!",
          IF);

  // Verify that the intrinsic prototype lines up with what the .td files
  // describe.
  FunctionType *IFTy = IF->getFunctionType();
  bool IsVarArg = IFTy->isVarArg();

  SmallVector<Intrinsic::IITDescriptor, 8> Table;
  getIntrinsicInfoTableEntries(ID, Table);
  ArrayRef<Intrinsic::IITDescriptor> TableRef = Table;

  SmallVector<Type *, 4> ArgTys;
  Assert1(!VerifyIntrinsicType(IFTy->getReturnType(), TableRef, ArgTys),
          "Intrinsic has incorrect return type!", IF);
  for (unsigned i = 0, e = IFTy->getNumParams(); i != e; ++i)
    Assert1(!VerifyIntrinsicType(IFTy->getParamType(i), TableRef, ArgTys),
            "Intrinsic has incorrect argument type!", IF);

  // Verify if the intrinsic call matches the vararg property.
  if (IsVarArg)
    Assert1(!VerifyIntrinsicIsVarArg(IsVarArg, TableRef),
            "Intrinsic was not defined with variable arguments!", IF);
  else
    Assert1(!VerifyIntrinsicIsVarArg(IsVarArg, TableRef),
            "Callsite was not defined with variable arguments!", IF);

  // All descriptors should be absorbed by now.
  Assert1(TableRef.empty(), "Intrinsic has too few arguments!", IF);

  // Now that we have the intrinsic ID and the actual argument types (and we
  // know they are legal for the intrinsic!) get the intrinsic name through the
  // usual means.  This allows us to verify the mangling of argument types into
  // the name.
  Assert1(Intrinsic::getName(ID, ArgTys) == IF->getName(),
          "Intrinsic name not mangled correctly for type arguments!", IF);

  // If the intrinsic takes MDNode arguments, verify that they are either global
  // or are local to *this* function.
  for (unsigned i = 0, e = CI.getNumArgOperands(); i != e; ++i)
    if (MDNode *MD = dyn_cast<MDNode>(CI.getArgOperand(i)))
      visitMDNode(*MD, CI.getParent()->getParent());

  switch (ID) {
  default:
    break;
  case Intrinsic::ctlz:  // llvm.ctlz
  case Intrinsic::cttz:  // llvm.cttz
    Assert1(isa<ConstantInt>(CI.getArgOperand(1)),
            "is_zero_undef argument of bit counting intrinsics must be a "
            "constant int", &CI);
    break;
  case Intrinsic::dbg_declare: {  // llvm.dbg.declare
    Assert1(CI.getArgOperand(0) && isa<MDNode>(CI.getArgOperand(0)),
                "invalid llvm.dbg.declare intrinsic call 1", &CI);
    MDNode *MD = cast<MDNode>(CI.getArgOperand(0));
    Assert1(MD->getNumOperands() == 1,
                "invalid llvm.dbg.declare intrinsic call 2", &CI);
    if (!DisableDebugInfoVerifier)
      Finder.processDeclare(*M, cast<DbgDeclareInst>(&CI));
  } break;
  case Intrinsic::dbg_value: { //llvm.dbg.value
    if (!DisableDebugInfoVerifier) {
      Assert1(CI.getArgOperand(0) && isa<MDNode>(CI.getArgOperand(0)),
              "invalid llvm.dbg.value intrinsic call 1", &CI);
      Finder.processValue(*M, cast<DbgValueInst>(&CI));
    }
    break;
  }
  case Intrinsic::memcpy:
  case Intrinsic::memmove:
  case Intrinsic::memset:
    Assert1(isa<ConstantInt>(CI.getArgOperand(3)),
            "alignment argument of memory intrinsics must be a constant int",
            &CI);
    Assert1(isa<ConstantInt>(CI.getArgOperand(4)),
            "isvolatile argument of memory intrinsics must be a constant int",
            &CI);
    break;
  case Intrinsic::gcroot:
  case Intrinsic::gcwrite:
  case Intrinsic::gcread:
    if (ID == Intrinsic::gcroot) {
      AllocaInst *AI =
        dyn_cast<AllocaInst>(CI.getArgOperand(0)->stripPointerCasts());
      Assert1(AI, "llvm.gcroot parameter #1 must be an alloca.", &CI);
      Assert1(isa<Constant>(CI.getArgOperand(1)),
              "llvm.gcroot parameter #2 must be a constant.", &CI);
      if (!AI->getType()->getElementType()->isPointerTy()) {
        Assert1(!isa<ConstantPointerNull>(CI.getArgOperand(1)),
                "llvm.gcroot parameter #1 must either be a pointer alloca, "
                "or argument #2 must be a non-null constant.", &CI);
      }
    }

    Assert1(CI.getParent()->getParent()->hasGC(),
            "Enclosing function does not use GC.", &CI);
    break;
  case Intrinsic::init_trampoline:
    Assert1(isa<Function>(CI.getArgOperand(1)->stripPointerCasts()),
            "llvm.init_trampoline parameter #2 must resolve to a function.",
            &CI);
    break;
  case Intrinsic::prefetch:
    Assert1(isa<ConstantInt>(CI.getArgOperand(1)) &&
            isa<ConstantInt>(CI.getArgOperand(2)) &&
            cast<ConstantInt>(CI.getArgOperand(1))->getZExtValue() < 2 &&
            cast<ConstantInt>(CI.getArgOperand(2))->getZExtValue() < 4,
            "invalid arguments to llvm.prefetch",
            &CI);
    break;
  case Intrinsic::stackprotector:
    Assert1(isa<AllocaInst>(CI.getArgOperand(1)->stripPointerCasts()),
            "llvm.stackprotector parameter #2 must resolve to an alloca.",
            &CI);
    break;
  case Intrinsic::lifetime_start:
  case Intrinsic::lifetime_end:
  case Intrinsic::invariant_start:
    Assert1(isa<ConstantInt>(CI.getArgOperand(0)),
            "size argument of memory use markers must be a constant integer",
            &CI);
    break;
  case Intrinsic::invariant_end:
    Assert1(isa<ConstantInt>(CI.getArgOperand(1)),
            "llvm.invariant.end parameter #2 must be a constant integer", &CI);
    break;
  }
}

void Verifier::verifyDebugInfo() {
  // Verify Debug Info.
  if (!DisableDebugInfoVerifier) {
    for (DebugInfoFinder::iterator I = Finder.compile_unit_begin(),
         E = Finder.compile_unit_end(); I != E; ++I)
      Assert1(DICompileUnit(*I).Verify(), "DICompileUnit does not Verify!", *I);
    for (DebugInfoFinder::iterator I = Finder.subprogram_begin(),
         E = Finder.subprogram_end(); I != E; ++I)
      Assert1(DISubprogram(*I).Verify(), "DISubprogram does not Verify!", *I);
    for (DebugInfoFinder::iterator I = Finder.global_variable_begin(),
         E = Finder.global_variable_end(); I != E; ++I)
      Assert1(DIGlobalVariable(*I).Verify(),
              "DIGlobalVariable does not Verify!", *I);
    for (DebugInfoFinder::iterator I = Finder.type_begin(),
         E = Finder.type_end(); I != E; ++I)
      Assert1(DIType(*I).Verify(), "DIType does not Verify!", *I);
    for (DebugInfoFinder::iterator I = Finder.scope_begin(),
         E = Finder.scope_end(); I != E; ++I)
      Assert1(DIScope(*I).Verify(), "DIScope does not Verify!", *I);
  }
}

//===----------------------------------------------------------------------===//
//  Implement the public interfaces to this file...
//===----------------------------------------------------------------------===//

bool llvm::verifyFunction(const Function &f, raw_ostream *OS) {
  Function &F = const_cast<Function &>(f);
  assert(!F.isDeclaration() && "Cannot verify external functions");

  raw_null_ostream NullStr;
  Verifier V(OS ? *OS : NullStr);

  // Note that this function's return value is inverted from what you would
  // expect of a function called "verify".
  return !V.verify(F);
}

bool llvm::verifyModule(const Module &M, raw_ostream *OS) {
  raw_null_ostream NullStr;
  Verifier V(OS ? *OS : NullStr);

  bool Broken = false;
  for (Module::const_iterator I = M.begin(), E = M.end(); I != E; ++I)
    if (!I->isDeclaration())
      Broken |= !V.verify(*I);

  // Note that this function's return value is inverted from what you would
  // expect of a function called "verify".
  return !V.verify(M) || Broken;
}

namespace {
struct VerifierLegacyPass : public FunctionPass {
  static char ID;

  Verifier V;
  bool FatalErrors;

  VerifierLegacyPass() : FunctionPass(ID), FatalErrors(true) {
    initializeVerifierLegacyPassPass(*PassRegistry::getPassRegistry());
  }
  explicit VerifierLegacyPass(bool FatalErrors)
      : FunctionPass(ID), V(dbgs()), FatalErrors(FatalErrors) {
    initializeVerifierLegacyPassPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override {
    if (!V.verify(F) && FatalErrors)
      report_fatal_error("Broken function found, compilation aborted!");

    return false;
  }

  bool doFinalization(Module &M) override {
    if (!V.verify(M) && FatalErrors)
      report_fatal_error("Broken module found, compilation aborted!");

    return false;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }
};
}

char VerifierLegacyPass::ID = 0;
INITIALIZE_PASS(VerifierLegacyPass, "verify", "Module Verifier", false, false)

FunctionPass *llvm::createVerifierPass(bool FatalErrors) {
  return new VerifierLegacyPass(FatalErrors);
}

PreservedAnalyses VerifierPass::run(Module *M) {
  if (verifyModule(*M, &dbgs()) && FatalErrors)
    report_fatal_error("Broken module found, compilation aborted!");

  return PreservedAnalyses::all();
}

PreservedAnalyses VerifierPass::run(Function *F) {
  if (verifyFunction(*F, &dbgs()) && FatalErrors)
    report_fatal_error("Broken function found, compilation aborted!");

  return PreservedAnalyses::all();
}
