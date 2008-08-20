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
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/IRBuilder.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprObjC.h"

#include <vector>
#include <map>

namespace llvm {
  class BasicBlock;
  class Module;
}

namespace clang {
  class ASTContext;
  class Decl;
  class EnumConstantDecl;
  class FunctionDecl;
  class FunctionTypeProto;
  class LabelStmt;
  class ObjCMethodDecl;
  class TargetInfo;
  class VarDecl;

namespace CodeGen {
  class CodeGenModule;
  class CodeGenTypes;
  class CGRecordLayout;  

/// RValue - This trivial value class is used to represent the result of an
/// expression that is evaluated.  It can be one of three things: either a
/// simple LLVM SSA value, a pair of SSA values for complex numbers, or the
/// address of an aggregate value in memory.
class RValue {
  llvm::Value *V1, *V2;
  // TODO: Encode this into the low bit of pointer for more efficient
  // return-by-value.
  enum { Scalar, Complex, Aggregate } Flavor;
  
  // FIXME: Aggregate rvalues need to retain information about whether they are
  // volatile or not.
public:
  
  bool isScalar() const { return Flavor == Scalar; }
  bool isComplex() const { return Flavor == Complex; }
  bool isAggregate() const { return Flavor == Aggregate; }
  
  /// getScalar() - Return the Value* of this scalar value.
  llvm::Value *getScalarVal() const {
    assert(isScalar() && "Not a scalar!");
    return V1;
  }

  /// getComplexVal - Return the real/imag components of this complex value.
  ///
  std::pair<llvm::Value *, llvm::Value *> getComplexVal() const {
    return std::pair<llvm::Value *, llvm::Value *>(V1, V2);
  }
  
  /// getAggregateAddr() - Return the Value* of the address of the aggregate.
  llvm::Value *getAggregateAddr() const {
    assert(isAggregate() && "Not an aggregate!");
    return V1;
  }
  
  static RValue get(llvm::Value *V) {
    RValue ER;
    ER.V1 = V;
    ER.Flavor = Scalar;
    return ER;
  }
  static RValue getComplex(llvm::Value *V1, llvm::Value *V2) {
    RValue ER;
    ER.V1 = V1;
    ER.V2 = V2;
    ER.Flavor = Complex;
    return ER;
  }
  static RValue getComplex(const std::pair<llvm::Value *, llvm::Value *> &C) {
    RValue ER;
    ER.V1 = C.first;
    ER.V2 = C.second;
    ER.Flavor = Complex;
    return ER;
  }
  static RValue getAggregate(llvm::Value *V) {
    RValue ER;
    ER.V1 = V;
    ER.Flavor = Aggregate;
    return ER;
  }
};


/// LValue - This represents an lvalue references.  Because C/C++ allow
/// bitfields, this is not a simple LLVM pointer, it may be a pointer plus a
/// bitrange.
class LValue {
  // FIXME: alignment?
  
  enum {
    Simple,       // This is a normal l-value, use getAddress().
    VectorElt,    // This is a vector element l-value (V[i]), use getVector*
    BitField,     // This is a bitfield l-value, use getBitfield*.
    ExtVectorElt  // This is an extended vector subset, use getExtVectorComp
  } LVType;
  
  llvm::Value *V;
  
  union {
    // Index into a vector subscript: V[i]
    llvm::Value *VectorIdx;

    // ExtVector element subset: V.xyx
    llvm::Constant *VectorElts;
    
    // BitField start bit and size
    struct {
      unsigned short StartBit;
      unsigned short Size;
      bool IsSigned;
    } BitfieldData;
  };

  bool Volatile:1;
  // FIXME: set but never used, what effect should it have?
  bool Restrict:1;

private:
  static void SetQualifiers(unsigned Qualifiers, LValue& R) {
    R.Volatile = (Qualifiers&QualType::Volatile)!=0;
    R.Restrict = (Qualifiers&QualType::Restrict)!=0;
  }

public:
  bool isSimple() const { return LVType == Simple; }
  bool isVectorElt() const { return LVType == VectorElt; }
  bool isBitfield() const { return LVType == BitField; }
  bool isExtVectorElt() const { return LVType == ExtVectorElt; }
  
  bool isVolatileQualified() const { return Volatile; }
  bool isRestrictQualified() const { return Restrict; }

  // simple lvalue
  llvm::Value *getAddress() const { assert(isSimple()); return V; }
  // vector elt lvalue
  llvm::Value *getVectorAddr() const { assert(isVectorElt()); return V; }
  llvm::Value *getVectorIdx() const { assert(isVectorElt()); return VectorIdx; }
  // extended vector elements.
  llvm::Value *getExtVectorAddr() const { assert(isExtVectorElt()); return V; }
  llvm::Constant *getExtVectorElts() const {
    assert(isExtVectorElt());
    return VectorElts;
  }
  // bitfield lvalue
  llvm::Value *getBitfieldAddr() const { assert(isBitfield()); return V; }
  unsigned short getBitfieldStartBit() const {
    assert(isBitfield());
    return BitfieldData.StartBit;
  }
  unsigned short getBitfieldSize() const {
    assert(isBitfield());
    return BitfieldData.Size;
  }
  bool isBitfieldSigned() const {
    assert(isBitfield());
    return BitfieldData.IsSigned;
  }

  static LValue MakeAddr(llvm::Value *V, unsigned Qualifiers) {
    LValue R;
    R.LVType = Simple;
    R.V = V;
    SetQualifiers(Qualifiers,R);
    return R;
  }
  
  static LValue MakeVectorElt(llvm::Value *Vec, llvm::Value *Idx,
                              unsigned Qualifiers) {
    LValue R;
    R.LVType = VectorElt;
    R.V = Vec;
    R.VectorIdx = Idx;
    SetQualifiers(Qualifiers,R);
    return R;
  }
  
  static LValue MakeExtVectorElt(llvm::Value *Vec, llvm::Constant *Elts,
                                 unsigned Qualifiers) {
    LValue R;
    R.LVType = ExtVectorElt;
    R.V = Vec;
    R.VectorElts = Elts;
    SetQualifiers(Qualifiers,R);
    return R;
  }

  static LValue MakeBitfield(llvm::Value *V, unsigned short StartBit,
                             unsigned short Size, bool IsSigned,
                             unsigned Qualifiers) {
    LValue R;
    R.LVType = BitField;
    R.V = V;
    R.BitfieldData.StartBit = StartBit;
    R.BitfieldData.Size = Size;
    R.BitfieldData.IsSigned = IsSigned;
    SetQualifiers(Qualifiers,R);
    return R;
  }
};

/// CodeGenFunction - This class organizes the per-function state that is used
/// while generating LLVM code.
class CodeGenFunction {
public:
  CodeGenModule &CGM;  // Per-module state.
  TargetInfo &Target;
  
  typedef std::pair<llvm::Value *, llvm::Value *> ComplexPairTy;
  llvm::IRBuilder<> Builder;
  
  // Holds the Decl for the current function or method
  const Decl *CurFuncDecl;
  QualType FnRetTy;
  llvm::Function *CurFn;

  /// AllocaInsertPoint - This is an instruction in the entry block before which
  /// we prefer to insert allocas.
  llvm::Instruction *AllocaInsertPt;

  const llvm::Type *LLVMIntTy;
  uint32_t LLVMPointerWidth;
  
private:
  /// LabelIDs - Track arbitrary ids assigned to labels for use in
  /// implementing the GCC address-of-label extension and indirect
  /// goto. IDs are assigned to labels inside getIDForAddrOfLabel().
  std::map<const LabelStmt*, unsigned> LabelIDs;

  /// IndirectSwitches - Record the list of switches for indirect
  /// gotos. Emission of the actual switching code needs to be delayed
  /// until all AddrLabelExprs have been seen.
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
  
  /// SwitchInsn - This is nearest current switch instruction. It is null if
  /// if current context is not in a switch.
  llvm::SwitchInst *SwitchInsn;

  /// CaseRangeBlock - This block holds if condition check for last case 
  /// statement range in current switch instruction.
  llvm::BasicBlock *CaseRangeBlock;

public:
  CodeGenFunction(CodeGenModule &cgm);
  
  ASTContext &getContext() const;

  void GenerateObjCMethod(const ObjCMethodDecl *OMD);
  void GenerateCode(const FunctionDecl *FD,
                    llvm::Function *Fn);
  void GenerateFunction(const Stmt *Body);
  
  const llvm::Type *ConvertType(QualType T);

  /// LoadObjCSelf - Load the value of self. This function is only
  /// valid while generating code for an Objective-C method.
  llvm::Value *LoadObjCSelf();

  /// isObjCPointerType - Return true if the specificed AST type will map onto
  /// some Objective-C pointer type.
  static bool isObjCPointerType(QualType T);

  /// hasAggregateLLVMType - Return true if the specified AST type will map into
  /// an aggregate LLVM type or is void.
  static bool hasAggregateLLVMType(QualType T);
  
  /// getBasicBlockForLabel - Return the LLVM basicblock that the specified
  /// label maps to.
  llvm::BasicBlock *getBasicBlockForLabel(const LabelStmt *S);
  
  
  void EmitBlock(llvm::BasicBlock *BB);
  
  /// ErrorUnsupported - Print out an error that codegen doesn't support the
  /// specified stmt yet.
  void ErrorUnsupported(const Stmt *S, const char *Type);

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
  RValue EmitAnyExpr(const Expr *E, llvm::Value *AggLoc = 0, 
                     bool isAggLocVolatile = false);

  /// isDummyBlock - Return true if BB is an empty basic block
  /// with no predecessors.
  static bool isDummyBlock(const llvm::BasicBlock *BB);

  /// StartBlock - Start new block named N. If insert block is a dummy block
  /// then reuse it.
  void StartBlock(const char *N);

  /// getCGRecordLayout - Return record layout info.
  const CGRecordLayout *getCGRecordLayout(CodeGenTypes &CGT, QualType RTy);

  /// GetAddrOfStaticLocalVar - Return the address of a static local variable.
  llvm::Constant *GetAddrOfStaticLocalVar(const VarDecl *BVD);

  /// getAccessedFieldNo - Given an encoded value and a result number, return
  /// the input field number being accessed.
  static unsigned getAccessedFieldNo(unsigned Idx, const llvm::Constant *Elts);

  unsigned GetIDForAddrOfLabel(const LabelStmt *L);

  //===--------------------------------------------------------------------===//
  //                            Declaration Emission
  //===--------------------------------------------------------------------===//
  
  void EmitDecl(const Decl &D);
  void EmitEnumConstantDecl(const EnumConstantDecl &D);
  void EmitBlockVarDecl(const VarDecl &D);
  void EmitLocalBlockVarDecl(const VarDecl &D);
  void EmitStaticBlockVarDecl(const VarDecl &D);

  /// EmitParmDecl - Emit a ParmVarDecl or an ImplicitParamDecl.
  void EmitParmDecl(const VarDecl &D, llvm::Value *Arg);
  
  //===--------------------------------------------------------------------===//
  //                             Statement Emission
  //===--------------------------------------------------------------------===//

  void EmitStmt(const Stmt *S);
  RValue EmitCompoundStmt(const CompoundStmt &S, bool GetLast = false,
                          llvm::Value *AggLoc = 0, bool isAggVol = false);
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
  void EmitBreakStmt();
  void EmitContinueStmt();
  void EmitSwitchStmt(const SwitchStmt &S);
  void EmitDefaultStmt(const DefaultStmt &S);
  void EmitCaseStmt(const CaseStmt &S);
  void EmitCaseStmtRange(const CaseStmt &S);
  void EmitAsmStmt(const AsmStmt &S);
  
  //===--------------------------------------------------------------------===//
  //                         LValue Expression Emission
  //===--------------------------------------------------------------------===//

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
  
  /// EmitLoadOfLValue - Given an expression that represents a value lvalue,
  /// this method emits the address of the lvalue, then loads the result as an
  /// rvalue, returning the rvalue.
  RValue EmitLoadOfLValue(LValue V, QualType LVType);
  RValue EmitLoadOfExtVectorElementLValue(LValue V, QualType LVType);
  RValue EmitLoadOfBitfieldLValue(LValue LV, QualType ExprType);

  
  /// EmitStoreThroughLValue - Store the specified rvalue into the specified
  /// lvalue, where both are guaranteed to the have the same type, and that type
  /// is 'Ty'.
  void EmitStoreThroughLValue(RValue Src, LValue Dst, QualType Ty);
  void EmitStoreThroughExtVectorComponentLValue(RValue Src, LValue Dst,
                                                QualType Ty);
  void EmitStoreThroughBitfieldLValue(RValue Src, LValue Dst, QualType Ty);
   
  // Note: only availabe for agg return types
  LValue EmitCallExprLValue(const CallExpr *E);
  
  LValue EmitDeclRefLValue(const DeclRefExpr *E);
  LValue EmitStringLiteralLValue(const StringLiteral *E);
  LValue EmitPredefinedLValue(const PredefinedExpr *E);
  LValue EmitUnaryOpLValue(const UnaryOperator *E);
  LValue EmitArraySubscriptExpr(const ArraySubscriptExpr *E);
  LValue EmitExtVectorElementExpr(const ExtVectorElementExpr *E);
  LValue EmitMemberExpr(const MemberExpr *E);
  LValue EmitCompoundLiteralLValue(const CompoundLiteralExpr *E);

  LValue EmitLValueForField(llvm::Value* Base, FieldDecl* Field,
                            bool isUnion, unsigned CVRQualifiers);
      
  LValue EmitObjCIvarRefLValue(const ObjCIvarRefExpr *E);
  //===--------------------------------------------------------------------===//
  //                         Scalar Expression Emission
  //===--------------------------------------------------------------------===//

  RValue EmitCallExpr(const CallExpr *E);

  RValue EmitCallExpr(Expr *FnExpr, CallExpr::const_arg_iterator ArgBeg,
                      CallExpr::const_arg_iterator ArgEnd);

  RValue EmitCallExpr(llvm::Value *Callee, QualType FnType,
                      CallExpr::const_arg_iterator ArgBeg,
                      CallExpr::const_arg_iterator ArgEnd);
  
  RValue EmitBuiltinExpr(unsigned BuiltinID, const CallExpr *E);

  llvm::Value *EmitX86BuiltinExpr(unsigned BuiltinID, const CallExpr *E);
  llvm::Value *EmitPPCBuiltinExpr(unsigned BuiltinID, const CallExpr *E);
  
  llvm::Value *EmitShuffleVector(llvm::Value* V1, llvm::Value *V2, ...);
  llvm::Value *EmitVector(llvm::Value * const *Vals, unsigned NumVals,
                          bool isSplat = false);
  
  llvm::Value *EmitObjCProtocolExpr(const ObjCProtocolExpr *E);
  llvm::Value *EmitObjCStringLiteral(const ObjCStringLiteral *E);
  llvm::Value *EmitObjCSelectorExpr(const ObjCSelectorExpr *E);
  llvm::Value *EmitObjCMessageExpr(const ObjCMessageExpr *E);


  //===--------------------------------------------------------------------===//
  //                           Expression Emission
  //===--------------------------------------------------------------------===//

  // Expressions are broken into three classes: scalar, complex, aggregate.
  
  /// EmitScalarExpr - Emit the computation of the specified expression of
  /// LLVM scalar type, returning the result.
  llvm::Value *EmitScalarExpr(const Expr *E);
  
  /// EmitScalarConversion - Emit a conversion from the specified type to the
  /// specified destination type, both of which are LLVM scalar types.
  llvm::Value *EmitScalarConversion(llvm::Value *Src, QualType SrcTy,
                                    QualType DstTy);
  
  /// EmitComplexToScalarConversion - Emit a conversion from the specified
  /// complex type to the specified destination type, where the destination
  /// type is an LLVM scalar type.
  llvm::Value *EmitComplexToScalarConversion(ComplexPairTy Src, QualType SrcTy,
                                             QualType DstTy);
  
  
  /// EmitAggExpr - Emit the computation of the specified expression of
  /// aggregate type.  The result is computed into DestPtr.  Note that if
  /// DestPtr is null, the value of the aggregate expression is not needed.
  void EmitAggExpr(const Expr *E, llvm::Value *DestPtr, bool VolatileDest);
  
  /// EmitComplexExpr - Emit the computation of the specified expression of
  /// complex type, returning the result.
  ComplexPairTy EmitComplexExpr(const Expr *E);
  
  /// EmitComplexExprIntoAddr - Emit the computation of the specified expression
  /// of complex type, storing into the specified Value*.
  void EmitComplexExprIntoAddr(const Expr *E, llvm::Value *DestAddr,
                               bool DestIsVolatile);
  /// LoadComplexFromAddr - Load a complex number from the specified address.
  ComplexPairTy LoadComplexFromAddr(llvm::Value *SrcAddr, bool SrcIsVolatile);

  /// GenerateStaticBlockVarDecl - return the the static
  /// declaration of local variable. 
  llvm::GlobalValue *GenerateStaticBlockVarDecl(const VarDecl &D,
                                                bool NoInit,
                                                const char *Separator);

  //===--------------------------------------------------------------------===//
  //                             Internal Helpers
  //===--------------------------------------------------------------------===//
 
private:
  /// EmitIndirectSwitches - Emit code for all of the switch
  /// instructions in IndirectSwitches.
  void EmitIndirectSwitches();
};
}  // end namespace CodeGen
}  // end namespace clang

#endif
