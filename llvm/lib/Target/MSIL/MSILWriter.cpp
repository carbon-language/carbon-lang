//===-- MSILWriter.cpp - Library for converting LLVM code to MSIL ---------===//
//
//		       The LLVM Compiler Infrastructure
//
// This file was developed by Roman Samoilov and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This library converts LLVM code to MSIL code.
//
//===----------------------------------------------------------------------===//

#include "MSILWriter.h"
#include "llvm/CallingConv.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Intrinsics.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/ParameterAttributes.h"
#include "llvm/TypeSymbolTable.h"
#include "llvm/Analysis/ConstantsScanner.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/InstVisitor.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/ADT/StringExtras.h"

namespace {
  // TargetMachine for the MSIL 
  struct VISIBILITY_HIDDEN MSILTarget : public TargetMachine {
    const TargetData DataLayout;       // Calculates type size & alignment

    MSILTarget(const Module &M, const std::string &FS)
      : DataLayout(&M) {}

    virtual bool WantsWholeFile() const { return true; }
    virtual bool addPassesToEmitWholeFile(PassManager &PM, std::ostream &Out,
                                         CodeGenFileType FileType, bool Fast);

    // This class always works, but shouldn't be the default in most cases.
    static unsigned getModuleMatchQuality(const Module &M) { return 1; }

    virtual const TargetData *getTargetData() const { return &DataLayout; }
  };
}


RegisterTarget<MSILTarget> X("msil", "  MSIL backend");

bool MSILModule::runOnModule(Module &M) {
  ModulePtr = &M;
  TD = &getAnalysis<TargetData>();
  bool Changed = false;
  // Find named types.  
  TypeSymbolTable& Table = M.getTypeSymbolTable();
  std::set<const Type *> Types = getAnalysis<FindUsedTypes>().getTypes();
  for (TypeSymbolTable::iterator I = Table.begin(), E = Table.end(); I!=E; ) {
    if (!isa<StructType>(I->second) && !isa<OpaqueType>(I->second))
      Table.remove(I++);
    else {
      std::set<const Type *>::iterator T = Types.find(I->second);
      if (T==Types.end())
        Table.remove(I++);
      else {
        Types.erase(T);
        ++I;
      }
    }
  }
  // Find unnamed types.
  unsigned RenameCounter = 0;
  for (std::set<const Type *>::const_iterator I = Types.begin(),
       E = Types.end(); I!=E; ++I)
    if (const StructType *STy = dyn_cast<StructType>(*I)) {
      while (ModulePtr->addTypeName("unnamed$"+utostr(RenameCounter), STy))
        ++RenameCounter;
      Changed = true;
    }
  // Pointer for FunctionPass.
  UsedTypes = &getAnalysis<FindUsedTypes>().getTypes();
  return Changed;
}


bool MSILWriter::runOnFunction(Function &F) {
  if (F.isDeclaration()) return false;
  LInfo = &getAnalysis<LoopInfo>();
  printFunction(F);
  return false;
}


bool MSILWriter::doInitialization(Module &M) {
  ModulePtr = &M;
  Mang = new Mangler(M); 
  Out << ".assembly extern mscorlib {}\n";
  Out << ".assembly MSIL {}\n\n";
  Out << "// External\n";
  printExternals();
  Out << "// Declarations\n";
  printDeclarations(M.getTypeSymbolTable());
  Out << "// Definitions\n";
  printGlobalVariables();
  return false;
}


bool MSILWriter::doFinalization(Module &M) {
  delete Mang;
  return false;
}


bool MSILWriter::isZeroValue(const Value* V) {
  if (const Constant *C = dyn_cast<Constant>(V))
    return C->isNullValue();
  return false;
}


std::string MSILWriter::getValueName(const Value* V) {
  // Name into the quotes allow control and space characters.
  return "'"+Mang->getValueName(V)+"'";
}


std::string MSILWriter::getLabelName(const std::string& Name) {
  if (Name.find('.')!=std::string::npos) {
    std::string Tmp(Name);
    // Replace unaccepable characters in the label name.
    for (std::string::iterator I = Tmp.begin(), E = Tmp.end(); I!=E; ++I)
      if (*I=='.') *I = '@';
    return Tmp;
  }
  return Name;
}


std::string MSILWriter::getLabelName(const Value* V) {
  return getLabelName(Mang->getValueName(V));
}


std::string MSILWriter::getConvModopt(unsigned CallingConvID) {
  switch (CallingConvID) {
  case CallingConv::C:
  case CallingConv::Cold:
  case CallingConv::Fast:
    return "modopt([mscorlib]System.Runtime.CompilerServices.CallConvCdecl) ";
  case CallingConv::X86_FastCall:
    return "modopt([mscorlib]System.Runtime.CompilerServices.CallConvFastcall) ";
  case CallingConv::X86_StdCall:
    return "modopt([mscorlib]System.Runtime.CompilerServices.CallConvStdcall) ";
  default:
    cerr << "CallingConvID = " << CallingConvID << '\n';
    assert(0 && "Unsupported calling convention");
  }
}


std::string MSILWriter::getArrayTypeName(Type::TypeID TyID, const Type* Ty) {
  std::string Tmp = "";
  const Type* ElemTy = Ty;
  assert(Ty->getTypeID()==TyID && "Invalid type passed");
  // Walk trought array element types.
  for (;;) {
    // Multidimensional array.
    if (ElemTy->getTypeID()==TyID) {
      if (const ArrayType* ATy = dyn_cast<ArrayType>(ElemTy))
        Tmp += utostr(ATy->getNumElements());
      else if (const VectorType* VTy = dyn_cast<VectorType>(ElemTy))
        Tmp += utostr(VTy->getNumElements());
      ElemTy = cast<SequentialType>(ElemTy)->getElementType();
    }
    // Base element type found.
    if (ElemTy->getTypeID()!=TyID) break;
    Tmp += ",";
  }
  return getTypeName(ElemTy)+"["+Tmp+"]";
}


std::string MSILWriter::getPrimitiveTypeName(const Type* Ty, bool isSigned) {
  unsigned NumBits = 0;
  switch (Ty->getTypeID()) {
  case Type::VoidTyID:
    return "void ";
  case Type::IntegerTyID:
    NumBits = getBitWidth(Ty);
    if(NumBits==1)
      return "bool ";
    if (!isSigned)
      return "unsigned int"+utostr(NumBits)+" ";
    return "int"+utostr(NumBits)+" ";
  case Type::FloatTyID:
    return "float32 ";
  case Type::DoubleTyID:
    return "float64 "; 
  default:
    cerr << "Type = " << *Ty << '\n';
    assert(0 && "Invalid primitive type");
  }
}


std::string MSILWriter::getTypeName(const Type* Ty, bool isSigned) {
  if (Ty->isPrimitiveType() || Ty->isInteger())
    return getPrimitiveTypeName(Ty,isSigned);
  // FIXME: "OpaqueType" support
  switch (Ty->getTypeID()) {
  case Type::PointerTyID:
    return "void* ";
  case Type::StructTyID:
    return "valuetype '"+ModulePtr->getTypeName(Ty)+"' ";
  case Type::ArrayTyID:
    return "valuetype '"+getArrayTypeName(Ty->getTypeID(),Ty)+"' ";
  case Type::VectorTyID:
    return "valuetype '"+getArrayTypeName(Ty->getTypeID(),Ty)+"' ";
  default:
    cerr << "Type = " << *Ty << '\n';
    assert(0 && "Invalid type in getTypeName()");
  }
}


MSILWriter::ValueType MSILWriter::getValueLocation(const Value* V) {
  // Function argument
  if (isa<Argument>(V))
    return ArgumentVT;
  // Function
  else if (const Function* F = dyn_cast<Function>(V))
    return F->hasInternalLinkage() ? InternalVT : GlobalVT;
  // Variable
  else if (const GlobalVariable* G = dyn_cast<GlobalVariable>(V))
    return G->hasInternalLinkage() ? InternalVT : GlobalVT;
  // Constant
  else if (isa<Constant>(V))
    return isa<ConstantExpr>(V) ? ConstExprVT : ConstVT;
  // Local variable
  return LocalVT;
}


std::string MSILWriter::getTypePostfix(const Type* Ty, bool Expand,
                                       bool isSigned) {
  unsigned NumBits = 0;
  switch (Ty->getTypeID()) {
  // Integer constant, expanding for stack operations.
  case Type::IntegerTyID:
    NumBits = getBitWidth(Ty);
    // Expand integer value to "int32" or "int64".
    if (Expand) return (NumBits<=32 ? "i4" : "i8");
    if (NumBits==1) return "i1";
    return (isSigned ? "i" : "u")+utostr(NumBits/8);
  // Float constant.
  case Type::FloatTyID:
    return "r4";
  case Type::DoubleTyID:
    return "r8";
  case Type::PointerTyID:
    return "i"+utostr(TD->getTypeSize(Ty));
  default:
    cerr << "TypeID = " << Ty->getTypeID() << '\n';
    assert(0 && "Invalid type in TypeToPostfix()");
  }
}


void MSILWriter::printPtrLoad(uint64_t N) {
  switch (ModulePtr->getPointerSize()) {
  case Module::Pointer32:
    printSimpleInstruction("ldc.i4",utostr(N).c_str());
    // FIXME: Need overflow test?
    assert(N<0xFFFFFFFF && "32-bit pointer overflowed");
    break;
  case Module::Pointer64:
    printSimpleInstruction("ldc.i8",utostr(N).c_str());
    break;
  default:
    assert(0 && "Module use not supporting pointer size");
  }
}


void MSILWriter::printConstLoad(const Constant* C) {
  if (const ConstantInt* CInt = dyn_cast<ConstantInt>(C)) {
    // Integer constant
    Out << "\tldc." << getTypePostfix(C->getType(),true) << '\t';
    if (CInt->isMinValue(true))
      Out << CInt->getSExtValue();
    else
      Out << CInt->getZExtValue();
  } else if (const ConstantFP* CFp = dyn_cast<ConstantFP>(C)) {
    // Float constant
    Out << "\tldc." << getTypePostfix(C->getType(),true) << '\t' <<
      CFp->getValue();
  } else {
    cerr << "Constant = " << *C << '\n';
    assert(0 && "Invalid constant value");
  }
  Out << '\n';
}


void MSILWriter::printValueLoad(const Value* V) {
  switch (getValueLocation(V)) {
  // Global variable or function address.
  case GlobalVT:
  case InternalVT:
    if (const Function* F = dyn_cast<Function>(V)) {
      std::string Name = getConvModopt(F->getCallingConv())+getValueName(F);
      printSimpleInstruction("ldftn",
        getCallSignature(F->getFunctionType(),NULL,Name).c_str());
    } else {
      const Type* ElemTy = cast<PointerType>(V->getType())->getElementType();
      std::string Tmp = getTypeName(ElemTy)+getValueName(V);
      printSimpleInstruction("ldsflda",Tmp.c_str());
    }
    break;
  // Function argument.
  case ArgumentVT:
    printSimpleInstruction("ldarg",getValueName(V).c_str());
    break;
  // Local function variable.
  case LocalVT:
    printSimpleInstruction("ldloc",getValueName(V).c_str());
    break;
  // Constant value.
  case ConstVT:
    if (isa<ConstantPointerNull>(V))
      printPtrLoad(0);
    else
      printConstLoad(cast<Constant>(V));
    break;
  // Constant expression.
  case ConstExprVT:
    printConstantExpr(cast<ConstantExpr>(V));
    break;
  default:
    cerr << "Value = " << *V << '\n';
    assert(0 && "Invalid value location");
  }
}


void MSILWriter::printValueSave(const Value* V) {
  switch (getValueLocation(V)) {
  case ArgumentVT:
    printSimpleInstruction("starg",getValueName(V).c_str());
    break;
  case LocalVT:
    printSimpleInstruction("stloc",getValueName(V).c_str());
    break;
  default:
    cerr << "Value  = " << *V << '\n';
    assert(0 && "Invalid value location");
  }
}


void MSILWriter::printBinaryInstruction(const char* Name, const Value* Left,
                                        const Value* Right) {
  printValueLoad(Left);
  printValueLoad(Right);
  Out << '\t' << Name << '\n';
}


void MSILWriter::printSimpleInstruction(const char* Inst, const char* Operand) {
  if(Operand) 
    Out << '\t' << Inst << '\t' << Operand << '\n';
  else
    Out << '\t' << Inst << '\n';
}


void MSILWriter::printPHICopy(const BasicBlock* Src, const BasicBlock* Dst) {
  for (BasicBlock::const_iterator I = Dst->begin(), E = Dst->end();
       isa<PHINode>(I); ++I) {
    const PHINode* Phi = cast<PHINode>(I);
    const Value* Val = Phi->getIncomingValueForBlock(Src);
    if (isa<UndefValue>(Val)) continue;
    printValueLoad(Val);
    printValueSave(Phi);
  }
}


void MSILWriter::printBranchToBlock(const BasicBlock* CurrBB,
                                    const BasicBlock* TrueBB,
                                    const BasicBlock* FalseBB) {
  if (TrueBB==FalseBB) {
    // "TrueBB" and "FalseBB" destination equals
    printPHICopy(CurrBB,TrueBB);
    printSimpleInstruction("pop");
    printSimpleInstruction("br",getLabelName(TrueBB).c_str());
  } else if (FalseBB==NULL) {
    // If "FalseBB" not used the jump have condition
    printPHICopy(CurrBB,TrueBB);
    printSimpleInstruction("brtrue",getLabelName(TrueBB).c_str());
  } else if (TrueBB==NULL) {
    // If "TrueBB" not used the jump is unconditional
    printPHICopy(CurrBB,FalseBB);
    printSimpleInstruction("br",getLabelName(FalseBB).c_str());
  } else {
    // Copy PHI instructions for each block
    std::string TmpLabel;
    // Print PHI instructions for "TrueBB"
    if (isa<PHINode>(TrueBB->begin())) {
      TmpLabel = getLabelName(TrueBB)+"$phi_"+utostr(getUniqID());
      printSimpleInstruction("brtrue",TmpLabel.c_str());
    } else {
      printSimpleInstruction("brtrue",getLabelName(TrueBB).c_str());
    }
    // Print PHI instructions for "FalseBB"
    if (isa<PHINode>(FalseBB->begin())) {
      printPHICopy(CurrBB,FalseBB);
      printSimpleInstruction("br",getLabelName(FalseBB).c_str());
    } else {
      printSimpleInstruction("br",getLabelName(FalseBB).c_str());
    }
    if (isa<PHINode>(TrueBB->begin())) {
      // Handle "TrueBB" PHI Copy
      Out << TmpLabel << ":\n";
      printPHICopy(CurrBB,TrueBB);
      printSimpleInstruction("br",getLabelName(TrueBB).c_str());
    }
  }
}


void MSILWriter::printBranchInstruction(const BranchInst* Inst) {
  if (Inst->isUnconditional()) {
    printBranchToBlock(Inst->getParent(),NULL,Inst->getSuccessor(0));
  } else {
    printValueLoad(Inst->getCondition());
    printBranchToBlock(Inst->getParent(),Inst->getSuccessor(0),
                       Inst->getSuccessor(1));
  }
}


void MSILWriter::printSelectInstruction(const Value* Cond, const Value* VTrue,
                                        const Value* VFalse) {
  std::string TmpLabel = std::string("select$true_")+utostr(getUniqID());
  printValueLoad(VTrue);
  printValueLoad(Cond);
  printSimpleInstruction("brtrue",TmpLabel.c_str());
  printSimpleInstruction("pop");
  printValueLoad(VFalse);
  Out << TmpLabel << ":\n";
}


void MSILWriter::printIndirectLoad(const Value* V) {
  printValueLoad(V);
  std::string Tmp = "ldind."+getTypePostfix(V->getType(),false);
  printSimpleInstruction(Tmp.c_str());
}


void MSILWriter::printStoreInstruction(const Instruction* Inst) {
  const Value* Val = Inst->getOperand(0);
  const Value* Ptr = Inst->getOperand(1);
  // Load destination address.
  printValueLoad(Ptr);
  // Load value.
  printValueLoad(Val);
  // Instruction need signed postfix for any type.
  std::string postfix = getTypePostfix(Val->getType(),false);
  if (*postfix.begin()=='u') *postfix.begin() = 'i';
  postfix = "stind."+postfix;
  printSimpleInstruction(postfix.c_str());
}


void MSILWriter::printCastInstruction(unsigned int Op, const Value* V,
                                      const Type* Ty) {
  std::string Tmp("");
  printValueLoad(V);
  switch (Op) {
  // Signed
  case Instruction::SExt:
  case Instruction::SIToFP:
  case Instruction::FPToSI:
    Tmp = "conv."+getTypePostfix(Ty,false,true);
    printSimpleInstruction(Tmp.c_str());
    break;
  // Unsigned
  case Instruction::FPTrunc:
  case Instruction::FPExt:
  case Instruction::UIToFP:
  case Instruction::Trunc:
  case Instruction::ZExt:
  case Instruction::FPToUI:
  case Instruction::PtrToInt:
  case Instruction::IntToPtr:
    Tmp = "conv."+getTypePostfix(Ty,false);
    printSimpleInstruction(Tmp.c_str());
    break;
  // Do nothing
  case Instruction::BitCast:
    // FIXME: meaning that ld*/st* instruction do not change data format.
    break;
  default:
    cerr << "Opcode = " << Op << '\n';
    assert(0 && "Invalid conversion instruction");
  }
}


void MSILWriter::printGepInstruction(const Value* V, gep_type_iterator I,
                                     gep_type_iterator E) {
  // Load address
  printValueLoad(V);
  // Calculate element offset.
  unsigned TySize;
  for (++I; I!=E; ++I){
    const Type* Ty = I.getIndexedType();
    const Value* Idx = I.getOperand();
    // Get size of type.
    switch (Ty->getTypeID()) {
    case Type::IntegerTyID:
    case Type::FloatTyID:
    case Type::DoubleTyID:
    case Type::PointerTyID:
      TySize = TD->getTypeSize(Ty);
      break;
    case Type::StructTyID:
      TySize = 0;
      break;
    case Type::ArrayTyID:
      TySize = TD->getTypeSize(cast<ArrayType>(Ty)->getElementType());
      break;
    case Type::VectorTyID:
      TySize = TD->getTypeSize(cast<VectorType>(Ty)->getElementType());
      break;
    default:
      cerr << "Type = " << *Ty << '\n';
      assert(0 && "Invalid index type in printGepInstruction()");
    }
    // Calculate offset to structure field.
    if (const StructType* STy = dyn_cast<StructType>(Ty)) {
      TySize = 0;
      uint64_t FieldIdx = cast<ConstantInt>(Idx)->getZExtValue();
      // Offset is the summ of all previous structure fields.
      for (uint64_t F = 0; F<FieldIdx; ++F)
        TySize += TD->getTypeSize(STy->getContainedType(unsigned(F)));
      // Add field offset to stack top.
      printPtrLoad(TySize);
      printSimpleInstruction("add");
      continue;
    }
    // Add offset of current element to stack top.
    if (!isZeroValue(Idx)) {
      uint64_t TySize = TD->getTypeSize(I.getIndexedType());
      // Constant optimization
      if (const ConstantInt* CInt = dyn_cast<ConstantInt>(Idx)) {
        printPtrLoad(CInt->getZExtValue()*TySize);
      } else {
        printPtrLoad(TySize);
        printValueLoad(Idx);
        printSimpleInstruction("mul");
      }
      printSimpleInstruction("add");
    }
  }
}


std::string MSILWriter::getCallSignature(const FunctionType* Ty,
                                         const Instruction* Inst,
                                         std::string Name) {
  std::string Tmp = "";
  if (Ty->isVarArg()) Tmp += "vararg ";
  // Name and return type.
  Tmp += getTypeName(Ty->getReturnType())+Name+"(";
  // Function argument type list.
  unsigned NumParams = Ty->getNumParams();
  for (unsigned I = 0; I!=NumParams; ++I) {
    if (I!=0) Tmp += ",";
    Tmp += getTypeName(Ty->getParamType(I));
  }
  // CLR needs to know the exact amount of parameters received by vararg
  // function, because caller cleans the stack.
  if (Ty->isVarArg() && Inst) {
    // Origin to function arguments in "CallInst" or "InvokeInst"
    unsigned Org = isa<InvokeInst>(Inst) ? 3 : 1;
    // Print variable argument types.
    unsigned NumOperands = Inst->getNumOperands()-Org;
    if (NumParams<NumOperands) {
      if (NumParams!=0) Tmp += ", ";
      Tmp += "... , ";
      for (unsigned J = NumParams; J!=NumOperands; ++J) {
        if (J!=NumParams) Tmp += ", ";
        Tmp += getTypeName(Inst->getOperand(J+Org)->getType());
      }
    }
  }
  return Tmp+")";
}


void MSILWriter::printFunctionCall(const Value* FnVal,
                                   const Instruction* Inst) {
  // Get function calling convention
  std::string Name = "";
  if (const CallInst* Call = dyn_cast<CallInst>(Inst))
    Name = getConvModopt(Call->getCallingConv());
  else if (const InvokeInst* Invoke = dyn_cast<InvokeInst>(Inst))
    Name = getConvModopt(Invoke->getCallingConv());
  else {
    cerr << "Instruction = " << Inst->getName() << '\n';
    assert(0 && "Need \"Invoke\" or \"Call\" instruction only");
  }
  
  if (const Function* F = dyn_cast<Function>(FnVal)) {
    // Direct call
    Name += getValueName(F);
    printSimpleInstruction("call",
      getCallSignature(F->getFunctionType(),Inst,Name).c_str());
  } else {
    // Indirect function call
    const PointerType* PTy = cast<PointerType>(FnVal->getType());
    const FunctionType* FTy = cast<FunctionType>(PTy->getElementType());
    // Load function address
    printValueLoad(FnVal);
    printSimpleInstruction("calli",getCallSignature(FTy,Inst,Name).c_str());
  }
}


void MSILWriter::printCallInstruction(const Instruction* Inst) {
  // Load arguments to stack
  for (int I = 1, E = Inst->getNumOperands(); I!=E; ++I)
    printValueLoad(Inst->getOperand(I));
  printFunctionCall(Inst->getOperand(0),Inst);
}


void MSILWriter::printICmpInstruction(unsigned Predicate, const Value* Left,
                                      const Value* Right) {
  switch (Predicate) {
  case ICmpInst::ICMP_EQ:
    printBinaryInstruction("ceq",Left,Right);
    break;
  case ICmpInst::ICMP_NE:
    // Emulate = not (Op1 eq Op2)
    printBinaryInstruction("ceq",Left,Right);
    printSimpleInstruction("not");
    break;
  case ICmpInst::ICMP_ULE:
  case ICmpInst::ICMP_SLE:
    // Emulate = (Op1 eq Op2) or (Op1 lt Op2)
    printBinaryInstruction("ceq",Left,Right);
    if (Predicate==ICmpInst::ICMP_ULE)
      printBinaryInstruction("clt.un",Left,Right);
    else
      printBinaryInstruction("clt",Left,Right);
    printSimpleInstruction("or");
    break;
  case ICmpInst::ICMP_UGE:
  case ICmpInst::ICMP_SGE:
    // Emulate = (Op1 eq Op2) or (Op1 gt Op2)
    printBinaryInstruction("ceq",Left,Right);
    if (Predicate==ICmpInst::ICMP_UGE)
      printBinaryInstruction("cgt.un",Left,Right);
    else
      printBinaryInstruction("cgt",Left,Right);
    printSimpleInstruction("or");
    break;
  case ICmpInst::ICMP_ULT:
    printBinaryInstruction("clt.un",Left,Right);
    break;
  case ICmpInst::ICMP_SLT:
    printBinaryInstruction("clt",Left,Right);
    break;
  case ICmpInst::ICMP_UGT:
    printBinaryInstruction("cgt.un",Left,Right);
  case ICmpInst::ICMP_SGT:
    printBinaryInstruction("cgt",Left,Right);
    break;
  default:
    cerr << "Predicate = " << Predicate << '\n';
    assert(0 && "Invalid icmp predicate");
  }
}


void MSILWriter::printFCmpInstruction(unsigned Predicate, const Value* Left,
                                      const Value* Right) {
  // FIXME: Correct comparison
  std::string NanFunc = "bool [mscorlib]System.Double::IsNaN(float64)";
  switch (Predicate) {
  case FCmpInst::FCMP_UGT:
    // X >  Y || llvm_fcmp_uno(X, Y)
    printBinaryInstruction("cgt",Left,Right);
    printFCmpInstruction(FCmpInst::FCMP_UNO,Left,Right);
    printSimpleInstruction("or");
    break;
  case FCmpInst::FCMP_OGT:
    // X >  Y
    printBinaryInstruction("cgt",Left,Right);
    break;
  case FCmpInst::FCMP_UGE:
    // X >= Y || llvm_fcmp_uno(X, Y)
    printBinaryInstruction("ceq",Left,Right);
    printBinaryInstruction("cgt",Left,Right);
    printSimpleInstruction("or");
    printFCmpInstruction(FCmpInst::FCMP_UNO,Left,Right);
    printSimpleInstruction("or");
    break;
  case FCmpInst::FCMP_OGE:
    // X >= Y
    printBinaryInstruction("ceq",Left,Right);
    printBinaryInstruction("cgt",Left,Right);
    printSimpleInstruction("or");
    break;
  case FCmpInst::FCMP_ULT:
    // X <  Y || llvm_fcmp_uno(X, Y)
    printBinaryInstruction("clt",Left,Right);
    printFCmpInstruction(FCmpInst::FCMP_UNO,Left,Right);
    printSimpleInstruction("or");
    break;
  case FCmpInst::FCMP_OLT:
    // X <  Y
    printBinaryInstruction("clt",Left,Right);
    break;
  case FCmpInst::FCMP_ULE:
    // X <= Y || llvm_fcmp_uno(X, Y)
    printBinaryInstruction("ceq",Left,Right);
    printBinaryInstruction("clt",Left,Right);
    printSimpleInstruction("or");
    printFCmpInstruction(FCmpInst::FCMP_UNO,Left,Right);
    printSimpleInstruction("or");
    break;
  case FCmpInst::FCMP_OLE:
    // X <= Y
    printBinaryInstruction("ceq",Left,Right);
    printBinaryInstruction("clt",Left,Right);
    printSimpleInstruction("or");
    break;
  case FCmpInst::FCMP_UEQ:
    // X == Y || llvm_fcmp_uno(X, Y)
    printBinaryInstruction("ceq",Left,Right);
    printFCmpInstruction(FCmpInst::FCMP_UNO,Left,Right);
    printSimpleInstruction("or");
    break;
  case FCmpInst::FCMP_OEQ:
    // X == Y
    printBinaryInstruction("ceq",Left,Right);
    break;
  case FCmpInst::FCMP_UNE:
    // X != Y
    printBinaryInstruction("ceq",Left,Right);
    printSimpleInstruction("not");
    break;
  case FCmpInst::FCMP_ONE:
    // X != Y && llvm_fcmp_ord(X, Y)
    printBinaryInstruction("ceq",Left,Right);
    printSimpleInstruction("not");
    break;
  case FCmpInst::FCMP_ORD:
    // return X == X && Y == Y
    printBinaryInstruction("ceq",Left,Left);
    printBinaryInstruction("ceq",Right,Right);
    printSimpleInstruction("or");
    break;
  case FCmpInst::FCMP_UNO:
    // X != X || Y != Y
    printBinaryInstruction("ceq",Left,Left);
    printSimpleInstruction("not");
    printBinaryInstruction("ceq",Right,Right);
    printSimpleInstruction("not");
    printSimpleInstruction("or");
    break;
  default:
    assert(0 && "Illegal FCmp predicate");
  }
}


void MSILWriter::printInvokeInstruction(const InvokeInst* Inst) {
  std::string Label = "leave$normal_"+utostr(getUniqID());
  Out << ".try {\n";
  // Load arguments
  for (int I = 3, E = Inst->getNumOperands(); I!=E; ++I)
    printValueLoad(Inst->getOperand(I));
  // Print call instruction
  printFunctionCall(Inst->getOperand(0),Inst);
  // Save function result and leave "try" block
  printValueSave(Inst);
  printSimpleInstruction("leave",Label.c_str());
  Out << "}\n";
  Out << "catch [mscorlib]System.Exception {\n";
  // Redirect to unwind block
  printSimpleInstruction("pop");
  printBranchToBlock(Inst->getParent(),NULL,Inst->getUnwindDest());
  Out << "}\n" << Label << ":\n";
  // Redirect to continue block
  printBranchToBlock(Inst->getParent(),NULL,Inst->getNormalDest());
}


void MSILWriter::printSwitchInstruction(const SwitchInst* Inst) {
  // FIXME: Emulate with IL "switch" instruction
  // Emulate = if () else if () else if () else ...
  for (unsigned int I = 1, E = Inst->getNumCases(); I!=E; ++I) {
    printValueLoad(Inst->getCondition());
    printValueLoad(Inst->getCaseValue(I));
    printSimpleInstruction("ceq");
    // Condition jump to successor block
    printBranchToBlock(Inst->getParent(),Inst->getSuccessor(I),NULL);
  }
  // Jump to default block
  printBranchToBlock(Inst->getParent(),NULL,Inst->getDefaultDest());
}


void MSILWriter::printInstruction(const Instruction* Inst) {
  const Value *Left = 0, *Right = 0;
  if (Inst->getNumOperands()>=1) Left = Inst->getOperand(0);
  if (Inst->getNumOperands()>=2) Right = Inst->getOperand(1);
  // Print instruction
  // FIXME: "ShuffleVector","ExtractElement","InsertElement","VAArg" support.
  switch (Inst->getOpcode()) {
  // Terminator
  case Instruction::Ret:
    if (Inst->getNumOperands()) {
      printValueLoad(Left);
      printSimpleInstruction("ret");
    } else
      printSimpleInstruction("ret");
    break;
  case Instruction::Br:
    printBranchInstruction(cast<BranchInst>(Inst));
    break;
  // Binary
  case Instruction::Add:
    printBinaryInstruction("add",Left,Right);
    break;
  case Instruction::Sub:
    printBinaryInstruction("sub",Left,Right);
    break;
  case Instruction::Mul:  
    printBinaryInstruction("mul",Left,Right);
    break;
  case Instruction::UDiv:
    printBinaryInstruction("div.un",Left,Right);
    break;
  case Instruction::SDiv:
  case Instruction::FDiv:
    printBinaryInstruction("div",Left,Right);
    break;
  case Instruction::URem:
    printBinaryInstruction("rem.un",Left,Right);
    break;
  case Instruction::SRem:
  case Instruction::FRem:
    printBinaryInstruction("rem",Left,Right);
    break;
  // Binary Condition
  case Instruction::ICmp:
    printICmpInstruction(cast<ICmpInst>(Inst)->getPredicate(),Left,Right);
    break;
  case Instruction::FCmp:
    printFCmpInstruction(cast<FCmpInst>(Inst)->getPredicate(),Left,Right);
    break;
  // Bitwise Binary
  case Instruction::And:
    printBinaryInstruction("and",Left,Right);
    break;
  case Instruction::Or:
    printBinaryInstruction("or",Left,Right);
    break;
  case Instruction::Xor:
    printBinaryInstruction("xor",Left,Right);
    break;
  case Instruction::Shl:
    printBinaryInstruction("shl",Left,Right);
    break;
  case Instruction::LShr:
    printBinaryInstruction("shr.un",Left,Right);
    break;
  case Instruction::AShr:
    printBinaryInstruction("shr",Left,Right);
    break;
  case Instruction::Select:
    printSelectInstruction(Inst->getOperand(0),Inst->getOperand(1),Inst->getOperand(2));
    break;
  case Instruction::Load:
    printIndirectLoad(Inst->getOperand(0));
    break;
  case Instruction::Store:
    printStoreInstruction(Inst);
    break;
  case Instruction::Trunc:
  case Instruction::ZExt:
  case Instruction::SExt:
  case Instruction::FPTrunc:
  case Instruction::FPExt:
  case Instruction::UIToFP:
  case Instruction::SIToFP:
  case Instruction::FPToUI:
  case Instruction::FPToSI:
  case Instruction::PtrToInt:
  case Instruction::IntToPtr:
  case Instruction::BitCast:
    printCastInstruction(Inst->getOpcode(),Left,
                         cast<CastInst>(Inst)->getDestTy());
    break;
  case Instruction::GetElementPtr:
    printGepInstruction(Inst->getOperand(0),gep_type_begin(Inst),
                        gep_type_end(Inst));
    break;
  case Instruction::Call:
    printCallInstruction(cast<CallInst>(Inst));
    break;
  case Instruction::Invoke:
    printInvokeInstruction(cast<InvokeInst>(Inst));
    break;
  case Instruction::Unwind: {
    std::string Class = "instance void [mscorlib]System.Exception::.ctor()";
    printSimpleInstruction("newobj",Class.c_str());
    printSimpleInstruction("throw");
    break;
  }
  case Instruction::Switch:
    printSwitchInstruction(cast<SwitchInst>(Inst));
    break;
  case Instruction::Alloca:
    printValueLoad(Inst->getOperand(0));
    printSimpleInstruction("localloc");
    break;
  case Instruction::Malloc:
    assert(0 && "LowerAllocationsPass used");
    break;
  case Instruction::Free:
    assert(0 && "LowerAllocationsPass used");
    break;
  case Instruction::Unreachable:
    printSimpleInstruction("ldnull");
    printSimpleInstruction("throw");
    break;
  default:
    cerr << "Instruction = " << Inst->getName() << '\n';
    assert(0 && "Unsupported instruction");
  }
}


void MSILWriter::printLoop(const Loop* L) {
  Out << getLabelName(L->getHeader()->getName()) << ":\n";
  const std::vector<BasicBlock*>& blocks = L->getBlocks();
  for (unsigned I = 0, E = blocks.size(); I!=E; I++) {
    BasicBlock* BB = blocks[I];
    Loop* BBLoop = LInfo->getLoopFor(BB);
    if (BBLoop == L)
      printBasicBlock(BB);
    else if (BB==BBLoop->getHeader() && BBLoop->getParentLoop()==L)
      printLoop(BBLoop);
  }
  printSimpleInstruction("br",getLabelName(L->getHeader()->getName()).c_str());
}


void MSILWriter::printBasicBlock(const BasicBlock* BB) {
  Out << getLabelName(BB) << ":\n";
  for (BasicBlock::const_iterator I = BB->begin(), E = BB->end(); I!=E; ++I) {
    const Instruction* Inst = I;
    // Comment llvm original instruction
    Out << "\n//" << *Inst << "\n";
    // Do not handle PHI instruction in current block
    if (Inst->getOpcode()==Instruction::PHI) continue;
    // Print instruction
    printInstruction(Inst);
    // Save result
    if (Inst->getType()!=Type::VoidTy) {
      // Do not save value after invoke, it done in "try" block
      if (Inst->getOpcode()==Instruction::Invoke) continue;
      printValueSave(Inst);
    }
  }
}


void MSILWriter::printLocalVariables(const Function& F) {
  std::string Name;
  const Type* Ty = NULL;
  // Find variables
  for (const_inst_iterator I = inst_begin(&F), E = inst_end(&F); I!=E; ++I) {
    const AllocaInst* AI = dyn_cast<AllocaInst>(&*I);
    if (AI && !isa<GlobalVariable>(AI)) {
      Ty = PointerType::get(AI->getAllocatedType());
      Name = getValueName(AI);
    } else if (I->getType()!=Type::VoidTy) {
      Ty = I->getType();
      Name = getValueName(&*I);
    } else continue;
    Out << "\t.locals (" << getTypeName(Ty) << Name << ")\n";
  }
}


void MSILWriter::printFunctionBody(const Function& F) {
  // Print body
  for (Function::const_iterator I = F.begin(), E = F.end(); I!=E; ++I) {
    if (Loop *L = LInfo->getLoopFor(I)) {
      if (L->getHeader()==I && L->getParentLoop()==0)
        printLoop(L);
    } else {
      printBasicBlock(I);
    }
  }
}


void MSILWriter::printConstantExpr(const ConstantExpr* CE) {
  const Value *left = 0, *right = 0;
  if (CE->getNumOperands()>=1) left = CE->getOperand(0);
  if (CE->getNumOperands()>=2) right = CE->getOperand(1);
  // Print instruction
  switch (CE->getOpcode()) {
  case Instruction::Trunc:
  case Instruction::ZExt:
  case Instruction::SExt:
  case Instruction::FPTrunc:
  case Instruction::FPExt:
  case Instruction::UIToFP:
  case Instruction::SIToFP:
  case Instruction::FPToUI:
  case Instruction::FPToSI:
  case Instruction::PtrToInt:
  case Instruction::IntToPtr:
  case Instruction::BitCast:
    printCastInstruction(CE->getOpcode(),left,CE->getType());
    break;
  case Instruction::GetElementPtr:
    printGepInstruction(CE->getOperand(0),gep_type_begin(CE),gep_type_end(CE));
    break;
  case Instruction::ICmp:
    printICmpInstruction(CE->getPredicate(),left,right);
    break;
  case Instruction::FCmp:
    printFCmpInstruction(CE->getPredicate(),left,right);
    break;
  case Instruction::Select:
    printSelectInstruction(CE->getOperand(0),CE->getOperand(1),CE->getOperand(2));
    break;
  case Instruction::Add:
    printBinaryInstruction("add",left,right);
    break;
  case Instruction::Sub:
    printBinaryInstruction("sub",left,right);
    break;
  case Instruction::Mul:
    printBinaryInstruction("mul",left,right);
    break;
  case Instruction::UDiv:
    printBinaryInstruction("div.un",left,right);
    break;
  case Instruction::SDiv:
  case Instruction::FDiv:
    printBinaryInstruction("div",left,right);
    break;
  case Instruction::URem:
    printBinaryInstruction("rem.un",left,right);
    break;
  case Instruction::SRem:
  case Instruction::FRem:
    printBinaryInstruction("rem",left,right);
    break;
  case Instruction::And:
    printBinaryInstruction("and",left,right);
    break;
  case Instruction::Or:
    printBinaryInstruction("or",left,right);
    break;
  case Instruction::Xor:
    printBinaryInstruction("xor",left,right);
    break;
  case Instruction::Shl:
    printBinaryInstruction("shl",left,right);
    break;
  case Instruction::LShr:
    printBinaryInstruction("shr.un",left,right);
    break;
  case Instruction::AShr:
    printBinaryInstruction("shr",left,right);
    break;
  default:
    cerr << "Expression = " << *CE << "\n";
    assert(0 && "Invalid constant expression");
  }
}


void MSILWriter::printStaticInitializerList() {
  // List of global variables with uninitialized fields.
  for (std::map<const GlobalVariable*,std::vector<StaticInitializer> >::iterator
       VarI = StaticInitList.begin(), VarE = StaticInitList.end(); VarI!=VarE;
       ++VarI) {
    const std::vector<StaticInitializer>& InitList = VarI->second;
    if (InitList.empty()) continue;
    // For each uninitialized field.
    for (std::vector<StaticInitializer>::const_iterator I = InitList.begin(),
         E = InitList.end(); I!=E; ++I) {
      if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(I->constant)) {
        Out << "\n// Init " << getValueName(VarI->first) << ", offset " <<
          utostr(I->offset) << ", type "<< *I->constant->getType() << "\n\n";
        // Load variable address
        printValueLoad(VarI->first);
        // Add offset
        if (I->offset!=0) {
          printPtrLoad(I->offset);
          printSimpleInstruction("add");
        }
        // Load value
        printConstantExpr(CE);
        // Save result at offset
        std::string postfix = getTypePostfix(CE->getType(),true);
        if (*postfix.begin()=='u') *postfix.begin() = 'i';
        postfix = "stind."+postfix;
        printSimpleInstruction(postfix.c_str());
      } else {
        cerr << "Constant = " << *I->constant << '\n';
        assert(0 && "Invalid static initializer");
      }
    }
  }
}


void MSILWriter::printFunction(const Function& F) {
  const FunctionType* FTy = F.getFunctionType();
  const ParamAttrsList *Attrs = FTy->getParamAttrs();
  bool isSigned = Attrs && Attrs->paramHasAttr(0, ParamAttr::SExt);
  Out << "\n.method static ";
  Out << (F.hasInternalLinkage() ? "private " : "public ");
  if (F.isVarArg()) Out << "vararg ";
  Out << getTypeName(F.getReturnType(),isSigned) << 
    getConvModopt(F.getCallingConv()) << getValueName(&F) << '\n';
  // Arguments
  Out << "\t(";
  unsigned ArgIdx = 1;
  for (Function::const_arg_iterator I = F.arg_begin(), E = F.arg_end(); I!=E;
       ++I, ++ArgIdx) {
    isSigned = Attrs && Attrs->paramHasAttr(ArgIdx, ParamAttr::SExt);
    if (I!=F.arg_begin()) Out << ", ";
    Out << getTypeName(I->getType(),isSigned) << getValueName(I);
  }
  Out << ") cil managed\n";
  // Body
  Out << "{\n";
  // FIXME: Convert "string[]" to "argc,argv"
  if (F.getName()=="main") {
    printSimpleInstruction(".entrypoint");
    printLocalVariables(F);
    printStaticInitializerList();
  } else {
    printLocalVariables(F);  
  }
  printFunctionBody(F);
  Out << "}\n";
}


void MSILWriter::printDeclarations(const TypeSymbolTable& ST) {
  std::string Name;
  std::set<const Type*> Printed;
  //cerr << "UsedTypes = " << UsedTypes << '\n';
  for (std::set<const Type*>::const_iterator
       UI = UsedTypes->begin(), UE = UsedTypes->end(); UI!=UE; ++UI) {
    const Type* Ty = *UI;
    if (isa<ArrayType>(Ty))
      Name = getArrayTypeName(Ty->getTypeID(),Ty);
    else if (isa<VectorType>(Ty))
      Name = getArrayTypeName(Ty->getTypeID(),Ty);
    else if (isa<StructType>(Ty))
      Name = ModulePtr->getTypeName(Ty);
    // Type with no need to declare.
    else continue;
    // Print not duplicated type
    if (Printed.insert(Ty).second) {
      Out << ".class value explicit ansi sealed '" << Name << "'";
      Out << " { .pack " << 1 << " .size " << TD->getTypeSize(Ty) << " }\n\n";
    }
  }
}


unsigned int MSILWriter::getBitWidth(const Type* Ty) {
  unsigned int N = Ty->getPrimitiveSizeInBits();
  assert(N!=0 && "Invalid type in getBitWidth()");
  switch (N) {
  case 1:
  case 8:
  case 16:
  case 32:
  case 64:
    return N;
  default:
    cerr << "Bits = " << N << '\n';
    assert(0 && "Unsupported integer width");
  }
}


void MSILWriter::printStaticConstant(const Constant* C, uint64_t& Offset) {
  uint64_t TySize = 0;
  const Type* Ty = C->getType();
  // Print zero initialized constant.
  if (isa<ConstantAggregateZero>(C) || C->isNullValue()) {
    TySize = TD->getTypeSize(C->getType());
    Offset += TySize;
    Out << "int8 (0) [" << TySize << "]";
    return;
  }
  // Print constant initializer
  switch (Ty->getTypeID()) {
  case Type::IntegerTyID: {
    TySize = TD->getTypeSize(Ty);
    const ConstantInt* Int = cast<ConstantInt>(C);
    Out << getPrimitiveTypeName(Ty,true) << "(" << Int->getSExtValue() << ")";
    break;
  }
  case Type::FloatTyID:
  case Type::DoubleTyID: {
    TySize = TD->getTypeSize(Ty);
    const ConstantFP* CFp = cast<ConstantFP>(C);
    Out << getPrimitiveTypeName(Ty,true) << "(" << CFp->getValue() << ")";
    break;
  }
  case Type::ArrayTyID:
  case Type::VectorTyID:
  case Type::StructTyID:
    for (unsigned I = 0, E = C->getNumOperands(); I<E; I++) {
      if (I!=0) Out << ",\n";
      printStaticConstant(C->getOperand(I),Offset);
    }
    break;
  case Type::PointerTyID:
    TySize = TD->getTypeSize(C->getType());
    // Initialize with global variable address
    if (const GlobalVariable *G = dyn_cast<GlobalVariable>(C)) {
      std::string name = getValueName(G);
      Out << "&(" << name.insert(name.length()-1,"$data") << ")";
    } else {
      // Dynamic initialization
      if (!isa<ConstantPointerNull>(C) && !C->isNullValue())
        InitListPtr->push_back(StaticInitializer(C,Offset));
      // Null pointer initialization
      if (TySize==4) Out << "int32 (0)";
      else if (TySize==8) Out << "int64 (0)";
      else assert(0 && "Invalid pointer size");
    }
    break;
  default:
    cerr << "TypeID = " << Ty->getTypeID() << '\n';
    assert(0 && "Invalid type in printStaticConstant()");
  }
  // Increase offset.
  Offset += TySize;
}


void MSILWriter::printStaticInitializer(const Constant* C,
                                        const std::string& Name) {
  switch (C->getType()->getTypeID()) {
  case Type::IntegerTyID:
  case Type::FloatTyID:
  case Type::DoubleTyID: 
    Out << getPrimitiveTypeName(C->getType(),true);
    break;
  case Type::ArrayTyID:
  case Type::VectorTyID:
  case Type::StructTyID:
  case Type::PointerTyID:
    Out << getTypeName(C->getType());
    break;
  default:
    cerr << "Type = " << *C << "\n";
    assert(0 && "Invalid constant type");
  }
  // Print initializer
  std::string label = Name;
  label.insert(label.length()-1,"$data");
  Out << Name << " at " << label << '\n';
  Out << ".data " << label << " = {\n";
  uint64_t offset = 0;
  printStaticConstant(C,offset);
  Out << "\n}\n\n";
}


void MSILWriter::printVariableDefinition(const GlobalVariable* G) {
  const Constant* C = G->getInitializer();
  if (C->isNullValue() || isa<ConstantAggregateZero>(C) || isa<UndefValue>(C))
    InitListPtr = 0;
  else
    InitListPtr = &StaticInitList[G];
  printStaticInitializer(C,getValueName(G));
}


void MSILWriter::printGlobalVariables() {
  if (ModulePtr->global_empty()) return;
  Module::global_iterator I,E;
  for (I = ModulePtr->global_begin(), E = ModulePtr->global_end(); I!=E; ++I) {
    // Variable definition
    if (I->isDeclaration()) continue;
    Out << ".field static " << (I->hasExternalLinkage() ? "public " :
                                                          "private ");
    printVariableDefinition(&*I);
  }
}


void MSILWriter::printExternals() {
  Module::const_iterator I,E;
  for (I=ModulePtr->begin(),E=ModulePtr->end(); I!=E; ++I) {
    // Skip intrisics
    if (I->getIntrinsicID()) continue;
    // FIXME: Treat as standard library function
    if (I->isDeclaration()) {
      const Function* F = &*I; 
      const FunctionType* FTy = F->getFunctionType();
      std::string Name = getConvModopt(F->getCallingConv())+getValueName(F);
      std::string Sig = getCallSignature(FTy,NULL,Name);
      Out << ".method static hidebysig pinvokeimpl(\"msvcrt.dll\" cdecl)\n\t" 
        << Sig << " preservesig {}\n\n";
    }
  }
}

//===----------------------------------------------------------------------===//
//			 External Interface declaration
//===----------------------------------------------------------------------===//

bool MSILTarget::addPassesToEmitWholeFile(PassManager &PM, std::ostream &o,
                                          CodeGenFileType FileType, bool Fast)
{
  if (FileType != TargetMachine::AssemblyFile) return true;
  MSILWriter* Writer = new MSILWriter(o);
  PM.add(createLowerGCPass());
  PM.add(createLowerAllocationsPass(true));
  // FIXME: Handle switch trougth native IL instruction "switch"
  PM.add(createLowerSwitchPass());
  PM.add(createCFGSimplificationPass());
  PM.add(new MSILModule(Writer->UsedTypes,Writer->TD));
  PM.add(Writer);
  return false;
}
