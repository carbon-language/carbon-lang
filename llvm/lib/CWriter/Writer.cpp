//===-- Writer.cpp - Library for converting LLVM code to C ----------------===//
//
// This library converts LLVM code to C code, compilable by GCC.
//
//===----------------------------------------------------------------------===//

#include "llvm/Assembly/CWriter.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/SymbolTable.h"
#include "llvm/Intrinsics.h"
#include "llvm/Analysis/FindUsedTypes.h"
#include "llvm/Analysis/ConstantsScanner.h"
#include "llvm/Support/InstVisitor.h"
#include "llvm/Support/InstIterator.h"
#include "llvm/Support/CallSite.h"
#include "llvm/Support/Mangler.h"
#include "Support/StringExtras.h"
#include "Support/STLExtras.h"
#include <algorithm>
#include <sstream>

namespace {
  class CWriter : public Pass, public InstVisitor<CWriter> {
    std::ostream &Out; 
    Mangler *Mang;
    const Module *TheModule;
    std::map<const Type *, std::string> TypeNames;
    std::set<const Value*> MangledGlobals;
    bool needsMalloc, emittedInvoke;

    std::map<const ConstantFP *, unsigned> FPConstantMap;
  public:
    CWriter(std::ostream &o) : Out(o) {}

    void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
      AU.addRequired<FindUsedTypes>();
    }

    virtual bool run(Module &M) {
      // Initialize
      TheModule = &M;

      // Ensure that all structure types have names...
      bool Changed = nameAllUsedStructureTypes(M);
      Mang = new Mangler(M);

      // Run...
      printModule(&M);

      // Free memory...
      delete Mang;
      TypeNames.clear();
      MangledGlobals.clear();
      return false;
    }

    std::ostream &printType(std::ostream &Out, const Type *Ty,
                            const std::string &VariableName = "",
                            bool IgnoreName = false, bool namedContext = true);

    void writeOperand(Value *Operand);
    void writeOperandInternal(Value *Operand);

  private :
    bool nameAllUsedStructureTypes(Module &M);
    void printModule(Module *M);
    void printSymbolTable(const SymbolTable &ST);
    void printContainedStructs(const Type *Ty, std::set<const StructType *> &);
    void printFunctionSignature(const Function *F, bool Prototype);

    void printFunction(Function *);

    void printConstant(Constant *CPV);
    void printConstantArray(ConstantArray *CPA);

    // isInlinableInst - Attempt to inline instructions into their uses to build
    // trees as much as possible.  To do this, we have to consistently decide
    // what is acceptable to inline, so that variable declarations don't get
    // printed and an extra copy of the expr is not emitted.
    //
    static bool isInlinableInst(const Instruction &I) {
      // Must be an expression, must be used exactly once.  If it is dead, we
      // emit it inline where it would go.
      if (I.getType() == Type::VoidTy || I.use_size() != 1 ||
          isa<TerminatorInst>(I) || isa<CallInst>(I) || isa<PHINode>(I) || 
          isa<LoadInst>(I) || isa<VarArgInst>(I))
        // Don't inline a load across a store or other bad things!
        return false;

      // Only inline instruction it it's use is in the same BB as the inst.
      return I.getParent() == cast<Instruction>(I.use_back())->getParent();
    }

    // isDirectAlloca - Define fixed sized allocas in the entry block as direct
    // variables which are accessed with the & operator.  This causes GCC to
    // generate significantly better code than to emit alloca calls directly.
    //
    static const AllocaInst *isDirectAlloca(const Value *V) {
      const AllocaInst *AI = dyn_cast<AllocaInst>(V);
      if (!AI) return false;
      if (AI->isArrayAllocation())
        return 0;   // FIXME: we can also inline fixed size array allocas!
      if (AI->getParent() != &AI->getParent()->getParent()->getEntryBlock())
        return 0;
      return AI;
    }

    // Instruction visitation functions
    friend class InstVisitor<CWriter>;

    void visitReturnInst(ReturnInst &I);
    void visitBranchInst(BranchInst &I);
    void visitSwitchInst(SwitchInst &I);
    void visitInvokeInst(InvokeInst &I);
    void visitUnwindInst(UnwindInst &I);

    void visitPHINode(PHINode &I);
    void visitBinaryOperator(Instruction &I);

    void visitCastInst (CastInst &I);
    void visitCallInst (CallInst &I);
    void visitCallSite (CallSite CS);
    void visitShiftInst(ShiftInst &I) { visitBinaryOperator(I); }

    void visitMallocInst(MallocInst &I);
    void visitAllocaInst(AllocaInst &I);
    void visitFreeInst  (FreeInst   &I);
    void visitLoadInst  (LoadInst   &I);
    void visitStoreInst (StoreInst  &I);
    void visitGetElementPtrInst(GetElementPtrInst &I);
    void visitVarArgInst(VarArgInst &I);

    void visitInstruction(Instruction &I) {
      std::cerr << "C Writer does not know about " << I;
      abort();
    }

    void outputLValue(Instruction *I) {
      Out << "  " << Mang->getValueName(I) << " = ";
    }
    void printBranchToBlock(BasicBlock *CurBlock, BasicBlock *SuccBlock,
                            unsigned Indent);
    void printIndexingExpression(Value *Ptr, User::op_iterator I,
                                 User::op_iterator E);
  };
}

// A pointer type should not use parens around *'s alone, e.g., (**)
inline bool ptrTypeNameNeedsParens(const std::string &NameSoFar) {
  return (NameSoFar.find_last_not_of('*') != std::string::npos);
}

// Pass the Type* and the variable name and this prints out the variable
// declaration.
//
std::ostream &CWriter::printType(std::ostream &Out, const Type *Ty,
                                 const std::string &NameSoFar,
                                 bool IgnoreName, bool namedContext) {
  if (Ty->isPrimitiveType())
    switch (Ty->getPrimitiveID()) {
    case Type::VoidTyID:   return Out << "void "               << NameSoFar;
    case Type::BoolTyID:   return Out << "bool "               << NameSoFar;
    case Type::UByteTyID:  return Out << "unsigned char "      << NameSoFar;
    case Type::SByteTyID:  return Out << "signed char "        << NameSoFar;
    case Type::UShortTyID: return Out << "unsigned short "     << NameSoFar;
    case Type::ShortTyID:  return Out << "short "              << NameSoFar;
    case Type::UIntTyID:   return Out << "unsigned "           << NameSoFar;
    case Type::IntTyID:    return Out << "int "                << NameSoFar;
    case Type::ULongTyID:  return Out << "unsigned long long " << NameSoFar;
    case Type::LongTyID:   return Out << "signed long long "   << NameSoFar;
    case Type::FloatTyID:  return Out << "float "              << NameSoFar;
    case Type::DoubleTyID: return Out << "double "             << NameSoFar;
    default :
      std::cerr << "Unknown primitive type: " << Ty << "\n";
      abort();
    }
  
  // Check to see if the type is named.
  if (!IgnoreName || isa<OpaqueType>(Ty)) {
    std::map<const Type *, std::string>::iterator I = TypeNames.find(Ty);
    if (I != TypeNames.end()) return Out << I->second << " " << NameSoFar;
  }

  switch (Ty->getPrimitiveID()) {
  case Type::FunctionTyID: {
    const FunctionType *MTy = cast<FunctionType>(Ty);
    std::stringstream FunctionInnards; 
    FunctionInnards << " (" << NameSoFar << ") (";
    for (FunctionType::ParamTypes::const_iterator
           I = MTy->getParamTypes().begin(),
           E = MTy->getParamTypes().end(); I != E; ++I) {
      if (I != MTy->getParamTypes().begin())
        FunctionInnards << ", ";
      printType(FunctionInnards, *I, "");
    }
    if (MTy->isVarArg()) {
      if (!MTy->getParamTypes().empty()) 
    	FunctionInnards << ", ...";
    } else if (MTy->getParamTypes().empty()) {
      FunctionInnards << "void";
    }
    FunctionInnards << ")";
    std::string tstr = FunctionInnards.str();
    printType(Out, MTy->getReturnType(), tstr);
    return Out;
  }
  case Type::StructTyID: {
    const StructType *STy = cast<StructType>(Ty);
    Out << NameSoFar + " {\n";
    unsigned Idx = 0;
    for (StructType::ElementTypes::const_iterator
           I = STy->getElementTypes().begin(),
           E = STy->getElementTypes().end(); I != E; ++I) {
      Out << "  ";
      printType(Out, *I, "field" + utostr(Idx++));
      Out << ";\n";
    }
    return Out << "}";
  }  

  case Type::PointerTyID: {
    const PointerType *PTy = cast<PointerType>(Ty);
    std::string ptrName = "*" + NameSoFar;

    // Do not need parens around "* NameSoFar" if NameSoFar consists only
    // of zero or more '*' chars *and* this is not an unnamed pointer type
    // such as the result type in a cast statement.  Otherwise, enclose in ( ).
    if (ptrTypeNameNeedsParens(NameSoFar) || !namedContext || 
        PTy->getElementType()->getPrimitiveID() == Type::ArrayTyID)
      ptrName = "(" + ptrName + ")";    // 

    return printType(Out, PTy->getElementType(), ptrName);
  }

  case Type::ArrayTyID: {
    const ArrayType *ATy = cast<ArrayType>(Ty);
    unsigned NumElements = ATy->getNumElements();
    return printType(Out, ATy->getElementType(),
                     NameSoFar + "[" + utostr(NumElements) + "]");
  }

  case Type::OpaqueTyID: {
    static int Count = 0;
    std::string TyName = "struct opaque_" + itostr(Count++);
    assert(TypeNames.find(Ty) == TypeNames.end());
    TypeNames[Ty] = TyName;
    return Out << TyName << " " << NameSoFar;
  }
  default:
    assert(0 && "Unhandled case in getTypeProps!");
    abort();
  }

  return Out;
}

void CWriter::printConstantArray(ConstantArray *CPA) {

  // As a special case, print the array as a string if it is an array of
  // ubytes or an array of sbytes with positive values.
  // 
  const Type *ETy = CPA->getType()->getElementType();
  bool isString = (ETy == Type::SByteTy || ETy == Type::UByteTy);

  // Make sure the last character is a null char, as automatically added by C
  if (isString && (CPA->getNumOperands() == 0 ||
                   !cast<Constant>(*(CPA->op_end()-1))->isNullValue()))
    isString = false;
  
  if (isString) {
    Out << "\"";
    // Keep track of whether the last number was a hexadecimal escape
    bool LastWasHex = false;

    // Do not include the last character, which we know is null
    for (unsigned i = 0, e = CPA->getNumOperands()-1; i != e; ++i) {
      unsigned char C = cast<ConstantInt>(CPA->getOperand(i))->getRawValue();
      
      // Print it out literally if it is a printable character.  The only thing
      // to be careful about is when the last letter output was a hex escape
      // code, in which case we have to be careful not to print out hex digits
      // explicitly (the C compiler thinks it is a continuation of the previous
      // character, sheesh...)
      //
      if (isprint(C) && (!LastWasHex || !isxdigit(C))) {
        LastWasHex = false;
        if (C == '"' || C == '\\')
          Out << "\\" << C;
        else
          Out << C;
      } else {
        LastWasHex = false;
        switch (C) {
        case '\n': Out << "\\n"; break;
        case '\t': Out << "\\t"; break;
        case '\r': Out << "\\r"; break;
        case '\v': Out << "\\v"; break;
        case '\a': Out << "\\a"; break;
        case '\"': Out << "\\\""; break;
        case '\'': Out << "\\\'"; break;           
        default:
          Out << "\\x";
          Out << (char)(( C/16  < 10) ? ( C/16 +'0') : ( C/16 -10+'A'));
          Out << (char)(((C&15) < 10) ? ((C&15)+'0') : ((C&15)-10+'A'));
          LastWasHex = true;
          break;
        }
      }
    }
    Out << "\"";
  } else {
    Out << "{";
    if (CPA->getNumOperands()) {
      Out << " ";
      printConstant(cast<Constant>(CPA->getOperand(0)));
      for (unsigned i = 1, e = CPA->getNumOperands(); i != e; ++i) {
        Out << ", ";
        printConstant(cast<Constant>(CPA->getOperand(i)));
      }
    }
    Out << " }";
  }
}

/// FPCSafeToPrint - Returns true if we may assume that CFP may be
/// written out textually as a double (rather than as a reference to a
/// stack-allocated variable). We decide this by converting CFP to a
/// string and back into a double, and then checking whether the
/// conversion results in a bit-equal double to the original value of
/// CFP. This depends on us and the target C compiler agreeing on the
/// conversion process (which is pretty likely since we only deal in
/// IEEE FP.) This is adapted from similar code in
/// lib/VMCore/AsmWriter.cpp:WriteConstantInt().
static bool FPCSafeToPrint (const ConstantFP *CFP) {
  std::string StrVal = ftostr(CFP->getValue());
  // Check to make sure that the stringized number is not some string like
  // "Inf" or NaN, that atof will accept, but the lexer will not.  Check that
  // the string matches the "[-+]?[0-9]" regex.
  if ((StrVal[0] >= '0' && StrVal[0] <= '9') ||
      ((StrVal[0] == '-' || StrVal[0] == '+') &&
       (StrVal[1] >= '0' && StrVal[1] <= '9')))
    // Reparse stringized version!
    return (atof(StrVal.c_str()) == CFP->getValue());
  return false;
}

// printConstant - The LLVM Constant to C Constant converter.
void CWriter::printConstant(Constant *CPV) {
  if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(CPV)) {
    switch (CE->getOpcode()) {
    case Instruction::Cast:
      Out << "((";
      printType(Out, CPV->getType());
      Out << ")";
      printConstant(CE->getOperand(0));
      Out << ")";
      return;

    case Instruction::GetElementPtr:
      Out << "(&(";
      printIndexingExpression(CE->getOperand(0),
                              CPV->op_begin()+1, CPV->op_end());
      Out << "))";
      return;
    case Instruction::Add:
    case Instruction::Sub:
    case Instruction::Mul:
    case Instruction::Div:
    case Instruction::Rem:
    case Instruction::SetEQ:
    case Instruction::SetNE:
    case Instruction::SetLT:
    case Instruction::SetLE:
    case Instruction::SetGT:
    case Instruction::SetGE:
      Out << "(";
      printConstant(CE->getOperand(0));
      switch (CE->getOpcode()) {
      case Instruction::Add: Out << " + "; break;
      case Instruction::Sub: Out << " - "; break;
      case Instruction::Mul: Out << " * "; break;
      case Instruction::Div: Out << " / "; break;
      case Instruction::Rem: Out << " % "; break;
      case Instruction::SetEQ: Out << " == "; break;
      case Instruction::SetNE: Out << " != "; break;
      case Instruction::SetLT: Out << " < "; break;
      case Instruction::SetLE: Out << " <= "; break;
      case Instruction::SetGT: Out << " > "; break;
      case Instruction::SetGE: Out << " >= "; break;
      default: assert(0 && "Illegal opcode here!");
      }
      printConstant(CE->getOperand(1));
      Out << ")";
      return;

    default:
      std::cerr << "CWriter Error: Unhandled constant expression: "
                << CE << "\n";
      abort();
    }
  }

  switch (CPV->getType()->getPrimitiveID()) {
  case Type::BoolTyID:
    Out << (CPV == ConstantBool::False ? "0" : "1"); break;
  case Type::SByteTyID:
  case Type::ShortTyID:
    Out << cast<ConstantSInt>(CPV)->getValue(); break;
  case Type::IntTyID:
    if ((int)cast<ConstantSInt>(CPV)->getValue() == (int)0x80000000)
      Out << "((int)0x80000000)";   // Handle MININT specially to avoid warning
    else
      Out << cast<ConstantSInt>(CPV)->getValue();
    break;

  case Type::LongTyID:
    Out << cast<ConstantSInt>(CPV)->getValue() << "ll"; break;

  case Type::UByteTyID:
  case Type::UShortTyID:
    Out << cast<ConstantUInt>(CPV)->getValue(); break;
  case Type::UIntTyID:
    Out << cast<ConstantUInt>(CPV)->getValue() << "u"; break;
  case Type::ULongTyID:
    Out << cast<ConstantUInt>(CPV)->getValue() << "ull"; break;

  case Type::FloatTyID:
  case Type::DoubleTyID: {
    ConstantFP *FPC = cast<ConstantFP>(CPV);
    std::map<const ConstantFP*, unsigned>::iterator I = FPConstantMap.find(FPC);
    if (I != FPConstantMap.end()) {
      // Because of FP precision problems we must load from a stack allocated
      // value that holds the value in hex.
      Out << "(*(" << (FPC->getType() == Type::FloatTy ? "float" : "double")
          << "*)&FloatConstant" << I->second << ")";
    } else {
      if (FPCSafeToPrint (FPC)) {
        Out << ftostr (FPC->getValue ());
      } else {
        Out << FPC->getValue(); // Who knows? Give it our best shot...
      }
    }
    break;
  }

  case Type::ArrayTyID:
    printConstantArray(cast<ConstantArray>(CPV));
    break;

  case Type::StructTyID: {
    Out << "{";
    if (CPV->getNumOperands()) {
      Out << " ";
      printConstant(cast<Constant>(CPV->getOperand(0)));
      for (unsigned i = 1, e = CPV->getNumOperands(); i != e; ++i) {
        Out << ", ";
        printConstant(cast<Constant>(CPV->getOperand(i)));
      }
    }
    Out << " }";
    break;
  }

  case Type::PointerTyID:
    if (isa<ConstantPointerNull>(CPV)) {
      Out << "((";
      printType(Out, CPV->getType());
      Out << ")/*NULL*/0)";
      break;
    } else if (ConstantPointerRef *CPR = dyn_cast<ConstantPointerRef>(CPV)) {
      writeOperand(CPR->getValue());
      break;
    }
    // FALL THROUGH
  default:
    std::cerr << "Unknown constant type: " << CPV << "\n";
    abort();
  }
}

void CWriter::writeOperandInternal(Value *Operand) {
  if (Instruction *I = dyn_cast<Instruction>(Operand))
    if (isInlinableInst(*I) && !isDirectAlloca(I)) {
      // Should we inline this instruction to build a tree?
      Out << "(";
      visit(*I);
      Out << ")";    
      return;
    }
  
  if (Constant *CPV = dyn_cast<Constant>(Operand)) {
    printConstant(CPV); 
  } else {
    Out << Mang->getValueName(Operand);
  }
}

void CWriter::writeOperand(Value *Operand) {
  if (isa<GlobalVariable>(Operand) || isDirectAlloca(Operand))
    Out << "(&";  // Global variables are references as their addresses by llvm

  writeOperandInternal(Operand);

  if (isa<GlobalVariable>(Operand) || isDirectAlloca(Operand))
    Out << ")";
}

// nameAllUsedStructureTypes - If there are structure types in the module that
// are used but do not have names assigned to them in the symbol table yet then
// we assign them names now.
//
bool CWriter::nameAllUsedStructureTypes(Module &M) {
  // Get a set of types that are used by the program...
  std::set<const Type *> UT = getAnalysis<FindUsedTypes>().getTypes();

  // Loop over the module symbol table, removing types from UT that are already
  // named.
  //
  SymbolTable &MST = M.getSymbolTable();
  if (MST.find(Type::TypeTy) != MST.end())
    for (SymbolTable::type_iterator I = MST.type_begin(Type::TypeTy),
           E = MST.type_end(Type::TypeTy); I != E; ++I)
      UT.erase(cast<Type>(I->second));

  // UT now contains types that are not named.  Loop over it, naming structure
  // types.
  //
  bool Changed = false;
  for (std::set<const Type *>::const_iterator I = UT.begin(), E = UT.end();
       I != E; ++I)
    if (const StructType *ST = dyn_cast<StructType>(*I)) {
      ((Value*)ST)->setName("unnamed", &MST);
      Changed = true;
    }
  return Changed;
}

// generateCompilerSpecificCode - This is where we add conditional compilation
// directives to cater to specific compilers as need be.
//
static void generateCompilerSpecificCode(std::ostream& Out) {
  // Alloca is hard to get, and we don't want to include stdlib.h here...
  Out << "/* get a declaration for alloca */\n"
      << "#ifdef sun\n"
      << "extern void *__builtin_alloca(unsigned long);\n"
      << "#define alloca(x) __builtin_alloca(x)\n"
      << "#else\n"
      << "#ifndef __FreeBSD__\n"
      << "#include <alloca.h>\n"
      << "#endif\n"
      << "#endif\n\n";

  // We output GCC specific attributes to preserve 'linkonce'ness on globals.
  // If we aren't being compiled with GCC, just drop these attributes.
  Out << "#ifndef __GNUC__  /* Can only support \"linkonce\" vars with GCC */\n"
      << "#define __attribute__(X)\n"
      << "#endif\n";
}

void CWriter::printModule(Module *M) {
  // Calculate which global values have names that will collide when we throw
  // away type information.
  {  // Scope to delete the FoundNames set when we are done with it...
    std::set<std::string> FoundNames;
    for (Module::iterator I = M->begin(), E = M->end(); I != E; ++I)
      if (I->hasName())                      // If the global has a name...
        if (FoundNames.count(I->getName()))  // And the name is already used
          MangledGlobals.insert(I);          // Mangle the name
        else
          FoundNames.insert(I->getName());   // Otherwise, keep track of name

    for (Module::giterator I = M->gbegin(), E = M->gend(); I != E; ++I)
      if (I->hasName())                      // If the global has a name...
        if (FoundNames.count(I->getName()))  // And the name is already used
          MangledGlobals.insert(I);          // Mangle the name
        else
          FoundNames.insert(I->getName());   // Otherwise, keep track of name
  }

  // get declaration for alloca
  Out << "/* Provide Declarations */\n";
  Out << "#include <stdarg.h>\n";
  Out << "#include <setjmp.h>\n";
  generateCompilerSpecificCode(Out);
  
  // Provide a definition for `bool' if not compiling with a C++ compiler.
  Out << "\n"
      << "#ifndef __cplusplus\ntypedef unsigned char bool;\n#endif\n"
    
      << "\n\n/* Support for floating point constants */\n"
      << "typedef unsigned long long ConstantDoubleTy;\n"
      << "typedef unsigned int        ConstantFloatTy;\n"
    
      << "\n\n/* Support for the invoke instruction */\n"
      << "extern struct __llvm_jmpbuf_list_t {\n"
      << "  jmp_buf buf; struct __llvm_jmpbuf_list_t *next;\n"
      << "} *__llvm_jmpbuf_list;\n"

      << "\n\n/* Global Declarations */\n";

  // First output all the declarations for the program, because C requires
  // Functions & globals to be declared before they are used.
  //

  // Loop over the symbol table, emitting all named constants...
  printSymbolTable(M->getSymbolTable());

  // Global variable declarations...
  if (!M->gempty()) {
    Out << "\n/* External Global Variable Declarations */\n";
    for (Module::giterator I = M->gbegin(), E = M->gend(); I != E; ++I) {
      if (I->hasExternalLinkage()) {
        Out << "extern ";
        printType(Out, I->getType()->getElementType(), Mang->getValueName(I));
        Out << ";\n";
      }
    }
  }

  // Function declarations
  if (!M->empty()) {
    Out << "\n/* Function Declarations */\n";
    needsMalloc = true;
    for (Module::iterator I = M->begin(), E = M->end(); I != E; ++I) {
      // If the function is external and the name collides don't print it.
      // Sometimes the bytecode likes to have multiple "declarations" for
      // external functions
      if ((I->hasInternalLinkage() || !MangledGlobals.count(I)) &&
          !I->getIntrinsicID()) {
        printFunctionSignature(I, true);
        Out << ";\n";
      }
    }
  }

  // Print Malloc prototype if needed
  if (needsMalloc) {
    Out << "\n/* Malloc to make sun happy */\n";
    Out << "extern void * malloc();\n\n";
  }

  // Output the global variable declarations
  if (!M->gempty()) {
    Out << "\n\n/* Global Variable Declarations */\n";
    for (Module::giterator I = M->gbegin(), E = M->gend(); I != E; ++I)
      if (!I->isExternal()) {
        Out << "extern ";
        printType(Out, I->getType()->getElementType(), Mang->getValueName(I));
      
        Out << ";\n";
      }
  }

  // Output the global variable definitions and contents...
  if (!M->gempty()) {
    Out << "\n\n/* Global Variable Definitions and Initialization */\n";
    for (Module::giterator I = M->gbegin(), E = M->gend(); I != E; ++I)
      if (!I->isExternal()) {
        if (I->hasInternalLinkage())
          Out << "static ";
        printType(Out, I->getType()->getElementType(), Mang->getValueName(I));
        if (I->hasLinkOnceLinkage())
          Out << " __attribute__((common))";
        if (!I->getInitializer()->isNullValue()) {
          Out << " = " ;
          writeOperand(I->getInitializer());
        }
        Out << ";\n";
      }
  }

  // Output all of the functions...
  emittedInvoke = false;
  if (!M->empty()) {
    Out << "\n\n/* Function Bodies */\n";
    for (Module::iterator I = M->begin(), E = M->end(); I != E; ++I)
      printFunction(I);
  }

  // If the program included an invoke instruction, we need to output the
  // support code for it here!
  if (emittedInvoke) {
    Out << "\n/* More support for the invoke instruction */\n"
        << "struct __llvm_jmpbuf_list_t *__llvm_jmpbuf_list "
        << "__attribute__((common)) = 0;\n";
  }
}


/// printSymbolTable - Run through symbol table looking for type names.  If a
/// type name is found, emit it's declaration...
///
void CWriter::printSymbolTable(const SymbolTable &ST) {
  // If there are no type names, exit early.
  if (ST.find(Type::TypeTy) == ST.end())
    return;

  // We are only interested in the type plane of the symbol table...
  SymbolTable::type_const_iterator I   = ST.type_begin(Type::TypeTy);
  SymbolTable::type_const_iterator End = ST.type_end(Type::TypeTy);
  
  // Print out forward declarations for structure types before anything else!
  Out << "/* Structure forward decls */\n";
  for (; I != End; ++I)
    if (const Type *STy = dyn_cast<StructType>(I->second)) {
      std::string Name = "struct l_" + Mangler::makeNameProper(I->first);
      Out << Name << ";\n";
      TypeNames.insert(std::make_pair(STy, Name));
    }

  Out << "\n";

  // Now we can print out typedefs...
  Out << "/* Typedefs */\n";
  for (I = ST.type_begin(Type::TypeTy); I != End; ++I) {
    const Type *Ty = cast<Type>(I->second);
    std::string Name = "l_" + Mangler::makeNameProper(I->first);
    Out << "typedef ";
    printType(Out, Ty, Name);
    Out << ";\n";
  }

  Out << "\n";

  // Keep track of which structures have been printed so far...
  std::set<const StructType *> StructPrinted;

  // Loop over all structures then push them into the stack so they are
  // printed in the correct order.
  //
  Out << "/* Structure contents */\n";
  for (I = ST.type_begin(Type::TypeTy); I != End; ++I)
    if (const StructType *STy = dyn_cast<StructType>(I->second))
      printContainedStructs(STy, StructPrinted);
}

// Push the struct onto the stack and recursively push all structs
// this one depends on.
void CWriter::printContainedStructs(const Type *Ty,
                                    std::set<const StructType*> &StructPrinted){
  if (const StructType *STy = dyn_cast<StructType>(Ty)) {
    //Check to see if we have already printed this struct
    if (StructPrinted.count(STy) == 0) {
      // Print all contained types first...
      for (StructType::ElementTypes::const_iterator
             I = STy->getElementTypes().begin(),
             E = STy->getElementTypes().end(); I != E; ++I) {
        const Type *Ty1 = I->get();
        if (isa<StructType>(Ty1) || isa<ArrayType>(Ty1))
          printContainedStructs(*I, StructPrinted);
      }
      
      //Print structure type out..
      StructPrinted.insert(STy);
      std::string Name = TypeNames[STy];  
      printType(Out, STy, Name, true);
      Out << ";\n\n";
    }

    // If it is an array, check contained types and continue
  } else if (const ArrayType *ATy = dyn_cast<ArrayType>(Ty)){
    const Type *Ty1 = ATy->getElementType();
    if (isa<StructType>(Ty1) || isa<ArrayType>(Ty1))
      printContainedStructs(Ty1, StructPrinted);
  }
}


void CWriter::printFunctionSignature(const Function *F, bool Prototype) {
  // If the program provides its own malloc prototype we don't need
  // to include the general one.  
  if (Mang->getValueName(F) == "malloc")
    needsMalloc = false;

  if (F->hasInternalLinkage()) Out << "static ";
  if (F->hasLinkOnceLinkage()) Out << "inline ";
  
  // Loop over the arguments, printing them...
  const FunctionType *FT = cast<FunctionType>(F->getFunctionType());
  
  std::stringstream FunctionInnards; 
    
  // Print out the name...
  FunctionInnards << Mang->getValueName(F) << "(";
    
  if (!F->isExternal()) {
    if (!F->aempty()) {
      std::string ArgName;
      if (F->abegin()->hasName() || !Prototype)
        ArgName = Mang->getValueName(F->abegin());
      printType(FunctionInnards, F->afront().getType(), ArgName);
      for (Function::const_aiterator I = ++F->abegin(), E = F->aend();
           I != E; ++I) {
        FunctionInnards << ", ";
        if (I->hasName() || !Prototype)
          ArgName = Mang->getValueName(I);
        else 
          ArgName = "";
        printType(FunctionInnards, I->getType(), ArgName);
      }
    }
  } else {
    // Loop over the arguments, printing them...
    for (FunctionType::ParamTypes::const_iterator I = 
	   FT->getParamTypes().begin(),
	   E = FT->getParamTypes().end(); I != E; ++I) {
      if (I != FT->getParamTypes().begin()) FunctionInnards << ", ";
      printType(FunctionInnards, *I);
    }
  }

  // Finish printing arguments... if this is a vararg function, print the ...,
  // unless there are no known types, in which case, we just emit ().
  //
  if (FT->isVarArg() && !FT->getParamTypes().empty()) {
    if (FT->getParamTypes().size()) FunctionInnards << ", ";
    FunctionInnards << "...";  // Output varargs portion of signature!
  }
  FunctionInnards << ")";
  // Print out the return type and the entire signature for that matter
  printType(Out, F->getReturnType(), FunctionInnards.str());
}

void CWriter::printFunction(Function *F) {
  if (F->isExternal()) return;

  printFunctionSignature(F, false);
  Out << " {\n";

  // print local variable information for the function
  for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I)
    if (const AllocaInst *AI = isDirectAlloca(*I)) {
      Out << "  ";
      printType(Out, AI->getAllocatedType(), Mang->getValueName(AI));
      Out << ";    /* Address exposed local */\n";
    } else if ((*I)->getType() != Type::VoidTy && !isInlinableInst(**I)) {
      Out << "  ";
      printType(Out, (*I)->getType(), Mang->getValueName(*I));
      Out << ";\n";
      
      if (isa<PHINode>(*I)) {  // Print out PHI node temporaries as well...
        Out << "  ";
        printType(Out, (*I)->getType(), Mang->getValueName(*I)+"__PHI_TEMPORARY");
        Out << ";\n";
      }
    }

  Out << "\n";

  // Scan the function for floating point constants.  If any FP constant is used
  // in the function, we want to redirect it here so that we do not depend on
  // the precision of the printed form, unless the printed form preserves
  // precision.
  //
  unsigned FPCounter = 0;
  for (constant_iterator I = constant_begin(F), E = constant_end(F); I != E;++I)
    if (const ConstantFP *FPC = dyn_cast<ConstantFP>(*I))
      if ((!FPCSafeToPrint(FPC)) // Do not put in FPConstantMap if safe.
	  && (FPConstantMap.find(FPC) == FPConstantMap.end())) {
        double Val = FPC->getValue();
        
        FPConstantMap[FPC] = FPCounter;  // Number the FP constants

        if (FPC->getType() == Type::DoubleTy)
          Out << "  const ConstantDoubleTy FloatConstant" << FPCounter++
              << " = 0x" << std::hex << *(unsigned long long*)&Val << std::dec
              << "ULL;    /* " << Val << " */\n";
        else if (FPC->getType() == Type::FloatTy) {
          float fVal = Val;
          Out << "  const ConstantFloatTy FloatConstant" << FPCounter++
              << " = 0x" << std::hex << *(unsigned*)&fVal << std::dec
              << "U;    /* " << Val << " */\n";
        } else
          assert(0 && "Unknown float type!");
      }

  Out << "\n";
 
  // print the basic blocks
  for (Function::iterator BB = F->begin(), E = F->end(); BB != E; ++BB) {
    BasicBlock *Prev = BB->getPrev();

    // Don't print the label for the basic block if there are no uses, or if the
    // only terminator use is the predecessor basic block's terminator.  We have
    // to scan the use list because PHI nodes use basic blocks too but do not
    // require a label to be generated.
    //
    bool NeedsLabel = false;
    for (Value::use_iterator UI = BB->use_begin(), UE = BB->use_end();
         UI != UE; ++UI)
      if (TerminatorInst *TI = dyn_cast<TerminatorInst>(*UI))
        if (TI != Prev->getTerminator() ||
            isa<SwitchInst>(Prev->getTerminator()) ||
            isa<InvokeInst>(Prev->getTerminator())) {
          NeedsLabel = true;
          break;        
        }

    if (NeedsLabel) Out << Mang->getValueName(BB) << ":\n";

    // Output all of the instructions in the basic block...
    for (BasicBlock::iterator II = BB->begin(), E = --BB->end(); II != E; ++II){
      if (!isInlinableInst(*II) && !isDirectAlloca(II)) {
        if (II->getType() != Type::VoidTy)
          outputLValue(II);
        else
          Out << "  ";
        visit(*II);
        Out << ";\n";
      }
    }

    // Don't emit prefix or suffix for the terminator...
    visit(*BB->getTerminator());
  }
  
  Out << "}\n\n";
  FPConstantMap.clear();
}

// Specific Instruction type classes... note that all of the casts are
// necessary because we use the instruction classes as opaque types...
//
void CWriter::visitReturnInst(ReturnInst &I) {
  // Don't output a void return if this is the last basic block in the function
  if (I.getNumOperands() == 0 && 
      &*--I.getParent()->getParent()->end() == I.getParent() &&
      !I.getParent()->size() == 1) {
    return;
  }

  Out << "  return";
  if (I.getNumOperands()) {
    Out << " ";
    writeOperand(I.getOperand(0));
  }
  Out << ";\n";
}

void CWriter::visitSwitchInst(SwitchInst &SI) {
  Out << "  switch (";
  writeOperand(SI.getOperand(0));
  Out << ") {\n  default:\n";
  printBranchToBlock(SI.getParent(), SI.getDefaultDest(), 2);
  Out << ";\n";
  for (unsigned i = 2, e = SI.getNumOperands(); i != e; i += 2) {
    Out << "  case ";
    writeOperand(SI.getOperand(i));
    Out << ":\n";
    BasicBlock *Succ = cast<BasicBlock>(SI.getOperand(i+1));
    printBranchToBlock(SI.getParent(), Succ, 2);
    if (Succ == SI.getParent()->getNext())
      Out << "    break;\n";
  }
  Out << "  }\n";
}

void CWriter::visitInvokeInst(InvokeInst &II) {
  Out << "  {\n"
      << "    struct __llvm_jmpbuf_list_t Entry;\n"
      << "    Entry.next = __llvm_jmpbuf_list;\n"
      << "    if (setjmp(Entry.buf)) {\n"
      << "      __llvm_jmpbuf_list = Entry.next;\n";
  printBranchToBlock(II.getParent(), II.getExceptionalDest(), 4);
  Out << "    }\n"
      << "    __llvm_jmpbuf_list = &Entry;\n"
      << "    ";

  if (II.getType() != Type::VoidTy) outputLValue(&II);
  visitCallSite(&II);
  Out << ";\n"
      << "    __llvm_jmpbuf_list = Entry.next;\n"
      << "  }\n";
  printBranchToBlock(II.getParent(), II.getNormalDest(), 0);
  emittedInvoke = true;
}


void CWriter::visitUnwindInst(UnwindInst &I) {
  // The unwind instructions causes a control flow transfer out of the current
  // function, unwinding the stack until a caller who used the invoke
  // instruction is found.  In this context, we code generated the invoke
  // instruction to add an entry to the top of the jmpbuf_list.  Thus, here we
  // just have to longjmp to the specified handler.
  Out << "  if (__llvm_jmpbuf_list == 0) {  /* unwind */\n"
      << "    extern write();\n"
      << "    ((void (*)(int, void*, unsigned))write)(2,\n"
      << "           \"throw found with no handler!\\n\", 31); abort();\n"
      << "  }\n"
      << "  longjmp(__llvm_jmpbuf_list->buf, 1);\n";
  emittedInvoke = true;
}

static bool isGotoCodeNeccessary(BasicBlock *From, BasicBlock *To) {
  // If PHI nodes need copies, we need the copy code...
  if (isa<PHINode>(To->front()) ||
      From->getNext() != To)      // Not directly successor, need goto
    return true;

  // Otherwise we don't need the code.
  return false;
}

void CWriter::printBranchToBlock(BasicBlock *CurBB, BasicBlock *Succ,
                                 unsigned Indent) {
  for (BasicBlock::iterator I = Succ->begin();
       PHINode *PN = dyn_cast<PHINode>(I); ++I) {
    //  now we have to do the printing
    Out << std::string(Indent, ' ');
    Out << "  " << Mang->getValueName(I) << "__PHI_TEMPORARY = ";
    writeOperand(PN->getIncomingValue(PN->getBasicBlockIndex(CurBB)));
    Out << ";   /* for PHI node */\n";
  }

  if (CurBB->getNext() != Succ || isa<InvokeInst>(CurBB->getTerminator())) {
    Out << std::string(Indent, ' ') << "  goto ";
    writeOperand(Succ);
    Out << ";\n";
  }
}

// Branch instruction printing - Avoid printing out a branch to a basic block
// that immediately succeeds the current one.
//
void CWriter::visitBranchInst(BranchInst &I) {
  if (I.isConditional()) {
    if (isGotoCodeNeccessary(I.getParent(), I.getSuccessor(0))) {
      Out << "  if (";
      writeOperand(I.getCondition());
      Out << ") {\n";
      
      printBranchToBlock(I.getParent(), I.getSuccessor(0), 2);
      
      if (isGotoCodeNeccessary(I.getParent(), I.getSuccessor(1))) {
        Out << "  } else {\n";
        printBranchToBlock(I.getParent(), I.getSuccessor(1), 2);
      }
    } else {
      // First goto not necessary, assume second one is...
      Out << "  if (!";
      writeOperand(I.getCondition());
      Out << ") {\n";

      printBranchToBlock(I.getParent(), I.getSuccessor(1), 2);
    }

    Out << "  }\n";
  } else {
    printBranchToBlock(I.getParent(), I.getSuccessor(0), 0);
  }
  Out << "\n";
}

// PHI nodes get copied into temporary values at the end of predecessor basic
// blocks.  We now need to copy these temporary values into the REAL value for
// the PHI.
void CWriter::visitPHINode(PHINode &I) {
  writeOperand(&I);
  Out << "__PHI_TEMPORARY";
}


void CWriter::visitBinaryOperator(Instruction &I) {
  // binary instructions, shift instructions, setCond instructions.
  assert(!isa<PointerType>(I.getType()));

  // We must cast the results of binary operations which might be promoted.
  bool needsCast = false;
  if ((I.getType() == Type::UByteTy) || (I.getType() == Type::SByteTy)
      || (I.getType() == Type::UShortTy) || (I.getType() == Type::ShortTy)
      || (I.getType() == Type::FloatTy)) {
    needsCast = true;
    Out << "((";
    printType(Out, I.getType(), "", false, false);
    Out << ")(";
  }
      
  writeOperand(I.getOperand(0));

  switch (I.getOpcode()) {
  case Instruction::Add: Out << " + "; break;
  case Instruction::Sub: Out << " - "; break;
  case Instruction::Mul: Out << "*"; break;
  case Instruction::Div: Out << "/"; break;
  case Instruction::Rem: Out << "%"; break;
  case Instruction::And: Out << " & "; break;
  case Instruction::Or: Out << " | "; break;
  case Instruction::Xor: Out << " ^ "; break;
  case Instruction::SetEQ: Out << " == "; break;
  case Instruction::SetNE: Out << " != "; break;
  case Instruction::SetLE: Out << " <= "; break;
  case Instruction::SetGE: Out << " >= "; break;
  case Instruction::SetLT: Out << " < "; break;
  case Instruction::SetGT: Out << " > "; break;
  case Instruction::Shl : Out << " << "; break;
  case Instruction::Shr : Out << " >> "; break;
  default: std::cerr << "Invalid operator type!" << I; abort();
  }

  writeOperand(I.getOperand(1));

  if (needsCast) {
    Out << "))";
  }
}

void CWriter::visitCastInst(CastInst &I) {
  if (I.getType() == Type::BoolTy) {
    Out << "(";
    writeOperand(I.getOperand(0));
    Out << " != 0)";
    return;
  }
  Out << "(";
  printType(Out, I.getType(), "", /*ignoreName*/false, /*namedContext*/false);
  Out << ")";
  if (isa<PointerType>(I.getType())&&I.getOperand(0)->getType()->isIntegral() ||
      isa<PointerType>(I.getOperand(0)->getType())&&I.getType()->isIntegral()) {
    // Avoid "cast to pointer from integer of different size" warnings
    Out << "(long)";  
  }
  
  writeOperand(I.getOperand(0));
}

void CWriter::visitCallInst(CallInst &I) {
  // Handle intrinsic function calls first...
  if (Function *F = I.getCalledFunction())
    if (LLVMIntrinsic::ID ID = (LLVMIntrinsic::ID)F->getIntrinsicID()) {
      switch (ID) {
      default:  assert(0 && "Unknown LLVM intrinsic!");
      case LLVMIntrinsic::va_start: 
        Out << "va_start(*(va_list*)";
        writeOperand(I.getOperand(1));
        Out << ", ";
        // Output the last argument to the enclosing function...
        writeOperand(&I.getParent()->getParent()->aback());
        Out << ")";
        return;
      case LLVMIntrinsic::va_end:
        Out << "va_end(*(va_list*)";
        writeOperand(I.getOperand(1));
        Out << ")";
        return;
      case LLVMIntrinsic::va_copy:
        Out << "va_copy(*(va_list*)";
        writeOperand(I.getOperand(1));
        Out << ", (va_list)";
        writeOperand(I.getOperand(2));
        Out << ")";
        return;

      case LLVMIntrinsic::setjmp:
      case LLVMIntrinsic::sigsetjmp:
        // This intrinsic should never exist in the program, but until we get
        // setjmp/longjmp transformations going on, we should codegen it to
        // something reasonable.  This will allow code that never calls longjmp
        // to work.
        Out << "0";
        return;
      case LLVMIntrinsic::longjmp:
      case LLVMIntrinsic::siglongjmp:
        // Longjmp is not implemented, and never will be.  It would cause an
        // exception throw.
        Out << "abort()";
        return;
      }
    }
  visitCallSite(&I);
}

void CWriter::visitCallSite(CallSite CS) {
  const PointerType  *PTy   = cast<PointerType>(CS.getCalledValue()->getType());
  const FunctionType *FTy   = cast<FunctionType>(PTy->getElementType());
  const Type         *RetTy = FTy->getReturnType();
  
  writeOperand(CS.getCalledValue());
  Out << "(";

  if (CS.arg_begin() != CS.arg_end()) {
    CallSite::arg_iterator AI = CS.arg_begin(), AE = CS.arg_end();
    writeOperand(*AI);

    for (++AI; AI != AE; ++AI) {
      Out << ", ";
      writeOperand(*AI);
    }
  }
  Out << ")";
}  

void CWriter::visitMallocInst(MallocInst &I) {
  Out << "(";
  printType(Out, I.getType());
  Out << ")malloc(sizeof(";
  printType(Out, I.getType()->getElementType());
  Out << ")";

  if (I.isArrayAllocation()) {
    Out << " * " ;
    writeOperand(I.getOperand(0));
  }
  Out << ")";
}

void CWriter::visitAllocaInst(AllocaInst &I) {
  Out << "(";
  printType(Out, I.getType());
  Out << ") alloca(sizeof(";
  printType(Out, I.getType()->getElementType());
  Out << ")";
  if (I.isArrayAllocation()) {
    Out << " * " ;
    writeOperand(I.getOperand(0));
  }
  Out << ")";
}

void CWriter::visitFreeInst(FreeInst &I) {
  Out << "free((char*)";
  writeOperand(I.getOperand(0));
  Out << ")";
}

void CWriter::printIndexingExpression(Value *Ptr, User::op_iterator I,
                                      User::op_iterator E) {
  bool HasImplicitAddress = false;
  // If accessing a global value with no indexing, avoid *(&GV) syndrome
  if (GlobalValue *V = dyn_cast<GlobalValue>(Ptr)) {
    HasImplicitAddress = true;
  } else if (ConstantPointerRef *CPR = dyn_cast<ConstantPointerRef>(Ptr)) {
    HasImplicitAddress = true;
    Ptr = CPR->getValue();         // Get to the global...
  } else if (isDirectAlloca(Ptr)) {
    HasImplicitAddress = true;
  }

  if (I == E) {
    if (!HasImplicitAddress)
      Out << "*";  // Implicit zero first argument: '*x' is equivalent to 'x[0]'

    writeOperandInternal(Ptr);
    return;
  }

  const Constant *CI = dyn_cast<Constant>(I);
  if (HasImplicitAddress && (!CI || !CI->isNullValue()))
    Out << "(&";

  writeOperandInternal(Ptr);

  if (HasImplicitAddress && (!CI || !CI->isNullValue())) {
    Out << ")";
    HasImplicitAddress = false;  // HIA is only true if we haven't addressed yet
  }

  assert(!HasImplicitAddress || (CI && CI->isNullValue()) &&
         "Can only have implicit address with direct accessing");

  if (HasImplicitAddress) {
    ++I;
  } else if (CI && CI->isNullValue() && I+1 != E) {
    // Print out the -> operator if possible...
    if ((*(I+1))->getType() == Type::UByteTy) {
      Out << (HasImplicitAddress ? "." : "->");
      Out << "field" << cast<ConstantUInt>(*(I+1))->getValue();
      I += 2;
    } 
  }

  for (; I != E; ++I)
    if ((*I)->getType() == Type::LongTy) {
      Out << "[";
      writeOperand(*I);
      Out << "]";
    } else {
      Out << ".field" << cast<ConstantUInt>(*I)->getValue();
    }
}

void CWriter::visitLoadInst(LoadInst &I) {
  Out << "*";
  writeOperand(I.getOperand(0));
}

void CWriter::visitStoreInst(StoreInst &I) {
  Out << "*";
  writeOperand(I.getPointerOperand());
  Out << " = ";
  writeOperand(I.getOperand(0));
}

void CWriter::visitGetElementPtrInst(GetElementPtrInst &I) {
  Out << "&";
  printIndexingExpression(I.getPointerOperand(), I.idx_begin(), I.idx_end());
}

void CWriter::visitVarArgInst(VarArgInst &I) {
  Out << "va_arg((va_list)*";
  writeOperand(I.getOperand(0));
  Out << ", ";
  printType(Out, I.getType(), "", /*ignoreName*/false, /*namedContext*/false);
  Out << ")";  
}


//===----------------------------------------------------------------------===//
//                       External Interface declaration
//===----------------------------------------------------------------------===//

Pass *createWriteToCPass(std::ostream &o) { return new CWriter(o); }
