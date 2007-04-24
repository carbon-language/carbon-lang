//===-- Analyzer.cpp - Analysis and Dumping of Bytecode ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the AnalyzerHandler class and PrintBytecodeAnalysis
//  function which together comprise the basic functionality of the llmv-abcd
//  tool. The AnalyzerHandler collects information about the bytecode file into
//  the BytecodeAnalysis structure. The PrintBytecodeAnalysis function prints
//  out the content of that structure.
//  @see include/llvm/Bytecode/Analysis.h
//
//===----------------------------------------------------------------------===//

#include "Reader.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Module.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Bytecode/BytecodeHandler.h"
#include "llvm/Assembly/Writer.h"
#include <iomanip>
#include <sstream>
#include <ios>
using namespace llvm;

namespace {

/// @brief Bytecode reading handler for analyzing bytecode.
class AnalyzerHandler : public BytecodeHandler {
  BytecodeAnalysis& bca;     ///< The structure in which data is recorded
  std::ostream* os;        ///< A convenience for osing data.
  /// @brief Keeps track of current function
  BytecodeAnalysis::BytecodeFunctionInfo* currFunc;
  Module* M; ///< Keeps track of current module

/// @name Constructor
/// @{
public:
  /// The only way to construct an AnalyzerHandler. All that is needed is a
  /// reference to the BytecodeAnalysis structure where the output will be
  /// placed.
  AnalyzerHandler(BytecodeAnalysis& TheBca, std::ostream* output)
    : bca(TheBca)
    , os(output)
    , currFunc(0)
    { }

/// @}
/// @name BytecodeHandler Implementations
/// @{
public:
  virtual void handleError(const std::string& str ) {
    if (os)
      *os << "ERROR: " << str << "\n";
  }

  virtual void handleStart( Module* Mod, unsigned theSize ) {
    M = Mod;
    if (os)
      *os << "Bytecode {\n";
    bca.byteSize = theSize;
    bca.ModuleId.clear();
    bca.numBlocks = 0;
    bca.numTypes = 0;
    bca.numValues = 0;
    bca.numFunctions = 0;
    bca.numConstants = 0;
    bca.numGlobalVars = 0;
    bca.numInstructions = 0;
    bca.numBasicBlocks = 0;
    bca.numOperands = 0;
    bca.numCmpctnTables = 0;
    bca.numSymTab = 0;
    bca.numLibraries = 0;
    bca.libSize = 0;
    bca.maxTypeSlot = 0;
    bca.maxValueSlot = 0;
    bca.numAlignment = 0;
    bca.fileDensity = 0.0;
    bca.globalsDensity = 0.0;
    bca.functionDensity = 0.0;
    bca.instructionSize = 0;
    bca.longInstructions = 0;
    bca.FunctionInfo.clear();
    bca.BlockSizes[BytecodeFormat::Reserved_DoNotUse] = 0;
    bca.BlockSizes[BytecodeFormat::ModuleBlockID] = theSize;
    bca.BlockSizes[BytecodeFormat::FunctionBlockID] = 0;
    bca.BlockSizes[BytecodeFormat::ConstantPoolBlockID] = 0;
    bca.BlockSizes[BytecodeFormat::ValueSymbolTableBlockID] = 0;
    bca.BlockSizes[BytecodeFormat::ModuleGlobalInfoBlockID] = 0;
    bca.BlockSizes[BytecodeFormat::GlobalTypePlaneBlockID] = 0;
    bca.BlockSizes[BytecodeFormat::InstructionListBlockID] = 0;
    bca.BlockSizes[BytecodeFormat::TypeSymbolTableBlockID] = 0;
  }

  virtual void handleFinish() {
    if (os)
      *os << "} End Bytecode\n";

    bca.fileDensity = double(bca.byteSize) / double( bca.numTypes + bca.numValues );
    double globalSize = 0.0;
    globalSize += double(bca.BlockSizes[BytecodeFormat::ConstantPoolBlockID]);
    globalSize += double(bca.BlockSizes[BytecodeFormat::ModuleGlobalInfoBlockID]);
    globalSize += double(bca.BlockSizes[BytecodeFormat::GlobalTypePlaneBlockID]);
    bca.globalsDensity = globalSize / double( bca.numTypes + bca.numConstants +
      bca.numGlobalVars );
    bca.functionDensity = double(bca.BlockSizes[BytecodeFormat::FunctionBlockID]) /
      double(bca.numFunctions);

    if (bca.progressiveVerify) {
      std::string msg;
      if (verifyModule(*M, ReturnStatusAction, &msg))
        bca.VerifyInfo += "Verify@Finish: " + msg + "\n";
    }
  }

  virtual void handleModuleBegin(const std::string& id) {
    if (os)
      *os << "  Module " << id << " {\n";
    bca.ModuleId = id;
  }

  virtual void handleModuleEnd(const std::string& id) {
    if (os)
      *os << "  } End Module " << id << "\n";
    if (bca.progressiveVerify) {
      std::string msg;
      if (verifyModule(*M, ReturnStatusAction, &msg))
        bca.VerifyInfo += "Verify@EndModule: " + msg + "\n";
    }
  }

  virtual void handleVersionInfo(
    unsigned char RevisionNum        ///< Byte code revision number
  ) {
    if (os)
      *os << "    RevisionNum: " << int(RevisionNum) << "\n";
    bca.version = RevisionNum;
  }

  virtual void handleModuleGlobalsBegin() {
    if (os)
      *os << "    BLOCK: ModuleGlobalInfo {\n";
  }

  virtual void handleGlobalVariable(
    const Type* ElemType,
    bool isConstant,
    GlobalValue::LinkageTypes Linkage,
    GlobalValue::VisibilityTypes Visibility,
    unsigned SlotNum,
    unsigned initSlot,
    bool isThreadLocal
  ) {
    if (os) {
      *os << "      GV: "
          << ( initSlot == 0 ? "Uni" : "I" ) << "nitialized, "
          << ( isConstant? "Constant, " : "Variable, ")
          << " Thread Local = " << ( isThreadLocal? "yes, " : "no, ")
          << " Linkage=" << Linkage
          << " Visibility="<< Visibility
          << " Type=";
      //WriteTypeSymbolic(*os, ElemType, M);
      *os << " Slot=" << SlotNum << " InitSlot=" << initSlot
          << "\n";
    }

    bca.numGlobalVars++;
    bca.numValues++;
    if (SlotNum > bca.maxValueSlot)
      bca.maxValueSlot = SlotNum;
    if (initSlot > bca.maxValueSlot)
      bca.maxValueSlot = initSlot;

  }

  virtual void handleTypeList(unsigned numEntries) {
    bca.maxTypeSlot = numEntries - 1;
  }

  virtual void handleType( const Type* Ty ) {
    bca.numTypes++;
    if (os) {
      *os << "      Type: ";
      //WriteTypeSymbolic(*os,Ty,M);
      *os << "\n";
    }
  }

  virtual void handleFunctionDeclaration(
    Function* Func            ///< The function
  ) {
    bca.numFunctions++;
    bca.numValues++;
    if (os) {
      *os << "      Function Decl: ";
      //WriteTypeSymbolic(*os,Func->getType(),M);
      *os <<", Linkage=" << Func->getLinkage();
      *os <<", Visibility=" << Func->getVisibility();
      *os << "\n";
    }
  }

  virtual void handleGlobalInitializer(GlobalVariable* GV, Constant* CV) {
    if (os) {
      *os << "    Initializer: GV=";
      GV->print(*os);
      *os << "      CV=";
      CV->print(*os);
      *os << "\n";
    }
  }

  virtual void handleDependentLibrary(const std::string& libName) {
    bca.numLibraries++;
    bca.libSize += libName.size() + (libName.size() < 128 ? 1 : 2);
    if (os)
      *os << "      Library: '" << libName << "'\n";
  }

  virtual void handleModuleGlobalsEnd() {
    if (os)
      *os << "    } END BLOCK: ModuleGlobalInfo\n";
    if (bca.progressiveVerify) {
      std::string msg;
      if (verifyModule(*M, ReturnStatusAction, &msg))
        bca.VerifyInfo += "Verify@EndModuleGlobalInfo: " + msg + "\n";
    }
  }

  virtual void handleTypeSymbolTableBegin(TypeSymbolTable* ST) {
    bca.numSymTab++;
    if (os)
      *os << "    BLOCK: TypeSymbolTable {\n";
  }
  virtual void handleValueSymbolTableBegin(Function* CF, ValueSymbolTable* ST) {
    bca.numSymTab++;
    if (os)
      *os << "    BLOCK: ValueSymbolTable {\n";
  }

  virtual void handleSymbolTableType(unsigned i, unsigned TypSlot,
    const std::string& name ) {
    if (os)
      *os << "        Type " << i << " Slot=" << TypSlot
         << " Name: " << name << "\n";
  }

  virtual void handleSymbolTableValue(unsigned TySlot, unsigned ValSlot, 
                                      const char *Name, unsigned NameLen) {
    if (os)
      *os << "        Value " << TySlot << " Slot=" << ValSlot
          << " Name: " << std::string(Name, Name+NameLen) << "\n";
    if (ValSlot > bca.maxValueSlot)
      bca.maxValueSlot = ValSlot;
  }

  virtual void handleValueSymbolTableEnd() {
    if (os)
      *os << "    } END BLOCK: ValueSymbolTable\n";
  }

  virtual void handleTypeSymbolTableEnd() {
    if (os)
      *os << "    } END BLOCK: TypeSymbolTable\n";
  }

  virtual void handleFunctionBegin(Function* Func, unsigned Size) {
    if (os) {
      *os << "    BLOCK: Function {\n"
          << "      Linkage: " << Func->getLinkage() << "\n"
          << "      Visibility: " << Func->getVisibility() << "\n"
          << "      Type: ";
      //WriteTypeSymbolic(*os,Func->getType(),M);
      *os << "\n";
    }

    currFunc = &bca.FunctionInfo[Func];
    std::ostringstream tmp;
    //WriteTypeSymbolic(tmp,Func->getType(),M);
    currFunc->description = tmp.str();
    currFunc->name = Func->getName();
    currFunc->byteSize = Size;
    currFunc->numInstructions = 0;
    currFunc->numBasicBlocks = 0;
    currFunc->numPhis = 0;
    currFunc->numOperands = 0;
    currFunc->density = 0.0;
    currFunc->instructionSize = 0;
    currFunc->longInstructions = 0;

  }

  virtual void handleFunctionEnd( Function* Func) {
    if (os)
      *os << "    } END BLOCK: Function\n";
    currFunc->density = double(currFunc->byteSize) /
      double(currFunc->numInstructions);

    if (bca.progressiveVerify) {
      std::string msg;
      if (verifyModule(*M, ReturnStatusAction, &msg))
        bca.VerifyInfo += "Verify@EndFunction: " + msg + "\n";
    }
  }

  virtual void handleBasicBlockBegin( unsigned blocknum) {
    if (os)
      *os << "      BLOCK: BasicBlock #" << blocknum << "{\n";
    bca.numBasicBlocks++;
    bca.numValues++;
    if ( currFunc ) currFunc->numBasicBlocks++;
  }

  virtual bool handleInstruction( unsigned Opcode, const Type* iType,
                                unsigned *Operands, unsigned NumOps, 
                                Instruction *Inst,
                                unsigned Size){
    if (os) {
      *os << "        INST: OpCode="
         << Instruction::getOpcodeName(Opcode);
      for (unsigned i = 0; i != NumOps; ++i)
        *os << " Op(" << Operands[i] << ")";
      *os << *Inst;
    }

    bca.numInstructions++;
    bca.numValues++;
    bca.instructionSize += Size;
    if (Size > 4 ) bca.longInstructions++;
    bca.numOperands += NumOps;
    for (unsigned i = 0; i != NumOps; ++i)
      if (Operands[i] > bca.maxValueSlot)
        bca.maxValueSlot = Operands[i];
    if ( currFunc ) {
      currFunc->numInstructions++;
      currFunc->instructionSize += Size;
      if (Size > 4 ) currFunc->longInstructions++;
      if (Opcode == Instruction::PHI) currFunc->numPhis++;
    }
    return Instruction::isTerminator(Opcode);
  }

  virtual void handleBasicBlockEnd(unsigned blocknum) {
    if (os)
      *os << "      } END BLOCK: BasicBlock #" << blocknum << "\n";
  }

  virtual void handleGlobalConstantsBegin() {
    if (os)
      *os << "    BLOCK: GlobalConstants {\n";
  }

  virtual void handleConstantExpression(unsigned Opcode,
      Constant**ArgVec, unsigned NumArgs, Constant* C) {
    if (os) {
      *os << "      EXPR: " << Instruction::getOpcodeName(Opcode) << "\n";
      for ( unsigned i = 0; i != NumArgs; ++i ) {
        *os << "        Arg#" << i << " "; ArgVec[i]->print(*os);
        *os << "\n";
      }
      *os << "        Value=";
      C->print(*os);
      *os << "\n";
    }
    bca.numConstants++;
    bca.numValues++;
  }

  virtual void handleConstantValue( Constant * c ) {
    if (os) {
      *os << "      VALUE: ";
      c->print(*os);
      *os << "\n";
    }
    bca.numConstants++;
    bca.numValues++;
  }

  virtual void handleConstantArray( const ArrayType* AT,
          Constant**Elements, unsigned NumElts,
          unsigned TypeSlot,
          Constant* ArrayVal ) {
    if (os) {
      *os << "      ARRAY: ";
      //WriteTypeSymbolic(*os,AT,M);
      *os << " TypeSlot=" << TypeSlot << "\n";
      for (unsigned i = 0; i != NumElts; ++i) {
        *os << "        #" << i;
        Elements[i]->print(*os);
        *os << "\n";
      }
      *os << "        Value=";
      ArrayVal->print(*os);
      *os << "\n";
    }

    bca.numConstants++;
    bca.numValues++;
  }

  virtual void handleConstantStruct(
        const StructType* ST,
        Constant**Elements, unsigned NumElts,
        Constant* StructVal)
  {
    if (os) {
      *os << "      STRUC: ";
      //WriteTypeSymbolic(*os,ST,M);
      *os << "\n";
      for ( unsigned i = 0; i != NumElts; ++i) {
        *os << "        #" << i << " "; Elements[i]->print(*os);
        *os << "\n";
      }
      *os << "        Value=";
      StructVal->print(*os);
      *os << "\n";
    }
    bca.numConstants++;
    bca.numValues++;
  }

  virtual void handleConstantVector(
    const VectorType* PT,
    Constant**Elements, unsigned NumElts,
    unsigned TypeSlot,
    Constant* VectorVal)
  {
    if (os) {
      *os << "      PACKD: ";
      //WriteTypeSymbolic(*os,PT,M);
      *os << " TypeSlot=" << TypeSlot << "\n";
      for ( unsigned i = 0; i != NumElts; ++i ) {
        *os << "        #" << i;
        Elements[i]->print(*os);
        *os << "\n";
      }
      *os << "        Value=";
      VectorVal->print(*os);
      *os << "\n";
    }

    bca.numConstants++;
    bca.numValues++;
  }

  virtual void handleConstantPointer( const PointerType* PT,
      unsigned Slot, GlobalValue* GV ) {
    if (os) {
      *os << "       PNTR: ";
      //WriteTypeSymbolic(*os,PT,M);
      *os << " Slot=" << Slot << " GlobalValue=";
      GV->print(*os);
      *os << "\n";
    }
    bca.numConstants++;
    bca.numValues++;
  }

  virtual void handleConstantString( const ConstantArray* CA ) {
    if (os) {
      *os << "      STRNG: ";
      CA->print(*os);
      *os << "\n";
    }
    bca.numConstants++;
    bca.numValues++;
  }

  virtual void handleGlobalConstantsEnd() {
    if (os)
      *os << "    } END BLOCK: GlobalConstants\n";

    if (bca.progressiveVerify) {
      std::string msg;
      if (verifyModule(*M, ReturnStatusAction, &msg))
        bca.VerifyInfo += "Verify@EndGlobalConstants: " + msg + "\n";
    }
  }

  virtual void handleAlignment(unsigned numBytes) {
    bca.numAlignment += numBytes;
  }

  virtual void handleBlock(
    unsigned BType, const unsigned char* StartPtr, unsigned Size) {
    bca.numBlocks++;
    assert(BType >= BytecodeFormat::ModuleBlockID);
    assert(BType < BytecodeFormat::NumberOfBlockIDs);
    bca.BlockSizes[
      llvm::BytecodeFormat::BytecodeBlockIdentifiers(BType)] += Size;

    if (bca.version < 3) // Check for long block headers versions
      bca.BlockSizes[llvm::BytecodeFormat::Reserved_DoNotUse] += 8;
    else
      bca.BlockSizes[llvm::BytecodeFormat::Reserved_DoNotUse] += 4;
  }

};
} // end anonymous namespace

/// @brief Utility for printing a titled unsigned value with
/// an aligned colon.
inline static void print(std::ostream& Out, const char*title,
  unsigned val, bool nl = true ) {
  Out << std::setw(30) << std::right << title
      << std::setw(0) << ": "
      << std::setw(9) << val << "\n";
}

/// @brief Utility for printing a titled double value with an
/// aligned colon
inline static void print(std::ostream&Out, const char*title,
  double val ) {
  Out << std::setw(30) << std::right << title
      << std::setw(0) << ": "
      << std::setw(9) << std::setprecision(6) << val << "\n" ;
}

/// @brief Utility for printing a titled double value with a
/// percentage and aligned colon.
inline static void print(std::ostream&Out, const char*title,
  double top, double bot ) {
  Out << std::setw(30) << std::right << title
      << std::setw(0) << ": "
      << std::setw(9) << std::setprecision(6) << top
      << " (" << std::left << std::setw(0) << std::setprecision(4)
      << (top/bot)*100.0 << "%)\n";
}

/// @brief Utility for printing a titled string value with
/// an aligned colon.
inline static void print(std::ostream&Out, const char*title,
  std::string val, bool nl = true) {
  Out << std::setw(30) << std::right << title
      << std::setw(0) << ": "
      << std::left << val << (nl ? "\n" : "");
}

/// This function prints the contents of rhe BytecodeAnalysis structure in
/// a human legible form.
/// @brief Print BytecodeAnalysis structure to an ostream
void llvm::PrintBytecodeAnalysis(BytecodeAnalysis& bca, std::ostream& Out )
{
  Out << "\nSummary Analysis Of " << bca.ModuleId << ": \n\n";
  print(Out, "Bytecode Analysis Of Module",     bca.ModuleId);
  print(Out, "Bytecode Version Number",         bca.version);
  print(Out, "File Size",                       bca.byteSize);
  print(Out, "Module Bytes",
        double(bca.BlockSizes[BytecodeFormat::ModuleBlockID]),
        double(bca.byteSize));
  print(Out, "Function Bytes",
        double(bca.BlockSizes[BytecodeFormat::FunctionBlockID]),
        double(bca.byteSize));
  print(Out, "Global Types Bytes",
        double(bca.BlockSizes[BytecodeFormat::GlobalTypePlaneBlockID]),
        double(bca.byteSize));
  print(Out, "Constant Pool Bytes",
        double(bca.BlockSizes[BytecodeFormat::ConstantPoolBlockID]),
        double(bca.byteSize));
  print(Out, "Module Globals Bytes",
        double(bca.BlockSizes[BytecodeFormat::ModuleGlobalInfoBlockID]),
        double(bca.byteSize));
  print(Out, "Instruction List Bytes",
        double(bca.BlockSizes[BytecodeFormat::InstructionListBlockID]),
        double(bca.byteSize));
  print(Out, "Value Symbol Table Bytes",
        double(bca.BlockSizes[BytecodeFormat::ValueSymbolTableBlockID]),
        double(bca.byteSize));
  print(Out, "Type Symbol Table Bytes",
        double(bca.BlockSizes[BytecodeFormat::TypeSymbolTableBlockID]),
        double(bca.byteSize));
  print(Out, "Alignment Bytes",
        double(bca.numAlignment), double(bca.byteSize));
  print(Out, "Block Header Bytes",
        double(bca.BlockSizes[BytecodeFormat::Reserved_DoNotUse]),
        double(bca.byteSize));
  print(Out, "Dependent Libraries Bytes", double(bca.libSize),
        double(bca.byteSize));
  print(Out, "Number Of Bytecode Blocks",       bca.numBlocks);
  print(Out, "Number Of Functions",             bca.numFunctions);
  print(Out, "Number Of Types",                 bca.numTypes);
  print(Out, "Number Of Constants",             bca.numConstants);
  print(Out, "Number Of Global Variables",      bca.numGlobalVars);
  print(Out, "Number Of Values",                bca.numValues);
  print(Out, "Number Of Basic Blocks",          bca.numBasicBlocks);
  print(Out, "Number Of Instructions",          bca.numInstructions);
  print(Out, "Number Of Long Instructions",     bca.longInstructions);
  print(Out, "Number Of Operands",              bca.numOperands);
  print(Out, "Number Of Symbol Tables",         bca.numSymTab);
  print(Out, "Number Of Dependent Libs",        bca.numLibraries);
  print(Out, "Total Instruction Size",          bca.instructionSize);
  print(Out, "Average Instruction Size",
        double(bca.instructionSize)/double(bca.numInstructions));

  print(Out, "Maximum Type Slot Number",        bca.maxTypeSlot);
  print(Out, "Maximum Value Slot Number",       bca.maxValueSlot);
  print(Out, "Bytes Per Value ",                bca.fileDensity);
  print(Out, "Bytes Per Global",                bca.globalsDensity);
  print(Out, "Bytes Per Function",              bca.functionDensity);

  if (bca.detailedResults) {
    Out << "\nDetailed Analysis Of " << bca.ModuleId << " Functions:\n";

    std::map<const Function*,BytecodeAnalysis::BytecodeFunctionInfo>::iterator I =
      bca.FunctionInfo.begin();
    std::map<const Function*,BytecodeAnalysis::BytecodeFunctionInfo>::iterator E =
      bca.FunctionInfo.end();

    while ( I != E ) {
      Out << std::left << std::setw(0) << "\n";
      if (I->second.numBasicBlocks == 0) Out << "External ";
      Out << "Function: " << I->second.name << "\n";
      print(Out, "Type:", I->second.description);
      print(Out, "Byte Size", I->second.byteSize);
      if (I->second.numBasicBlocks) {
        print(Out, "Basic Blocks", I->second.numBasicBlocks);
        print(Out, "Instructions", I->second.numInstructions);
        print(Out, "Long Instructions", I->second.longInstructions);
        print(Out, "Operands", I->second.numOperands);
        print(Out, "Instruction Size", I->second.instructionSize);
        print(Out, "Average Instruction Size",
              double(I->second.instructionSize) / I->second.numInstructions);
        print(Out, "Bytes Per Instruction", I->second.density);
      }
      ++I;
    }
  }

  if ( bca.progressiveVerify )
    Out << bca.VerifyInfo;
}

// AnalyzeBytecodeFile - analyze one file
Module* llvm::AnalyzeBytecodeFile(const std::string &Filename,  ///< File to analyze
                                  BytecodeAnalysis& bca,        ///< Statistical output
                                  BCDecompressor_t *BCDC,
                                  std::string *ErrMsg,          ///< Error output
                                  std::ostream* output          ///< Dump output
                                  ) {
  BytecodeHandler* AH = new AnalyzerHandler(bca, output);
  ModuleProvider* MP = getBytecodeModuleProvider(Filename, BCDC, ErrMsg, AH);
  if (!MP) return 0;
  Module *M = MP->releaseModule(ErrMsg);
  delete MP;
  return M;
}
