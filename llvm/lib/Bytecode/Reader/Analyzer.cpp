//===-- Analyzer.cpp - Analysis and Dumping of Bytecode 000000---*- C++ -*-===//
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
    bca.vbrCount32 = 0;
    bca.vbrCount64 = 0;
    bca.vbrCompBytes = 0;
    bca.vbrExpdBytes = 0;
    bca.FunctionInfo.clear();
    bca.BlockSizes[BytecodeFormat::Reserved_DoNotUse] = 0;
    bca.BlockSizes[BytecodeFormat::ModuleBlockID] = theSize;
    bca.BlockSizes[BytecodeFormat::FunctionBlockID] = 0;
    bca.BlockSizes[BytecodeFormat::ConstantPoolBlockID] = 0;
    bca.BlockSizes[BytecodeFormat::SymbolTableBlockID] = 0;
    bca.BlockSizes[BytecodeFormat::ModuleGlobalInfoBlockID] = 0;
    bca.BlockSizes[BytecodeFormat::GlobalTypePlaneBlockID] = 0;
    bca.BlockSizes[BytecodeFormat::InstructionListBlockID] = 0;
    bca.BlockSizes[BytecodeFormat::CompactionTableBlockID] = 0;
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
    unsigned char RevisionNum,        ///< Byte code revision number
    Module::Endianness Endianness,    ///< Endianness indicator
    Module::PointerSize PointerSize   ///< PointerSize indicator
  ) {
    if (os)
      *os << "    RevisionNum: " << int(RevisionNum)
         << " Endianness: " << Endianness
         << " PointerSize: " << PointerSize << "\n";
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
    unsigned SlotNum,
    unsigned initSlot
  ) {
    if (os) {
      *os << "      GV: "
          << ( initSlot == 0 ? "Uni" : "I" ) << "nitialized, "
          << ( isConstant? "Constant, " : "Variable, ")
          << " Linkage=" << Linkage << " Type=";
      WriteTypeSymbolic(*os, ElemType, M);
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
      WriteTypeSymbolic(*os,Ty,M);
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
      WriteTypeSymbolic(*os,Func->getType(),M);
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

  virtual void handleCompactionTableBegin() {
    if (os)
      *os << "      BLOCK: CompactionTable {\n";
    bca.numCmpctnTables++;
  }

  virtual void handleCompactionTablePlane( unsigned Ty, unsigned NumEntries) {
    if (os)
      *os << "        Plane: Ty=" << Ty << " Size=" << NumEntries << "\n";
  }

  virtual void handleCompactionTableType( unsigned i, unsigned TypSlot,
      const Type* Ty ) {
    if (os) {
      *os << "          Type: " << i << " Slot:" << TypSlot << " is ";
      WriteTypeSymbolic(*os,Ty,M);
      *os << "\n";
    }
  }

  virtual void handleCompactionTableValue(unsigned i, unsigned TypSlot,
                                          unsigned ValSlot) {
    if (os)
      *os << "          Value: " << i << " TypSlot: " << TypSlot
         << " ValSlot:" << ValSlot << "\n";
    if (ValSlot > bca.maxValueSlot)
      bca.maxValueSlot = ValSlot;
  }

  virtual void handleCompactionTableEnd() {
    if (os)
      *os << "      } END BLOCK: CompactionTable\n";
  }

  virtual void handleSymbolTableBegin(Function* CF, SymbolTable* ST) {
    bca.numSymTab++;
    if (os)
      *os << "    BLOCK: SymbolTable {\n";
  }

  virtual void handleSymbolTablePlane(unsigned Ty, unsigned NumEntries,
    const Type* Typ) {
    if (os) {
      *os << "      Plane: Ty=" << Ty << " Size=" << NumEntries << " Type: ";
      WriteTypeSymbolic(*os,Typ,M);
      *os << "\n";
    }
  }

  virtual void handleSymbolTableType(unsigned i, unsigned TypSlot,
    const std::string& name ) {
    if (os)
      *os << "        Type " << i << " Slot=" << TypSlot
         << " Name: " << name << "\n";
  }

  virtual void handleSymbolTableValue(unsigned i, unsigned ValSlot,
    const std::string& name ) {
    if (os)
      *os << "        Value " << i << " Slot=" << ValSlot
         << " Name: " << name << "\n";
    if (ValSlot > bca.maxValueSlot)
      bca.maxValueSlot = ValSlot;
  }

  virtual void handleSymbolTableEnd() {
    if (os)
      *os << "    } END BLOCK: SymbolTable\n";
  }

  virtual void handleFunctionBegin(Function* Func, unsigned Size) {
    if (os) {
      *os << "    BLOCK: Function {\n"
          << "      Linkage: " << Func->getLinkage() << "\n"
          << "      Type: ";
      WriteTypeSymbolic(*os,Func->getType(),M);
      *os << "\n";
    }

    currFunc = &bca.FunctionInfo[Func];
    std::ostringstream tmp;
    WriteTypeSymbolic(tmp,Func->getType(),M);
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
    currFunc->vbrCount32 = 0;
    currFunc->vbrCount64 = 0;
    currFunc->vbrCompBytes = 0;
    currFunc->vbrExpdBytes = 0;

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
                                std::vector<unsigned>& Operands, unsigned Size){
    if (os) {
      *os << "        INST: OpCode="
         << Instruction::getOpcodeName(Opcode) << " Type=\"";
      WriteTypeSymbolic(*os,iType,M);
      *os << "\"";
      for ( unsigned i = 0; i < Operands.size(); ++i )
        *os << " Op(" << i << ")=Slot(" << Operands[i] << ")";
      *os << "\n";
    }

    bca.numInstructions++;
    bca.numValues++;
    bca.instructionSize += Size;
    if (Size > 4 ) bca.longInstructions++;
    bca.numOperands += Operands.size();
    for (unsigned i = 0; i < Operands.size(); ++i )
      if (Operands[i] > bca.maxValueSlot)
        bca.maxValueSlot = Operands[i];
    if ( currFunc ) {
      currFunc->numInstructions++;
      currFunc->instructionSize += Size;
      if (Size > 4 ) currFunc->longInstructions++;
      if ( Opcode == Instruction::PHI ) currFunc->numPhis++;
    }
    return Instruction::isTerminator(Opcode);
  }

  virtual void handleBasicBlockEnd(unsigned blocknum) {
    if (os)
      *os << "      } END BLOCK: BasicBlock #" << blocknum << "{\n";
  }

  virtual void handleGlobalConstantsBegin() {
    if (os)
      *os << "    BLOCK: GlobalConstants {\n";
  }

  virtual void handleConstantExpression( unsigned Opcode,
      std::vector<Constant*> ArgVec, Constant* C ) {
    if (os) {
      *os << "      EXPR: " << Instruction::getOpcodeName(Opcode) << "\n";
      for ( unsigned i = 0; i < ArgVec.size(); ++i ) {
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
          std::vector<Constant*>& Elements,
          unsigned TypeSlot,
          Constant* ArrayVal ) {
    if (os) {
      *os << "      ARRAY: ";
      WriteTypeSymbolic(*os,AT,M);
      *os << " TypeSlot=" << TypeSlot << "\n";
      for ( unsigned i = 0; i < Elements.size(); ++i ) {
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
        std::vector<Constant*>& Elements,
        Constant* StructVal)
  {
    if (os) {
      *os << "      STRUC: ";
      WriteTypeSymbolic(*os,ST,M);
      *os << "\n";
      for ( unsigned i = 0; i < Elements.size(); ++i ) {
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

  virtual void handleConstantPacked(
    const PackedType* PT,
    std::vector<Constant*>& Elements,
    unsigned TypeSlot,
    Constant* PackedVal)
  {
    if (os) {
      *os << "      PACKD: ";
      WriteTypeSymbolic(*os,PT,M);
      *os << " TypeSlot=" << TypeSlot << "\n";
      for ( unsigned i = 0; i < Elements.size(); ++i ) {
        *os << "        #" << i;
        Elements[i]->print(*os);
        *os << "\n";
      }
      *os << "        Value=";
      PackedVal->print(*os);
      *os << "\n";
    }

    bca.numConstants++;
    bca.numValues++;
  }

  virtual void handleConstantPointer( const PointerType* PT,
      unsigned Slot, GlobalValue* GV ) {
    if (os) {
      *os << "       PNTR: ";
      WriteTypeSymbolic(*os,PT,M);
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
      llvm::BytecodeFormat::CompressedBytecodeBlockIdentifiers(BType)] += Size;

    if (bca.version < 3) // Check for long block headers versions
      bca.BlockSizes[llvm::BytecodeFormat::Reserved_DoNotUse] += 8;
    else
      bca.BlockSizes[llvm::BytecodeFormat::Reserved_DoNotUse] += 4;
  }

  virtual void handleVBR32(unsigned Size ) {
    bca.vbrCount32++;
    bca.vbrCompBytes += Size;
    bca.vbrExpdBytes += sizeof(uint32_t);
    if (currFunc) {
      currFunc->vbrCount32++;
      currFunc->vbrCompBytes += Size;
      currFunc->vbrExpdBytes += sizeof(uint32_t);
    }
  }

  virtual void handleVBR64(unsigned Size ) {
    bca.vbrCount64++;
    bca.vbrCompBytes += Size;
    bca.vbrExpdBytes += sizeof(uint64_t);
    if ( currFunc ) {
      currFunc->vbrCount64++;
      currFunc->vbrCompBytes += Size;
      currFunc->vbrExpdBytes += sizeof(uint64_t);
    }
  }
};


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

}

namespace llvm {

/// This function prints the contents of rhe BytecodeAnalysis structure in
/// a human legible form.
/// @brief Print BytecodeAnalysis structure to an ostream
void PrintBytecodeAnalysis(BytecodeAnalysis& bca, std::ostream& Out )
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
  print(Out, "Compaction Table Bytes",
        double(bca.BlockSizes[BytecodeFormat::CompactionTableBlockID]),
        double(bca.byteSize));
  print(Out, "Symbol Table Bytes",
        double(bca.BlockSizes[BytecodeFormat::SymbolTableBlockID]),
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
  print(Out, "Number Of Compaction Tables",     bca.numCmpctnTables);
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
  print(Out, "# of VBR 32-bit Integers",   bca.vbrCount32);
  print(Out, "# of VBR 64-bit Integers",   bca.vbrCount64);
  print(Out, "# of VBR Compressed Bytes",  bca.vbrCompBytes);
  print(Out, "# of VBR Expanded Bytes",    bca.vbrExpdBytes);
  print(Out, "Bytes Saved With VBR",
        double(bca.vbrExpdBytes)-double(bca.vbrCompBytes),
        double(bca.vbrExpdBytes));

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
        print(Out, "# of VBR 32-bit Integers",   I->second.vbrCount32);
        print(Out, "# of VBR 64-bit Integers",   I->second.vbrCount64);
        print(Out, "# of VBR Compressed Bytes",  I->second.vbrCompBytes);
        print(Out, "# of VBR Expanded Bytes",    I->second.vbrExpdBytes);
        print(Out, "Bytes Saved With VBR",
              double(I->second.vbrExpdBytes) - I->second.vbrCompBytes),
              double(I->second.vbrExpdBytes);
      }
      ++I;
    }
  }

  if ( bca.progressiveVerify )
    Out << bca.VerifyInfo;
}

BytecodeHandler* createBytecodeAnalyzerHandler(BytecodeAnalysis& bca,
                                               std::ostream* output)
{
  return new AnalyzerHandler(bca,output);
}

}

