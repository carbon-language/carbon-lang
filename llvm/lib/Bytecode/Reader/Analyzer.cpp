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
#include "llvm/Bytecode/Analyzer.h"
#include "llvm/Bytecode/BytecodeHandler.h"
#include <iomanip>
#include <sstream>

using namespace llvm;

namespace {

/// @brief Bytecode reading handler for analyzing bytecode.
class AnalyzerHandler : public BytecodeHandler {
  BytecodeAnalysis& bca;     ///< The structure in which data is recorded
  std::ostringstream dump;   ///< A convenience for dumping data.
  /// @brief Keeps track of current function
  BytecodeAnalysis::BytecodeFunctionInfo* currFunc; 
  Module* M; ///< Keeps track of current module

/// @name Constructor
/// @{
public:
  /// The only way to construct an AnalyzerHandler. All that is needed is a
  /// reference to the BytecodeAnalysis structure where the output will be
  /// placed.
  AnalyzerHandler(BytecodeAnalysis& TheBca) 
    : bca(TheBca) 
    , dump()
    , currFunc(0)
    { }  

/// @}
/// @name BytecodeHandler Implementations
/// @{
public:
  virtual void handleError(const std::string& str ) { 
    dump << "ERROR: " << str << "\n";
    bca.BytecodeDump = dump.str() ;
  }

  virtual void handleStart( Module* Mod, unsigned theSize ) {
    M = Mod;
    dump << "Bytecode {\n";
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
    bca.BytecodeDump.clear();
    bca.BlockSizes[BytecodeFormat::Module] = 0;
    bca.BlockSizes[BytecodeFormat::Function] = 0;
    bca.BlockSizes[BytecodeFormat::ConstantPool] = 0;
    bca.BlockSizes[BytecodeFormat::SymbolTable] = 0;
    bca.BlockSizes[BytecodeFormat::ModuleGlobalInfo] = 0;
    bca.BlockSizes[BytecodeFormat::GlobalTypePlane] = 0;
    bca.BlockSizes[BytecodeFormat::BasicBlock] = 0;
    bca.BlockSizes[BytecodeFormat::InstructionList] = 0;
    bca.BlockSizes[BytecodeFormat::CompactionTable] = 0;
  }

  virtual void handleFinish() {
    dump << "} End Bytecode\n"; 
    bca.BytecodeDump = dump.str() ;

    bca.fileDensity = double(bca.byteSize) / double( bca.numTypes + bca.numValues );
    double globalSize = 0.0;
    globalSize += double(bca.BlockSizes[BytecodeFormat::ConstantPool]);
    globalSize += double(bca.BlockSizes[BytecodeFormat::ModuleGlobalInfo]);
    globalSize += double(bca.BlockSizes[BytecodeFormat::GlobalTypePlane]);
    bca.globalsDensity = globalSize / double( bca.numTypes + bca.numConstants + 
      bca.numGlobalVars );
    bca.functionDensity = double(bca.BlockSizes[BytecodeFormat::Function]) / 
      double(bca.numFunctions);

    if ( bca.progressiveVerify ) {
      try {
	verifyModule(*M, ThrowExceptionAction);
      } catch ( std::string& msg ) {
	bca.VerifyInfo += "Verify@Finish: " + msg + "\n";
      }
    }
  }

  virtual void handleModuleBegin(const std::string& id) {
    dump << "  Module " << id << " {\n";
    bca.ModuleId = id;
  }

  virtual void handleModuleEnd(const std::string& id) { 
    dump << "  } End Module " << id << "\n";
    if ( bca.progressiveVerify ) {
      try {
	verifyModule(*M, ThrowExceptionAction);
      } catch ( std::string& msg ) {
	bca.VerifyInfo += "Verify@EndModule: " + msg + "\n";
      }
    }
  }

  virtual void handleVersionInfo(
    unsigned char RevisionNum,        ///< Byte code revision number
    Module::Endianness Endianness,    ///< Endianness indicator
    Module::PointerSize PointerSize   ///< PointerSize indicator
  ) { 
    dump << "    RevisionNum: " << int(RevisionNum) 
	 << " Endianness: " << Endianness
	 << " PointerSize: " << PointerSize << "\n";
  }

  virtual void handleModuleGlobalsBegin() { 
    dump << "    BLOCK: ModuleGlobalInfo {\n";
  }

  virtual void handleGlobalVariable( 
    const Type* ElemType,     
    bool isConstant,          
    GlobalValue::LinkageTypes Linkage,
    unsigned SlotNum,
    unsigned initSlot
  ) {
    bca.numGlobalVars++;
    bca.numValues++;

    dump << "      GV: "
         << ( initSlot == 0 ? "Uni" : "I" ) << "nitialized, "
	 << ( isConstant? "Constant, " : "Variable, ")
	 << " Linkage=" << Linkage << " Type=" 
	 << ElemType->getDescription() 
	 << " Slot=" << SlotNum << " InitSlot=" << initSlot 
	 << "\n";
  }

  virtual void handleType( const Type* Ty ) { 
    bca.numTypes++; 
    dump << "      Type: " << Ty->getDescription() << "\n";
  }

  virtual void handleFunctionDeclaration( 
    Function* Func	    ///< The function
  ) {
    bca.numFunctions++;
    bca.numValues++;
    dump << "      Function Decl: " << Func->getType()->getDescription() << "\n";
  }

  virtual void handleGlobalInitializer(GlobalVariable* GV, Constant* CV) {
    dump << "      Initializer: GV=";
    GV->print(dump);
    dump << " CV=";
    CV->print(dump);
    dump << "\n";
  }

  virtual void handleModuleGlobalsEnd() { 
    dump << "    } END BLOCK: ModuleGlobalInfo\n";
    if ( bca.progressiveVerify ) {
      try {
	verifyModule(*M, ThrowExceptionAction);
      } catch ( std::string& msg ) {
	bca.VerifyInfo += "Verify@EndModuleGlobalInfo: " + msg + "\n";
      }
    }
  }

  virtual void handleCompactionTableBegin() { 
    dump << "    BLOCK: CompactionTable {\n";
  }

  virtual void handleCompactionTablePlane( unsigned Ty, unsigned NumEntries) {
    bca.numCmpctnTables++;
    dump << "      Plane: Ty=" << Ty << " Size=" << NumEntries << "\n";
  }

  virtual void handleCompactionTableType( unsigned i, unsigned TypSlot, 
      const Type* Ty ) {
    dump << "        Type: " << i << " Slot:" << TypSlot 
	      << " is " << Ty->getDescription() << "\n"; 
  }

  virtual void handleCompactionTableValue( 
    unsigned i, 
    unsigned TypSlot,
    unsigned ValSlot, 
    const Type* Ty ) { 
    dump << "        Value: " << i << " TypSlot: " << TypSlot 
	 << " ValSlot:" << ValSlot << " is " << Ty->getDescription() 
	 << "\n";
  }

  virtual void handleCompactionTableEnd() { 
    dump << "    } END BLOCK: CompactionTable\n";
  }

  virtual void handleSymbolTableBegin(Function* CF, SymbolTable* ST) { 
    bca.numSymTab++; 
    dump << "    BLOCK: SymbolTable {\n";
  }

  virtual void handleSymbolTablePlane(unsigned Ty, unsigned NumEntries, 
    const Type* Typ) { 
    dump << "      Plane: Ty=" << Ty << " Size=" << NumEntries
	 << " Type: " << Typ->getDescription() << "\n"; 
  }

  virtual void handleSymbolTableType(unsigned i, unsigned slot, 
    const std::string& name ) { 
    dump << "        Type " << i << " Slot=" << slot
	      << " Name: " << name << "\n"; 
  }

  virtual void handleSymbolTableValue(unsigned i, unsigned slot, 
    const std::string& name ) { 
    dump << "        Value " << i << " Slot=" << slot
	      << " Name: " << name << "\n";
  }

  virtual void handleSymbolTableEnd() { 
    dump << "    } END BLOCK: SymbolTable\n";
  }

  virtual void handleFunctionBegin(Function* Func, unsigned Size) {
    dump << "BLOCK: Function {\n";
    dump << "  Linkage: " << Func->getLinkage() << "\n";
    dump << "  Type: " << Func->getType()->getDescription() << "\n";
    const FunctionType* FType = 
      cast<FunctionType>(Func->getType()->getElementType());
    currFunc = &bca.FunctionInfo[Func];
    currFunc->description = FType->getDescription();
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
    dump << "} END BLOCK: Function\n";
    currFunc->density = double(currFunc->byteSize) /
      double(currFunc->numInstructions+currFunc->numBasicBlocks);

    if ( bca.progressiveVerify ) {
      try {
	verifyModule(*M, ThrowExceptionAction);
      } catch ( std::string& msg ) {
	bca.VerifyInfo += "Verify@EndFunction: " + msg + "\n";
      }
    }
  }

  virtual void handleBasicBlockBegin( unsigned blocknum) {
    dump << "  BLOCK: BasicBlock #" << blocknum << "{\n";
    bca.numBasicBlocks++;
    bca.numValues++;
    if ( currFunc ) currFunc->numBasicBlocks++;
  }

  virtual bool handleInstruction( unsigned Opcode, const Type* iType, 
				std::vector<unsigned>& Operands, unsigned Size){
    dump << "    INST: OpCode=" 
	 << Instruction::getOpcodeName(Opcode) << " Type=" 
	 << iType->getDescription() << "\n";
    for ( unsigned i = 0; i < Operands.size(); ++i ) 
      dump << "      Op#" << i << " Slot=" << Operands[i] << "\n";

    bca.numInstructions++;
    bca.numValues++;
    bca.instructionSize += Size;
    if (Size > 4 ) bca.longInstructions++;
    bca.numOperands += Operands.size();
    if ( currFunc ) {
      currFunc->numInstructions++;
      currFunc->instructionSize += Size;
      if (Size > 4 ) currFunc->longInstructions++;
      if ( Opcode == Instruction::PHI ) currFunc->numPhis++;
    }
    return Instruction::isTerminator(Opcode); 
  }

  virtual void handleBasicBlockEnd(unsigned blocknum) { 
    dump << "  } END BLOCK: BasicBlock #" << blocknum << "{\n";
  }

  virtual void handleGlobalConstantsBegin() { 
    dump << "    BLOCK: GlobalConstants {\n";
  }

  virtual void handleConstantExpression( unsigned Opcode, 
      std::vector<Constant*> ArgVec, Constant* C ) {
    dump << "      EXPR: " << Instruction::getOpcodeName(Opcode) << "\n";
    for ( unsigned i = 0; i < ArgVec.size(); ++i )  {
      dump << "        Arg#" << i << " "; ArgVec[i]->print(dump); dump << "\n";
    }
    dump << "        Value=";
    C->print(dump);
    dump << "\n";
    bca.numConstants++;
    bca.numValues++;
  }

  virtual void handleConstantValue( Constant * c ) {
    dump << "      VALUE: ";
    c->print(dump);
    dump << "\n";
    bca.numConstants++;
    bca.numValues++;
  }

  virtual void handleConstantArray( const ArrayType* AT, 
          std::vector<Constant*>& Elements,
	  unsigned TypeSlot,
	  Constant* ArrayVal ) {
    dump << "      ARRAY: " << AT->getDescription() 
         << " TypeSlot=" << TypeSlot << "\n";
    for ( unsigned i = 0; i < Elements.size(); ++i ) {
      dump << "        #" << i;
      Elements[i]->print(dump);
      dump << "\n";
    }
    dump << "        Value=";
    ArrayVal->print(dump);
    dump << "\n";

    bca.numConstants++;
    bca.numValues++;
  }

  virtual void handleConstantStruct(
        const StructType* ST,
        std::vector<Constant*>& Elements,
	Constant* StructVal)
  {
    dump << "      STRUC: " << ST->getDescription() << "\n";
    for ( unsigned i = 0; i < Elements.size(); ++i ) {
      dump << "        #" << i << " "; Elements[i]->print(dump); dump << "\n";
    }
    dump << "        Value=";
    StructVal->print(dump);
    dump << "\n";
    bca.numConstants++;
    bca.numValues++;
  }

  virtual void handleConstantPointer( const PointerType* PT, 
      unsigned Slot, GlobalValue* GV, Constant* PtrVal) {
    dump << "       PNTR: " << PT->getDescription() 
	 << " Slot=" << Slot << " GlobalValue=";
    GV->print(dump);
    dump << "\n        Value=";
    PtrVal->print(dump);
    dump << "\n";
    bca.numConstants++;
    bca.numValues++;
  }

  virtual void handleConstantString( const ConstantArray* CA ) {
    dump << "      STRNG: ";
    CA->print(dump); 
    dump << "\n";
    bca.numConstants++;
    bca.numValues++;
  }

  virtual void handleGlobalConstantsEnd() { 
    dump << "    } END BLOCK: GlobalConstants\n";
    if ( bca.progressiveVerify ) {
      try {
	verifyModule(*M, ThrowExceptionAction);
      } catch ( std::string& msg ) {
	bca.VerifyInfo += "Verify@EndGlobalConstants: " + msg + "\n";
      }
    }
  }

  virtual void handleAlignment(unsigned numBytes) {
    bca.numAlignment += numBytes;
  }

  virtual void handleBlock(
    unsigned BType, const unsigned char* StartPtr, unsigned Size) {
    bca.numBlocks++;
    bca.BlockSizes[llvm::BytecodeFormat::FileBlockIDs(BType)] += Size;
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
  print(Out, "Bytecode Analysis Of Module",     bca.ModuleId);
  print(Out, "File Size",                       bca.byteSize);
  print(Out, "Bytecode Compression Index",std::string("TBD"));
  print(Out, "Number Of Bytecode Blocks",       bca.numBlocks);
  print(Out, "Number Of Types",                 bca.numTypes);
  print(Out, "Number Of Values",                bca.numValues);
  print(Out, "Number Of Constants",             bca.numConstants);
  print(Out, "Number Of Global Variables",      bca.numGlobalVars);
  print(Out, "Number Of Functions",             bca.numFunctions);
  print(Out, "Number Of Basic Blocks",          bca.numBasicBlocks);
  print(Out, "Number Of Instructions",          bca.numInstructions);
  print(Out, "Number Of Operands",              bca.numOperands);
  print(Out, "Number Of Compaction Tables",     bca.numCmpctnTables);
  print(Out, "Number Of Symbol Tables",         bca.numSymTab);
  print(Out, "Long Instructions", bca.longInstructions);
  print(Out, "Instruction Size", bca.instructionSize);
  print(Out, "Average Instruction Size", 
    double(bca.instructionSize)/double(bca.numInstructions));
  print(Out, "Maximum Type Slot Number",        bca.maxTypeSlot);
  print(Out, "Maximum Value Slot Number",       bca.maxValueSlot);
  print(Out, "Bytes Thrown To Alignment",       double(bca.numAlignment), 
    double(bca.byteSize));
  print(Out, "File Density (bytes/def)",        bca.fileDensity);
  print(Out, "Globals Density (bytes/def)",     bca.globalsDensity);
  print(Out, "Function Density (bytes/func)",   bca.functionDensity);
  print(Out, "Number of VBR 32-bit Integers",   bca.vbrCount32);
  print(Out, "Number of VBR 64-bit Integers",   bca.vbrCount64);
  print(Out, "Number of VBR Compressed Bytes",  bca.vbrCompBytes);
  print(Out, "Number of VBR Expanded Bytes",    bca.vbrExpdBytes);
  print(Out, "VBR Savings", 
    double(bca.vbrExpdBytes)-double(bca.vbrCompBytes),
    double(bca.byteSize));

  if ( bca.detailedResults ) {
    print(Out, "Module Bytes",
        double(bca.BlockSizes[BytecodeFormat::Module]),
        double(bca.byteSize));
    print(Out, "Function Bytes", 
        double(bca.BlockSizes[BytecodeFormat::Function]),
        double(bca.byteSize));
    print(Out, "Constant Pool Bytes", 
        double(bca.BlockSizes[BytecodeFormat::ConstantPool]),
        double(bca.byteSize));
    print(Out, "Symbol Table Bytes", 
        double(bca.BlockSizes[BytecodeFormat::SymbolTable]),
        double(bca.byteSize));
    print(Out, "Module Global Info Bytes", 
        double(bca.BlockSizes[BytecodeFormat::ModuleGlobalInfo]),
        double(bca.byteSize));
    print(Out, "Global Type Plane Bytes", 
        double(bca.BlockSizes[BytecodeFormat::GlobalTypePlane]),
        double(bca.byteSize));
    print(Out, "Basic Block Bytes", 
        double(bca.BlockSizes[BytecodeFormat::BasicBlock]),
        double(bca.byteSize));
    print(Out, "Instruction List Bytes", 
        double(bca.BlockSizes[BytecodeFormat::InstructionList]),
        double(bca.byteSize));
    print(Out, "Compaction Table Bytes", 
        double(bca.BlockSizes[BytecodeFormat::CompactionTable]),
        double(bca.byteSize));

    std::map<const Function*,BytecodeAnalysis::BytecodeFunctionInfo>::iterator I = 
      bca.FunctionInfo.begin();
    std::map<const Function*,BytecodeAnalysis::BytecodeFunctionInfo>::iterator E = 
      bca.FunctionInfo.end();

    while ( I != E ) {
      Out << std::left << std::setw(0);
      Out << "Function: " << I->second.name << "\n";
      print(Out, "Type:", I->second.description);
      print(Out, "Byte Size", I->second.byteSize);
      print(Out, "Instructions", I->second.numInstructions);
      print(Out, "Long Instructions", I->second.longInstructions);
      print(Out, "Instruction Size", I->second.instructionSize);
      print(Out, "Average Instruction Size", 
        double(I->second.instructionSize)/double(I->second.numInstructions));
      print(Out, "Basic Blocks", I->second.numBasicBlocks);
      print(Out, "Operand", I->second.numOperands);
      print(Out, "Function Density", I->second.density);
      print(Out, "Number of VBR 32-bit Integers",   I->second.vbrCount32);
      print(Out, "Number of VBR 64-bit Integers",   I->second.vbrCount64);
      print(Out, "Number of VBR Compressed Bytes",  I->second.vbrCompBytes);
      print(Out, "Number of VBR Expanded Bytes",    I->second.vbrExpdBytes);
      print(Out, "VBR Savings", 
        double(I->second.vbrExpdBytes)-double(I->second.vbrCompBytes),
        double(I->second.byteSize));
      ++I;
    }
  }

  if ( bca.dumpBytecode )
    Out << bca.BytecodeDump;

  if ( bca.progressiveVerify )
    Out << bca.VerifyInfo;
}

BytecodeHandler* createBytecodeAnalyzerHandler(BytecodeAnalysis& bca)
{
  return new AnalyzerHandler(bca);
}

}

// vim: sw=2
