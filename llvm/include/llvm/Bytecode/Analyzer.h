//===-- llvm/Bytecode/Analyzer.h - Analyzer for Bytecode files --*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the 
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This functionality is implemented by the lib/Bytecode/Analysis library.
// This library is used to read VM bytecode files from a file or memory buffer
// and print out a diagnostic analysis of the contents of the file. It is 
// intended for three uses: (a) understanding the bytecode format, (b) ensuring 
// correctness of bytecode format, (c) statistical analysis of generated 
// bytecode files.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_BYTECODE_ANALYZER_H
#define LLVM_BYTECODE_ANALYZER_H

#include "llvm/Bytecode/Format.h"
#include <string>
#include <map>

namespace llvm {

class Function;

/// This structure is used to contain the output of the Bytecode Analysis 
/// library. It simply contains fields to hold each item of the analysis 
/// results.
/// @brief Bytecode Analysis results structure
struct BytecodeAnalysis {
  std::string ModuleId;     ///< Identification of the module
  unsigned byteSize;        ///< The size of the bytecode file in bytes
  unsigned numTypes;        ///< The number of types
  unsigned numValues;       ///< The number of values
  unsigned numBlocks;       ///< The number of *bytecode* blocks
  unsigned numFunctions;    ///< The number of functions defined
  unsigned numConstants;    ///< The number of constants
  unsigned numGlobalVars;   ///< The number of global variables
  unsigned numInstructions; ///< The number of instructions in all functions
  unsigned numBasicBlocks;  ///< The number of BBs in all functions
  unsigned numOperands;     ///< The number of BBs in all functions
  unsigned numCmpctnTables; ///< The number of compaction tables
  unsigned numSymTab;       ///< The number of symbol tables
  unsigned numAlignment;    ///< The number of alignment bytes
  unsigned maxTypeSlot;     ///< The maximum slot number for types
  unsigned maxValueSlot;    ///< The maximum slot number for values
  double   fileDensity;     ///< Density of file (bytes/definition)
    ///< This is the density of the bytecode file. It is the ratio of
    ///< the number of bytes to the number of definitions in the file. Smaller
    ///< numbers mean the file is more compact (denser). Larger numbers mean
    ///< the file is more sparse.
  double   globalsDensity;  ///< density of global defs (bytes/definition)
  double   functionDensity; ///< Average density of functions (bytes/function)
  unsigned instructionSize; ///< Size of instructions in bytes
  unsigned longInstructions;///< Number of instructions > 4 bytes
  unsigned vbrCount32;      ///< Number of 32-bit vbr values
  unsigned vbrCount64;      ///< Number of 64-bit vbr values
  unsigned vbrCompBytes;    ///< Number of vbr bytes (compressed)
  unsigned vbrExpdBytes;    ///< Number of vbr bytes (expanded)

  typedef std::map<BytecodeFormat::FileBlockIDs,unsigned> BlockSizeMap;
  BlockSizeMap BlockSizes;

  /// A structure that contains various pieces of information related to
  /// an analysis of a single function.
  struct BytecodeFunctionInfo {
    std::string description;  ///< Function type description
    std::string name;         ///< Name of function if it has one
    unsigned byteSize;        ///< The size of the function in bytecode bytes
    unsigned numInstructions; ///< The number of instructions in the function
    unsigned numBasicBlocks;  ///< The number of basic blocks in the function
    unsigned numPhis;         ///< Number of Phi Nodes in Instructions
    unsigned numOperands;     ///< The number of operands in the function
    double   density;         ///< Density of function
    unsigned instructionSize; ///< Size of instructions in bytes
    unsigned longInstructions;///< Number of instructions > 4 bytes
    unsigned vbrCount32;      ///< Number of 32-bit vbr values
    unsigned vbrCount64;      ///< Number of 64-bit vbr values
    unsigned vbrCompBytes;    ///< Number of vbr bytes (compressed)
    unsigned vbrExpdBytes;    ///< Number of vbr bytes (expanded)
  };

  /// A mapping of function slot numbers to the collected information about 
  /// the function.
  std::map<const Function*,BytecodeFunctionInfo> FunctionInfo; 

  /// The content of the bytecode dump
  std::string BytecodeDump;

  /// Flags for what should be done
  bool dumpBytecode;     ///< If true, BytecodeDump has contents
  bool detailedResults;  ///< If true, FunctionInfo has contents 
};

/// This function is the main entry point into the bytecode analysis library. It
/// allows you to simply provide a \p filename and storage for the \p Results 
/// that will be filled in with the analysis results.
/// @brief Analyze contents of a bytecode File
void AnalyzeBytecodeFile(
      const std::string& Filename, ///< The name of the bytecode file to read
      BytecodeAnalysis& Results,   ///< The results of the analysis
      std::string* ErrorStr = 0    ///< Errors, if any.
    );

/// This function is an alternate entry point into the bytecode analysis
/// library. It allows you to provide an arbitrary memory buffer which is
/// assumed to contain a complete bytecode file. The \p Buffer is analyzed and
/// the \p Results are filled in.
/// @brief Analyze contents of a bytecode buffer.
void AnalyzeBytecodeBuffer(
       const unsigned char* Buffer, ///< Pointer to start of bytecode buffer
       unsigned BufferSize,         ///< Size of the bytecode buffer
       BytecodeAnalysis& Results,   ///< The results of the analysis
       std::string* ErrorStr = 0    ///< Errors, if any.
     );

/// This function prints the contents of rhe BytecodeAnalysis structure in
/// a human legible form.
/// @brief Print BytecodeAnalysis structure to an ostream
void PrintBytecodeAnalysis(BytecodeAnalysis& bca, std::ostream& Out );

/// @brief std::ostream inserter for BytecodeAnalysis structure
inline std::ostream& operator<<(std::ostream& Out, BytecodeAnalysis& bca ) {
    PrintBytecodeAnalysis(bca,Out);
    return Out;
}

} // End llvm namespace

#endif
