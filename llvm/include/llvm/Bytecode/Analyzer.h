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

#include <string>
#include <map>

namespace llvm {

/// This structure is used to contain the output of the Bytecode Analysis 
/// library. It simply contains fields to hold each item of the analysis 
/// results.
/// @brief Bytecode Analysis results structure
struct BytecodeAnalysis {
  unsigned byteSize;            ///< The size of the bytecode file in bytes
  unsigned numTypes;        ///< The number of types
  unsigned numValues;       ///< The number of values
  unsigned numFunctions;    ///< The number of functions defined
  unsigned numConstants;    ///< The number of constants
  unsigned numGlobalVars;   ///< The number of global variables
  unsigned numInstructions; ///< The number of instructions in all functions
  unsigned numBasicBlocks;  ///< The number of BBs in all functions
  unsigned numOperands;     ///< The number of BBs in all functions
  unsigned maxTypeSlot;     ///< The maximum slot number for types
  unsigned maxValueSlot;    ///< The maximum slot number for values
  double   density;         ///< Density of file (bytes/defs) 

  /// A structure that contains various pieces of information related to
  /// an analysis of a single function.
  struct BytecodeFunctionInfo {
    unsigned byteSize;        ///< The size of the function in bytecode bytes
    unsigned numInstructions; ///< The number of instructions in the function
    unsigned numBasicBlocks;  ///< The number of basic blocks in the function
    unsigned numOperands;     ///< The number of operands in the function
    double density;           ///< Density of function
    double vbrEffectiveness;  ///< Effectiveness of variable bit rate encoding.
    ///< This is the average number of bytes per unsigned value written in the
    ///< vbr encoding. A "perfect" score of 1.0 means all vbr values were 
    ///< encoded in one byte. A score between 1.0 and 4.0 means that some
    ///< savings were achieved. A score of 4.0 means vbr didn't help. A score
    ///< greater than 4.0 means vbr negatively impacted size of the file.
  };

  /// A mapping of function names to the collected information about the 
  /// function.
  std::map<std::string,BytecodeFunctionInfo> FunctionInfo; 

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
