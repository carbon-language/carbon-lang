//===- lto-c.cpp - LLVM Link Time Optimizer C Wrappers --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chandler Carruth and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements a C wrapper API for the Link Time Optimization
// library.
//
//===----------------------------------------------------------------------===//

#include "llvm-c/LinkTimeOptimizer.h"
#include "llvm/LinkTimeOptimizer.h"
using namespace llvm;


/// Create an instance of the LLVM LTO object for performing the link
/// time optimizations.
extern "C"
llvm_lto_t llvm_create_optimizer() {
  return new llvm::LTO();
}

/// Destroy an instance of the LLVM LTO object
extern "C"
void llvm_destroy_optimizer(llvm_lto_t lto) {
  delete (llvm::LTO*)lto;
}

/// Read an LLVM bytecode file using LTO::readLLVMObjectFile.
extern "C"
llvm_lto_status
llvm_read_object_file(llvm_lto_t lto, const char *input_filename) {
  llvm::LTO *l = (llvm::LTO*)lto;

  if (input_filename == NULL)
    return LLVM_LTO_READ_FAILURE;

  std::string InputFilename(input_filename);
  llvm::LTO::NameToSymbolMap symbols;
  std::set<std::string> references;
  return (llvm_lto_status)((int)(l->readLLVMObjectFile(InputFilename, symbols,
                                                       references)));
}


/// Optimize and output object code using LTO::optimizeModules.
extern "C"
llvm_lto_status
llvm_optimize_modules(llvm_lto_t lto, const char *output_filename) {
  llvm::LTO *l = (llvm::LTO*)lto;

  std::string OutputFilename(output_filename);
  std::vector<const char *> exportList;
  std::string targetTriple;

  return (llvm_lto_status)((int)(
    l->optimizeModules(OutputFilename, exportList,
                       targetTriple, false, "")));
}



