//===-LTOCodeGenerator.h - LLVM Link Time Optimizer -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the LTOCodeGenerator class.
//
//   LTO compilation consists of three phases: Pre-IPO, IPO and Post-IPO. 
//
//   The Pre-IPO phase compiles source code into bitcode file. The resulting
// bitcode files, along with object files and libraries, will be fed to the
// linker to through the IPO and Post-IPO phases. By using obj-file extension,
// the resulting bitcode file disguises itself as an object file, and therefore
// obviates the need of writing a special set of the make-rules only for LTO
// compilation.
//
//   The IPO phase perform inter-procedural analyses and optimizations, and
// the Post-IPO consists two sub-phases: intra-procedural scalar optimizations
// (SOPT), and intra-procedural target-dependent code generator (CG).
// 
//   As of this writing, we don't separate IPO and the Post-IPO SOPT. They
// are intermingled together, and are driven by a single pass manager (see
// PassManagerBuilder::populateLTOPassManager()).
// 
//   The "LTOCodeGenerator" is the driver for the IPO and Post-IPO stages. 
// The "CodeGenerator" here is bit confusing. Don't confuse the "CodeGenerator"
// with the machine specific code generator.
//
//===----------------------------------------------------------------------===//

#ifndef LTO_CODE_GENERATOR_H
#define LTO_CODE_GENERATOR_H

#include "llvm-c/lto.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Linker.h"
#include <string>
#include <vector>

namespace llvm {
  class LLVMContext;
  class GlobalValue;
  class Mangler;
  class MemoryBuffer;
  class TargetMachine;
  class raw_ostream;
}

//===----------------------------------------------------------------------===//
/// LTOCodeGenerator - C++ class which implements the opaque lto_code_gen_t
/// type.
///
struct LTOCodeGenerator {
  static const char *getVersionString();

  LTOCodeGenerator();
  ~LTOCodeGenerator();

  // Merge given module, return true on success.
  bool addModule(struct LTOModule*, std::string &errMsg);

  void setDebugInfo(lto_debug_model);
  void setCodePICModel(lto_codegen_model);

  void setCpu(const char* mCpu) { _mCpu = mCpu; }

  void addMustPreserveSymbol(const char* sym) {
    _mustPreserveSymbols[sym] = 1;
  }

  // To pass options to the driver and optimization passes. These options are
  // not necessarily for debugging purpose (The function name is misleading).
  // This function should be called before LTOCodeGenerator::compilexxx(),
  // and LTOCodeGenerator::writeMergedModules().
  //
  void setCodeGenDebugOptions(const char *opts);

  // Write the merged module to the file specified by the given path.
  // Return true on success.
  bool writeMergedModules(const char *path, std::string &errMsg);

  // Compile the merged module into a *single* object file; the path to object
  // file is returned to the caller via argument "name". Return true on
  // success.
  //
  // NOTE that it is up to the linker to remove the intermediate object file.
  //  Do not try to remove the object file in LTOCodeGenerator's destructor
  //  as we don't who (LTOCodeGenerator or the obj file) will last longer.
  // 
  bool compile_to_file(const char **name, std::string &errMsg);

  // As with compile_to_file(), this function compiles the merged module into
  // single object file. Instead of returning the object-file-path to the caller
  // (linker), it brings the object to a buffer, and return the buffer to the
  // caller. This function should delete intermediate object file once its content
  // is brought to memory. Return NULL is the compilation was not successful. 
  //
  const void *compile(size_t *length, std::string &errMsg);

private:
  void initializeLTOPasses();

  bool generateObjectFile(llvm::raw_ostream &out, std::string &errMsg);
  void applyScopeRestrictions();
  void applyRestriction(llvm::GlobalValue &GV,
                        std::vector<const char*> &mustPreserveList,
                        llvm::SmallPtrSet<llvm::GlobalValue*, 8> &asmUsed,
                        llvm::Mangler &mangler);
  bool determineTarget(std::string &errMsg);

  typedef llvm::StringMap<uint8_t> StringSet;

  llvm::LLVMContext&          _context;
  llvm::Linker                _linker;
  llvm::TargetMachine*        _target;
  bool                        _emitDwarfDebugInfo;
  bool                        _scopeRestrictionsDone;
  lto_codegen_model           _codeModel;
  StringSet                   _mustPreserveSymbols;
  StringSet                   _asmUndefinedRefs;
  llvm::MemoryBuffer*         _nativeObjectFile;
  std::vector<char*>          _codegenOptions;
  std::string                 _mCpu;
  std::string                 _nativeObjectPath;
};

#endif // LTO_CODE_GENERATOR_H
