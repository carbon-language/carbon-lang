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
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Target/TargetOptions.h"
#include <string>
#include <vector>

namespace llvm {
  class LLVMContext;
  class DiagnosticInfo;
  class GlobalValue;
  class Mangler;
  class MemoryBuffer;
  class TargetLibraryInfo;
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

  void setTargetOptions(llvm::TargetOptions options);
  void setDebugInfo(lto_debug_model);
  void setCodePICModel(lto_codegen_model);

  void setCpu(const char *mCpu) { MCpu = mCpu; }
  void setAttr(const char *mAttr) { MAttr = mAttr; }

  void addMustPreserveSymbol(const char *sym) { MustPreserveSymbols[sym] = 1; }

  // To pass options to the driver and optimization passes. These options are
  // not necessarily for debugging purpose (The function name is misleading).
  // This function should be called before LTOCodeGenerator::compilexxx(),
  // and LTOCodeGenerator::writeMergedModules().
  void setCodeGenDebugOptions(const char *opts);

  // Parse the options set in setCodeGenDebugOptions. Like
  // setCodeGenDebugOptions, this must be called before
  // LTOCodeGenerator::compilexxx() and LTOCodeGenerator::writeMergedModules()
  void parseCodeGenDebugOptions();

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
  bool compile_to_file(const char **name,
                       bool disableOpt,
                       bool disableInline,
                       bool disableGVNLoadPRE,
                       std::string &errMsg);

  // As with compile_to_file(), this function compiles the merged module into
  // single object file. Instead of returning the object-file-path to the caller
  // (linker), it brings the object to a buffer, and return the buffer to the
  // caller. This function should delete intermediate object file once its content
  // is brought to memory. Return NULL if the compilation was not successful.
  const void *compile(size_t *length,
                      bool disableOpt,
                      bool disableInline,
                      bool disableGVNLoadPRE,
                      std::string &errMsg);

  void setDiagnosticHandler(lto_diagnostic_handler_t, void *);

private:
  void initializeLTOPasses();

  bool generateObjectFile(llvm::raw_ostream &out,
                          bool disableOpt,
                          bool disableInline,
                          bool disableGVNLoadPRE,
                          std::string &errMsg);
  void applyScopeRestrictions();
  void applyRestriction(llvm::GlobalValue &GV,
                        const llvm::ArrayRef<llvm::StringRef> &Libcalls,
                        std::vector<const char*> &MustPreserveList,
                        llvm::SmallPtrSet<llvm::GlobalValue*, 8> &AsmUsed,
                        llvm::Mangler &Mangler);
  bool determineTarget(std::string &errMsg);

  static void DiagnosticHandler(const llvm::DiagnosticInfo &DI, void *Context);

  void DiagnosticHandler2(const llvm::DiagnosticInfo &DI);

  typedef llvm::StringMap<uint8_t> StringSet;

  llvm::LLVMContext &Context;
  llvm::Linker Linker;
  llvm::TargetMachine *TargetMach;
  bool EmitDwarfDebugInfo;
  bool ScopeRestrictionsDone;
  lto_codegen_model CodeModel;
  StringSet MustPreserveSymbols;
  StringSet AsmUndefinedRefs;
  llvm::MemoryBuffer *NativeObjectFile;
  std::vector<char *> CodegenOptions;
  std::string MCpu;
  std::string MAttr;
  std::string NativeObjectPath;
  llvm::TargetOptions Options;
  lto_diagnostic_handler_t DiagHandler;
  void *DiagContext;
};

#endif // LTO_CODE_GENERATOR_H
