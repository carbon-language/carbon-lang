//===- lib/ReaderWriter/MachO/WriterMachO.cpp -----------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lld/ReaderWriter/Writer.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileOutputBuffer.h"
#include "llvm/Support/MachO.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/system_error.h"

#include "lld/Core/File.h"
#include "lld/ReaderWriter/MachOLinkingContext.h"


#include "MachONormalizedFile.h"
#include "ExecutableAtoms.hpp"

using lld::mach_o::normalized::NormalizedFile;

namespace lld {
namespace mach_o {

class MachOWriter : public Writer {
public:
  MachOWriter(const MachOLinkingContext &ctxt) : _context(ctxt) { }

  virtual error_code writeFile(const lld::File &file, StringRef path) {
    // Construct empty normalized file from atoms.
    ErrorOr<std::unique_ptr<NormalizedFile>> nFile = 
                                normalized::normalizedFromAtoms(file, _context);
    if (!nFile)
      return nFile;
    
    // For debugging, write out yaml form of normalized file.
    //writeYaml(*nFile->get(), llvm::errs());
    
    // Write normalized file as mach-o binary.
    return writeBinary(*nFile->get(), path);
  }
  
  virtual bool createImplicitFiles(std::vector<std::unique_ptr<File> > &r) {
    if (_context.outputFileType() == llvm::MachO::MH_EXECUTE) {
      // When building main executables, add _main as required entry point.
      r.emplace_back(new CRuntimeFile(_context));
    }
    return true;
  }
private:
   const MachOLinkingContext  &_context;
 };


} // namespace mach_o

std::unique_ptr<Writer> createWriterMachO(const MachOLinkingContext &context) {
  return std::unique_ptr<Writer>(new lld::mach_o::MachOWriter(context));
}

} // namespace lld
