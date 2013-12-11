//===- lib/ReaderWriter/PECOFF/ReaderImportHeader.h -----------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_PE_COFF_READER_IMPORT_HEADER_H
#define LLD_READER_WRITER_PE_COFF_READER_IMPORT_HEADER_H

#include <memory>
#include <vector>

namespace llvm {
class MemoryBuffer;
class error_code;
}

namespace lld {
class LinkingContext;
class File;

namespace coff {

error_code parseCOFFImportLibrary(const LinkingContext &context,
                                  std::unique_ptr<MemoryBuffer> &mb,
                                  std::vector<std::unique_ptr<File> > &result);
}
}

#endif
