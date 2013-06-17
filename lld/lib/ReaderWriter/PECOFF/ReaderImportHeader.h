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
class TargetInfo;
class File;

namespace coff {

llvm::error_code
    parseCOFFImportLibrary(const TargetInfo &targetInfo,
                           std::unique_ptr<llvm::MemoryBuffer> &mb,
                           std::vector<std::unique_ptr<File> > &result);
}
}

#endif
