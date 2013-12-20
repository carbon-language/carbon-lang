//===- lib/ReaderWriter/PECOFF/WriterImportLibrary.h ----------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_PE_COFF_WRITER_IMPORT_LIBRARY_H
#define LLD_READER_WRITER_PE_COFF_WRITER_IMPORT_LIBRARY_H

namespace lld {
class PECOFFLinkingContext;

namespace pecoff {

void writeImportLibrary(const PECOFFLinkingContext &ctx);

} // end namespace pecoff
} // end namespace lld

#endif
