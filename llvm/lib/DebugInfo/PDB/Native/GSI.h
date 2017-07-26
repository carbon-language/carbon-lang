//===- GSI.h - Common Declarations for GlobalsStream and PublicsStream ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// The data structures defined in this file are based on the reference
// implementation which is available at
// https://github.com/Microsoft/microsoft-pdb/blob/master/PDB/dbi/gsi.h
//
// When you are reading the reference source code, you'd find the
// information below useful.
//
//  - ppdb1->m_fMinimalDbgInfo seems to be always true.
//  - SMALLBUCKETS macro is defined.
//
// The reference doesn't compile, so I learned just by reading code.
// It's not guaranteed to be correct.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_DEBUGINFO_PDB_RAW_GSI_H
#define LLVM_LIB_DEBUGINFO_PDB_RAW_GSI_H

#include "llvm/DebugInfo/PDB/Native/RawTypes.h"
#include "llvm/Support/BinaryStreamArray.h"

#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"

namespace llvm {

class BinaryStreamReader;

namespace pdb {

Error readGSIHashBuckets(FixedStreamArray<support::ulittle32_t> &HashBuckets,
                         ArrayRef<uint8_t> &HashBitmap,
                         const GSIHashHeader *HashHdr,
                         BinaryStreamReader &Reader);
Error readGSIHashHeader(const GSIHashHeader *&HashHdr,
                        BinaryStreamReader &Reader);
Error readGSIHashRecords(FixedStreamArray<PSHashRecord> &HashRecords,
                         const GSIHashHeader *HashHdr,
                         BinaryStreamReader &Reader);
}
}

#endif
