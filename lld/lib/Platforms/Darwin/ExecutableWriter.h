//===- Platforms/Darwin/ExecutableWriter.h --------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#include "lld/Core/LLVM.h"


#ifndef LLD_PLATFORM_DARWIN_EXECUTABLE_WRITER_H_
#define LLD_PLATFORM_DARWIN_EXECUTABLE_WRITER_H_

namespace lld {

class File;

namespace darwin {

class DarwinPlatform;

void writeExecutable(const lld::File &file, DarwinPlatform &platform, 
                                            raw_ostream &out);


} // namespace darwin
} // namespace lld



#endif // LLD_PLATFORM_DARWIN_EXECUTABLE_WRITER_H_

