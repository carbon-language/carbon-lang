//===- Platform/PlatformDarwin.h - Darwin Platform Implementation ---------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_PLATFORM_PLATFORM_H_
#define LLD_PLATFORM_PLATFORM_H_

#include "lld/Platform/Platform.h"

namespace lld {

class PlatformDarwin : public Platform {
  virtual void initialize();

  // keep track of: ObjC GC-ness, if any .o file cannot be scattered,
  // cpu-sub-type
  virtual void fileAdded(const File &file);

  virtual bool deadCodeStripping();
};

} // namespace lld

#endif // LLD_PLATFORM_PLATFORM_H_
