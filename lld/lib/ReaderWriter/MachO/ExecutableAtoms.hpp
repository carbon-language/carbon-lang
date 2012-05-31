//===- lib/ReaderWriter/MachO/ExecutableAtoms.hpp -------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_READER_WRITER_MACHO_EXECUTABLE_ATOM_H_
#define LLD_READER_WRITER_MACHO_EXECUTABLE_ATOM_H_


#include "lld/Core/DefinedAtom.h"
#include "lld/Core/UndefinedAtom.h"
#include "lld/Core/File.h"
#include "lld/Core/Reference.h"

#include "lld/ReaderWriter/WriterMachO.h"

#include "SimpleAtoms.hpp"

namespace lld {
namespace mach_o {


//
// CRuntimeFile adds an UndefinedAtom for "_main" so that the Resolving
// phase will fail if "_main" is undefined.
//
class CRuntimeFile : public SimpleFile {
public:
    CRuntimeFile(const WriterOptionsMachO &options) 
      : SimpleFile("C runtime"), _undefMain(*this, "_main") {
      // only main executables need _main
      if ( options.outputKind() == WriterOptionsMachO::outputDynamicExecutable)
        this->addAtom(_undefMain);
   }
        
private:
  SimpleUndefinedAtom   _undefMain;
};



} // namespace mach_o 
} // namespace lld 


#endif // LLD_READER_WRITER_MACHO_EXECUTABLE_ATOM_H_
