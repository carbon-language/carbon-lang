//===- Platforms/Darwin/x86_64StubAtom.hpp --------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_PLATFORM_DARWIN_EXECUTABLE_ATOM_H_
#define LLD_PLATFORM_DARWIN_EXECUTABLE_ATOM_H_


#include "lld/Core/DefinedAtom.h"
#include "lld/Core/UndefinedAtom.h"
#include "lld/Core/File.h"
#include "lld/Core/Reference.h"

#include "DarwinReferenceKinds.h"
#include "SimpleAtoms.hpp"

namespace lld {
namespace darwin {


//
// EntryPointReferenceAtom is used to:
//  1) Require "_main" is defined.
//  2) Give Darwin Platform a pointer to the atom named "_main"
//
class EntryPointReferenceAtom : public SimpleDefinedAtom {
public:
        EntryPointReferenceAtom(const File &file) 
                       : SimpleDefinedAtom(file)
                       , _mainUndefAtom(file, "_main") {
          this->addReference(ReferenceKind::none, 0, &_mainUndefAtom, 0);
        }

  virtual ContentType contentType() const  {
    return DefinedAtom::typeCode;
  }

  virtual uint64_t size() const {
    return 0;
  }

  virtual ContentPermissions permissions() const  {
    return DefinedAtom::permR_X;
  }
  
  virtual ArrayRef<uint8_t> rawContent() const {
    return ArrayRef<uint8_t>();
  }
private:
  friend class CRuntimeFile;
  SimpleUndefinedAtom   _mainUndefAtom;
};


class CRuntimeFile : public SimpleFile {
public:
    CRuntimeFile() 
          : SimpleFile("C runtime")
          , _atom(*this) {
      this->addAtom(_atom);
      this->addAtom(_atom._mainUndefAtom);
   }
    
    const Atom *mainAtom() {
      const Reference *ref = *(_atom.begin());
      return ref->target();
    }
    
private:
  EntryPointReferenceAtom   _atom;
};



} // namespace darwin 
} // namespace lld 


#endif // LLD_PLATFORM_DARWIN_EXECUTABLE_ATOM_H_
