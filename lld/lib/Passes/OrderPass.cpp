//===- Passes/OrderPass.cpp - sort atoms ----------------------------------===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This linker pass sorts atoms.  By default atoms are sort first by the
// order their .o files were found on the command line, then by order of the
// atoms (address) in the .o file.  But some atoms have a prefered location
// in their section (such as pinned to the start or end of the section), so
// the sort must take that into account too.
//
//===----------------------------------------------------------------------===//


#include "lld/Core/DefinedAtom.h"
#include "lld/Core/File.h"
#include "lld/Core/LLVM.h"
#include "lld/Core/Pass.h"
#include "lld/Core/Reference.h"
#include "llvm/ADT/DenseMap.h"


namespace lld {

static bool compare(const DefinedAtom *left, const DefinedAtom *right) {
  if ( left == right )
    return false;

  // Sort same permissions together.
  DefinedAtom::ContentPermissions leftPerms  = left->permissions();
  DefinedAtom::ContentPermissions rightPerms = right->permissions();
  if (leftPerms != rightPerms) 
		return leftPerms < rightPerms;

  
  // Sort same content types together.
  DefinedAtom::ContentType leftType  = left->contentType();
  DefinedAtom::ContentType rightType = right->contentType();
  if (leftType != rightType) 
		return leftType < rightType;


  // TO DO: Sort atoms in customs sections together.


  // Sort by section position preference.
  DefinedAtom::SectionPosition leftPos  = left->sectionPosition();
  DefinedAtom::SectionPosition rightPos = right->sectionPosition();
  bool leftSpecialPos  = (leftPos  != DefinedAtom::sectionPositionAny); 
  bool rightSpecialPos = (rightPos != DefinedAtom::sectionPositionAny); 
  if (leftSpecialPos || rightSpecialPos) {
    if (leftPos != rightPos)
      return leftPos < rightPos;
  }
  
  // Sort by .o order.
  const File *leftFile  = &left->file();
  const File *rightFile = &right->file();
  if ( leftFile != rightFile ) 
		return leftFile->ordinal() < rightFile->ordinal();
  
  // Sort by atom order with .o file.
  uint64_t leftOrdinal  = left->ordinal();
  uint64_t rightOrdinal = right->ordinal();
  if ( leftOrdinal != rightOrdinal ) 
		return leftOrdinal < rightOrdinal;
 
  return false;
}



void OrderPass::perform(MutableFile &mergedFile) {

  MutableFile::DefinedAtomRange atomRange = mergedFile.definedAtoms();
  
  std::sort(atomRange.begin(), atomRange.end(), compare);
}

} // namespace
