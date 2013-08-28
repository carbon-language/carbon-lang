//===-- llvm/CodeGen/DIEHash.cpp - Dwarf Hashing Framework ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for DWARF4 hashing of DIEs.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "dwarfdebug"

#include "DIE.h"
#include "DIEHash.h"
#include "DwarfCompileUnit.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/MD5.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

/// \brief Grabs the string in whichever attribute is passed in and returns
/// a reference to it.
static StringRef getDIEStringAttr(DIE *Die, uint16_t Attr) {
  const SmallVectorImpl<DIEValue *> &Values = Die->getValues();
  const DIEAbbrev &Abbrevs = Die->getAbbrev();

  // Iterate through all the attributes until we find the one we're
  // looking for, if we can't find it return an empty string.
  for (size_t i = 0; i < Values.size(); ++i) {
    if (Abbrevs.getData()[i].getAttribute() == Attr) {
      DIEValue *V = Values[i];
      assert(isa<DIEString>(V) && "String requested. Not a string.");
      DIEString *S = cast<DIEString>(V);
      return S->getString();
    }
  }
  return StringRef("");
}

/// \brief Adds the string in \p Str to the hash. This also hashes
/// a trailing NULL with the string.
void DIEHash::addString(StringRef Str) {
  DEBUG(dbgs() << "Adding string " << Str << " to hash.\n");
  Hash.update(Str);
  Hash.update(makeArrayRef((uint8_t)'\0'));
}

// FIXME: The LEB128 routines are copied and only slightly modified out of
// LEB128.h.

/// \brief Adds the unsigned in \p Value to the hash encoded as a ULEB128.
void DIEHash::addULEB128(uint64_t Value) {
  DEBUG(dbgs() << "Adding ULEB128 " << Value << " to hash.\n");
  do {
    uint8_t Byte = Value & 0x7f;
    Value >>= 7;
    if (Value != 0)
      Byte |= 0x80; // Mark this byte to show that more bytes will follow.
    Hash.update(Byte);
  } while (Value != 0);
}

/// \brief Including \p Parent adds the context of Parent to the hash..
void DIEHash::addParentContext(DIE *Parent) {

  DEBUG(dbgs() << "Adding parent context to hash...\n");

  // [7.27.2] For each surrounding type or namespace beginning with the
  // outermost such construct...
  SmallVector<DIE *, 1> Parents;
  while (Parent->getTag() != dwarf::DW_TAG_compile_unit) {
    Parents.push_back(Parent);
    Parent = Parent->getParent();
  }

  // Reverse iterate over our list to go from the outermost construct to the
  // innermost.
  for (SmallVectorImpl<DIE *>::reverse_iterator I = Parents.rbegin(),
                                                E = Parents.rend();
       I != E; ++I) {
    DIE *Die = *I;

    // ... Append the letter "C" to the sequence...
    addULEB128('C');

    // ... Followed by the DWARF tag of the construct...
    addULEB128(Die->getTag());

    // ... Then the name, taken from the DW_AT_name attribute.
    StringRef Name = getDIEStringAttr(Die, dwarf::DW_AT_name);
    DEBUG(dbgs() << "... adding context: " << Name << "\n");
    if (!Name.empty())
      addString(Name);
  }
}

// Collect all of the attributes for a particular DIE in single structure.
void DIEHash::collectAttributes(DIE *Die, DIEAttrs &Attrs) {
  const SmallVectorImpl<DIEValue *> &Values = Die->getValues();
  const DIEAbbrev &Abbrevs = Die->getAbbrev();

#define COLLECT_ATTR(NAME)                                                     \
  Attrs.NAME.Val = Values[i];                                                  \
  Attrs.NAME.Desc = &Abbrevs.getData()[i];

  for (size_t i = 0, e = Values.size(); i != e; ++i) {
    DEBUG(dbgs() << "Attribute: "
                 << dwarf::AttributeString(Abbrevs.getData()[i].getAttribute())
                 << " added.\n");
    switch (Abbrevs.getData()[i].getAttribute()) {
    case dwarf::DW_AT_name:
      COLLECT_ATTR(DW_AT_name);
      break;
    case dwarf::DW_AT_language:
      COLLECT_ATTR(DW_AT_language);
      break;
    default:
      break;
    }
  }
}

// Hash an individual attribute \param Attr based on the type of attribute and
// the form.
void DIEHash::hashAttribute(AttrEntry Attr) {
  const DIEValue *Value = Attr.Val;
  const DIEAbbrevData *Desc = Attr.Desc;

  // TODO: Add support for types.

  // Add the letter A to the hash.
  addULEB128('A');

  // Then the attribute code and form.
  addULEB128(Desc->getAttribute());
  addULEB128(Desc->getForm());

  // TODO: Add support for additional forms.
  switch (Desc->getForm()) {
  // TODO: We'll want to add DW_FORM_string here if we start emitting them again.
  case dwarf::DW_FORM_strp:
    addString(cast<DIEString>(Value)->getString());
    break;
  case dwarf::DW_FORM_data1:
  case dwarf::DW_FORM_data2:
  case dwarf::DW_FORM_data4:
  case dwarf::DW_FORM_data8:
  case dwarf::DW_FORM_udata:
    addULEB128(cast<DIEInteger>(Value)->getValue());
    break;
  }
}

// Go through the attributes from \param Attrs in the order specified in 7.27.4
// and hash them.
void DIEHash::hashAttributes(const DIEAttrs &Attrs) {
#define ADD_ATTR(ATTR)                                                         \
  {                                                                            \
    if (ATTR.Val != 0)                                                         \
      hashAttribute(ATTR);                                                     \
  }

  // FIXME: Add the rest.
  ADD_ATTR(Attrs.DW_AT_name);
  ADD_ATTR(Attrs.DW_AT_language);
}

// Add all of the attributes for \param Die to the hash.
void DIEHash::addAttributes(DIE *Die) {
  DIEAttrs Attrs;
  memset(&Attrs, 0, sizeof(Attrs));
  collectAttributes(Die, Attrs);
  hashAttributes(Attrs);
}

// Compute the hash of a DIE. This is based on the type signature computation
// given in section 7.27 of the DWARF4 standard. It is the md5 hash of a
// flattened description of the DIE.
void DIEHash::computeHash(DIE *Die) {

  // Append the letter 'D', followed by the DWARF tag of the DIE.
  addULEB128('D');
  addULEB128(Die->getTag());

  // Add each of the attributes of the DIE.
  addAttributes(Die);

  // Then hash each of the children of the DIE.
  for (std::vector<DIE *>::const_iterator I = Die->getChildren().begin(),
                                          E = Die->getChildren().end();
       I != E; ++I)
    computeHash(*I);
}

/// This is based on the type signature computation given in section 7.27 of the
/// DWARF4 standard. It is the md5 hash of a flattened description of the DIE
/// with the exception that we are hashing only the context and the name of the
/// type.
uint64_t DIEHash::computeDIEODRSignature(DIE *Die) {

  // Add the contexts to the hash. We won't be computing the ODR hash for
  // function local types so it's safe to use the generic context hashing
  // algorithm here.
  // FIXME: If we figure out how to account for linkage in some way we could
  // actually do this with a slight modification to the parent hash algorithm.
  DIE *Parent = Die->getParent();
  if (Parent)
    addParentContext(Parent);

  // Add the current DIE information.

  // Add the DWARF tag of the DIE.
  addULEB128(Die->getTag());

  // Add the name of the type to the hash.
  addString(getDIEStringAttr(Die, dwarf::DW_AT_name));

  // Now get the result.
  MD5::MD5Result Result;
  Hash.final(Result);

  // ... take the least significant 8 bytes and return those. Our MD5
  // implementation always returns its results in little endian, swap bytes
  // appropriately.
  return *reinterpret_cast<support::ulittle64_t *>(Result + 8);
}

/// This is based on the type signature computation given in section 7.27 of the
/// DWARF4 standard. It is an md5 hash of the flattened description of the DIE
/// with the inclusion of the full CU and all top level CU entities.
uint64_t DIEHash::computeCUSignature(DIE *Die) {

  // Hash the DIE.
  computeHash(Die);

  // Now return the result.
  MD5::MD5Result Result;
  Hash.final(Result);

  // ... take the least significant 8 bytes and return those. Our MD5
  // implementation always returns its results in little endian, swap bytes
  // appropriately.
  return *reinterpret_cast<support::ulittle64_t *>(Result + 8);
}
