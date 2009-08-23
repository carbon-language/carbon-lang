//===--- lib/CodeGen/DwarfLabel.h - Dwarf Label -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// DWARF Labels.
// 
//===----------------------------------------------------------------------===//

#ifndef CODEGEN_ASMPRINTER_DWARFLABEL_H__
#define CODEGEN_ASMPRINTER_DWARFLABEL_H__

namespace llvm {
  class FoldingSetNodeID;
  class raw_ostream;

  //===--------------------------------------------------------------------===//
  /// DWLabel - Labels are used to track locations in the assembler file.
  /// Labels appear in the form @verbatim <prefix><Tag><Number> @endverbatim,
  /// where the tag is a category of label (Ex. location) and number is a value
  /// unique in that category.
  class DWLabel {
    /// Tag - Label category tag. Should always be a statically declared C
    /// string.
    /// 
    const char *Tag;

    /// Number - Value to make label unique.
    /// 
    unsigned Number;
  public:
    DWLabel(const char *T, unsigned N) : Tag(T), Number(N) {}

    // Accessors.
    const char *getTag() const { return Tag; }
    unsigned getNumber() const { return Number; }

    /// Profile - Used to gather unique data for the folding set.
    ///
    void Profile(FoldingSetNodeID &ID) const;

#ifndef NDEBUG
    void print(raw_ostream &O) const;
#endif
  };
} // end llvm namespace

#endif
