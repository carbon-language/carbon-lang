//===--- lib/CodeGen/DIE.h - DWARF Info Entries -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Data structures for DWARF info entries.
// 
//===----------------------------------------------------------------------===//

#ifndef CODEGEN_ASMPRINTER_DIE_H__
#define CODEGEN_ASMPRINTER_DIE_H__

#include "DwarfLabel.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Dwarf.h"
#include <vector>

namespace llvm {
  class AsmPrinter;
  class Dwarf;
  class TargetData;

  //===--------------------------------------------------------------------===//
  /// DIEAbbrevData - Dwarf abbreviation data, describes the one attribute of a
  /// Dwarf abbreviation.
  class VISIBILITY_HIDDEN DIEAbbrevData {
    /// Attribute - Dwarf attribute code.
    ///
    unsigned Attribute;

    /// Form - Dwarf form code.
    ///
    unsigned Form;
  public:
    DIEAbbrevData(unsigned A, unsigned F) : Attribute(A), Form(F) {}

    // Accessors.
    unsigned getAttribute() const { return Attribute; }
    unsigned getForm()      const { return Form; }

    /// Profile - Used to gather unique data for the abbreviation folding set.
    ///
    void Profile(FoldingSetNodeID &ID) const;
  };

  //===--------------------------------------------------------------------===//
  /// DIEAbbrev - Dwarf abbreviation, describes the organization of a debug
  /// information object.
  class VISIBILITY_HIDDEN DIEAbbrev : public FoldingSetNode {
    /// Tag - Dwarf tag code.
    ///
    unsigned Tag;

    /// Unique number for node.
    ///
    unsigned Number;

    /// ChildrenFlag - Dwarf children flag.
    ///
    unsigned ChildrenFlag;

    /// Data - Raw data bytes for abbreviation.
    ///
    SmallVector<DIEAbbrevData, 8> Data;
  public:
    DIEAbbrev(unsigned T, unsigned C) : Tag(T), ChildrenFlag(C), Data() {}
    virtual ~DIEAbbrev() {}

    // Accessors.
    unsigned getTag() const { return Tag; }
    unsigned getNumber() const { return Number; }
    unsigned getChildrenFlag() const { return ChildrenFlag; }
    const SmallVector<DIEAbbrevData, 8> &getData() const { return Data; }
    void setTag(unsigned T) { Tag = T; }
    void setChildrenFlag(unsigned CF) { ChildrenFlag = CF; }
    void setNumber(unsigned N) { Number = N; }

    /// AddAttribute - Adds another set of attribute information to the
    /// abbreviation.
    void AddAttribute(unsigned Attribute, unsigned Form) {
      Data.push_back(DIEAbbrevData(Attribute, Form));
    }

    /// AddFirstAttribute - Adds a set of attribute information to the front
    /// of the abbreviation.
    void AddFirstAttribute(unsigned Attribute, unsigned Form) {
      Data.insert(Data.begin(), DIEAbbrevData(Attribute, Form));
    }

    /// Profile - Used to gather unique data for the abbreviation folding set.
    ///
    void Profile(FoldingSetNodeID &ID) const;

    /// Emit - Print the abbreviation using the specified asm printer.
    ///
    void Emit(const AsmPrinter *Asm) const;

#ifndef NDEBUG
    void print(raw_ostream &O);
    void dump();
#endif
  };

  //===--------------------------------------------------------------------===//
  /// DIE - A structured debug information entry.  Has an abbreviation which
  /// describes it's organization.
  class CompileUnit;
  class DIEValue;

  class VISIBILITY_HIDDEN DIE : public FoldingSetNode {
  protected:
    /// Abbrev - Buffer for constructing abbreviation.
    ///
    DIEAbbrev Abbrev;

    /// Offset - Offset in debug info section.
    ///
    unsigned Offset;

    /// Size - Size of instance + children.
    ///
    unsigned Size;

    /// Children DIEs.
    ///
    std::vector<DIE *> Children;

    /// Attributes values.
    ///
    SmallVector<DIEValue*, 32> Values;

    /// Abstract compile unit.
    CompileUnit *AbstractCU;
    
    // Private data for print()
    mutable unsigned IndentCount;
  public:
    explicit DIE(unsigned Tag)
      : Abbrev(Tag, dwarf::DW_CHILDREN_no), Offset(0),
        Size(0), IndentCount(0) {}
    virtual ~DIE();

    // Accessors.
    DIEAbbrev &getAbbrev() { return Abbrev; }
    unsigned getAbbrevNumber() const { return Abbrev.getNumber(); }
    unsigned getTag() const { return Abbrev.getTag(); }
    unsigned getOffset() const { return Offset; }
    unsigned getSize() const { return Size; }
    const std::vector<DIE *> &getChildren() const { return Children; }
    SmallVector<DIEValue*, 32> &getValues() { return Values; }
    CompileUnit *getAbstractCompileUnit() const { return AbstractCU; }

    void setTag(unsigned Tag) { Abbrev.setTag(Tag); }
    void setOffset(unsigned O) { Offset = O; }
    void setSize(unsigned S) { Size = S; }
    void setAbstractCompileUnit(CompileUnit *CU) { AbstractCU = CU; }

    /// AddValue - Add a value and attributes to a DIE.
    ///
    void AddValue(unsigned Attribute, unsigned Form, DIEValue *Value) {
      Abbrev.AddAttribute(Attribute, Form);
      Values.push_back(Value);
    }

    /// SiblingOffset - Return the offset of the debug information entry's
    /// sibling.
    unsigned SiblingOffset() const { return Offset + Size; }

    /// AddSiblingOffset - Add a sibling offset field to the front of the DIE.
    ///
    void AddSiblingOffset();

    /// AddChild - Add a child to the DIE.
    ///
    void AddChild(DIE *Child) {
      Abbrev.setChildrenFlag(dwarf::DW_CHILDREN_yes);
      Children.push_back(Child);
    }

    /// Detach - Detaches objects connected to it after copying.
    ///
    void Detach() {
      Children.clear();
    }

    /// Profile - Used to gather unique data for the value folding set.
    ///
    void Profile(FoldingSetNodeID &ID) ;

#ifndef NDEBUG
    void print(raw_ostream &O, unsigned IncIndent = 0);
    void dump();
#endif
  };

  //===--------------------------------------------------------------------===//
  /// DIEValue - A debug information entry value.
  ///
  class VISIBILITY_HIDDEN DIEValue : public FoldingSetNode {
  public:
    enum {
      isInteger,
      isString,
      isLabel,
      isAsIsLabel,
      isSectionOffset,
      isDelta,
      isEntry,
      isBlock
    };
  protected:
    /// Type - Type of data stored in the value.
    ///
    unsigned Type;
  public:
    explicit DIEValue(unsigned T) : Type(T) {}
    virtual ~DIEValue() {}

    // Accessors
    unsigned getType()  const { return Type; }

    /// EmitValue - Emit value via the Dwarf writer.
    ///
    virtual void EmitValue(Dwarf *D, unsigned Form) const = 0;

    /// SizeOf - Return the size of a value in bytes.
    ///
    virtual unsigned SizeOf(const TargetData *TD, unsigned Form) const = 0;

    /// Profile - Used to gather unique data for the value folding set.
    ///
    virtual void Profile(FoldingSetNodeID &ID) = 0;

    // Implement isa/cast/dyncast.
    static bool classof(const DIEValue *) { return true; }

#ifndef NDEBUG
    virtual void print(raw_ostream &O) = 0;
    void dump();
#endif
  };

  //===--------------------------------------------------------------------===//
  /// DIEInteger - An integer value DIE.
  ///
  class VISIBILITY_HIDDEN DIEInteger : public DIEValue {
    uint64_t Integer;
  public:
    explicit DIEInteger(uint64_t I) : DIEValue(isInteger), Integer(I) {}

    /// BestForm - Choose the best form for integer.
    ///
    static unsigned BestForm(bool IsSigned, uint64_t Int) {
      if (IsSigned) {
        if ((char)Int == (signed)Int)   return dwarf::DW_FORM_data1;
        if ((short)Int == (signed)Int)  return dwarf::DW_FORM_data2;
        if ((int)Int == (signed)Int)    return dwarf::DW_FORM_data4;
      } else {
        if ((unsigned char)Int == Int)  return dwarf::DW_FORM_data1;
        if ((unsigned short)Int == Int) return dwarf::DW_FORM_data2;
        if ((unsigned int)Int == Int)   return dwarf::DW_FORM_data4;
      }
      return dwarf::DW_FORM_data8;
    }

    /// EmitValue - Emit integer of appropriate size.
    ///
    virtual void EmitValue(Dwarf *D, unsigned Form) const;

    /// SizeOf - Determine size of integer value in bytes.
    ///
    virtual unsigned SizeOf(const TargetData *TD, unsigned Form) const;

    /// Profile - Used to gather unique data for the value folding set.
    ///
    static void Profile(FoldingSetNodeID &ID, unsigned Int);
    virtual void Profile(FoldingSetNodeID &ID);

    // Implement isa/cast/dyncast.
    static bool classof(const DIEInteger *) { return true; }
    static bool classof(const DIEValue *I) { return I->getType() == isInteger; }

#ifndef NDEBUG
    virtual void print(raw_ostream &O);
#endif
  };

  //===--------------------------------------------------------------------===//
  /// DIEString - A string value DIE.
  ///
  class VISIBILITY_HIDDEN DIEString : public DIEValue {
    const std::string Str;
  public:
    explicit DIEString(const std::string &S) : DIEValue(isString), Str(S) {}

    /// EmitValue - Emit string value.
    ///
    virtual void EmitValue(Dwarf *D, unsigned Form) const;

    /// SizeOf - Determine size of string value in bytes.
    ///
    virtual unsigned SizeOf(const TargetData *, unsigned /*Form*/) const {
      return Str.size() + sizeof(char); // sizeof('\0');
    }

    /// Profile - Used to gather unique data for the value folding set.
    ///
    static void Profile(FoldingSetNodeID &ID, const std::string &Str);
    virtual void Profile(FoldingSetNodeID &ID);

    // Implement isa/cast/dyncast.
    static bool classof(const DIEString *) { return true; }
    static bool classof(const DIEValue *S) { return S->getType() == isString; }

#ifndef NDEBUG
    virtual void print(raw_ostream &O);
#endif
  };

  //===--------------------------------------------------------------------===//
  /// DIEDwarfLabel - A Dwarf internal label expression DIE.
  //
  class VISIBILITY_HIDDEN DIEDwarfLabel : public DIEValue {
    const DWLabel Label;
  public:
    explicit DIEDwarfLabel(const DWLabel &L) : DIEValue(isLabel), Label(L) {}

    /// EmitValue - Emit label value.
    ///
    virtual void EmitValue(Dwarf *D, unsigned Form) const;

    /// SizeOf - Determine size of label value in bytes.
    ///
    virtual unsigned SizeOf(const TargetData *TD, unsigned Form) const;

    /// Profile - Used to gather unique data for the value folding set.
    ///
    static void Profile(FoldingSetNodeID &ID, const DWLabel &Label);
    virtual void Profile(FoldingSetNodeID &ID);

    // Implement isa/cast/dyncast.
    static bool classof(const DIEDwarfLabel *)  { return true; }
    static bool classof(const DIEValue *L) { return L->getType() == isLabel; }

#ifndef NDEBUG
    virtual void print(raw_ostream &O);
#endif
  };

  //===--------------------------------------------------------------------===//
  /// DIEObjectLabel - A label to an object in code or data.
  //
  class VISIBILITY_HIDDEN DIEObjectLabel : public DIEValue {
    const std::string Label;
  public:
    explicit DIEObjectLabel(const std::string &L)
      : DIEValue(isAsIsLabel), Label(L) {}

    /// EmitValue - Emit label value.
    ///
    virtual void EmitValue(Dwarf *D, unsigned Form) const;

    /// SizeOf - Determine size of label value in bytes.
    ///
    virtual unsigned SizeOf(const TargetData *TD, unsigned Form) const;

    /// Profile - Used to gather unique data for the value folding set.
    ///
    static void Profile(FoldingSetNodeID &ID, const std::string &Label);
    virtual void Profile(FoldingSetNodeID &ID);

    // Implement isa/cast/dyncast.
    static bool classof(const DIEObjectLabel *) { return true; }
    static bool classof(const DIEValue *L) {
      return L->getType() == isAsIsLabel;
    }

#ifndef NDEBUG
    virtual void print(raw_ostream &O);
#endif
  };

  //===--------------------------------------------------------------------===//
  /// DIESectionOffset - A section offset DIE.
  ///
  class VISIBILITY_HIDDEN DIESectionOffset : public DIEValue {
    const DWLabel Label;
    const DWLabel Section;
    bool IsEH : 1;
    bool UseSet : 1;
  public:
    DIESectionOffset(const DWLabel &Lab, const DWLabel &Sec,
                     bool isEH = false, bool useSet = true)
      : DIEValue(isSectionOffset), Label(Lab), Section(Sec),
        IsEH(isEH), UseSet(useSet) {}

    /// EmitValue - Emit section offset.
    ///
    virtual void EmitValue(Dwarf *D, unsigned Form) const;

    /// SizeOf - Determine size of section offset value in bytes.
    ///
    virtual unsigned SizeOf(const TargetData *TD, unsigned Form) const;

    /// Profile - Used to gather unique data for the value folding set.
    ///
    static void Profile(FoldingSetNodeID &ID, const DWLabel &Label,
                        const DWLabel &Section);
    virtual void Profile(FoldingSetNodeID &ID);

    // Implement isa/cast/dyncast.
    static bool classof(const DIESectionOffset *)  { return true; }
    static bool classof(const DIEValue *D) {
      return D->getType() == isSectionOffset;
    }

#ifndef NDEBUG
    virtual void print(raw_ostream &O);
#endif
  };

  //===--------------------------------------------------------------------===//
  /// DIEDelta - A simple label difference DIE.
  ///
  class VISIBILITY_HIDDEN DIEDelta : public DIEValue {
    const DWLabel LabelHi;
    const DWLabel LabelLo;
  public:
    DIEDelta(const DWLabel &Hi, const DWLabel &Lo)
      : DIEValue(isDelta), LabelHi(Hi), LabelLo(Lo) {}

    /// EmitValue - Emit delta value.
    ///
    virtual void EmitValue(Dwarf *D, unsigned Form) const;

    /// SizeOf - Determine size of delta value in bytes.
    ///
    virtual unsigned SizeOf(const TargetData *TD, unsigned Form) const;

    /// Profile - Used to gather unique data for the value folding set.
    ///
    static void Profile(FoldingSetNodeID &ID, const DWLabel &LabelHi,
                        const DWLabel &LabelLo);
    virtual void Profile(FoldingSetNodeID &ID);

    // Implement isa/cast/dyncast.
    static bool classof(const DIEDelta *)  { return true; }
    static bool classof(const DIEValue *D) { return D->getType() == isDelta; }

#ifndef NDEBUG
    virtual void print(raw_ostream &O);
#endif
  };

  //===--------------------------------------------------------------------===//
  /// DIEntry - A pointer to another debug information entry.  An instance of
  /// this class can also be used as a proxy for a debug information entry not
  /// yet defined (ie. types.)
  class VISIBILITY_HIDDEN DIEEntry : public DIEValue {
    DIE *Entry;
  public:
    explicit DIEEntry(DIE *E) : DIEValue(isEntry), Entry(E) {}

    DIE *getEntry() const { return Entry; }
    void setEntry(DIE *E) { Entry = E; }

    /// EmitValue - Emit debug information entry offset.
    ///
    virtual void EmitValue(Dwarf *D, unsigned Form) const;

    /// SizeOf - Determine size of debug information entry in bytes.
    ///
    virtual unsigned SizeOf(const TargetData *TD, unsigned Form) const {
      return sizeof(int32_t);
    }

    /// Profile - Used to gather unique data for the value folding set.
    ///
    static void Profile(FoldingSetNodeID &ID, DIE *Entry);
    virtual void Profile(FoldingSetNodeID &ID);

    // Implement isa/cast/dyncast.
    static bool classof(const DIEEntry *)  { return true; }
    static bool classof(const DIEValue *E) { return E->getType() == isEntry; }

#ifndef NDEBUG
    virtual void print(raw_ostream &O);
#endif
  };

  //===--------------------------------------------------------------------===//
  /// DIEBlock - A block of values.  Primarily used for location expressions.
  //
  class VISIBILITY_HIDDEN DIEBlock : public DIEValue, public DIE {
    unsigned Size;                // Size in bytes excluding size header.
  public:
    DIEBlock()
      : DIEValue(isBlock), DIE(0), Size(0) {}
    virtual ~DIEBlock() {}

    /// ComputeSize - calculate the size of the block.
    ///
    unsigned ComputeSize(const TargetData *TD);

    /// BestForm - Choose the best form for data.
    ///
    unsigned BestForm() const {
      if ((unsigned char)Size == Size)  return dwarf::DW_FORM_block1;
      if ((unsigned short)Size == Size) return dwarf::DW_FORM_block2;
      if ((unsigned int)Size == Size)   return dwarf::DW_FORM_block4;
      return dwarf::DW_FORM_block;
    }

    /// EmitValue - Emit block data.
    ///
    virtual void EmitValue(Dwarf *D, unsigned Form) const;

    /// SizeOf - Determine size of block data in bytes.
    ///
    virtual unsigned SizeOf(const TargetData *TD, unsigned Form) const;

    /// Profile - Used to gather unique data for the value folding set.
    ///
    virtual void Profile(FoldingSetNodeID &ID);

    // Implement isa/cast/dyncast.
    static bool classof(const DIEBlock *)  { return true; }
    static bool classof(const DIEValue *E) { return E->getType() == isBlock; }

#ifndef NDEBUG
    virtual void print(raw_ostream &O);
#endif
  };

} // end llvm namespace

#endif
