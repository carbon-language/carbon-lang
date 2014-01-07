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

#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Dwarf.h"
#include <vector>

namespace llvm {
  class AsmPrinter;
  class MCSymbol;
  class MCSymbolRefExpr;
  class raw_ostream;
  class DwarfTypeUnit;

  //===--------------------------------------------------------------------===//
  /// DIEAbbrevData - Dwarf abbreviation data, describes one attribute of a
  /// Dwarf abbreviation.
  class DIEAbbrevData {
    /// Attribute - Dwarf attribute code.
    ///
    dwarf::Attribute Attribute;

    /// Form - Dwarf form code.
    ///
    dwarf::Form Form;
  public:
    DIEAbbrevData(dwarf::Attribute A, dwarf::Form F) : Attribute(A), Form(F) {}

    // Accessors.
    dwarf::Attribute getAttribute() const { return Attribute; }
    dwarf::Form getForm() const { return Form; }

    /// Profile - Used to gather unique data for the abbreviation folding set.
    ///
    void Profile(FoldingSetNodeID &ID) const;
  };

  //===--------------------------------------------------------------------===//
  /// DIEAbbrev - Dwarf abbreviation, describes the organization of a debug
  /// information object.
  class DIEAbbrev : public FoldingSetNode {
    /// Tag - Dwarf tag code.
    ///
    dwarf::Tag Tag;

    /// ChildrenFlag - Dwarf children flag.
    ///
    uint16_t ChildrenFlag;

    /// Unique number for node.
    ///
    unsigned Number;

    /// Data - Raw data bytes for abbreviation.
    ///
    SmallVector<DIEAbbrevData, 12> Data;

  public:
    DIEAbbrev(dwarf::Tag T, uint16_t C) : Tag(T), ChildrenFlag(C), Data() {}

    // Accessors.
    dwarf::Tag getTag() const { return Tag; }
    unsigned getNumber() const { return Number; }
    uint16_t getChildrenFlag() const { return ChildrenFlag; }
    const SmallVectorImpl<DIEAbbrevData> &getData() const { return Data; }
    void setChildrenFlag(uint16_t CF) { ChildrenFlag = CF; }
    void setNumber(unsigned N) { Number = N; }

    /// AddAttribute - Adds another set of attribute information to the
    /// abbreviation.
    void AddAttribute(dwarf::Attribute Attribute, dwarf::Form Form) {
      Data.push_back(DIEAbbrevData(Attribute, Form));
    }

    /// Profile - Used to gather unique data for the abbreviation folding set.
    ///
    void Profile(FoldingSetNodeID &ID) const;

    /// Emit - Print the abbreviation using the specified asm printer.
    ///
    void Emit(AsmPrinter *AP) const;

#ifndef NDEBUG
    void print(raw_ostream &O);
    void dump();
#endif
  };

  //===--------------------------------------------------------------------===//
  /// DIE - A structured debug information entry.  Has an abbreviation which
  /// describes its organization.
  class DIEValue;

  class DIE {
  protected:
    /// Offset - Offset in debug info section.
    ///
    unsigned Offset;

    /// Size - Size of instance + children.
    ///
    unsigned Size;

    /// Abbrev - Buffer for constructing abbreviation.
    ///
    DIEAbbrev Abbrev;

    /// Children DIEs.
    ///
    std::vector<DIE *> Children;

    DIE *Parent;

    /// Attribute values.
    ///
    SmallVector<DIEValue*, 12> Values;

  public:
    explicit DIE(unsigned Tag)
        : Offset(0), Size(0), Abbrev((dwarf::Tag)Tag, dwarf::DW_CHILDREN_no),
          Parent(0) {}
    ~DIE();

    // Accessors.
    DIEAbbrev &getAbbrev() { return Abbrev; }
    const DIEAbbrev &getAbbrev() const { return Abbrev; }
    unsigned getAbbrevNumber() const { return Abbrev.getNumber(); }
    dwarf::Tag getTag() const { return Abbrev.getTag(); }
    unsigned getOffset() const { return Offset; }
    unsigned getSize() const { return Size; }
    const std::vector<DIE *> &getChildren() const { return Children; }
    const SmallVectorImpl<DIEValue*> &getValues() const { return Values; }
    DIE *getParent() const { return Parent; }
    /// Climb up the parent chain to get the compile or type unit DIE this DIE
    /// belongs to.
    const DIE *getUnit() const;
    /// Similar to getUnit, returns null when DIE is not added to an
    /// owner yet.
    const DIE *getUnitOrNull() const;
    void setOffset(unsigned O) { Offset = O; }
    void setSize(unsigned S) { Size = S; }

    /// addValue - Add a value and attributes to a DIE.
    ///
    void addValue(dwarf::Attribute Attribute, dwarf::Form Form,
                  DIEValue *Value) {
      Abbrev.AddAttribute(Attribute, Form);
      Values.push_back(Value);
    }

    /// addChild - Add a child to the DIE.
    ///
    void addChild(DIE *Child) {
      assert(!Child->getParent());
      Abbrev.setChildrenFlag(dwarf::DW_CHILDREN_yes);
      Children.push_back(Child);
      Child->Parent = this;
    }

    /// findAttribute - Find a value in the DIE with the attribute given,
    /// returns NULL if no such attribute exists.
    DIEValue *findAttribute(uint16_t Attribute) const;

#ifndef NDEBUG
    void print(raw_ostream &O, unsigned IndentCount = 0) const;
    void dump();
#endif
  };

  //===--------------------------------------------------------------------===//
  /// DIEValue - A debug information entry value.
  ///
  class DIEValue {
    virtual void anchor();
  public:
    enum {
      isInteger,
      isString,
      isExpr,
      isLabel,
      isDelta,
      isEntry,
      isTypeSignature,
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
    virtual void EmitValue(AsmPrinter *AP, dwarf::Form Form) const = 0;

    /// SizeOf - Return the size of a value in bytes.
    ///
    virtual unsigned SizeOf(AsmPrinter *AP, dwarf::Form Form) const = 0;

#ifndef NDEBUG
    virtual void print(raw_ostream &O) const = 0;
    void dump() const;
#endif
  };

  //===--------------------------------------------------------------------===//
  /// DIEInteger - An integer value DIE.
  ///
  class DIEInteger : public DIEValue {
    uint64_t Integer;
  public:
    explicit DIEInteger(uint64_t I) : DIEValue(isInteger), Integer(I) {}

    /// BestForm - Choose the best form for integer.
    ///
    static dwarf::Form BestForm(bool IsSigned, uint64_t Int) {
      if (IsSigned) {
        const int64_t SignedInt = Int;
        if ((char)Int == SignedInt)     return dwarf::DW_FORM_data1;
        if ((short)Int == SignedInt)    return dwarf::DW_FORM_data2;
        if ((int)Int == SignedInt)      return dwarf::DW_FORM_data4;
      } else {
        if ((unsigned char)Int == Int)  return dwarf::DW_FORM_data1;
        if ((unsigned short)Int == Int) return dwarf::DW_FORM_data2;
        if ((unsigned int)Int == Int)   return dwarf::DW_FORM_data4;
      }
      return dwarf::DW_FORM_data8;
    }

    /// EmitValue - Emit integer of appropriate size.
    ///
    virtual void EmitValue(AsmPrinter *AP, dwarf::Form Form) const;

    uint64_t getValue() const { return Integer; }

    /// SizeOf - Determine size of integer value in bytes.
    ///
    virtual unsigned SizeOf(AsmPrinter *AP, dwarf::Form Form) const;

    // Implement isa/cast/dyncast.
    static bool classof(const DIEValue *I) { return I->getType() == isInteger; }

#ifndef NDEBUG
    virtual void print(raw_ostream &O) const;
#endif
  };

  //===--------------------------------------------------------------------===//
  /// DIEExpr - An expression DIE.
  //
  class DIEExpr : public DIEValue {
    const MCExpr *Expr;
  public:
    explicit DIEExpr(const MCExpr *E) : DIEValue(isExpr), Expr(E) {}

    /// EmitValue - Emit expression value.
    ///
    virtual void EmitValue(AsmPrinter *AP, dwarf::Form Form) const;

    /// getValue - Get MCExpr.
    ///
    const MCExpr *getValue() const { return Expr; }

    /// SizeOf - Determine size of expression value in bytes.
    ///
    virtual unsigned SizeOf(AsmPrinter *AP, dwarf::Form Form) const;

    // Implement isa/cast/dyncast.
    static bool classof(const DIEValue *E) { return E->getType() == isExpr; }

#ifndef NDEBUG
    virtual void print(raw_ostream &O) const;
#endif
  };

  //===--------------------------------------------------------------------===//
  /// DIELabel - A label DIE.
  //
  class DIELabel : public DIEValue {
    const MCSymbol *Label;
  public:
    explicit DIELabel(const MCSymbol *L) : DIEValue(isLabel), Label(L) {}

    /// EmitValue - Emit label value.
    ///
    virtual void EmitValue(AsmPrinter *AP, dwarf::Form Form) const;

    /// getValue - Get MCSymbol.
    ///
    const MCSymbol *getValue() const { return Label; }

    /// SizeOf - Determine size of label value in bytes.
    ///
    virtual unsigned SizeOf(AsmPrinter *AP, dwarf::Form Form) const;

    // Implement isa/cast/dyncast.
    static bool classof(const DIEValue *L) { return L->getType() == isLabel; }

#ifndef NDEBUG
    virtual void print(raw_ostream &O) const;
#endif
  };

  //===--------------------------------------------------------------------===//
  /// DIEDelta - A simple label difference DIE.
  ///
  class DIEDelta : public DIEValue {
    const MCSymbol *LabelHi;
    const MCSymbol *LabelLo;
  public:
    DIEDelta(const MCSymbol *Hi, const MCSymbol *Lo)
      : DIEValue(isDelta), LabelHi(Hi), LabelLo(Lo) {}

    /// EmitValue - Emit delta value.
    ///
    virtual void EmitValue(AsmPrinter *AP, dwarf::Form Form) const;

    /// SizeOf - Determine size of delta value in bytes.
    ///
    virtual unsigned SizeOf(AsmPrinter *AP, dwarf::Form Form) const;

    // Implement isa/cast/dyncast.
    static bool classof(const DIEValue *D) { return D->getType() == isDelta; }

#ifndef NDEBUG
    virtual void print(raw_ostream &O) const;
#endif
  };

  //===--------------------------------------------------------------------===//
  /// DIEString - A container for string values.
  ///
  class DIEString : public DIEValue {
    const DIEValue *Access;
    const StringRef Str;

  public:
    DIEString(const DIEValue *Acc, const StringRef S)
        : DIEValue(isString), Access(Acc), Str(S) {}

    /// getString - Grab the string out of the object.
    StringRef getString() const { return Str; }

    /// EmitValue - Emit delta value.
    ///
    virtual void EmitValue(AsmPrinter *AP, dwarf::Form Form) const;

    /// SizeOf - Determine size of delta value in bytes.
    ///
    virtual unsigned SizeOf(AsmPrinter *AP, dwarf::Form Form) const;

    // Implement isa/cast/dyncast.
    static bool classof(const DIEValue *D) { return D->getType() == isString; }

  #ifndef NDEBUG
    virtual void print(raw_ostream &O) const;
  #endif
  };

  //===--------------------------------------------------------------------===//
  /// DIEEntry - A pointer to another debug information entry.  An instance of
  /// this class can also be used as a proxy for a debug information entry not
  /// yet defined (ie. types.)
  class DIEEntry : public DIEValue {
    DIE *const Entry;
  public:
    explicit DIEEntry(DIE *E) : DIEValue(isEntry), Entry(E) {
      assert(E && "Cannot construct a DIEEntry with a null DIE");
    }

    DIE *getEntry() const { return Entry; }

    /// EmitValue - Emit debug information entry offset.
    ///
    virtual void EmitValue(AsmPrinter *AP, dwarf::Form Form) const;

    /// SizeOf - Determine size of debug information entry in bytes.
    ///
    virtual unsigned SizeOf(AsmPrinter *AP, dwarf::Form Form) const {
      return Form == dwarf::DW_FORM_ref_addr ? getRefAddrSize(AP)
                                             : sizeof(int32_t);
    }

    /// Returns size of a ref_addr entry.
    static unsigned getRefAddrSize(AsmPrinter *AP);

    // Implement isa/cast/dyncast.
    static bool classof(const DIEValue *E) { return E->getType() == isEntry; }

#ifndef NDEBUG
    virtual void print(raw_ostream &O) const;
#endif
  };

  //===--------------------------------------------------------------------===//
  /// \brief A signature reference to a type unit.
  class DIETypeSignature : public DIEValue {
    const DwarfTypeUnit &Unit;
  public:
    explicit DIETypeSignature(const DwarfTypeUnit &Unit)
        : DIEValue(isTypeSignature), Unit(Unit) {}

    /// \brief Emit type unit signature.
    virtual void EmitValue(AsmPrinter *Asm, dwarf::Form Form) const;

    /// Returns size of a ref_sig8 entry.
    virtual unsigned SizeOf(AsmPrinter *AP, dwarf::Form Form) const {
      assert(Form == dwarf::DW_FORM_ref_sig8);
      return 8;
    }

    // \brief Implement isa/cast/dyncast.
    static bool classof(const DIEValue *E) {
      return E->getType() == isTypeSignature;
    }
#ifndef NDEBUG
    virtual void print(raw_ostream &O) const;
    void dump() const;
#endif
  };

  //===--------------------------------------------------------------------===//
  /// DIEBlock - A block of values.  Primarily used for location expressions.
  //
  class DIEBlock : public DIEValue, public DIE {
    unsigned Size;                // Size in bytes excluding size header.
  public:
    DIEBlock() : DIEValue(isBlock), DIE(0), Size(0) {}

    /// ComputeSize - calculate the size of the block.
    ///
    unsigned ComputeSize(AsmPrinter *AP);

    /// BestForm - Choose the best form for data.
    ///
    dwarf::Form BestForm() const {
      if ((unsigned char)Size == Size)  return dwarf::DW_FORM_block1;
      if ((unsigned short)Size == Size) return dwarf::DW_FORM_block2;
      if ((unsigned int)Size == Size)   return dwarf::DW_FORM_block4;
      return dwarf::DW_FORM_block;
    }

    /// EmitValue - Emit block data.
    ///
    virtual void EmitValue(AsmPrinter *AP, dwarf::Form Form) const;

    /// SizeOf - Determine size of block data in bytes.
    ///
    virtual unsigned SizeOf(AsmPrinter *AP, dwarf::Form Form) const;

    // Implement isa/cast/dyncast.
    static bool classof(const DIEValue *E) { return E->getType() == isBlock; }

#ifndef NDEBUG
    virtual void print(raw_ostream &O) const;
#endif
  };

} // end llvm namespace

#endif
