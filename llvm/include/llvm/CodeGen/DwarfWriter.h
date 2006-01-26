//===-- llvm/CodeGen/DwarfWriter.h - Dwarf Framework ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by James M. Laskey and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing Dwarf debug info into asm files.  For
// Details on the Dwarf 3 specfication see DWARF Debugging Information Format
// V.3 reference manual http://dwarf.freestandards.org ,
//
// The role of the Dwarf Writer class is to extract debug information from the
// MachineDebugInfo object, organize it in Dwarf form and then emit it into asm
// the current asm file using data and high level Dwarf directives.
// 
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_DWARFWRITER_H
#define LLVM_CODEGEN_DWARFWRITER_H

#include "llvm/ADT/UniqueVector.h"

#include <iosfwd>
#include <string>


namespace llvm {
  
  //===--------------------------------------------------------------------===//
  // Forward declarations.
  //
  class AsmPrinter;
  class CompileUnitWrapper;
  class DIE;
  class DwarfWriter; 
  class DWContext;
  class MachineDebugInfo;
  class MachineFunction;
  class Module;
  class Type;
  
  //===--------------------------------------------------------------------===//
  // DWLabel - Labels are used to track locations in the assembler file.
  // Labels appear in the form <prefix>debug_<Tag><Number>, where the tag is a
  // category of label (Ex. location) and number is a value unique in that
  // category.
  class DWLabel {
  public:
    const char *Tag;                    // Label category tag. Should always be
                                        // a staticly declared C string.
    unsigned    Number;                 // Unique number.

    DWLabel(const char *T, unsigned N) : Tag(T), Number(N) {}
  };
  
  //===--------------------------------------------------------------------===//
  // DIEAbbrevData - Dwarf abbreviation data, describes the one attribute of a
  // Dwarf abbreviation.
  class DIEAbbrevData {
  private:
    unsigned Attribute;                 // Dwarf attribute code.
    unsigned Form;                      // Dwarf form code.
    
  public:
    DIEAbbrevData(unsigned A, unsigned F)
    : Attribute(A)
    , Form(F)
    {}
    
    // Accessors
    unsigned getAttribute() const { return Attribute; }
    unsigned getForm()      const { return Form; }
    
    /// operator== - Used by DIEAbbrev to locate entry.
    ///
    bool operator==(const DIEAbbrevData &DAD) const {
      return Attribute == DAD.Attribute && Form == DAD.Form;
    }

    /// operator!= - Used by DIEAbbrev to locate entry.
    ///
    bool operator!=(const DIEAbbrevData &DAD) const {
      return Attribute != DAD.Attribute || Form != DAD.Form;
    }
    
    /// operator< - Used by DIEAbbrev to locate entry.
    ///
    bool operator<(const DIEAbbrevData &DAD) const {
      return Attribute < DAD.Attribute ||
            (Attribute == DAD.Attribute && Form < DAD.Form);
    }
  };
  
  //===--------------------------------------------------------------------===//
  // DIEAbbrev - Dwarf abbreviation, describes the organization of a debug
  // information object.
  class DIEAbbrev {
  private:
    unsigned Tag;                       // Dwarf tag code.
    unsigned ChildrenFlag;              // Dwarf children flag.
    std::vector<DIEAbbrevData> Data;    // Raw data bytes for abbreviation.

  public:
  
    DIEAbbrev(unsigned T, unsigned C)
    : Tag(T)
    , ChildrenFlag(C)
    , Data()
    {}
    ~DIEAbbrev() {}
    
    // Accessors
    unsigned getTag()                           const { return Tag; }
    unsigned getChildrenFlag()                  const { return ChildrenFlag; }
    const std::vector<DIEAbbrevData> &getData() const { return Data; }

    /// operator== - Used by UniqueVector to locate entry.
    ///
    bool operator==(const DIEAbbrev &DA) const;

    /// operator< - Used by UniqueVector to locate entry.
    ///
    bool operator<(const DIEAbbrev &DA) const;

    /// AddAttribute - Adds another set of attribute information to the
    /// abbreviation.
    void AddAttribute(unsigned Attribute, unsigned Form) {
      Data.push_back(DIEAbbrevData(Attribute, Form));
    }
    
    /// Emit - Print the abbreviation using the specified Dwarf writer.
    ///
    void Emit(const DwarfWriter &DW) const; 
        
#ifndef NDEBUG
    void print(std::ostream &O);
    void dump();
#endif
  };

  //===--------------------------------------------------------------------===//
  // DIEValue - A debug information entry value.
  //
  class DIEValue {
  public:
    enum {
      isInteger,
      isString,
      isLabel,
      isAsIsLabel,
      isDelta,
      isEntry
    };
    
    unsigned Type;                      // Type of the value
    
    DIEValue(unsigned T) : Type(T) {}
    virtual ~DIEValue() {}
    
    // Implement isa/cast/dyncast.
    static bool classof(const DIEValue *) { return true; }
    
    /// EmitValue - Emit value via the Dwarf writer.
    ///
    virtual void EmitValue(const DwarfWriter &DW, unsigned Form) const = 0;
    
    /// SizeOf - Return the size of a value in bytes.
    ///
    virtual unsigned SizeOf(const DwarfWriter &DW, unsigned Form) const = 0;
  };

  //===--------------------------------------------------------------------===//
  // DWInteger - An integer value DIE.
  // 
  class DIEInteger : public DIEValue {
  private:
    int Integer;
    
  public:
    DIEInteger(int I) : DIEValue(isInteger), Integer(I) {}

    // Implement isa/cast/dyncast.
    static bool classof(const DIEInteger *) { return true; }
    static bool classof(const DIEValue *I)  { return I->Type == isInteger; }
    
    /// EmitValue - Emit integer of appropriate size.
    ///
    virtual void EmitValue(const DwarfWriter &DW, unsigned Form) const;
    
    /// SizeOf - Determine size of integer value in bytes.
    ///
    virtual unsigned SizeOf(const DwarfWriter &DW, unsigned Form) const;
  };

  //===--------------------------------------------------------------------===//
  // DIEString - A string value DIE.
  // 
  struct DIEString : public DIEValue {
    const std::string String;
    
    DIEString(const std::string &S) : DIEValue(isString), String(S) {}

    // Implement isa/cast/dyncast.
    static bool classof(const DIEString *) { return true; }
    static bool classof(const DIEValue *S) { return S->Type == isString; }
    
    /// EmitValue - Emit string value.
    ///
    virtual void EmitValue(const DwarfWriter &DW, unsigned Form) const;
    
    /// SizeOf - Determine size of string value in bytes.
    ///
    virtual unsigned SizeOf(const DwarfWriter &DW, unsigned Form) const;
  };

  //===--------------------------------------------------------------------===//
  // DIEDwarfLabel - A Dwarf internal label expression DIE.
  //
  struct DIEDwarfLabel : public DIEValue {
    const DWLabel Label;
    
    DIEDwarfLabel(const DWLabel &L) : DIEValue(isLabel), Label(L) {}

    // Implement isa/cast/dyncast.
    static bool classof(const DIEDwarfLabel *)  { return true; }
    static bool classof(const DIEValue *L) { return L->Type == isLabel; }
    
    /// EmitValue - Emit label value.
    ///
    virtual void EmitValue(const DwarfWriter &DW, unsigned Form) const;
    
    /// SizeOf - Determine size of label value in bytes.
    ///
    virtual unsigned SizeOf(const DwarfWriter &DW, unsigned Form) const;
  };


  //===--------------------------------------------------------------------===//
  // DIEObjectLabel - A label to an object in code or data.
  //
  struct DIEObjectLabel : public DIEValue {
    const std::string Label;
    
    DIEObjectLabel(const std::string &L) : DIEValue(isAsIsLabel), Label(L) {}

    // Implement isa/cast/dyncast.
    static bool classof(const DIEObjectLabel *) { return true; }
    static bool classof(const DIEValue *L)    { return L->Type == isAsIsLabel; }
    
    /// EmitValue - Emit label value.
    ///
    virtual void EmitValue(const DwarfWriter &DW, unsigned Form) const;
    
    /// SizeOf - Determine size of label value in bytes.
    ///
    virtual unsigned SizeOf(const DwarfWriter &DW, unsigned Form) const;
  };

  //===--------------------------------------------------------------------===//
  // DIEDelta - A simple label difference DIE.
  // 
  struct DIEDelta : public DIEValue {
    const DWLabel LabelHi;
    const DWLabel LabelLo;
    
    DIEDelta(const DWLabel &Hi, const DWLabel &Lo)
    : DIEValue(isDelta), LabelHi(Hi), LabelLo(Lo) {}

    // Implement isa/cast/dyncast.
    static bool classof(const DIEDelta *)  { return true; }
    static bool classof(const DIEValue *D) { return D->Type == isDelta; }
    
    /// EmitValue - Emit delta value.
    ///
    virtual void EmitValue(const DwarfWriter &DW, unsigned Form) const;
    
    /// SizeOf - Determine size of delta value in bytes.
    ///
    virtual unsigned SizeOf(const DwarfWriter &DW, unsigned Form) const;
  };
  
  //===--------------------------------------------------------------------===//
  // DIEntry - A pointer to a debug information entry.
  // 
  struct DIEntry : public DIEValue {
    DIE *Entry;
    
    DIEntry(DIE *E) : DIEValue(isEntry), Entry(E) {}

    // Implement isa/cast/dyncast.
    static bool classof(const DIEntry *)   { return true; }
    static bool classof(const DIEValue *E) { return E->Type == isEntry; }
    
    /// EmitValue - Emit delta value.
    ///
    virtual void EmitValue(const DwarfWriter &DW, unsigned Form) const;
    
    /// SizeOf - Determine size of delta value in bytes.
    ///
    virtual unsigned SizeOf(const DwarfWriter &DW, unsigned Form) const;
  };
  
  //===--------------------------------------------------------------------===//
  // DIE - A structured debug information entry.  Has an abbreviation which
  // describes it's organization.
  class DIE {
  private:
    DIEAbbrev *Abbrev;                    // Temporary buffer for abbreviation.
    unsigned AbbrevID;                    // Decribing abbreviation ID.
    unsigned Offset;                      // Offset in debug info section.
    unsigned Size;                        // Size of instance + children.
    DWContext *Context;                   // Context for types and values.
    std::vector<DIE *> Children;          // Children DIEs.
    std::vector<DIEValue *> Values;       // Attributes values.
    
  public:
    DIE(unsigned Tag, unsigned ChildrenFlag);
    ~DIE();
    
    // Accessors
    unsigned   getAbbrevID()                   const { return AbbrevID; }
    unsigned   getOffset()                     const { return Offset; }
    unsigned   getSize()                       const { return Size; }
    DWContext *getContext()                    const { return Context; }
    const std::vector<DIE *> &getChildren()    const { return Children; }
    const std::vector<DIEValue *> &getValues() const { return Values; }
    void setOffset(unsigned O)                 { Offset = O; }
    void setSize(unsigned S)                   { Size = S; }
    void setContext(DWContext *C)              { Context = C; }
    
    /// SiblingOffset - Return the offset of the debug information entry's
    /// sibling.
    unsigned SiblingOffset() const { return Offset + Size; }

    /// AddInt - Add a simple integer attribute data and value.
    ///
    void AddInt(unsigned Attribute, unsigned Form,
                int Integer, bool IsSigned = false);
        
    /// AddString - Add a std::string attribute data and value.
    ///
    void AddString(unsigned Attribute, unsigned Form,
                   const std::string &String);
        
    /// AddLabel - Add a Dwarf label attribute data and value.
    ///
    void AddLabel(unsigned Attribute, unsigned Form, const DWLabel &Label);
        
    /// AddObjectLabel - Add a non-Dwarf label attribute data and value.
    ///
    void AddObjectLabel(unsigned Attribute, unsigned Form,
                        const std::string &Label);
        
    /// AddDelta - Add a label delta attribute data and value.
    ///
    void AddDelta(unsigned Attribute, unsigned Form,
                  const DWLabel &Hi, const DWLabel &Lo);
        
    ///  AddDIEntry - Add a DIE attribute data and value.
    ///
    void AddDIEntry(unsigned Attribute, unsigned Form, DIE *Entry);

    /// Complete - Indicate that all attributes have been added and
    /// ready to get an abbreviation ID.
    ///
    void Complete(DwarfWriter &DW);
    
    /// AddChild - Add a child to the DIE.
    void AddChild(DIE *Child);
  };
  
  //===--------------------------------------------------------------------===//
  /// DWContext - Name context for types and values.
  ///
  class DWContext {
  private:
    DwarfWriter &DW;                    // DwarfWriter for global information.
    DWContext *Parent;                  // Next context level searched.
    DIE *Owner;                         // Owning debug information entry.
    std::map<const Type *, DIE*> Types; // Named types in context.
    std::map<std::string, DIE*> Variables;// Named variables in context.
    
  public:
    DWContext(DwarfWriter &D, DWContext *P, DIE *O)
    : DW(D)
    , Parent(P)
    , Owner(O)
    , Types()
    , Variables()
    {
      Owner->setContext(this);
    }
    ~DWContext() {}
    
    /// NewBasicType - Creates a new basic type, if necessary, then adds to the
    /// context and owner.
    DIE *NewBasicType(const Type *Ty, unsigned Size, unsigned Align);
                                               
    /// NewVariable - Creates a basic variable, if necessary, then adds to the
    /// context and owner.
    DIE *NewGlobalVariable(const std::string &Name,
                           const std::string &MangledName,
                           DIE *Type);
  };

  //===--------------------------------------------------------------------===//
  // DwarfWriter - Emits Dwarf debug and exception handling directives.
  //
  class DwarfWriter {
  protected:
  
    //===------------------------------------------------------------------===//
    // Core attributes used by the Dwarf  writer.
    //
    
    //
    /// O - Stream to .s file.
    ///
    std::ostream &O;

    /// Asm - Target of Dwarf emission.
    ///
    AsmPrinter *Asm;
    
    /// DebugInfo - Collected debug information.
    ///
    MachineDebugInfo *DebugInfo;
    
    /// didInitial - Flag to indicate if initial emission has been done.
    ///
    bool didInitial;
    
    //===------------------------------------------------------------------===//
    // Attributes used to construct specific Dwarf sections.
    //
    
    /// CompileUnits - All the compile units involved in this build.  The index
    /// of each entry in this vector corresponds to the sources in DebugInfo.
    std::vector<DIE *> CompileUnits;

    /// Abbreviations - A UniqueVector of TAG structure abbreviations.
    ///
    UniqueVector<DIEAbbrev> Abbreviations;
    
    /// GlobalTypes - A map of globally visible named types.
    ///
    std::map<std::string, DIE *> GlobalTypes;
    
    /// GlobalEntities - A map of globally visible named entities.
    ///
    std::map<std::string, DIE *> GlobalEntities;
     
    /// StringPool - A UniqueVector of strings used by indirect references.
    ///
    UniqueVector<std::string> StringPool;
    
    //===------------------------------------------------------------------===//
    // Properties to be set by the derived class ctor, used to configure the
    // Dwarf writer.
    //
    
    /// AddressSize - Size of addresses used in file.
    ///
    unsigned AddressSize;

    /// hasLEB128 - True if target asm supports leb128 directives.
    ///
    bool hasLEB128; /// Defaults to false.
    
    /// hasDotLoc - True if target asm supports .loc directives.
    ///
    bool hasDotLoc; /// Defaults to false.
    
    /// hasDotFile - True if target asm supports .file directives.
    ///
    bool hasDotFile; /// Defaults to false.
    
    /// needsSet - True if target asm can't compute addresses on data
    /// directives.
    bool needsSet; /// Defaults to false.
    
    /// DwarfAbbrevSection - Section directive for Dwarf abbrev.
    ///
    const char *DwarfAbbrevSection; /// Defaults to ".debug_abbrev".

    /// DwarfInfoSection - Section directive for Dwarf info.
    ///
    const char *DwarfInfoSection; /// Defaults to ".debug_info".

    /// DwarfLineSection - Section directive for Dwarf info.
    ///
    const char *DwarfLineSection; /// Defaults to ".debug_line".
    
    /// DwarfFrameSection - Section directive for Dwarf info.
    ///
    const char *DwarfFrameSection; /// Defaults to ".debug_frame".
    
    /// DwarfPubNamesSection - Section directive for Dwarf info.
    ///
    const char *DwarfPubNamesSection; /// Defaults to ".debug_pubnames".
    
    /// DwarfPubTypesSection - Section directive for Dwarf info.
    ///
    const char *DwarfPubTypesSection; /// Defaults to ".debug_pubtypes".
    
    /// DwarfStrSection - Section directive for Dwarf info.
    ///
    const char *DwarfStrSection; /// Defaults to ".debug_str".

    /// DwarfLocSection - Section directive for Dwarf info.
    ///
    const char *DwarfLocSection; /// Defaults to ".debug_loc".

    /// DwarfARangesSection - Section directive for Dwarf info.
    ///
    const char *DwarfARangesSection; /// Defaults to ".debug_aranges".

    /// DwarfRangesSection - Section directive for Dwarf info.
    ///
    const char *DwarfRangesSection; /// Defaults to ".debug_ranges".

    /// DwarfMacInfoSection - Section directive for Dwarf info.
    ///
    const char *DwarfMacInfoSection; /// Defaults to ".debug_macinfo".

    /// TextSection - Section directive for standard text.
    ///
    const char *TextSection; /// Defaults to ".text".
    
    /// DataSection - Section directive for standard data.
    ///
    const char *DataSection; /// Defaults to ".data".

    //===------------------------------------------------------------------===//
    // Emission and print routines
    //

public:
    /// getAddressSize - Return the size of a target address in bytes.
    ///
    unsigned getAddressSize() const { return AddressSize; }

    /// PrintHex - Print a value as a hexidecimal value.
    ///
    void PrintHex(int Value) const;

    /// EOL - Print a newline character to asm stream.  If a comment is present
    /// then it will be printed first.  Comments should not contain '\n'.
    void EOL(const std::string &Comment) const;
                                          
    /// EmitULEB128Bytes - Emit an assembler byte data directive to compose an
    /// unsigned leb128 value.
    void EmitULEB128Bytes(unsigned Value) const;
    
    /// EmitSLEB128Bytes - print an assembler byte data directive to compose a
    /// signed leb128 value.
    void EmitSLEB128Bytes(int Value) const;
    
    /// PrintULEB128 - Print a series of hexidecimal values (separated by
    /// commas) representing an unsigned leb128 value.
    void PrintULEB128(unsigned Value) const;

    /// SizeULEB128 - Compute the number of bytes required for an unsigned
    /// leb128 value.
    static unsigned SizeULEB128(unsigned Value);
    
    /// PrintSLEB128 - Print a series of hexidecimal values (separated by
    /// commas) representing a signed leb128 value.
    void PrintSLEB128(int Value) const;
    
    /// SizeSLEB128 - Compute the number of bytes required for a signed leb128
    /// value.
    static unsigned SizeSLEB128(int Value);
    
    /// EmitByte - Emit a byte directive and value.
    ///
    void EmitByte(int Value) const;

    /// EmitShort - Emit a short directive and value.
    ///
    void EmitShort(int Value) const;

    /// EmitLong - Emit a long directive and value.
    ///
    void EmitLong(int Value) const;
    
    /// EmitString - Emit a string with quotes and a null terminator.
    /// Special characters are emitted properly. (Eg. '\t')
    void EmitString(const std::string &String) const;

    /// PrintLabelName - Print label name in form used by Dwarf writer.
    ///
    void PrintLabelName(DWLabel Label) const {
      PrintLabelName(Label.Tag, Label.Number);
    }
    void PrintLabelName(const char *Tag, unsigned Number) const;
    
    /// EmitLabel - Emit location label for internal use by Dwarf.
    ///
    void EmitLabel(DWLabel Label) const {
      EmitLabel(Label.Tag, Label.Number);
    }
    void EmitLabel(const char *Tag, unsigned Number) const;
    
    /// EmitReference - Emit a reference to a label.
    ///
    void EmitReference(DWLabel Label) const {
      EmitReference(Label.Tag, Label.Number);
    }
    void EmitReference(const char *Tag, unsigned Number) const;
    void EmitReference(const std::string &Name) const;

    /// EmitDifference - Emit the difference between two labels.  Some
    /// assemblers do not behave with absolute expressions with data directives,
    /// so there is an option (needsSet) to use an intermediary set expression.
    void EmitDifference(DWLabel LabelHi, DWLabel LabelLo) const {
      EmitDifference(LabelHi.Tag, LabelHi.Number, LabelLo.Tag, LabelLo.Number);
    }
    void EmitDifference(const char *TagHi, unsigned NumberHi,
                        const char *TagLo, unsigned NumberLo) const;
                                   
    /// NewAbbreviation - Add the abbreviation to the Abbreviation vector.
    ///  
    unsigned NewAbbreviation(DIEAbbrev *Abbrev);
    
    /// NewString - Add a string to the constant pool and returns a label.
    ///
    DWLabel NewString(const std::string &String);
    
    /// NewGlobalType - Make the type visible globally using the given name.
    ///
    void NewGlobalType(const std::string &Name, DIE *Type);
    
    /// NewGlobalEntity - Make the entity visible globally using the given name.
    ///
    void NewGlobalEntity(const std::string &Name, DIE *Entity);
    
    /// NewGlobalVariable - Add a new global variable DIE to the context.
    ///
    void NewGlobalVariable(DWContext *Context,
                           const std::string &Name,
                           const std::string &MangledName,
                           const Type *Ty,
                           unsigned Size, unsigned Align);

private:

    /// NewCompileUnit - Create new compile unit information.
    ///
    DIE *DwarfWriter::NewCompileUnit(const CompileUnitWrapper &CompileUnit);

    /// EmitInitial - Emit initial Dwarf declarations.
    ///
    void EmitInitial() const;
    
    /// EmitDIE - Recusively Emits a debug information entry.
    ///
    void EmitDIE(DIE *Die) const;
    
    /// SizeAndOffsetDie - Compute the size and offset of a DIE.
    ///
    unsigned SizeAndOffsetDie(DIE *Die, unsigned Offset) const;

    /// SizeAndOffsets - Compute the size and offset of all the DIEs.
    ///
    void SizeAndOffsets();
    
    /// EmitDebugInfo - Emit the debug info section.
    ///
    void EmitDebugInfo() const;
    
    /// EmitAbbreviations - Emit the abbreviation section.
    ///
    void EmitAbbreviations() const;
    
    /// EmitDebugLines - Emit source line information.
    ///
    void EmitDebugLines() const;

    /// EmitDebugFrame - Emit info into a debug frame section.
    ///
    void EmitDebugFrame();
    
    /// EmitDebugPubNames - Emit info into a debug pubnames section.
    ///
    void EmitDebugPubNames();
    
    /// EmitDebugPubTypes - Emit info into a debug pubtypes section.
    ///
    void EmitDebugPubTypes();
    
    /// EmitDebugStr - Emit info into a debug str section.
    ///
    void EmitDebugStr();
    
    /// EmitDebugLoc - Emit info into a debug loc section.
    ///
    void EmitDebugLoc();
    
    /// EmitDebugARanges - Emit info into a debug aranges section.
    ///
    void EmitDebugARanges();
    
    /// EmitDebugRanges - Emit info into a debug ranges section.
    ///
    void EmitDebugRanges();
    
    /// EmitDebugMacInfo - Emit info into a debug macinfo section.
    ///
    void EmitDebugMacInfo();
    
    /// ConstructCompileUnitDIEs - Create a compile unit DIE for each source and
    /// header file.
    void ConstructCompileUnitDIEs();
    
    /// ConstructGlobalDIEs - Create DIEs for each of the externally visible global
    /// variables.
    void ConstructGlobalDIEs(Module &M);

    /// ShouldEmitDwarf - Returns true if Dwarf declarations should be made.
    /// When called it also checks to see if debug info is newly available.  if
    /// so the initial Dwarf headers are emitted.
    bool ShouldEmitDwarf();
  
  public:
    
    DwarfWriter(std::ostream &OS, AsmPrinter *A);
    virtual ~DwarfWriter();
    
    /// SetDebugInfo - Set DebugInfo when it's known that pass manager has
    /// created it.  Set by the target AsmPrinter.
    void SetDebugInfo(MachineDebugInfo *DI) { DebugInfo = DI; }

    //===------------------------------------------------------------------===//
    // Main entry points.
    //
    
    /// BeginModule - Emit all Dwarf sections that should come prior to the
    /// content.
    void BeginModule(Module &M);
    
    /// EndModule - Emit all Dwarf sections that should come after the content.
    ///
    void EndModule(Module &M);
    
    /// BeginFunction - Gather pre-function debug information.
    ///
    void BeginFunction(MachineFunction &MF);
    
    /// EndFunction - Gather and emit post-function debug information.
    ///
    void EndFunction(MachineFunction &MF);
  };

} // end llvm namespace

#endif
