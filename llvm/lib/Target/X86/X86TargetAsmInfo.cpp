//===-- X86TargetAsmInfo.cpp - X86 asm properties ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by James M. Laskey and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the X86TargetAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "X86TargetAsmInfo.h"
#include "X86TargetMachine.h"
#include "X86Subtarget.h"
#include "llvm/DerivedTypes.h"
#include "llvm/InlineAsm.h"
#include "llvm/Instructions.h"
#include "llvm/Intrinsics.h"
#include "llvm/Module.h"
#include "llvm/ADT/StringExtras.h"
using namespace llvm;

static const char* x86_asm_table[] = {"{si}", "S",
                                      "{di}", "D",
                                      "{ax}", "a",
                                      "{cx}", "c",
                                      "{memory}", "memory",
                                      "{flags}", "",
                                      "{dirflag}", "",
                                      "{fpsr}", "",
                                      "{cc}", "cc",
                                      0,0};

X86TargetAsmInfo::X86TargetAsmInfo(const X86TargetMachine &TM) {
  const X86Subtarget *Subtarget = &TM.getSubtarget<X86Subtarget>();
  
  // FIXME - Should be simplified.

  AsmTransCBE = x86_asm_table;
  
  switch (Subtarget->TargetType) {
  case X86Subtarget::isDarwin:
    AlignmentIsInBytes = false;
    GlobalPrefix = "_";
    if (!Subtarget->is64Bit())
      Data64bitsDirective = 0;       // we can't emit a 64-bit unit
    ZeroDirective = "\t.space\t";  // ".space N" emits N zeros.
    PrivateGlobalPrefix = "L";     // Marker for constant pool idxs
    BSSSection = 0;                       // no BSS section.
    ZeroFillDirective = "\t.zerofill\t";  // Uses .zerofill
    ConstantPoolSection = "\t.const\n";
    JumpTableDataSection = "\t.const\n";
    CStringSection = "\t.cstring";
    FourByteConstantSection = "\t.literal4\n";
    EightByteConstantSection = "\t.literal8\n";
    if (Subtarget->is64Bit())
      SixteenByteConstantSection = "\t.literal16\n";
    ReadOnlySection = "\t.const\n";
    LCOMMDirective = "\t.lcomm\t";
    COMMDirectiveTakesAlignment = false;
    HasDotTypeDotSizeDirective = false;
    if (TM.getRelocationModel() == Reloc::Static) {
      StaticCtorsSection = ".constructor";
      StaticDtorsSection = ".destructor";
    } else {
      StaticCtorsSection = ".mod_init_func";
      StaticDtorsSection = ".mod_term_func";
    }
    InlineAsmStart = "# InlineAsm Start";
    InlineAsmEnd = "# InlineAsm End";
    SetDirective = "\t.set";
    UsedDirective = "\t.no_dead_strip\t";
    WeakRefDirective = "\t.weak_reference\t";
    HiddenDirective = "\t.private_extern\t";
    
    // In non-PIC modes, emit a special label before jump tables so that the
    // linker can perform more accurate dead code stripping.
    if (TM.getRelocationModel() != Reloc::PIC_) {
      // Emit a local label that is preserved until the linker runs.
      JumpTableSpecialLabelPrefix = "l";
    }

    SupportsDebugInformation = true;
    NeedsSet = true;
    DwarfAbbrevSection = ".section __DWARF,__debug_abbrev,regular,debug";
    DwarfInfoSection = ".section __DWARF,__debug_info,regular,debug";
    DwarfLineSection = ".section __DWARF,__debug_line,regular,debug";
    DwarfFrameSection = ".section __DWARF,__debug_frame,regular,debug";
    DwarfPubNamesSection = ".section __DWARF,__debug_pubnames,regular,debug";
    DwarfPubTypesSection = ".section __DWARF,__debug_pubtypes,regular,debug";
    DwarfStrSection = ".section __DWARF,__debug_str,regular,debug";
    DwarfLocSection = ".section __DWARF,__debug_loc,regular,debug";
    DwarfARangesSection = ".section __DWARF,__debug_aranges,regular,debug";
    DwarfRangesSection = ".section __DWARF,__debug_ranges,regular,debug";
    DwarfMacInfoSection = ".section __DWARF,__debug_macinfo,regular,debug";
    break;

  case X86Subtarget::isELF:
    ReadOnlySection = "\t.section\t.rodata";
    FourByteConstantSection = "\t.section\t.rodata.cst4,\"aM\",@progbits,4";
    EightByteConstantSection = "\t.section\t.rodata.cst8,\"aM\",@progbits,8";
    SixteenByteConstantSection = "\t.section\t.rodata.cst16,\"aM\",@progbits,16";
    CStringSection = "\t.section\t.rodata.str1.1,\"aMS\",@progbits,1";
    PrivateGlobalPrefix = ".L";
    WeakRefDirective = "\t.weak\t";
    SetDirective = "\t.set\t";
    PCSymbol = ".";

    // Set up DWARF directives
    HasLEB128 = true;  // Target asm supports leb128 directives (little-endian)

    // Debug Information
    AbsoluteDebugSectionOffsets = true;
    SupportsDebugInformation = true;
    DwarfAbbrevSection =  "\t.section\t.debug_abbrev,\"\",@progbits";
    DwarfInfoSection =    "\t.section\t.debug_info,\"\",@progbits";
    DwarfLineSection =    "\t.section\t.debug_line,\"\",@progbits";
    DwarfFrameSection =   "\t.section\t.debug_frame,\"\",@progbits";
    DwarfPubNamesSection ="\t.section\t.debug_pubnames,\"\",@progbits";
    DwarfPubTypesSection ="\t.section\t.debug_pubtypes,\"\",@progbits";
    DwarfStrSection =     "\t.section\t.debug_str,\"\",@progbits";
    DwarfLocSection =     "\t.section\t.debug_loc,\"\",@progbits";
    DwarfARangesSection = "\t.section\t.debug_aranges,\"\",@progbits";
    DwarfRangesSection =  "\t.section\t.debug_ranges,\"\",@progbits";
    DwarfMacInfoSection = "\t.section\t.debug_macinfo,\"\",@progbits";

    // Exceptions handling
    if (!Subtarget->is64Bit())
      SupportsExceptionHandling = true;
    AbsoluteEHSectionOffsets = false;
    DwarfEHFrameSection = "\t.section\t.eh_frame,\"aw\",@progbits";
    DwarfExceptionSection = "\t.section\t.gcc_except_table,\"a\",@progbits";
    break;

  case X86Subtarget::isCygwin:
  case X86Subtarget::isMingw:
    GlobalPrefix = "_";
    LCOMMDirective = "\t.lcomm\t";
    COMMDirectiveTakesAlignment = false;
    HasDotTypeDotSizeDirective = false;
    StaticCtorsSection = "\t.section .ctors,\"aw\"";
    StaticDtorsSection = "\t.section .dtors,\"aw\"";
    HiddenDirective = NULL;
    PrivateGlobalPrefix = "L";  // Prefix for private global symbols
    WeakRefDirective = "\t.weak\t";
    SetDirective = "\t.set\t";

    // Set up DWARF directives
    HasLEB128 = true;  // Target asm supports leb128 directives (little-endian)
    AbsoluteDebugSectionOffsets = true;
    AbsoluteEHSectionOffsets = false;
    SupportsDebugInformation = true;
    DwarfSectionOffsetDirective = "\t.secrel32\t";
    DwarfAbbrevSection =  "\t.section\t.debug_abbrev,\"dr\"";
    DwarfInfoSection =    "\t.section\t.debug_info,\"dr\"";
    DwarfLineSection =    "\t.section\t.debug_line,\"dr\"";
    DwarfFrameSection =   "\t.section\t.debug_frame,\"dr\"";
    DwarfPubNamesSection ="\t.section\t.debug_pubnames,\"dr\"";
    DwarfPubTypesSection ="\t.section\t.debug_pubtypes,\"dr\"";
    DwarfStrSection =     "\t.section\t.debug_str,\"dr\"";
    DwarfLocSection =     "\t.section\t.debug_loc,\"dr\"";
    DwarfARangesSection = "\t.section\t.debug_aranges,\"dr\"";
    DwarfRangesSection =  "\t.section\t.debug_ranges,\"dr\"";
    DwarfMacInfoSection = "\t.section\t.debug_macinfo,\"dr\"";
    break;

  case X86Subtarget::isWindows:
    GlobalPrefix = "_";
    HasDotTypeDotSizeDirective = false;
    break;

  default: break;
  }
  
  if (Subtarget->isFlavorIntel()) {
    GlobalPrefix = "_";
    CommentString = ";";
  
    PrivateGlobalPrefix = "$";
    AlignDirective = "\talign\t";
    ZeroDirective = "\tdb\t";
    ZeroDirectiveSuffix = " dup(0)";
    AsciiDirective = "\tdb\t";
    AscizDirective = 0;
    Data8bitsDirective = "\tdb\t";
    Data16bitsDirective = "\tdw\t";
    Data32bitsDirective = "\tdd\t";
    Data64bitsDirective = "\tdq\t";
    HasDotTypeDotSizeDirective = false;
    
    TextSection = "_text";
    DataSection = "_data";
    JumpTableDataSection = NULL;
    SwitchToSectionDirective = "";
    TextSectionStartSuffix = "\tsegment 'CODE'";
    DataSectionStartSuffix = "\tsegment 'DATA'";
    SectionEndDirectiveSuffix = "\tends\n";
  }

  AssemblerDialect = Subtarget->getAsmFlavor();
}

bool X86TargetAsmInfo::LowerToBSwap(CallInst *CI) const {
  // FIXME: this should verify that we are targetting a 486 or better.  If not,
  // we will turn this bswap into something that will be lowered to logical ops
  // instead of emitting the bswap asm.  For now, we don't support 486 or lower
  // so don't worry about this.
  
  // Verify this is a simple bswap.
  if (CI->getNumOperands() != 2 ||
      CI->getType() != CI->getOperand(1)->getType() ||
      !CI->getType()->isInteger())
    return false;
  
  const IntegerType *Ty = dyn_cast<IntegerType>(CI->getType());
  if (!Ty || Ty->getBitWidth() % 16 != 0)
    return false;
  
  // Okay, we can do this xform, do so now.
  const Type *Tys[] = { Ty, Ty };
  Module *M = CI->getParent()->getParent()->getParent();
  Constant *Int = Intrinsic::getDeclaration(M, Intrinsic::bswap, Tys, 2);
  
  Value *Op = CI->getOperand(1);
  Op = new CallInst(Int, Op, CI->getName(), CI);
  
  CI->replaceAllUsesWith(Op);
  CI->eraseFromParent();
  return true;
}


bool X86TargetAsmInfo::ExpandInlineAsm(CallInst *CI) const {
  InlineAsm *IA = cast<InlineAsm>(CI->getCalledValue());
  std::vector<InlineAsm::ConstraintInfo> Constraints = IA->ParseConstraints();
  
  std::string AsmStr = IA->getAsmString();
  
  // TODO: should remove alternatives from the asmstring: "foo {a|b}" -> "foo a"
  std::vector<std::string> AsmPieces;
  SplitString(AsmStr, AsmPieces, "\n");  // ; as separator?
  
  switch (AsmPieces.size()) {
  default: return false;    
  case 1:
    AsmStr = AsmPieces[0];
    AsmPieces.clear();
    SplitString(AsmStr, AsmPieces, " \t");  // Split with whitespace.
    
    // bswap $0
    if (AsmPieces.size() == 2 && 
        AsmPieces[0] == "bswap" && AsmPieces[1] == "$0") {
      // No need to check constraints, nothing other than the equivalent of
      // "=r,0" would be valid here.
      return LowerToBSwap(CI);
    }
    break;
  case 3:
    if (CI->getType() == Type::Int64Ty && Constraints.size() >= 2 &&
        Constraints[0].Codes.size() == 1 && Constraints[0].Codes[0] == "A" &&
        Constraints[1].Codes.size() == 1 && Constraints[1].Codes[0] == "0") {
      // bswap %eax / bswap %edx / xchgl %eax, %edx  -> llvm.bswap.i64
      std::vector<std::string> Words;
      SplitString(AsmPieces[0], Words, " \t");
      if (Words.size() == 2 && Words[0] == "bswap" && Words[1] == "%eax") {
        Words.clear();
        SplitString(AsmPieces[1], Words, " \t");
        if (Words.size() == 2 && Words[0] == "bswap" && Words[1] == "%edx") {
          Words.clear();
          SplitString(AsmPieces[2], Words, " \t,");
          if (Words.size() == 3 && Words[0] == "xchgl" && Words[1] == "%eax" &&
              Words[2] == "%edx") {
            return LowerToBSwap(CI);
          }
        }
      }
    }
    break;
  }
  return false;
}
