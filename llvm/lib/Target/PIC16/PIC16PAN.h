//===-- PIC16PAN.h - PIC16 ABI Naming conventions --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source 
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the entry points for global functions defined in 
// the LLVM PIC16 back-end.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_PIC16PAN_H
#define LLVM_TARGET_PIC16PAN_H

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Function.h"
#include <iosfwd>
#include <cassert>
#include <cstring>
#include <string>

namespace llvm {
  // A Central class to manage all ABI naming conventions.
  // PAN - [P]ic16 [A]BI [N]ames
  class PAN {
    public:
    // Map the name of the symbol to its section name.
    // Current ABI:
    // -----------------------------------------------------
    // ALL Names are prefixed with the symobl '@'.
    // ------------------------------------------------------
    // Global variables do not have any '.' in their names.
    // These are maily function names and global variable names.
    // Example - @foo,  @i
    // -------------------------------------------------------
    // Functions and auto variables.
    // Names are mangled as <prefix><funcname>.<tag>.<varname>
    // Where <prefix> is '@' and <tag> is any one of
    // the following
    // .auto. - an automatic var of a function.
    // .temp. - temproray data of a function.
    // .ret.  - return value label for a function.
    // .frame. - Frame label for a function where retval, args
    //           and temps are stored.
    // .args. - Label used to pass arguments to a direct call.
    // Example - Function name:   @foo
    //           Its frame:       @foo.frame.
    //           Its retval:      @foo.ret.
    //           Its local vars:  @foo.auto.a
    //           Its temp data:   @foo.temp.
    //           Its arg passing: @foo.args.
    //----------------------------------------------
    // Libcall - compiler generated libcall names must start with .lib.
    //           This id will be used to emit extern decls for libcalls.
    // Example - libcall name:   @.lib.sra.i8
    //           To pass args:   @.lib.sra.i8.args.
    //           To return val:  @.lib.sra.i8.ret.
    //----------------------------------------------
    // SECTION Names
    // uninitialized globals - @udata.<num>.#
    // initialized globals - @idata.<num>.#
    // Function frame - @<func>.frame_section.
    // Function autos - @<func>.autos_section.
    // Declarations - Enclosed in comments. No section for them.
    //----------------------------------------------------------
    
    // Tags used to mangle different names. 
    enum TAGS {
      PREFIX_SYMBOL,
      GLOBAL,
      STATIC_LOCAL,
      AUTOS_LABEL,
      FRAME_LABEL,
      RET_LABEL,
      ARGS_LABEL,
      TEMPS_LABEL,
      
      LIBCALL,
      
      FRAME_SECTION,
      AUTOS_SECTION,
      CODE_SECTION
    };
    enum CallLine {
      MainLine,
      InterruptLine,
      SharedLine,
      UnspecifiedLine
    };

    // Textual names of the tags.
    inline static const char *getTagName(TAGS tag) {
      switch (tag) {
      default: return "";
      case PREFIX_SYMBOL:    return "@";
      case AUTOS_LABEL:       return ".auto.";
      case FRAME_LABEL:       return ".frame.";
      case TEMPS_LABEL:       return ".temp.";
      case ARGS_LABEL:       return ".args.";
      case RET_LABEL:       return ".ret.";
      case LIBCALL:       return ".lib.";
      case FRAME_SECTION:       return ".frame_section.";
      case AUTOS_SECTION:       return ".autos_section.";
      case CODE_SECTION:       return ".code_section.";
      }
    }

    inline static bool isISR(const Function *F) {
       if (F->getSection().find("interrupt") != std::string::npos)
         return true; 

       return false;
    } 
    inline static bool isInterruptLineFunction(const Function *F) {
       if (isISR(F)) return true;
       if (F->getSection().find("IL") != std::string::npos)
         return true; 

       return false;
    }
    inline static bool isMainLineFunction(const Function *F) {
       if (F->getSection().find("ML") != std::string::npos)
         return true; 

       return false;
    }
    inline static bool isSharedLineFunction(const Function *F) {
       if (F->getSection().find("SL") != std::string::npos)
         return true; 

       return false;
    }

    inline static const char *getUpdatedLibCallDecl(const char *Name, 
                                              const Function *F) {
       // If the current function is not an interrupt line function then 
       // there is no need to change the name.
       if (!isInterruptLineFunction(F))
          return Name;


       // CAUTION::This code may cause some memory leak and at times
       // use more memory than required.
       // (We will try to clean it sometime later)

       // InterruptLine functions should hold ".IL" suffix and
       char *NewName = (char *)malloc(strlen(Name) + 3 + 1);
       strcpy(NewName, Name);
       strcat(NewName, ".IL");
       return NewName;
    }

    inline static void updateCallLineAutos(std::string &Sym, std::string FuncName) {
      // If the function has ".IL" in its name then it must be
      // a cloned function and autos for such a function should also
      // have ".IL" in their name. So update them here.
      if (FuncName.find(".IL") != std::string::npos)
          Sym.replace(Sym.find(".auto"), 5, ".IL.auto");
    }

    // Insert ".IL" at proper location in the ARGS and RET symbols
    inline static void updateCallLineLibcall(std::string &Sym) {
       std::string SubStr;
       std::size_t pos;
       std::string Suffix;
       if (getSymbolTag(Sym) == ARGS_LABEL) {
          Suffix = getTagName(ARGS_LABEL);
          pos = Sym.find(Suffix);
          SubStr = Sym.substr(0,pos);
       } else if (getSymbolTag(Sym) == RET_LABEL) {
          Suffix = getTagName(RET_LABEL);
          pos = Sym.find(Suffix);
          SubStr = Sym.substr(0,pos);
       } else {
          SubStr = Sym;
          Suffix = "";
       }
       Sym = SubStr + ".IL" + Suffix;
    }

    inline static void updateCallLineSymbol(std::string &Sym, CallLine CLine) {
      if (isLIBCALLSymbol(Sym) && CLine == InterruptLine) {
         updateCallLineLibcall(Sym);
         return;
      }
         
       // UnMangle the function name - mangled in llvm-ld
       // For MainLine function the shared string should be removed
       // and for InterruptLine function the shared string should be 
       // replaced with ".IL"
       std::string ReplaceString="";
       if (CLine == MainLine || CLine == UnspecifiedLine)
         ReplaceString = "";
       else if (CLine == InterruptLine)
         ReplaceString = ".IL";
       std::string FindString = ".shared";
       if (Sym.find(FindString) != std::string::npos) {
         Sym.replace(Sym.find(FindString), FindString.length(), ReplaceString);
         return;
       }
    } 

    inline static void updateCallLineSymbol(std::string &Sym, 
                                            const Function *F) {
       // If it is an auto symbol then it should be updated accordingly
       if (Sym.find(".auto") != std::string::npos) {
         updateCallLineAutos(Sym, F->getName().str());
         return;
       } 

       if (isMainLineFunction(F))
         updateCallLineSymbol(Sym, MainLine);
       else if (isInterruptLineFunction(F))
         updateCallLineSymbol(Sym, InterruptLine);
       else
         updateCallLineSymbol(Sym, UnspecifiedLine);
    }

    inline static bool isLIBCALLSymbol(const std::string &Sym) {
      if (Sym.find(getTagName(LIBCALL)) != std::string::npos)
        return true;
     
      return false;
    }

    // Get tag type for the Symbol.
    inline static TAGS getSymbolTag(const std::string &Sym) {
      if (Sym.find(getTagName(TEMPS_LABEL)) != std::string::npos)
        return TEMPS_LABEL;

      if (Sym.find(getTagName(FRAME_LABEL)) != std::string::npos)
        return FRAME_LABEL;

      if (Sym.find(getTagName(RET_LABEL)) != std::string::npos)
        return RET_LABEL;

      if (Sym.find(getTagName(ARGS_LABEL)) != std::string::npos)
        return ARGS_LABEL;

      if (Sym.find(getTagName(AUTOS_LABEL)) != std::string::npos)
        return AUTOS_LABEL;

      if (Sym.find(getTagName(LIBCALL)) != std::string::npos)
        return LIBCALL;

      // It does not have any Tag. So its a true global or static local.
      if (Sym.find(".") == std::string::npos) 
        return GLOBAL;
      
      // If a . is there, then it may be static local.
      // We should mangle these as well in clang.
      if (Sym.find(".") != std::string::npos) 
        return STATIC_LOCAL;
 
      assert (0 && "Could not determine Symbol's tag");
      return PREFIX_SYMBOL; // Silence warning when assertions are turned off.
    }

    // addPrefix - add prefix symbol to a name if there isn't one already.
    inline static std::string addPrefix (const std::string &Name) {
      std::string prefix = getTagName (PREFIX_SYMBOL);

      // If this name already has a prefix, nothing to do.
      if (Name.compare(0, prefix.size(), prefix) == 0)
        return Name;

      return prefix + Name;
    }

    // Get mangled func name from a mangled sym name.
    // In all cases func name is the first component before a '.'.
    static inline std::string getFuncNameForSym(const std::string &Sym1) {
      assert (getSymbolTag(Sym1) != GLOBAL && "not belongs to a function");

      std::string Sym = addPrefix(Sym1);

      // Position of the . after func name. That's where func name ends.
      size_t func_name_end = Sym.find ('.');

      return Sym.substr (0, func_name_end);
    }

    // Get Frame start label for a func.
    static std::string getFrameLabel(const std::string &Func) {
      std::string Func1 = addPrefix(Func);
      std::string tag = getTagName(FRAME_LABEL);
      return Func1 + tag;
    }

    static std::string getRetvalLabel(const std::string &Func) {
      std::string Func1 = addPrefix(Func);
      std::string tag = getTagName(RET_LABEL);
      return Func1 + tag;
    }

    static std::string getArgsLabel(const std::string &Func) {
      std::string Func1 = addPrefix(Func);
      std::string tag = getTagName(ARGS_LABEL);
      return Func1 + tag;
    }

    static std::string getTempdataLabel(const std::string &Func) {
      std::string Func1 = addPrefix(Func);
      std::string tag = getTagName(TEMPS_LABEL);
      return Func1 + tag;
    }

    static std::string getFrameSectionName(const std::string &Func) {
      std::string Func1 = addPrefix(Func);
      std::string tag = getTagName(FRAME_SECTION);
      return Func1 + tag + "# UDATA_OVR";
    }

    static std::string getAutosSectionName(const std::string &Func) {
      std::string Func1 = addPrefix(Func);
      std::string tag = getTagName(AUTOS_SECTION);
      return Func1 + tag + "# UDATA_OVR";
    }

    static std::string getCodeSectionName(const std::string &Func, bool isInterrupt) {
      std::string Func1 = addPrefix(Func);
      std::string tag = getTagName(CODE_SECTION);
      std::string Name = Func1 + tag + "# CODE";

      // If this is an interrupt function then the code section should
      // be placed at address 0x4 (hard)
      if (isInterrupt)
         Name += "    0x4";

      return Name;
    }

    // udata, romdata and idata section names are generated by a given number.
    // @udata.<num>.# 
    static std::string getUdataSectionName(unsigned num, 
                                           std::string prefix = "") {
       std::ostringstream o;
       o << getTagName(PREFIX_SYMBOL) << prefix << "udata." << num 
         << ".# UDATA"; 
       return o.str(); 
    }

    static std::string getRomdataSectionName(unsigned num,
                                             std::string prefix = "") {
       std::ostringstream o;
       o << getTagName(PREFIX_SYMBOL) << prefix << "romdata." << num 
         << ".# ROMDATA";
       return o.str();
    }

    static std::string getIdataSectionName(unsigned num,
                                           std::string prefix = "") {
       std::ostringstream o;
       o << getTagName(PREFIX_SYMBOL) << prefix << "idata." << num 
         << ".# IDATA"; 
       return o.str(); 
    }

    inline static bool isLocalName (const std::string &Name) {
      if (getSymbolTag(Name) == AUTOS_LABEL)
        return true;

      return false;
    }

    inline static bool isMemIntrinsic (const std::string &Name) {
      if (Name.compare("@memcpy") == 0 || Name.compare("@memset") == 0 ||
          Name.compare("@memmove") == 0) {
        return true;
      }
      
      return false;
    }

    inline static bool isLocalToFunc (std::string &Func, std::string &Var) {
      if (! isLocalName(Var)) return false;

      std::string Func1 = addPrefix(Func);
      // Extract func name of the varilable.
      const std::string &fname = getFuncNameForSym(Var);

      if (fname.compare(Func1) == 0)
        return true;

      return false;
    }


    // Get the section for the given external symbol names.
    // This tries to find the type (Tag) of the symbol from its mangled name
    // and return appropriate section name for it.
    static inline std::string getSectionNameForSym(const std::string &Sym1) {
      std::string Sym = addPrefix(Sym1);

      std::string SectionName;
 
      std::string Fname = getFuncNameForSym (Sym);
      TAGS id = getSymbolTag (Sym);

      switch (id) {
        default : assert (0 && "Could not determine external symbol type");
        case FRAME_LABEL:
        case RET_LABEL:
        case TEMPS_LABEL:
        case ARGS_LABEL:  {
          return getFrameSectionName(Fname);
        }
        case AUTOS_LABEL: {
          return getAutosSectionName(Fname);
        }
      }
    }

    inline static std::string getAutosSectionForColor(std::string Color) {
      return Color.append("_AUTOS");
    }

  }; // class PAN.

} // end namespace llvm;

#endif
