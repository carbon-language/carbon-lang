//===- Configuration.cpp - Configuration Data Mgmt --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Reid Spencer and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the parsing of configuration files for the LLVM Compiler
// Driver (llvmc).
//
//===----------------------------------------------------------------------===//

#include "Configuration.h"
#include "ConfigLexer.h"
#include "CompilerDriver.h"
#include "llvm/Config/config.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/ADT/StringExtras.h"
#include <iostream>
#include <fstream>

using namespace llvm;

namespace sys {
  // From CompilerDriver.cpp (for now)
  extern bool FileIsReadable(const std::string& fname);
}

namespace llvm {
  ConfigLexerInfo ConfigLexerState;
  InputProvider* ConfigLexerInput = 0;

  InputProvider::~InputProvider() {}
  void InputProvider::error(const std::string& msg) {
    std::cerr << name << ":" << ConfigLexerState.lineNum << ": Error: " <<
      msg << "\n";
    errCount++;
  }

  void InputProvider::checkErrors() {
    if (errCount > 0) {
      std::cerr << name << " had " << errCount << " errors. Terminating.\n";
      exit(errCount);
    }
  }

}

namespace {

  class FileInputProvider : public InputProvider {
    public:
      FileInputProvider(const std::string & fname)
        : InputProvider(fname)
        , F(fname.c_str()) {
        ConfigLexerInput = this;
      }
      virtual ~FileInputProvider() { F.close(); ConfigLexerInput = 0; }
      virtual unsigned read(char *buffer, unsigned max_size) {
        if (F.good()) {
          F.read(buffer,max_size);
          if ( F.gcount() ) return F.gcount() - 1;
        }
        return 0;
      }

      bool okay() { return F.good(); }
    private:
      std::ifstream F;
  };

  cl::opt<bool> DumpTokens("dump-tokens", cl::Optional, cl::Hidden,
    cl::init(false), cl::desc("Dump lexical tokens (debug use only)."));

  struct Parser
  {
    Parser() {
      token = EOFTOK;
      provider = 0;
      confDat = 0;
      ConfigLexerState.lineNum = 1;
      ConfigLexerState.in_value = false;
      ConfigLexerState.StringVal.clear();
      ConfigLexerState.IntegerVal = 0;
    };

    ConfigLexerTokens token;
    InputProvider* provider;
    CompilerDriver::ConfigData* confDat;

    inline int next() {
      token = Configlex();
      if (DumpTokens)
        std::cerr << token << "\n";
      return token;
    }

    inline bool next_is_real() {
      next();
      return (token != EOLTOK) && (token != ERRORTOK) && (token != 0);
    }

    inline void eatLineRemnant() {
      while (next_is_real()) ;
    }

    void error(const std::string& msg, bool skip = true) {
      provider->error(msg);
      if (skip)
        eatLineRemnant();
    }

    bool parseCompleteItem(std::string& result) {
      result.clear();
      while (next_is_real()) {
        switch (token ) {
	case LLVMGCCDIR_SUBST:
	case LLVMGCCARCH_SUBST:
          case STRING :
          case OPTION :
            result += ConfigLexerState.StringVal;
            break;
          case SEPARATOR:
            result += ".";
            break;
          case SPACE:
            return true;
          default:
            return false;
        }
      }
      return false;
    }

    std::string parseName() {
      std::string result;
      if (next() == EQUALS) {
        if (parseCompleteItem(result))
          eatLineRemnant();
        if (result.empty())
          error("Name exepected");
      } else
        error("Expecting '='");
      return result;
    }

    bool parseBoolean() {
      bool result = true;
      if (next() == EQUALS) {
        if (next() == SPACE)
          next();
        if (token == FALSETOK) {
          result = false;
        } else if (token != TRUETOK) {
          error("Expecting boolean value");
          return false;
        }
        if (next() != EOLTOK && token != 0) {
          error("Extraneous tokens after boolean");
        }
      }
      else
        error("Expecting '='");
      return result;
    }

    bool parseSubstitution(CompilerDriver::StringVector& optList) {
      switch (token) {
        case ARGS_SUBST:        optList.push_back("%args%"); break;
        case BINDIR_SUBST:      optList.push_back("%bindir%"); break;
        case DEFS_SUBST:        optList.push_back("%defs%"); break;
        case IN_SUBST:          optList.push_back("%in%"); break;
        case INCLS_SUBST:       optList.push_back("%incls%"); break;
        case LIBDIR_SUBST:      optList.push_back("%libdir%"); break;
        case LIBS_SUBST:        optList.push_back("%libs%"); break;
        case OPT_SUBST:         optList.push_back("%opt%"); break;
        case OUT_SUBST:         optList.push_back("%out%"); break;
        case TARGET_SUBST:      optList.push_back("%target%"); break;
        case STATS_SUBST:       optList.push_back("%stats%"); break;
        case TIME_SUBST:        optList.push_back("%time%"); break;
        case VERBOSE_SUBST:     optList.push_back("%verbose%"); break;
        case FOPTS_SUBST:       optList.push_back("%fOpts%"); break;
        case MOPTS_SUBST:       optList.push_back("%Mopts%"); break;
        case WOPTS_SUBST:       optList.push_back("%Wopts%"); break;
        default:
          return false;
      }
      return true;
    }

    void parseOptionList(CompilerDriver::StringVector& optList ) {
      if (next() == EQUALS) {
        while (next_is_real()) {
          if (token == STRING || token == OPTION)
            optList.push_back(ConfigLexerState.StringVal);
          else if (!parseSubstitution(optList)) {
            error("Expecting a program argument or substitution", false);
            break;
          }
        }
      } else
        error("Expecting '='");
    }

    void parseVersion() {
      if (next() != EQUALS)
        error("Expecting '='");
      while (next_is_real()) {
        if (token == STRING || token == OPTION)
          confDat->version = ConfigLexerState.StringVal;
        else
          error("Expecting a version string");
      }
    }

    void parseLibs() {
      if (next() != EQUALS)
        error("Expecting '='");
      std::string lib;
      while (parseCompleteItem(lib)) {
        if (!lib.empty()) {
          confDat->libpaths.push_back(lib);
        }
      }
    }

    void parseLang() {
      if (next() != SEPARATOR)
        error("Expecting '.'");
      switch (next() ) {
        case LIBS:
          parseLibs();
          break;
        case NAME:
          confDat->langName = parseName();
          break;
        case OPT1:
          parseOptionList(confDat->opts[CompilerDriver::OPT_FAST_COMPILE]);
          break;
        case OPT2:
          parseOptionList(confDat->opts[CompilerDriver::OPT_SIMPLE]);
          break;
        case OPT3:
          parseOptionList(confDat->opts[CompilerDriver::OPT_AGGRESSIVE]);
          break;
        case OPT4:
          parseOptionList(confDat->opts[CompilerDriver::OPT_LINK_TIME]);
          break;
        case OPT5:
          parseOptionList(
            confDat->opts[CompilerDriver::OPT_AGGRESSIVE_LINK_TIME]);
          break;
        default:
          error("Expecting 'name' or 'optN' after 'lang.'");
          break;
      }
    }

    bool parseProgramName(std::string& str) {
      str.clear();
      do {
        switch (token) {
	case BINDIR_SUBST:
	case LLVMGCC_SUBST:
	case LLVMGXX_SUBST:
	case LLVMCC1_SUBST:
	case LLVMCC1PLUS_SUBST:
          case OPTION:
          case STRING:
          case ARGS_SUBST:
          case DEFS_SUBST:
          case IN_SUBST:
          case INCLS_SUBST:
          case LIBS_SUBST:
          case OPT_SUBST:
          case OUT_SUBST:
          case STATS_SUBST:
          case TARGET_SUBST:
          case TIME_SUBST:
          case VERBOSE_SUBST:
          case FOPTS_SUBST:
          case MOPTS_SUBST:
          case WOPTS_SUBST:
            str += ConfigLexerState.StringVal;
            break;
          case SEPARATOR:
            str += ".";
            break;
          case ASSEMBLY:
            str += "assembly";
            break;
          case BYTECODE:
            str += "bytecode";
            break;
          case TRUETOK:
            str += "true";
            break;
          case FALSETOK:
            str += "false";
            break;
          default:
            break;
        }
        next();
      } while (token != SPACE && token != EOFTOK && token != EOLTOK &&
               token != ERRORTOK);
      return !str.empty();
    }

    void parseCommand(CompilerDriver::Action& action) {
      if (next() != EQUALS)
        error("Expecting '='");
      switch (next()) {
        case EOLTOK:
          // no value (valid)
          action.program.clear();
          action.args.clear();
          break;
        case SPACE:
          next();
          /* FALL THROUGH */
        default:
        {
          std::string progname;
          if (parseProgramName(progname))
            action.program.set(progname);
          else
            error("Expecting a program name");

          // Get the options
          std::string anOption;
          while (next_is_real()) {
            switch (token) {
              case STRING:
              case OPTION:
                anOption += ConfigLexerState.StringVal;
                break;
              case ASSEMBLY:
                anOption += "assembly";
                break;
              case BYTECODE:
                anOption += "bytecode";
                break;
              case TRUETOK:
                anOption += "true";
                break;
              case FALSETOK:
                anOption += "false";
                break;
              case SEPARATOR:
                anOption += ".";
                break;
              case SPACE:
                action.args.push_back(anOption);
                anOption.clear();
                break;
              default:
                if (!parseSubstitution(action.args))
                  error("Expecting a program argument or substitution", false);
                break;
            }
          }
        }
      }
    }

    void parsePreprocessor() {
      if (next() != SEPARATOR)
        error("Expecting '.'");
      switch (next()) {
        case COMMAND:
          parseCommand(confDat->PreProcessor);
          break;
        case REQUIRED:
          if (parseBoolean())
            confDat->PreProcessor.set(CompilerDriver::REQUIRED_FLAG);
          else
            confDat->PreProcessor.clear(CompilerDriver::REQUIRED_FLAG);
          break;
        default:
          error("Expecting 'command' or 'required' but found '" +
              ConfigLexerState.StringVal);
          break;
      }
    }

    bool parseOutputFlag() {
      if (next() == EQUALS) {
        if (next() == SPACE)
          next();
        if (token == ASSEMBLY) {
          return true;
        } else if (token == BYTECODE) {
          return false;
        } else {
          error("Expecting output type value");
          return false;
        }
        if (next() != EOLTOK && token != 0) {
          error("Extraneous tokens after output value");
        }
      }
      else
        error("Expecting '='");
      return false;
    }

    void parseTranslator() {
      if (next() != SEPARATOR)
        error("Expecting '.'");
      switch (next()) {
        case COMMAND:
          parseCommand(confDat->Translator);
          break;
        case REQUIRED:
          if (parseBoolean())
            confDat->Translator.set(CompilerDriver::REQUIRED_FLAG);
          else
            confDat->Translator.clear(CompilerDriver::REQUIRED_FLAG);
          break;
        case PREPROCESSES:
          if (parseBoolean())
            confDat->Translator.set(CompilerDriver::PREPROCESSES_FLAG);
          else
            confDat->Translator.clear(CompilerDriver::PREPROCESSES_FLAG);
          break;
        case OUTPUT:
          if (parseOutputFlag())
            confDat->Translator.set(CompilerDriver::OUTPUT_IS_ASM_FLAG);
          else
            confDat->Translator.clear(CompilerDriver::OUTPUT_IS_ASM_FLAG);
          break;

        default:
          error("Expecting 'command', 'required', 'preprocesses', or "
                "'output' but found '" + ConfigLexerState.StringVal +
                "' instead");
          break;
      }
    }

    void parseOptimizer() {
      if (next() != SEPARATOR)
        error("Expecting '.'");
      switch (next()) {
        case COMMAND:
          parseCommand(confDat->Optimizer);
          break;
        case PREPROCESSES:
          if (parseBoolean())
            confDat->Optimizer.set(CompilerDriver::PREPROCESSES_FLAG);
          else
            confDat->Optimizer.clear(CompilerDriver::PREPROCESSES_FLAG);
          break;
        case TRANSLATES:
          if (parseBoolean())
            confDat->Optimizer.set(CompilerDriver::TRANSLATES_FLAG);
          else
            confDat->Optimizer.clear(CompilerDriver::TRANSLATES_FLAG);
          break;
        case REQUIRED:
          if (parseBoolean())
            confDat->Optimizer.set(CompilerDriver::REQUIRED_FLAG);
          else
            confDat->Optimizer.clear(CompilerDriver::REQUIRED_FLAG);
          break;
        case OUTPUT:
          if (parseOutputFlag())
            confDat->Translator.set(CompilerDriver::OUTPUT_IS_ASM_FLAG);
          else
            confDat->Translator.clear(CompilerDriver::OUTPUT_IS_ASM_FLAG);
          break;
        default:
          error(std::string("Expecting 'command', 'preprocesses', "
              "'translates' or 'output' but found '") +
              ConfigLexerState.StringVal + "' instead");
          break;
      }
    }

    void parseAssembler() {
      if (next() != SEPARATOR)
        error("Expecting '.'");
      switch(next()) {
        case COMMAND:
          parseCommand(confDat->Assembler);
          break;
        default:
          error("Expecting 'command'");
          break;
      }
    }

    void parseLinker() {
      if (next() != SEPARATOR)
        error("Expecting '.'");
      switch(next()) {
        case LIBS:
          break; //FIXME
        case LIBPATHS:
          break; //FIXME
        default:
          error("Expecting 'libs' or 'libpaths'");
          break;
      }
    }

    void parseAssignment() {
      switch (token) {
        case VERSION_TOK:   parseVersion(); break;
        case LANG:          parseLang(); break;
        case PREPROCESSOR:  parsePreprocessor(); break;
        case TRANSLATOR:    parseTranslator(); break;
        case OPTIMIZER:     parseOptimizer(); break;
        case ASSEMBLER:     parseAssembler(); break;
        case LINKER:        parseLinker(); break;
        case EOLTOK:        break; // just ignore
        case ERRORTOK:
        default:
          error("Invalid top level configuration item");
          break;
      }
    }

    void parseFile() {
      while ( next() != EOFTOK ) {
        if (token == ERRORTOK)
          error("Invalid token");
        else if (token != EOLTOK)
          parseAssignment();
      }
      provider->checkErrors();
    }
  };

void
ParseConfigData(InputProvider& provider, CompilerDriver::ConfigData& confDat) {
  Parser p;
  p.token = EOFTOK;
  p.provider = &provider;
  p.confDat = &confDat;
  p.parseFile();
  }

}

CompilerDriver::ConfigData*
LLVMC_ConfigDataProvider::ReadConfigData(const std::string& ftype) {
  CompilerDriver::ConfigData* result = 0;
  sys::Path confFile;
  if (configDir.isEmpty()) {
    // Try the environment variable
    const char* conf = getenv("LLVM_CONFIG_DIR");
    if (conf) {
      confFile.set(conf);
      confFile.appendComponent(ftype);
      if (!confFile.canRead())
        throw std::string("Configuration file for '") + ftype +
                          "' is not available.";
    } else {
      // Try the user's home directory
      confFile = sys::Path::GetUserHomeDirectory();
      if (!confFile.isEmpty()) {
        confFile.appendComponent(".llvm");
        confFile.appendComponent("etc");
        confFile.appendComponent(ftype);
        if (!confFile.canRead())
          confFile.clear();
      }
      if (confFile.isEmpty()) {
        // Okay, try the LLVM installation directory
        confFile = sys::Path::GetLLVMConfigDir();
        confFile.appendComponent(ftype);
        if (!confFile.canRead()) {
          // Okay, try the "standard" place
          confFile = sys::Path::GetLLVMDefaultConfigDir();
          confFile.appendComponent(ftype);
          if (!confFile.canRead()) {
            throw std::string("Configuration file for '") + ftype +
                              "' is not available.";
          }
        }
      }
    }
  } else {
    confFile = configDir;
    confFile.appendComponent(ftype);
    if (!confFile.canRead())
      throw std::string("Configuration file for '") + ftype +
                        "' is not available.";
  }
  FileInputProvider fip( confFile.toString() );
  if (!fip.okay()) {
    throw std::string("Configuration file for '") + ftype +
                      "' is not available.";
  }
  result = new CompilerDriver::ConfigData();
  ParseConfigData(fip,*result);
  return result;
}

LLVMC_ConfigDataProvider::~LLVMC_ConfigDataProvider()
{
  ConfigDataMap::iterator cIt = Configurations.begin();
  while (cIt != Configurations.end()) {
    CompilerDriver::ConfigData* cd = cIt->second;
    ++cIt;
    delete cd;
  }
  Configurations.clear();
}

CompilerDriver::ConfigData*
LLVMC_ConfigDataProvider::ProvideConfigData(const std::string& filetype) {
  CompilerDriver::ConfigData* result = 0;
  if (!Configurations.empty()) {
    ConfigDataMap::iterator cIt = Configurations.find(filetype);
    if ( cIt != Configurations.end() ) {
      // We found one in the case, return it.
      result = cIt->second;
    }
  }
  if (result == 0) {
    // The configuration data doesn't exist, we have to go read it.
    result = ReadConfigData(filetype);
    // If we got one, cache it
    if (result != 0)
      Configurations.insert(std::make_pair(filetype,result));
  }
  return result; // Might return 0
}
