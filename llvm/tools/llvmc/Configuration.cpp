//===- ConfigData.cpp - Configuration Data Mgmt -----------------*- C++ -*-===//
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
//===------------------------------------------------------------------------===

#include "ConfigData.h"
#include "CompilerDriver.h"
#include "Support/StringExtras.h"
#include <iostream>

using namespace llvm;

namespace {

// This array of strings provides the input for ".ll" files (LLVM Assembly)
// to the configuration file parser. This data is just "built-in" to 
// llvmc so it doesn't have to be read from a configuration file.
static const char* LL_Data[] = {
  "lang.name=LLVM Assembly", 
  "lang.translator.preprocesses=false",
  "lang.translator.optimizes=No",
  "lang.translator.groks_dash_O=No",
  "lang.preprocessor.needed=0",
  "preprocessor.prog=",
  "preprocessor.args=",
  "translator.prog=llvm-as",
  "translator.args=@in@ -o @out@",
  "optimizer.prog=opt",
  "optimizer.args=@in@ -o @out@",
  "assembler.prog=llc",
  "assembler.args=@in@ -o @out@",
  "linker.prog=llvm-link",
  "linker.args=@in@ -o @out@"
};

// This array of strings provides the input for ".st" files (Stacker).
static const char* ST_Data[] = {
  "lang.name=Stacker", 
  "lang.translator.preprocesses=false",
  "lang.translator.optimizes=true",
  "lang.translator.groks_dash_O=0",
  "lang.preprocessor.needed=0",
  "preprocessor.prog=cp",
  "preprocessor.args=@in@ @out@",
  "translator.prog=stkrc",
  "translator.args=@in@ -o @out@ -S 2048",
  "optimizer.prog=opt",
  "optimizer.args=@in@ -o @out@",
  "assembler.prog=llc",
  "assembler.args=@in@ -o @out@",
  "linker.prog=llvm-link",
  "linker.args=@in@ -o @out@"
};

class InputProvider {
  public:
    virtual bool getLine(std::string& line) = 0;
    virtual void error(const std::string& msg) = 0;
    virtual bool errorOccurred() = 0;
};

class StaticInputProvider : public InputProvider {
  public:
    StaticInputProvider(const char *data[], size_t count,
        const std::string& nam) { 
      TheData = data; 
      limit = count;
      where = 0;
      name = nam;
      errCount = 0;
    }
    virtual ~StaticInputProvider() {}
    virtual bool getLine(std::string& line) {
      if ( where >= limit ) return false;
      line = TheData[where++];
      return true;
    }

    virtual void error(const std::string& msg) {
      std::cerr << name << ":" << where << ": Error: " << msg << "\n";
      errCount++;
    }

    virtual bool errorOccurred() { return errCount > 0; };

  private:
    const char**TheData;
    size_t limit;
    size_t where;
    std::string name;
    size_t errCount;
};

inline bool recognize(const char*& p, const char*token) {
  while (*p == *token && *token != '\0')
    ++token, p++;
  return *token == '\0' && ((*p == '\0') || ( *p == '.' ) || (*p == '='));
}

inline bool getBoolean(const std::string& value) {
  switch (value[0]) {
    case 't':
    case 'T':
    case '1':
    case 'y':
    case 'Y':
      return true;
    default :
      return false;
  }
  return false;
}

inline void skipWhitespace( size_t& pos, const std::string& line ) {
  while (pos < line.size() && (
           line[pos] == ' ' ||  // Space
           line[pos] == '\t' || // Horizontal Tab
           line[pos] == '\n' || // New Line
           line[pos] == '\v' || // Vertical Tab
           line[pos] == '\f' || // Form Feed
           line[pos] == '\r')   // Carriate Return
        )
      pos++;
}

inline void parseArgs(CompilerDriver::Action& pat, 
                      const std::string& value, 
                      InputProvider& provider )
{
  const char* p = value.c_str();
  const char* argStart = p;
  while (*p != '\0') {
    switch (*p) {
      case ' ':
        if (argStart != p)
          pat.args.push_back(std::string(argStart, p-argStart));
        argStart = ++p;
        break;
      case '@' : 
        {
          if (argStart != p)
            pat.args.push_back(std::string(argStart,p-argStart));
          const char* token = ++p;
          while (*p != '@' && *p != 0) 
            p++;
          if ( *p != '@' ) {
            provider.error("Unterminated substitution token");
            return;
          } else {
            p++;
            bool legal = false;
            switch (token[0]) {
              case 'i':
                if (token[1] == 'n' && token[2] == '@' ) {
                  pat.inputAt = pat.args.size();
                  pat.args.push_back("in");
                  legal = true; 
                  argStart = p;
                }
                break;
              case 'o':
                if (token[1] == 'u' && token[2] == 't' && token[3] == '@') {
                  pat.outputAt = pat.args.size();
                  pat.args.push_back("out");
                  legal = true;
                  argStart = p;
                }
                break;
              default:
                break;
            }
            if (!legal) {
              provider.error("Invalid substitution token");
              return;
            }
          }
        }
        break;
      default :
        p++;
        break;
    }
  }
}

CompilerDriver::ConfigData*
ParseConfigData(InputProvider& provider) {
  std::string line;
  CompilerDriver::ConfigData data;
  while ( provider.getLine(line) ) {
    // Check line length first
    size_t lineLen = line.size();
    if (lineLen > 4096)
      provider.error("length of input line (" + utostr(lineLen) + 
                     ") is too long");

    // First, skip whitespace
    size_t stPos = 0;
    skipWhitespace(stPos, line);

    // See if there's a hash mark. It and everything after it is 
    // ignored so lets delete that now.
    size_t hashPos = line.find('#');
    if (hashPos != std::string::npos)
      line.erase(hashPos);

    // Make sure we have something left to parse
    if (line.size() == 0)
      continue; // ignore full-line comment or whitespace line

    // Find the equals sign
    size_t eqPos = line.find('=');
    if (eqPos == std::string::npos) 
      provider.error("Configuration directive is missing an =");

    // extract the item name
    std::string name(line, stPos, eqPos-stPos);

    // directives without names are illegal
    if (name.empty())
      provider.error("Configuration directive name is empty");

    // Skip whitespace in the value
    size_t valPos = eqPos + 1;
    skipWhitespace(valPos, line);

    // Skip white space at end of value
    size_t endPos = line.length() - 1;
    while (line[endPos] == ' ') 
      endPos--;
 
    // extract the item value
    std::string value(line, valPos, endPos-valPos+1);

    // Get the configuration item as a char pointer
    const char*p = name.c_str();

    // Indicate we haven't found an invalid item yet.
    bool invalidItem = false;

    // Parse the contents by examining first character and
    // using the recognize function strategically
    switch (*p++) {
      case 'l' :
        // could it be "lang."
        if (*p == 'a') {         // "lang." ?
          if (recognize(p,"ang")) {
            p++;
            switch (*p++) {
              case 'n':
                if (recognize(p,"ame"))
                  data.langName = value;
                else
                  invalidItem = true;
                break;
              case 't':
                if (recognize(p,"ranslator")) {
                  p++;
                  if (recognize(p,"preprocesses")) 
                    data.TranslatorPreprocesses = getBoolean(value);
                  else if (recognize(p, "optimizes")) 
                    data.TranslatorOptimizes = getBoolean(value);
                  else if (recognize(p, "groks_dash_O"))
                    data.TranslatorGroksDashO = getBoolean(value);
                  else
                    invalidItem = true;
                }
                else
                  invalidItem = true;
                break;
              case 'p':
                if (recognize(p,"reprocessor")) {
                  p++;
                  if (recognize(p,"needed")) {
                    data.PreprocessorNeeded = getBoolean(value);
                  } else
                    invalidItem = true;
                }
                else
                  invalidItem = true;
                break;
                  
              default:
                invalidItem = true;
                break;
            }
          }
        } else if (*p == 'i') {  // "linker." ?
          if (recognize(p,"inker")) {
            p++;
            if (recognize(p,"prog"))
              data.Linker.program = value;
            else if (recognize(p,"args"))
              parseArgs(data.Linker,value,provider);
            else
              invalidItem = true;
          }
          else
            invalidItem = true;
        } else {
          invalidItem = true;
        }
        break;

      case 'p' :
        if (*p == 'r') {         // "preprocessor." ?
          if (recognize(p, "reprocessor")) {
            p++;
            if (recognize(p,"prog")) 
              data.PreProcessor.program = value;
            else if (recognize(p,"args"))
              parseArgs(data.PreProcessor,value,provider);
            else
              invalidItem = true;
          } else
            invalidItem = true;
        } else {
          invalidItem = true;
        }
        break;

      case 't' :
        if (*p == 'r') {         // "translator." ?
          if (recognize(p, "ranslator")) {
            p++;
            if (recognize(p,"prog")) 
              data.Translator.program = value;
            else if (recognize(p,"args"))
              parseArgs(data.Translator,value,provider);
            else
              invalidItem = true;
          } else
            invalidItem = true;
        } else {
          invalidItem = true;
        }
        break;

      case 'o' :
        if (*p == 'p') {         // "optimizer." ?
          if (recognize(p, "ptimizer")) {
            p++;
            if (recognize(p,"prog")) 
              data.Optimizer.program = value;
            else if (recognize(p,"args"))
              parseArgs(data.Optimizer,value,provider);
            else
              invalidItem = true;
          } else
            invalidItem = true;
        } else {
          invalidItem = true;
        }
        break;
      case 'a' :
        if (*p == 's') {         // "assembler." ?
          if (recognize(p, "ssembler")) {
            p++;
            if (recognize(p,"prog")) 
              data.Assembler.program = value;
            else if (recognize(p,"args"))
              parseArgs(data.Assembler,value,provider);
            else
              invalidItem = true;
          } else
            invalidItem = true;
        } else {
          invalidItem = true;
        }
        break;
    }
    if (invalidItem)
      provider.error("Invalid configuration item: " + line.substr(stPos, eqPos-stPos));
  }
  return new CompilerDriver::ConfigData(data);
}

CompilerDriver::ConfigData*
ReadConfigData(const std::string& ftype) {
  if ( ftype == "ll" ) {
    StaticInputProvider sip(LL_Data, sizeof(LL_Data)/sizeof(LL_Data[0]), 
      "LLVM Assembly (internal)");
    return ParseConfigData(sip);
  } else if (ftype == "st") {
    StaticInputProvider sip(ST_Data, sizeof(ST_Data)/sizeof(ST_Data[0]),
      "Stacker (internal)");
    return ParseConfigData(sip);
  }
  return 0;
}

}

LLVMC_ConfigDataProvider::LLVMC_ConfigDataProvider() 
  : Configurations() 
  , configDir() 
{
  Configurations.clear();
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
    if ( result != 0 )
      Configurations.insert(std::make_pair(filetype,result));
  }
  return result; // Might return 0
}

// vim: sw=2 smartindent smarttab tw=80 autoindent expandtab
