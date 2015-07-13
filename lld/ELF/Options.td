include "llvm/Option/OptParser.td"

//===----------------------------------------------------------------------===//
/// Utility Functions
//===----------------------------------------------------------------------===//
// Single and multiple dash options combined
multiclass smDash<string opt1, string opt2, string help> {
  // Option
  def "" : Separate<["-"], opt1>, HelpText<help>;
  def opt1_eq : Joined<["-"], opt1#"=">,
      Alias<!cast<Option>(opt1)>;
  // Compatibility aliases
  def opt2_dashdash : Separate<["--"], opt2>,
      Alias<!cast<Option>(opt1)>;
  def opt2_dashdash_eq : Joined<["--"], opt2#"=">,
      Alias<!cast<Option>(opt1)>;
}

// Support -<option>,-<option>=
multiclass dashEq<string opt1, string opt2, string help> {
  // Option
  def "" : Separate<["-"], opt1>, HelpText<help>;
  // Compatibility aliases
  def opt2_eq : Joined<["-"], opt2#"=">,
      Alias<!cast<Option>(opt1)>;
}

// Support --<option>,--<option>=
multiclass mDashEq<string opt1, string help> {
  // Option
  def "" : Separate<["--"], opt1>, HelpText<help>;
  // Compatibility aliases
  def opt2_eq : Joined<["--"], opt1#"=">,
      Alias<!cast<Option>(opt1)>;
}

def output : Separate<["-"], "o">, MetaVarName<"<path>">,
     HelpText<"Path to file to write output">;

def L : Joined<["-"], "L">, MetaVarName<"<dir>">,
     HelpText<"Directory to search for libraries">;
def l : Joined<["-"], "l">, MetaVarName<"<libName>">,
     HelpText<"Root name of library to use">;
def noinhibit_exec : Flag<["--"], "noinhibit-exec">,
     HelpText<"Retain the executable output file whenever"
              " it is still usable">;
defm e : smDash<"e", "entry",
      "Name of entry point symbol">;

//===----------------------------------------------------------------------===//
/// Help
//===----------------------------------------------------------------------===//
def help : Flag<["--"], "help">,
     HelpText<"Display this help message">;
