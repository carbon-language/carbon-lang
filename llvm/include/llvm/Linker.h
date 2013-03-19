//===- llvm/Linker.h - Module Linker Interface ------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LINKER_H
#define LLVM_LINKER_H

#include <memory>
#include <string>
#include <vector>

namespace llvm {

class Module;
class LLVMContext;
class StringRef;

/// This class provides the core functionality of linking in LLVM. It retains a
/// Module object which is the composite of the modules and libraries linked
/// into it. The composite Module can be retrieved via the getModule() method.
/// In this case the Linker still retains ownership of the Module. If the
/// releaseModule() method is used, the ownership of the Module is transferred
/// to the caller and the Linker object is only suitable for destruction.
/// The Linker can link Modules from memory. By default, the linker
/// will generate error and warning messages to stderr but this capability can
/// be turned off with the QuietWarnings and QuietErrors flags. It can also be
/// instructed to verbosely print out the linking actions it is taking with
/// the Verbose flag.
/// @brief The LLVM Linker.
class Linker {

  /// @name Types
  /// @{
  public:
    /// This enumeration is used to control various optional features of the
    /// linker.
    enum ControlFlags {
      Verbose       = 1, ///< Print to stderr what steps the linker is taking
      QuietWarnings = 2, ///< Don't print warnings to stderr.
      QuietErrors   = 4  ///< Don't print errors to stderr.
    };

    enum LinkerMode {
      DestroySource = 0, // Allow source module to be destroyed.
      PreserveSource = 1 // Preserve the source module.
    };

  /// @}
  /// @name Constructors
  /// @{
  public:
    /// Construct the Linker with an empty module which will be given the
    /// name \p progname. \p progname will also be used for error messages.
    /// @brief Construct with empty module
    Linker(StringRef progname, ///< name of tool running linker
           StringRef modulename, ///< name of linker's end-result module
           LLVMContext &C, ///< Context for global info
           unsigned Flags = 0  ///< ControlFlags (one or more |'d together)
    );

    /// Construct the Linker with a previously defined module, \p aModule. Use
    /// \p progname for the name of the program in error messages.
    /// @brief Construct with existing module
    Linker(StringRef progname, Module* aModule, unsigned Flags = 0);

    /// Destruct the Linker.
    /// @brief Destructor
    ~Linker();

  /// @}
  /// @name Accessors
  /// @{
  public:
    /// This method gets the composite module into which linking is being
    /// done. The Composite module starts out empty and accumulates modules
    /// linked into it via the various LinkIn* methods. This method does not
    /// release the Module to the caller. The Linker retains ownership and will
    /// destruct the Module when the Linker is destructed.
    /// @see releaseModule
    /// @brief Get the linked/composite module.
    Module* getModule() const { return Composite; }

    /// This method releases the composite Module into which linking is being
    /// done. Ownership of the composite Module is transferred to the caller who
    /// must arrange for its destruct. After this method is called, the Linker
    /// terminates the linking session for the returned Module. It will no
    /// longer utilize the returned Module but instead resets itself for
    /// subsequent linking as if the constructor had been called.
    /// @brief Release the linked/composite module.
    Module* releaseModule();

    /// This method returns an error string suitable for printing to the user.
    /// The return value will be empty unless an error occurred in one of the
    /// LinkIn* methods. In those cases, the LinkIn* methods will have returned
    /// true, indicating an error occurred. At most one error is retained so
    /// this function always returns the last error that occurred. Note that if
    /// the Quiet control flag is not set, the error string will have already
    /// been printed to stderr.
    /// @brief Get the text of the last error that occurred.
    const std::string &getLastError() const { return Error; }

  /// @}
  /// @name Mutators
  /// @{
  public:
    /// This method links the \p Src module into the Linker's Composite module
    /// by calling LinkModules.
    /// @see LinkModules
    /// @returns True if an error occurs, false otherwise.
    /// @brief Link in a module.
    bool LinkInModule(
      Module* Src,              ///< Module linked into \p Dest
      std::string* ErrorMsg = 0 /// Error/diagnostic string
    ) {
      return LinkModules(Composite, Src, Linker::DestroySource, ErrorMsg);
    }

    /// This is the heart of the linker. This method will take unconditional
    /// control of the \p Src module and link it into the \p Dest module. The
    /// \p Src module will be destructed or subsumed by this method. In either
    /// case it is not usable by the caller after this method is invoked. Only
    /// the \p Dest module will remain. The \p Src module is linked into the
    /// Linker's composite module such that types, global variables, functions,
    /// and etc. are matched and resolved.  If an error occurs, this function
    /// returns true and ErrorMsg is set to a descriptive message about the
    /// error.
    /// @returns True if an error occurs, false otherwise.
    /// @brief Generically link two modules together.
    static bool LinkModules(Module* Dest, Module* Src, unsigned Mode,
                            std::string* ErrorMsg);

  /// @}
  /// @name Implementation
  /// @{
  private:
    bool warning(StringRef message);
    bool error(StringRef message);
    void verbose(StringRef message);

  /// @}
  /// @name Data
  /// @{
  private:
    LLVMContext& Context; ///< The context for global information
    Module* Composite; ///< The composite module linked together
    unsigned Flags;    ///< Flags to control optional behavior.
    std::string Error; ///< Text of error that occurred.
    std::string ProgramName; ///< Name of the program being linked
  /// @}

};

} // End llvm namespace

#endif
