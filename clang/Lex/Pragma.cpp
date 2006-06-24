//===--- Pragma.cpp - Pragma registration and handling --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the PragmaHandler and PragmaTable interfaces.
//
//===----------------------------------------------------------------------===//

#include "clang/Lex/Pragma.h"
#include "clang/Lex/Preprocessor.h"
using namespace llvm;
using namespace clang;

// Out-of-line destructor to provide a home for the class.
PragmaHandler::~PragmaHandler() {
}

void PragmaNamespace::HandlePragma(Preprocessor &PP, LexerToken &Tok) {
  // Read the 'namespace' that the directive is in, e.g. STDC.  Do not macro
  // expand it, the user can have a STDC #define, that should not affect this.
  PP.LexUnexpandedToken(Tok);
  
  // Get the handler for this token.  If there is no handler, ignore the pragma.
  PragmaHandler *Handler = FindHandler(Tok.getIdentifierInfo(), false);
  if (Handler == 0) return;
  
  // Otherwise, pass it down.
  Handler->HandlePragma(PP, Tok);
}



#if 0
/* Contains a registered pragma or pragma namespace.  */
typedef void (*pragma_cb) (cpp_reader *);
struct pragma_entry
{
  struct pragma_entry *next;
  const cpp_hashnode *pragma;   /* Name and length.  */
  bool is_nspace;
  bool is_internal;
  bool is_deferred;
  bool allow_expansion;
  union {
    pragma_cb handler;
    struct pragma_entry *space;
    unsigned int ident;
  } u;
};



/* Register a pragma NAME in namespace SPACE.  If SPACE is null, it
goes in the global namespace.  HANDLER is the handler it will call,
which must be non-NULL.  If ALLOW_EXPANSION is set, allow macro
expansion while parsing pragma NAME.  This function is exported
from libcpp. */
void
cpp_register_pragma (cpp_reader *pfile, const char *space, const char *name,
                     pragma_cb handler, bool allow_expansion)
{
  struct pragma_entry *entry;
  
  if (!handler)
  {
    cpp_error (pfile, CPP_DL_ICE, "registering pragma with NULL handler");
    return;
  }
  
  entry = register_pragma_1 (pfile, space, name, false);
  if (entry)
  {
    entry->allow_expansion = allow_expansion;
    entry->u.handler = handler;
  }
}

/* Similarly, but create mark the pragma for deferred processing.
When found, a CPP_PRAGMA token will be insertted into the stream
with IDENT in the token->u.pragma slot.  */
void
cpp_register_deferred_pragma (cpp_reader *pfile, const char *space,
                              const char *name, unsigned int ident,
                              bool allow_expansion, bool allow_name_expansion)
{
  struct pragma_entry *entry;
  
  entry = register_pragma_1 (pfile, space, name, allow_name_expansion);
  if (entry)
  {
    entry->is_deferred = true;
    entry->allow_expansion = allow_expansion;
    entry->u.ident = ident;
  }
}  


/* Register the pragmas the preprocessor itself handles.  */
void
_cpp_init_internal_pragmas (cpp_reader *pfile)
{
  /* Pragmas in the global namespace.  */
  register_pragma_internal (pfile, 0, "once", do_pragma_once);
  
  /* New GCC-specific pragmas should be put in the GCC namespace.  */
  register_pragma_internal (pfile, "GCC", "poison", do_pragma_poison);
  register_pragma_internal (pfile, "GCC", "system_header",
                            do_pragma_system_header);
  register_pragma_internal (pfile, "GCC", "dependency", do_pragma_dependency);
}
#endif
