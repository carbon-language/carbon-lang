//==--- DiagnosticCommentKinds.td - diagnostics related to comments -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

let Component = "Comment" in {
let CategoryName = "Documentation Issue" in {

// HTML parsing errors.  These are under -Wdocumentation to make sure the user
// knows that we didn't parse something as he might expect.

def warn_doc_html_start_tag_expected_quoted_string : Warning<
  "expected quoted string after equals sign">,
  InGroup<Documentation>, DefaultIgnore;

def warn_doc_html_start_tag_expected_ident_or_greater : Warning<
  "HTML start tag prematurely ended, expected attribute name or '>'">,
  InGroup<Documentation>, DefaultIgnore;

def note_doc_html_tag_started_here : Note<
  "HTML tag started here">;

// HTML semantic errors

def warn_doc_html_end_forbidden : Warning<
  "HTML end tag '%0' is forbidden">,
  InGroup<DocumentationHTML>, DefaultIgnore;

def warn_doc_html_end_unbalanced : Warning<
  "HTML end tag does not match any start tag">,
  InGroup<DocumentationHTML>, DefaultIgnore;

def warn_doc_html_start_end_mismatch : Warning<
  "HTML start tag '%0' closed by '%1'">,
  InGroup<DocumentationHTML>, DefaultIgnore;

def note_doc_html_end_tag : Note<
  "end tag">;

// Commands

def warn_doc_block_command_empty_paragraph : Warning<
  "empty paragraph passed to '\\%0' command">,
  InGroup<Documentation>, DefaultIgnore;

// \param command

def warn_doc_param_invalid_direction : Warning<
  "unrecognized parameter passing direction, "
  "valid directions are '[in]', '[out]' and '[in,out]'">,
  InGroup<Documentation>, DefaultIgnore;

def warn_doc_param_spaces_in_direction : Warning<
  "whitespace is not allowed in parameter passing direction">,
  InGroup<DocumentationPedantic>, DefaultIgnore;

def warn_doc_param_not_attached_to_a_function_decl : Warning<
  "'\\param' command used in a comment that is not attached to "
  "a function declaration">,
  InGroup<Documentation>, DefaultIgnore;

def warn_doc_param_not_found : Warning<
  "parameter '%0' not found in the function declaration">,
  InGroup<Documentation>, DefaultIgnore;

def note_doc_param_name_suggestion : Note<
  "did you mean '%0'?">;

} // end of documentation issue category
} // end of AST component
