/* RUN: clang-cc -fsyntax-only -std=c90 -pedantic -fixit %s -o - | clang-cc -pedantic -x c -std=c90 -Werror -
 */

/* This is a test of the various code modification hints that are
   provided as part of warning or extension diagnostics. All of the
   warnings will be fixed by -fixit, and the resulting file should
   compile cleanly with -Werror -pedantic. */

enum e0 {
  e1,
};
